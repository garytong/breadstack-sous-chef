// Package toolgen builds the Anthropic Messages API tool definitions Sous
// Chef exposes to the model (README §2) and dispatches the server-side ones.
//
// The ten tools are generated per manifest hash in a fixed order, because the
// tool list is the first part of the cached prompt prefix: ToolsJSON is
// byte-stable for a given manifest. Data-op schemas are derived from the
// browser engine's op schema (cookie-cutter launcher/client/src/sousChef/
// ops/schema.js, mirrored in opspec.go) so the model's tool_use input IS the
// op the browser applies.
//
// Strictness follows the claude-api skill's structured-outputs rules: a strict
// schema needs `additionalProperties: false` on every object, a `required`
// array, and no numeric/string constraints. `set_fields` (`values`,
// `assets`) and `edit_list` (`item`) carry open maps keyed by manifest field
// keys, which strict mode cannot express, so those two are `strict: false`
// and the browser engine validates them. The API also caps the optional
// parameters of ALL strict tools in one request at 24 — every property not
// in `required`, counted through nested objects (the `target` object is 4
// per embedding) — and union-typed parameters (`anyOf`, `type: [..,"null"]`)
// at 16; exceeding either is a 400 before any token is billed (seen live on
// int 2026-10-04 at 29/24). `lookup_fields` and `describe_target`, whose
// members are all optional, are therefore `strict: false` too (Dispatch
// validates them and answers self-correcting errors); the six remaining
// tools are `strict: true` and spend 12 of the 24
// (TestStrictOptionalParameterBudget). `tool_choice` is always `auto`
// (forced tool use 400s on Claude Sonnet 5.5 / Opus 5.5), which the loop sets
// — not this package.
package toolgen

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"github.com/breadstack/sous-chef/internal/manifest"
)

// Tool is one Anthropic tool definition in wire shape.
type Tool struct {
	Name         string         `json:"name"`
	Description  string         `json:"description"`
	InputSchema  map[string]any `json:"input_schema"`
	Strict       bool           `json:"strict,omitempty"`
	CacheControl *CacheControl  `json:"cache_control,omitempty"`
}

// CacheControl marks a prompt-cache breakpoint.
type CacheControl struct {
	Type string `json:"type"`
}

// Runs says where a tool executes.
type Runs string

const (
	// RunsBrowser: emitted to the browser as a tool.call event; the engine applies it.
	RunsBrowser Runs = "browser"
	// RunsServer: answered inline by Dispatch.
	RunsServer Runs = "server"
	// RunsSuspend: suspends the turn until the merchant acts (pick_asset).
	RunsSuspend Runs = "suspend"
)

// Tool names, in the fixed emission order.
const (
	SetFields          = "set_fields"
	EditList           = "edit_list"
	Reset              = "reset"
	RestoreDefault     = "restore_default"
	SetPageLayout      = "set_page_layout"
	ResetPinnedSection = "reset_pinned_section"
	PickAsset          = "pick_asset"
	LookupFields       = "lookup_fields"
	DescribeTarget     = "describe_target"
	ReportUnsupported  = "report_unsupported"
)

// Order is the fixed tool order (part of the cached prefix).
var Order = []string{
	SetFields, EditList, Reset, RestoreDefault, SetPageLayout, ResetPinnedSection,
	PickAsset, LookupFields, DescribeTarget, ReportUnsupported,
}

var runsOf = map[string]Runs{
	SetFields: RunsBrowser, EditList: RunsBrowser, Reset: RunsBrowser, RestoreDefault: RunsBrowser,
	SetPageLayout: RunsBrowser, ResetPinnedSection: RunsBrowser,
	PickAsset:    RunsSuspend,
	LookupFields: RunsServer, DescribeTarget: RunsServer,
	// report_unsupported is answered by the server but also echoed to the
	// browser as a tool.call so the UI can show it (README design note).
	ReportUnsupported: RunsServer,
}

// RunsWhere classifies a tool name; ok is false for an unknown tool.
func RunsWhere(name string) (Runs, bool) {
	r, ok := runsOf[name]
	return r, ok
}

// EchoedToBrowser reports whether a server tool's call is also emitted as a
// tool.call event (report_unsupported only).
func EchoedToBrowser(name string) bool { return name == ReportUnsupported }

// Tools builds the tool list for a manifest.
func Tools(m *manifest.Manifest) []Tool {
	ctx := newSchemaContext(m)
	out := make([]Tool, 0, len(Order))
	for _, name := range Order {
		spec, ok := opSpecs[name]
		if !ok {
			panic("toolgen: no spec for " + name)
		}
		out = append(out, spec.build(name, ctx))
	}
	return out
}

// ToolsJSON renders Tools(m) as canonical JSON (sorted keys, no whitespace)
// so two builds from the same manifest are byte-identical.
func ToolsJSON(m *manifest.Manifest) ([]byte, error) {
	return manifest.Canonical(Tools(m))
}

// WithCacheBreakpoint returns a copy of tools with `cache_control: ephemeral`
// on the last one, the breakpoint the loop places after the tool list.
func WithCacheBreakpoint(tools []Tool) []Tool {
	out := make([]Tool, len(tools))
	copy(out, tools)
	for i := range out {
		out[i].CacheControl = nil
	}
	if len(out) > 0 {
		out[len(out)-1].CacheControl = &CacheControl{Type: "ephemeral"}
	}
	return out
}

// ─── Dispatch ────────────────────────────────────────────────────────────────

// Result is a server tool's tool_result content.
type Result struct {
	Content string
	IsError bool
}

// Dispatch answers a server tool call. handled is false when `name` is not a
// server tool (the loop then emits it to the browser). Errors go back to the
// model as `{"code","message"}` with is_error so it can self-correct; input
// is parsed with encoding/json only.
func Dispatch(m *manifest.Manifest, name string, input json.RawMessage) (res Result, handled bool) {
	switch name {
	case LookupFields:
		var req manifest.LookupRequest
		if err := unmarshalInput(input, &req); err != nil {
			return errResult("E_OP", err.Error()), true
		}
		out, err := m.LookupFields(req)
		if err != nil {
			return queryErr(err), true
		}
		return jsonResult(out), true
	case DescribeTarget:
		var req struct {
			Target    *manifest.Target `json:"target"`
			Container string           `json:"container"`
		}
		if err := unmarshalInput(input, &req); err != nil {
			return errResult("E_OP", err.Error()), true
		}
		out, err := m.DescribeTarget(req.Target, req.Container)
		if err != nil {
			return queryErr(err), true
		}
		return jsonResult(out), true
	case ReportUnsupported:
		var req struct {
			Reason     string `json:"reason"`
			ManualHint string `json:"manual_hint"`
		}
		if err := unmarshalInput(input, &req); err != nil {
			return errResult("E_OP", err.Error()), true
		}
		if strings.TrimSpace(req.Reason) == "" {
			return errResult("E_OP", "report_unsupported needs a reason"), true
		}
		return jsonResult(map[string]any{
			"ok":                true,
			"shown_to_merchant": true,
			"next":              "Tell the merchant in one or two sentences that this cannot be changed here and that they can make the change in the editor or contact Breadstack support. Do not invent contact details.",
		}), true
	}
	return Result{}, false
}

func unmarshalInput(input json.RawMessage, into any) error {
	if len(strings.TrimSpace(string(input))) == 0 {
		input = json.RawMessage(`{}`)
	}
	if err := json.Unmarshal(input, into); err != nil {
		return fmt.Errorf("invalid tool input: %v", err)
	}
	return nil
}

func jsonResult(v any) Result {
	b, err := json.Marshal(v)
	if err != nil {
		return errResult("E_OP", err.Error())
	}
	return Result{Content: string(b)}
}

func errResult(code, msg string) Result {
	b, _ := json.Marshal(map[string]string{"code": code, "message": msg})
	return Result{Content: string(b), IsError: true}
}

func queryErr(err error) Result {
	if qe, ok := err.(*manifest.QueryError); ok {
		return errResult(qe.Code, qe.Message)
	}
	return errResult("E_OP", err.Error())
}

// ─── Manifest-derived schema context ─────────────────────────────────────────

// schemaContext carries what the manifest contributes to the tool schemas:
// container paths (lookup_fields enum), page slugs, section type ids, panel
// names and the per-container list keys for edit_list's description.
type schemaContext struct {
	containers   []string
	pages        []string
	sectionTypes []string
	panels       []string
	listsByCont  string
	pinnedPages  []string
	pinnedIDs    []string
}

func newSchemaContext(m *manifest.Manifest) schemaContext {
	ctx := schemaContext{}
	if m != nil {
		ctx.containers = append([]string{"globalPanels"}, m.ContainerPaths()...)
		sort.Strings(ctx.containers)
		for p := range m.Pages {
			if p != manifest.CommonPageKey {
				ctx.pages = append(ctx.pages, p)
			}
		}
		ctx.pages = append(ctx.pages, "home", manifest.SEOPage)
		sort.Strings(ctx.pages)
		for s := range m.Sections {
			ctx.sectionTypes = append(ctx.sectionTypes, s)
		}
		sort.Strings(ctx.sectionTypes)
		for p := range m.GlobalPanels {
			ctx.panels = append(ctx.panels, p)
		}
		sort.Strings(ctx.panels)
		var lines []string
		m.EachContainer(func(path string, c *manifest.Container) {
			if len(c.Lists) == 0 {
				return
			}
			keys := make([]string, 0, len(c.Lists))
			for k := range c.Lists {
				keys = append(keys, k)
			}
			sort.Strings(keys)
			lines = append(lines, path+": "+strings.Join(keys, ", "))
		})
		ctx.listsByCont = strings.Join(lines, "; ")
	}
	for p := range manifest.PinnedByPage {
		ctx.pinnedPages = append(ctx.pinnedPages, p)
	}
	sort.Strings(ctx.pinnedPages)
	for _, p := range ctx.pinnedPages {
		ctx.pinnedIDs = append(ctx.pinnedIDs, manifest.PinnedByPage[p].ID)
	}
	return ctx
}
