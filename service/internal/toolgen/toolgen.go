// Package toolgen builds the Anthropic Messages API tool definitions Sous
// Chef exposes to the model (README §2) and dispatches the server-side ones.
//
// The eleven tools are generated per manifest hash in a fixed order, because the
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
// validates them and answers self-correcting errors); the seven remaining
// tools are `strict: true` and spend 14 of the 24
// (TestStrictOptionalParameterBudget; 11 before create_page's slug, status
// and in_nav, 12 before `keys` became required on reset / restore_default,
// 10 until report_unsupported gained `where`). `tool_choice` is always `auto` (forced tool use
// 400s on Claude Sonnet 5.5 / Opus 5.5), which the loop sets — not this
// package.
//
// Browser ops are applied by the engine, which is the authority on their
// shape — with two server-side exceptions (ValidateBrowserOp), refused
// before the loop emits them: a reset / restore_default without `keys`,
// because the engine's keyless form clears the WHOLE target (every field
// of a section, page or panel) and the model must never be able to reach
// it; and a create_page without a usable title, which the engine would
// refuse anyway.
package toolgen

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/breadstack/sous-chef/internal/fields"
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
	CreatePage         = "create_page"
	ResetPinnedSection = "reset_pinned_section"
	PickAsset          = "pick_asset"
	LookupFields       = "lookup_fields"
	DescribeTarget     = "describe_target"
	ReportUnsupported  = "report_unsupported"
)

// Order is the fixed tool order (part of the cached prefix).
var Order = []string{
	SetFields, EditList, Reset, RestoreDefault, SetPageLayout, CreatePage, ResetPinnedSection,
	PickAsset, LookupFields, DescribeTarget, ReportUnsupported,
}

var runsOf = map[string]Runs{
	SetFields: RunsBrowser, EditList: RunsBrowser, Reset: RunsBrowser, RestoreDefault: RunsBrowser,
	SetPageLayout: RunsBrowser, CreatePage: RunsBrowser, ResetPinnedSection: RunsBrowser,
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

// UnsupportedWhere is report_unsupported's `where` enum, shared contract 1
// with the launcher (cookie-cutter launcher/client/src/sousChef/nav/
// deepLinks.js WHERE, same order): the browser maps a value to a fixed
// label and link. Which requests take which value is taught in system[0]
// (prompt.Rules, "What Sous Chef cannot change").
var UnsupportedWhere = []string{"domain", "sites", "pages", "assets", "seo", "themes", "products", "categories", "brands", "support"}

// wherePlace names each `where` the way the launcher labels it (deepLinks.js
// TAB_LINKS / deepLinkFor), for the reply the model writes after the call.
var wherePlace = map[string]string{
	"domain":     "Site Settings › Domain",
	"sites":      "the Sites tab",
	"pages":      "the Pages tab",
	"assets":     "the Asset Library tab",
	"seo":        "the SEO tab",
	"themes":     "the Themes tab",
	"products":   "the Products tab",
	"categories": "the Categories tab",
	"brands":     "the Brands tab",
	"support":    "Breadstack support",
}

// whereDesc is the `where` member's description. The meaning of each value
// is taught once, in system[0] (prompt.Rules), not repeated here.
const whereDesc = "Optional: where the merchant makes this change — the place system rules name for each value (\"What Sous Chef cannot change\"). The browser shows a link there. Use \"support\" when no other value fits."

// NormalizeWhere returns v as a member of UnsupportedWhere (trimmed, lower
// case, like the browser's normalizeWhere); ok is false for anything else.
func NormalizeWhere(v any) (string, bool) {
	s, isStr := v.(string)
	if !isStr {
		return "", false
	}
	s = strings.ToLower(strings.TrimSpace(s))
	if _, ok := wherePlace[s]; !ok {
		return "", false
	}
	return s, true
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
		// The merchant vocabulary's readings of the query ("site name" →
		// "store name"), matched after the query itself.
		req.Readings = fields.QueryReadings(req.Query)
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
			Where      any    `json:"where"`
		}
		if err := unmarshalInput(input, &req); err != nil {
			return errResult("E_OP", err.Error()), true
		}
		if strings.TrimSpace(req.Reason) == "" {
			return errResult("E_OP", "report_unsupported needs a reason"), true
		}
		out := map[string]any{"ok": true, "shown_to_merchant": true}
		if where, ok := NormalizeWhere(req.Where); ok {
			out["where"] = where
			out["next"] = "Tell the merchant in one sentence that Sous Chef cannot change this here and that it is done in " + wherePlace[where] + " (the link is shown with your reply). Do not invent contact details."
		} else {
			// No `where`, or one outside the enum (a non-strict route): the
			// browser shows the reason without a link (deepLinks.js
			// normalizeWhere → null), so the reply names no place.
			out["next"] = "Tell the merchant in one or two sentences that this cannot be changed here and that they can make the change in the editor or contact Breadstack support. Do not invent contact details."
		}
		return jsonResult(out), true
	}
	return Result{}, false
}

// ─── Browser-op guard ────────────────────────────────────────────────────────

// ValidateBrowserOp is the server-side check a browser op passes before the
// loop emits it as a tool.call. It refuses — refused is true and res is the
// is_error tool_result for the model — a reset / restore_default whose
// `keys` is missing, not an array, empty, or holds anything but non-blank
// strings: the engine applies such a call to EVERY field of the target
// (validate.js validateResetLike, `op.keys === undefined`), and the strict
// schema (keys required, minItems 1) is belt; this is braces, so the clear-
// all form cannot reach the browser however the input arrived. It also
// refuses a create_page whose `title` is missing, not a string, blank or
// longer than MaxPageTitle characters (the engine's own limit; strict mode
// cannot carry maxLength), so a malformed create never reaches the editor.
// Every other tool passes untouched (the engine validates it).
func ValidateBrowserOp(name string, input json.RawMessage) (res Result, refused bool) {
	if name == CreatePage {
		return validateCreatePage(input)
	}
	if name != Reset && name != RestoreDefault {
		return Result{}, false
	}
	var req struct {
		Keys *json.RawMessage `json:"keys"`
	}
	if err := unmarshalInput(input, &req); err != nil {
		return errResult("E_OP", err.Error()), true
	}
	need := fmt.Sprintf("%s needs keys: list the field keys to %s (at least one, strings). Clearing a whole target is not available; nothing was changed.", name, map[string]string{Reset: "clear", RestoreDefault: "restore"}[name])
	if req.Keys == nil || strings.TrimSpace(string(*req.Keys)) == "null" {
		return errResult("E_OP", need), true
	}
	var keys []any
	if err := json.Unmarshal(*req.Keys, &keys); err != nil || len(keys) == 0 {
		return errResult("E_OP", need), true
	}
	for _, k := range keys {
		s, ok := k.(string)
		if !ok || strings.TrimSpace(s) == "" {
			return errResult("E_OP", need), true
		}
	}
	return Result{}, false
}

// MaxPageTitle is create_page's title limit in characters (after trimming),
// the launcher engine's (ops/validate.js create_page).
const MaxPageTitle = 80

// validateCreatePage refuses a create_page without a usable title.
func validateCreatePage(input json.RawMessage) (Result, bool) {
	var req struct {
		Title any `json:"title"`
	}
	if err := unmarshalInput(input, &req); err != nil {
		return errResult("E_OP", err.Error()), true
	}
	need := fmt.Sprintf("create_page needs a title: the new page's name as the merchant said it, plain text, at most %d characters. Nothing was created.", MaxPageTitle)
	title, ok := req.Title.(string)
	// Trimmed and whitespace runs collapsed, as the engine counts it
	// (validate.js normTitle, then code points).
	title = strings.Join(strings.Fields(title), " ")
	if !ok || title == "" || utf8.RuneCountInString(title) > MaxPageTitle {
		return errResult("E_VALUE", need), true
	}
	return Result{}, false
}

// PageOf is the page a browser op's input aims at by slug — target.page,
// page (set_page_layout / reset_pinned_section) or for.target.page
// (pick_asset) — or "" (none named: the viewed page).
func PageOf(input json.RawMessage) string {
	var in struct {
		Page   any `json:"page"`
		Target struct {
			Page any `json:"page"`
		} `json:"target"`
		For struct {
			Target struct {
				Page any `json:"page"`
			} `json:"target"`
		} `json:"for"`
	}
	if json.Unmarshal(input, &in) != nil {
		return ""
	}
	for _, p := range []any{in.Target.Page, in.Page, in.For.Target.Page} {
		if s, ok := p.(string); ok && s != "" {
			return s
		}
	}
	return ""
}

// CreatePageSlugs are the slugs a create_page call would have created: its
// slug as passed and Slugify(slug ‖ title), the engine's own rule. A batch
// stops later ops aimed at them when the create is refused (loop dispatch),
// so a follow-up never lands on a page that already holds the slug.
func CreatePageSlugs(input json.RawMessage) []string {
	var in struct {
		Title any `json:"title"`
		Slug  any `json:"slug"`
	}
	if json.Unmarshal(input, &in) != nil {
		return nil
	}
	var out []string
	given, _ := in.Slug.(string)
	if strings.TrimSpace(given) == "" {
		given = ""
	} else {
		out = append(out, given)
	}
	base := given
	if base == "" {
		base, _ = in.Title.(string)
	}
	if s := Slugify(base); s != "" && s != given {
		out = append(out, s)
	}
	return out
}

// Slugify mirrors the launcher's slugify (lib/pageSlug.js): lowercase, keep
// only a-z, 0-9, whitespace and hyphens, trim, whitespace runs → one
// hyphen, hyphen runs → one.
func Slugify(s string) string {
	var b strings.Builder
	for _, r := range strings.ToLower(s) {
		switch {
		case r >= 'a' && r <= 'z', r >= '0' && r <= '9', r == '-':
			b.WriteRune(r)
		case unicode.IsSpace(r) || r == '\ufeff':
			b.WriteRune(' ')
		}
	}
	out := strings.Join(strings.Fields(b.String()), "-")
	for strings.Contains(out, "--") {
		out = strings.ReplaceAll(out, "--", "-")
	}
	return out
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
