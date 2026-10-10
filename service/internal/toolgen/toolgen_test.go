package toolgen

import (
	"encoding/json"
	"os"
	"strings"
	"sync"
	"testing"

	"github.com/breadstack/sous-chef/internal/manifest"
)

var (
	snapOnce sync.Once
	snapM    *manifest.Manifest
	snapErr  error
)

func snapshot(t *testing.T) *manifest.Manifest {
	t.Helper()
	snapOnce.Do(func() {
		var data []byte
		data, snapErr = os.ReadFile("../../testdata/manifest.snapshot.json")
		if snapErr != nil {
			return
		}
		snapM, snapErr = manifest.Parse(data)
	})
	if snapErr != nil {
		t.Fatalf("snapshot: %v", snapErr)
	}
	return snapM
}

func toolByName(t *testing.T, tools []Tool, name string) Tool {
	t.Helper()
	for _, tl := range tools {
		if tl.Name == name {
			return tl
		}
	}
	t.Fatalf("tool %s missing", name)
	return Tool{}
}

func props(t *testing.T, schema map[string]any) map[string]any {
	t.Helper()
	p, ok := schema["properties"].(map[string]any)
	if !ok {
		t.Fatalf("schema has no properties: %v", schema)
	}
	return p
}

// walk visits every nested schema object (maps) under s.
func walk(s any, fn func(path string, m map[string]any)) {
	var rec func(path string, v any)
	rec = func(path string, v any) {
		switch x := v.(type) {
		case map[string]any:
			fn(path, x)
			for k, c := range x {
				rec(path+"."+k, c)
			}
		case []any:
			for i, c := range x {
				rec(path+"["+string(rune('0'+i))+"]", c)
			}
		}
	}
	rec("", s)
}

// ─── Tool list ───────────────────────────────────────────────────────────────

func TestToolsOrderAndNames(t *testing.T) {
	tools := Tools(snapshot(t))
	if len(tools) != 10 || len(Order) != 10 {
		t.Fatalf("len = %d", len(tools))
	}
	seen := map[string]bool{}
	for i, tl := range tools {
		if tl.Name != Order[i] {
			t.Errorf("tools[%d] = %s, want %s", i, tl.Name, Order[i])
		}
		if seen[tl.Name] {
			t.Errorf("duplicate tool %s", tl.Name)
		}
		seen[tl.Name] = true
		if tl.Description == "" || tl.InputSchema == nil || tl.CacheControl != nil {
			t.Errorf("%s: incomplete definition", tl.Name)
		}
		if _, ok := RunsWhere(tl.Name); !ok {
			t.Errorf("%s: no Runs classification", tl.Name)
		}
	}
	if Order[0] != SetFields || Order[9] != ReportUnsupported {
		t.Fatal("order must start with set_fields and end with report_unsupported (cached prefix)")
	}
}

func TestRunsClassification(t *testing.T) {
	want := map[string]Runs{
		SetFields: RunsBrowser, EditList: RunsBrowser, Reset: RunsBrowser, RestoreDefault: RunsBrowser,
		SetPageLayout: RunsBrowser, ResetPinnedSection: RunsBrowser, PickAsset: RunsSuspend,
		LookupFields: RunsServer, DescribeTarget: RunsServer, ReportUnsupported: RunsServer,
	}
	for name, w := range want {
		if r, ok := RunsWhere(name); !ok || r != w {
			t.Errorf("%s → %s,%v want %s", name, r, ok, w)
		}
	}
	if _, ok := RunsWhere("nope"); ok {
		t.Fatal("unknown tool must not classify")
	}
	if !EchoedToBrowser(ReportUnsupported) || EchoedToBrowser(LookupFields) {
		t.Fatal("only report_unsupported is echoed to the browser")
	}
}

func TestStrictness(t *testing.T) {
	tools := Tools(snapshot(t))
	// set_fields / edit_list carry open maps; lookup_fields / describe_target
	// are all-optional server queries that would spend 17 of the API's 24
	// strict optional parameters (TestStrictOptionalParameterBudget).
	nonStrict := map[string]bool{SetFields: true, EditList: true, LookupFields: true, DescribeTarget: true}
	for _, tl := range tools {
		if tl.Strict == nonStrict[tl.Name] {
			t.Errorf("%s: strict=%v", tl.Name, tl.Strict)
		}
	}
	for _, name := range []string{Reset, RestoreDefault, SetPageLayout, ResetPinnedSection, PickAsset, ReportUnsupported} {
		if !toolByName(t, tools, name).Strict {
			t.Errorf("%s must stay strict (data op / merchant-facing)", name)
		}
	}
}

// The Messages API caps strict schemas per request: at most 24 optional
// parameters (every property not listed in `required`, counted through nested
// objects, summed over ALL tools sent with strict:true and any JSON output
// schema) and at most 16 union-typed parameters (`anyOf` or a `type` array).
// Exceeding either is a 400 before any token is billed — the int launch on
// 2026-10-04 failed every call with "Schemas contains too many optional
// parameters (29) ... (limit: 24)", which is exactly what this counter gives
// for the pre-fix strict set (the nested target's 4 optionals × 4 tools made
// the difference between the 13 a flat count sees and the API's 29).
func TestStrictOptionalParameterBudget(t *testing.T) {
	const optionalLimit, unionLimit = 24, 16
	for _, m := range []*manifest.Manifest{snapshot(t), nil} {
		totalOpt, totalUnion := 0, 0
		for _, tl := range Tools(m) {
			if !tl.Strict {
				continue
			}
			opt, union := countStrictBudget(tl.InputSchema)
			t.Logf("%-22s optional=%2d union=%d", tl.Name, opt, union)
			totalOpt += opt
			totalUnion += union
		}
		t.Logf("strict set: optional=%d/%d union=%d/%d (manifest=%v)", totalOpt, optionalLimit, totalUnion, unionLimit, m != nil)
		if totalOpt > optionalLimit {
			t.Fatalf("strict tools carry %d optional parameters, API limit %d", totalOpt, optionalLimit)
		}
		if totalUnion > unionLimit {
			t.Fatalf("strict tools carry %d union-typed parameters, API limit %d", totalUnion, unionLimit)
		}
		// Headroom: a target member costs one per strict tool embedding it
		// (two today), so keep at least that much room for the next one.
		if optionalLimit-totalOpt < 4 {
			t.Errorf("only %d optional parameters of headroom left", optionalLimit-totalOpt)
		}
	}
	// Regression pin: counted over the pre-fix strict set (everything but the
	// two open-map tools), the counter must give the 29 the API reported.
	// `keys` on reset / restore_default was optional then (one each) and is
	// required now, so those two are added back before comparing; the
	// optional `where` on report_unsupported (2026-10-10, shared contract 1)
	// did not exist then, so it is taken out.
	preFix := 0
	for _, tl := range Tools(snapshot(t)) {
		if tl.Name != SetFields && tl.Name != EditList {
			opt, _ := countStrictBudget(tl.InputSchema)
			preFix += opt
		}
	}
	if preFix+2-1 != 29 {
		t.Fatalf("pre-fix strict set counts %d (+2 for the then-optional keys) optional parameters, the API reported 29 — counter drifted from the API's rule", preFix)
	}
	// The strict set today: 11 (two target embeddings = 8, set_page_layout's
	// page, report_unsupported's manual_hint and where). 13 of headroom.
	today := 0
	for _, tl := range Tools(snapshot(t)) {
		if tl.Strict {
			opt, _ := countStrictBudget(tl.InputSchema)
			today += opt
		}
	}
	if today != 11 {
		t.Fatalf("strict set counts %d optional parameters, want 11", today)
	}
	// The counter itself, on the shape the API rejected: a nested object's
	// optionals count, required members do not, anyOf is one union.
	opt, union := countStrictBudget(map[string]any{
		"type": "object", "required": []string{"a"},
		"properties": map[string]any{
			"a": map[string]any{"type": "object", "required": []string{"k"}, "properties": map[string]any{
				"k": map[string]any{"type": "string"},
				"x": map[string]any{"type": "string"},
				"y": map[string]any{"type": []any{"string", "null"}},
			}},
			"b": map[string]any{"anyOf": []any{map[string]any{"type": "integer"}, map[string]any{"type": "array", "items": map[string]any{"type": "object", "properties": map[string]any{"z": map[string]any{"type": "string"}}, "required": []any{}}}}},
		},
	})
	if opt != 4 || union != 2 {
		t.Fatalf("counter: optional=%d union=%d, want 4 and 2", opt, union)
	}
}

// countStrictBudget counts, the way the API does, the optional parameters
// (properties not in `required`, recursively through properties, items and
// anyOf/allOf branches) and the union-typed parameters (a property with
// `anyOf` or a `type` array) of one schema.
func countStrictBudget(schema map[string]any) (optional, union int) {
	var rec func(m map[string]any)
	rec = func(m map[string]any) {
		if p, ok := m["properties"].(map[string]any); ok {
			req := map[string]bool{}
			switch r := m["required"].(type) {
			case []string:
				for _, k := range r {
					req[k] = true
				}
			case []any:
				for _, k := range r {
					if s, ok := k.(string); ok {
						req[s] = true
					}
				}
			}
			for name, v := range p {
				ps, _ := v.(map[string]any)
				if !req[name] {
					optional++
				}
				if _, isUnion := ps["anyOf"]; isUnion {
					union++
				} else if _, typeArr := ps["type"].([]any); typeArr {
					union++
				} else if _, typeArr := ps["type"].([]string); typeArr {
					union++
				}
				rec(ps)
			}
		}
		if items, ok := m["items"].(map[string]any); ok {
			rec(items)
		}
		for _, k := range []string{"anyOf", "allOf", "oneOf"} {
			if branches, ok := m[k].([]any); ok {
				for _, b := range branches {
					if bm, ok := b.(map[string]any); ok {
						rec(bm)
					}
				}
			}
		}
	}
	rec(schema)
	return optional, union
}

// Strict schemas must satisfy the structured-outputs rules: every object has
// additionalProperties:false and a required array; no numeric / string
// constraints anywhere; the only array constraint is minItems 0 or 1
// (platform.claude.com/docs/en/build-with-claude/structured-outputs:
// "Array minItems (only values 0 and 1 supported)"); additionalProperties
// never anything but false.
func TestStrictSchemasObeyStructuredOutputRules(t *testing.T) {
	for _, tl := range Tools(snapshot(t)) {
		walk(tl.InputSchema, func(path string, m map[string]any) {
			for _, bad := range []string{"minimum", "maximum", "multipleOf", "minLength", "maxLength", "maxItems", "uniqueItems", "pattern", "$ref"} {
				if _, ok := m[bad]; ok {
					t.Errorf("%s%s: unsupported constraint %s", tl.Name, path, bad)
				}
			}
			if mi, ok := m["minItems"]; ok && mi != 0 && mi != 1 {
				t.Errorf("%s%s: minItems %v — strict mode accepts only 0 or 1", tl.Name, path, mi)
			}
			if m["type"] != "object" {
				return
			}
			if !tl.Strict {
				return
			}
			if ap, ok := m["additionalProperties"]; !ok || ap != false {
				t.Errorf("%s%s: strict object needs additionalProperties:false, got %v", tl.Name, path, ap)
			}
			if _, ok := m["required"].([]string); !ok {
				t.Errorf("%s%s: strict object needs a required array, got %T", tl.Name, path, m["required"])
			}
			if _, ok := m["properties"].(map[string]any); !ok {
				t.Errorf("%s%s: strict object needs properties", tl.Name, path)
			}
		})
	}
}

func TestEveryPropertyHasAType(t *testing.T) {
	for _, tl := range Tools(snapshot(t)) {
		walk(tl.InputSchema, func(path string, m map[string]any) {
			p, ok := m["properties"].(map[string]any)
			if !ok {
				return
			}
			req, _ := m["required"].([]string)
			for _, r := range req {
				if _, ok := p[r]; !ok {
					t.Errorf("%s%s: required %q is not a property", tl.Name, path, r)
				}
			}
			for name, v := range p {
				ps := v.(map[string]any)
				_, hasType := ps["type"]
				_, hasAnyOf := ps["anyOf"]
				if !hasType && !hasAnyOf {
					t.Errorf("%s%s.%s: property without type/anyOf", tl.Name, path, name)
				}
				if d, _ := ps["description"].(string); d == "" {
					t.Errorf("%s%s.%s: property without description", tl.Name, path, name)
				}
			}
		})
	}
}

func TestToolsJSONIsByteStable(t *testing.T) {
	m := snapshot(t)
	a, err := ToolsJSON(m)
	if err != nil {
		t.Fatal(err)
	}
	b, _ := ToolsJSON(m)
	if string(a) != string(b) {
		t.Fatal("ToolsJSON differs between two builds of the same manifest")
	}
	data, _ := os.ReadFile("../../testdata/manifest.snapshot.json")
	m2, _ := manifest.Parse(data)
	c, _ := ToolsJSON(m2)
	if string(a) != string(c) {
		t.Fatal("ToolsJSON differs between two parses of the same document")
	}
	t.Logf("tools JSON: %d bytes (%d tokens at 4 B/t)", len(a), len(a)/4)
	if len(a) > 40*1024 {
		t.Fatalf("tool list is %d bytes; keep the cached prefix small", len(a))
	}
	// Canonical: sorted keys, no whitespace, valid JSON.
	var decoded []map[string]any
	if err := json.Unmarshal(a, &decoded); err != nil || len(decoded) != 10 {
		t.Fatalf("ToolsJSON not a 10-tool array: %v", err)
	}
	if strings.Contains(string(a), "\n") || !strings.HasPrefix(string(a), `[{"description":`) {
		t.Fatalf("ToolsJSON not canonical: %.80s", a)
	}
	if strings.Contains(string(a), `"strict":false`) {
		t.Fatal("strict:false must be omitted")
	}
}

func TestToolsDependOnManifest(t *testing.T) {
	m := snapshot(t)
	small, err := manifest.Parse([]byte(`{"version":1,"sections":{"zzNew":{"editorKey":"z","pages":["home"],"fields":{"a":{"path":"settings.a","label":"A","type":"text","source":"s"}},"lists":{"rows":{"path":"settings.rows","item":{},"ops":["add"]}},"structural":[]}},"pages":{"custom-thing":{"fields":{}}}}`))
	if err != nil {
		t.Fatal(err)
	}
	a, _ := ToolsJSON(m)
	b, _ := ToolsJSON(small)
	if string(a) == string(b) {
		t.Fatal("different manifests must yield different tool lists")
	}
	lf := toolByName(t, Tools(small), LookupFields)
	enum := props(t, lf.InputSchema)["container"].(map[string]any)["enum"].([]any)
	joined := make([]string, len(enum))
	for i, e := range enum {
		joined[i] = e.(string)
	}
	if strings.Join(joined, ",") != "branding,globalPanels,pages.custom-thing,sections.zzNew" {
		t.Fatalf("container enum = %v", joined)
	}
	el := toolByName(t, Tools(small), EditList)
	if !strings.Contains(el.Description, "sections.zzNew: rows") || !strings.Contains(el.Description, "Section types for sections.add: zzNew") {
		t.Fatalf("edit_list description = %s", el.Description)
	}
	spl := toolByName(t, Tools(small), SetPageLayout)
	if !strings.Contains(props(t, spl.InputSchema)["page"].(map[string]any)["description"].(string), "custom-thing") {
		t.Fatal("set_page_layout page description should list manifest pages")
	}
	// A nil manifest still yields a well-formed (manifest-free) tool list.
	if n := Tools(nil); len(n) != 10 {
		t.Fatal("Tools(nil)")
	}
}

func TestTargetSchema(t *testing.T) {
	tools := Tools(snapshot(t))
	for _, name := range []string{SetFields, EditList, Reset, RestoreDefault, LookupFields, DescribeTarget} {
		tl := toolByName(t, tools, name)
		tgt := props(t, tl.InputSchema)["target"].(map[string]any)
		if tgt["type"] != "object" || tgt["additionalProperties"] != false {
			t.Errorf("%s: target is not a closed object", name)
		}
		tp := props(t, tgt)
		for _, k := range []string{"kind", "page", "sectionId", "storeId", "panel"} {
			if _, ok := tp[k]; !ok {
				t.Errorf("%s: target lacks %s", name, k)
			}
		}
		kinds := tp["kind"].(map[string]any)["enum"].([]any)
		if len(kinds) != 4 || kinds[0] != "section" || kinds[3] != "page" {
			t.Errorf("%s: kind enum = %v", name, kinds)
		}
		req := tgt["required"].([]string)
		if len(req) != 1 || req[0] != "kind" {
			t.Errorf("%s: target required = %v", name, req)
		}
		if !strings.Contains(tp["page"].(map[string]any)["description"].(string), "Omit") {
			t.Errorf("%s: page description must say it may be omitted for the viewed page", name)
		}
		if !strings.Contains(tp["panel"].(map[string]any)["description"].(string), "HeaderSection") {
			t.Errorf("%s: panel description should list the panels", name)
		}
	}
}

func TestDataOpSchemasMirrorEngineOps(t *testing.T) {
	tools := Tools(snapshot(t))
	sf := toolByName(t, tools, SetFields)
	if req := sf.InputSchema["required"].([]string); strings.Join(req, ",") != "target,values" {
		t.Fatalf("set_fields required = %v", req)
	}
	if _, ok := props(t, sf.InputSchema)["assets"]; !ok {
		t.Fatal("set_fields lacks assets")
	}
	el := toolByName(t, tools, EditList)
	ep := props(t, el.InputSchema)
	if req := el.InputSchema["required"].([]string); strings.Join(req, ",") != "target,list,action" {
		t.Fatalf("edit_list required = %v", req)
	}
	actions := ep["action"].(map[string]any)["enum"].([]any)
	if len(actions) != len(manifest.ListActions) {
		t.Fatalf("actions = %v", actions)
	}
	for i, a := range manifest.ListActions {
		if actions[i] != a {
			t.Fatalf("actions = %v", actions)
		}
	}
	idx := ep["index"].(map[string]any)["anyOf"].([]any)
	if len(idx) != 2 || idx[0].(map[string]any)["type"] != "integer" || idx[1].(map[string]any)["type"] != "array" {
		t.Fatalf("index = %v", idx)
	}
	for _, k := range []string{"item", "order", "sectionId"} {
		if _, ok := ep[k]; !ok {
			t.Fatalf("edit_list lacks %s", k)
		}
	}
	// reset / restore_default: keys REQUIRED and non-empty — the engine's
	// keyless form clears the whole target and must not be offered.
	for _, name := range []string{Reset, RestoreDefault} {
		tl := toolByName(t, tools, name)
		if req := tl.InputSchema["required"].([]string); strings.Join(req, ",") != "target,keys" {
			t.Fatalf("%s required = %v, want target,keys", name, req)
		}
		keys := props(t, tl.InputSchema)["keys"].(map[string]any)
		if keys["type"] != "array" || keys["items"].(map[string]any)["type"] != "string" || keys["minItems"] != 1 {
			t.Fatalf("%s keys = %v (want array of string, minItems 1)", name, keys)
		}
		for _, bad := range []string{"Omit", "omit", "whole target is", "every field of the target is"} {
			if strings.Contains(tl.Description, bad) || strings.Contains(keys["description"].(string), bad) {
				t.Errorf("%s still describes the keyless clear-all form (%q): %s | %s", name, bad, tl.Description, keys["description"])
			}
		}
		for _, want := range []string{"not available", "list every key", "E_OP"} {
			if !strings.Contains(tl.Description, want) {
				t.Errorf("%s description must say %q: %s", name, want, tl.Description)
			}
		}
	}
	if !strings.Contains(toolByName(t, tools, Reset).Description, "unset_value") || !strings.Contains(toolByName(t, tools, RestoreDefault).Description, "default") {
		t.Fatal("reset / restore_default descriptions must say what they write")
	}
	spl := toolByName(t, tools, SetPageLayout)
	if req := spl.InputSchema["required"].([]string); strings.Join(req, ",") != "layout" {
		t.Fatalf("set_page_layout required = %v (page is optional = viewed page)", req)
	}
}

// int-run2 (300 cases): the only E_ORDER refusals were a partial order ([1, 0]
// on a 3-row footerCustomColumns[].links, "permutation of 0..2") and indexes
// on the id-keyed filterOrder ("unknown id(s) 1, 0"). The `order` member must
// state both rules the engine enforces (validate.js reorder / orderIdsError).
func TestEditListOrderDescription(t *testing.T) {
	for _, m := range []*manifest.Manifest{snapshot(t), nil} {
		el := toolByName(t, Tools(m), EditList)
		order := props(t, el.InputSchema)["order"].(map[string]any)
		desc := order["description"].(string)
		for _, want := range []string{
			"EVERY current row exactly once", // complete permutation
			"3-row list needs 3 entries",     // the run-2 shape, concretely
			"[1, 0, 2]",                      // move one row = full order
			"NEVER indexes",                  // id-keyed lists
			"`orders`",                       // how the manifest marks them (describe_target / lookup_fields)
			"describe_target",
			"filterOrder",                   // the list run-2 broke
			"sections list",                 // the page's own list is id-keyed too
			`["brand", "category", "price"`, // an id array, spelled out
			"E_ORDER",
		} {
			if !strings.Contains(desc, want) {
				t.Errorf("edit_list.order must say %q: %s", want, desc)
			}
		}
		// The shape is unchanged: an array of integer | string (indexes or ids).
		items := order["items"].(map[string]any)["anyOf"].([]any)
		if len(items) != 2 || items[0].(map[string]any)["type"] != "integer" || items[1].(map[string]any)["type"] != "string" {
			t.Fatalf("order items = %v", items)
		}
		if !strings.Contains(el.Description, "`reorder` writes the COMPLETE new order (see `order`)") {
			t.Errorf("edit_list description must point reorder at `order`: %s", el.Description)
		}
	}
}

func TestResetPinnedSectionEnums(t *testing.T) {
	tl := toolByName(t, Tools(snapshot(t)), ResetPinnedSection)
	p := props(t, tl.InputSchema)
	pages := p["page"].(map[string]any)["enum"].([]any)
	ids := p["sectionId"].(map[string]any)["enum"].([]any)
	if len(pages) != len(manifest.PinnedByPage) || len(ids) != len(manifest.PinnedByPage) {
		t.Fatalf("pages %d ids %d", len(pages), len(ids))
	}
	have := map[string]bool{}
	for _, id := range ids {
		have[id.(string)] = true
	}
	for page, def := range manifest.PinnedByPage {
		if !have[def.ID] {
			t.Errorf("pinned id for %s missing: %s", page, def.ID)
		}
	}
	if req := tl.InputSchema["required"].([]string); strings.Join(req, ",") != "page,sectionId" {
		t.Fatalf("required = %v", req)
	}
}

// {kind:"branding"} is the age gate, location picker and certifications;
// the editor's Branding panel is global BrandingSection (real turn
// 2026-10-06: "the name of my site is HashiTong" went to branding).
func TestTargetKindSaysWhatBrandingIs(t *testing.T) {
	kind := props(t, targetSchema(newSchemaContext(snapshot(t))))["kind"].(map[string]any)["description"].(string)
	for _, want := range []string{"branding = ONLY age gate, location picker, certifications", "Branding panel BrandingSection: store name, logo, favicon"} {
		if !strings.Contains(kind, want) {
			t.Errorf("target.kind description lacks %q: %s", want, kind)
		}
	}
}

func TestPickAssetAndReportUnsupportedSchemas(t *testing.T) {
	tools := Tools(snapshot(t))
	pa := toolByName(t, tools, PickAsset)
	at := props(t, pa.InputSchema)["asset_type"].(map[string]any)["enum"].([]any)
	if len(at) != 1 || at[0] != "image" {
		t.Fatalf("asset_type = %v", at)
	}
	if req := pa.InputSchema["required"].([]string); strings.Join(req, ",") != "purpose,asset_type" {
		t.Fatalf("pick_asset required = %v", req)
	}
	if !strings.Contains(pa.Description, "ALONE") {
		t.Fatal("pick_asset must tell the model to call it alone")
	}
	ru := toolByName(t, tools, ReportUnsupported)
	if req := ru.InputSchema["required"].([]string); strings.Join(req, ",") != "reason" {
		t.Fatalf("report_unsupported required = %v", req)
	}
	if _, ok := props(t, ru.InputSchema)["manual_hint"]; !ok {
		t.Fatal("manual_hint missing")
	}
	// Shared contract 1: an optional, closed `where` enum in the browser's
	// order (deepLinks.js WHERE).
	where, ok := props(t, ru.InputSchema)["where"].(map[string]any)
	if !ok {
		t.Fatal("where missing")
	}
	var got []string
	for _, v := range where["enum"].([]any) {
		got = append(got, v.(string))
	}
	if strings.Join(got, ",") != "domain,sites,pages,assets,seo,themes,products,categories,brands,support" || where["type"] != "string" {
		t.Fatalf("where = %v", where)
	}
	for _, w := range UnsupportedWhere {
		if wherePlace[w] == "" {
			t.Errorf("where %q has no place name", w)
		}
	}
	if len(wherePlace) != len(UnsupportedWhere) {
		t.Errorf("wherePlace has %d entries for %d where values", len(wherePlace), len(UnsupportedWhere))
	}
	lf := toolByName(t, tools, LookupFields)
	lp := props(t, lf.InputSchema)
	for _, k := range []string{"target", "container", "query", "keys", "list", "limit", "help"} {
		if _, ok := lp[k]; !ok {
			t.Errorf("lookup_fields lacks %s", k)
		}
	}
	if lp["limit"].(map[string]any)["type"] != "integer" || lp["help"].(map[string]any)["type"] != "boolean" {
		t.Fatal("lookup_fields limit/help types")
	}
	if len(lf.InputSchema["required"].([]string)) != 0 {
		t.Fatal("lookup_fields has no mandatory member (target or container)")
	}
}

func TestWithCacheBreakpoint(t *testing.T) {
	tools := Tools(snapshot(t))
	marked := WithCacheBreakpoint(tools)
	for i, tl := range marked {
		last := i == len(marked)-1
		if (tl.CacheControl != nil) != last {
			t.Errorf("tools[%d] cache_control = %v", i, tl.CacheControl)
		}
		if last && tl.CacheControl.Type != "ephemeral" {
			t.Error("breakpoint type")
		}
	}
	if tools[len(tools)-1].CacheControl != nil {
		t.Fatal("WithCacheBreakpoint must not mutate its input")
	}
	b, _ := json.Marshal(marked[len(marked)-1])
	if !strings.Contains(string(b), `"cache_control":{"type":"ephemeral"}`) {
		t.Fatalf("wire shape: %s", b)
	}
	if len(WithCacheBreakpoint(nil)) != 0 {
		t.Fatal("empty input")
	}
}

func TestToolWireShape(t *testing.T) {
	tl := toolByName(t, Tools(snapshot(t)), Reset)
	b, _ := json.Marshal(tl)
	var wire map[string]any
	if err := json.Unmarshal(b, &wire); err != nil {
		t.Fatal(err)
	}
	for _, k := range []string{"name", "description", "input_schema", "strict"} {
		if _, ok := wire[k]; !ok {
			t.Errorf("wire lacks %s: %s", k, b)
		}
	}
	if _, ok := wire["cache_control"]; ok {
		t.Error("cache_control must be omitted when unset")
	}
	if wire["strict"] != true {
		t.Error("reset must be strict on the wire")
	}
}

// ─── Dispatch ────────────────────────────────────────────────────────────────

func TestDispatchLookupFields(t *testing.T) {
	m := snapshot(t)
	res, handled := Dispatch(m, LookupFields, json.RawMessage(`{"target":{"kind":"section","page":"home","sectionId":"heroBanner-1"},"query":"headline","limit":3}`))
	if !handled || res.IsError {
		t.Fatalf("res = %+v handled=%v", res, handled)
	}
	var out manifest.LookupResult
	if err := json.Unmarshal([]byte(res.Content), &out); err != nil {
		t.Fatal(err)
	}
	if len(out.Fields) != 3 || out.Fields[0].Key != "headline" || out.Fields[0].List != "slides" || !out.Truncated {
		t.Fatalf("out = %+v", out)
	}
	if strings.Contains(res.Content, `"path"`) || strings.Contains(res.Content, `"source"`) {
		t.Fatal("compact output must not carry path/source")
	}
}

// lookup_fields reads the query in the merchant's words too: the
// vocabulary's readings ("site name" → "store name"); the model cannot set
// them.
func TestDispatchLookupFieldsReadsTheVocabulary(t *testing.T) {
	m := snapshot(t)
	res, handled := Dispatch(m, LookupFields, json.RawMessage(`{"container":"globalPanels","query":"site name","Readings":["favicon"]}`))
	if !handled || res.IsError {
		t.Fatalf("res = %+v handled=%v", res, handled)
	}
	var out manifest.LookupResult
	if err := json.Unmarshal([]byte(res.Content), &out); err != nil {
		t.Fatal(err)
	}
	found := false
	for _, f := range out.Fields {
		found = found || (f.Key == "storeName" && f.Panel == "BrandingSection")
		if f.Key == "favicon" {
			t.Errorf("a reading the input carried was used: %s", f.Key)
		}
	}
	if !found {
		t.Fatalf("storeName not found: %s", res.Content)
	}
}

// "rename category": the vocabulary reads "rename" as name / label (51
// fields match through that reading alone); the keyed category name, whose
// prefix the query names in its own words, comes first (verify pass,
// 2026-10-10: it was not among the 40 returned).
func TestDispatchLookupRenameCategoryFindsTheKeyedName(t *testing.T) {
	m := snapshot(t)
	for _, in := range []string{`{"target":{"kind":"global"},"query":"rename category"}`, `{"container":"globalPanels","query":"rename a category"}`} {
		res, handled := Dispatch(m, LookupFields, json.RawMessage(in))
		if !handled || res.IsError {
			t.Fatalf("res = %+v handled=%v", res, handled)
		}
		var out manifest.LookupResult
		if err := json.Unmarshal([]byte(res.Content), &out); err != nil {
			t.Fatal(err)
		}
		if len(out.Fields) == 0 || out.Fields[0].Key != "categories.<categorySlug>.name" {
			first := ""
			if len(out.Fields) > 0 {
				first = out.Fields[0].Key
			}
			t.Errorf("%s: first field %q, want categories.<categorySlug>.name", in, first)
		}
	}
}

func TestDispatchErrorsAreSelfCorrecting(t *testing.T) {
	m := snapshot(t)
	res, handled := Dispatch(m, LookupFields, json.RawMessage(`{"target":{"kind":"section","page":"home","sectionId":"nope"}}`))
	if !handled || !res.IsError || !strings.Contains(res.Content, `"code":"E_TARGET"`) || !strings.Contains(res.Content, "heroBanner") {
		t.Fatalf("res = %+v", res)
	}
	res, _ = Dispatch(m, LookupFields, json.RawMessage(`{not json`))
	if !res.IsError || !strings.Contains(res.Content, `"code":"E_OP"`) {
		t.Fatalf("invalid json → %+v", res)
	}
	res, _ = Dispatch(m, LookupFields, json.RawMessage(`{}`))
	if !res.IsError || !strings.Contains(res.Content, `"code":"E_OP"`) {
		t.Fatalf("empty → %+v", res)
	}
	res, _ = Dispatch(m, LookupFields, json.RawMessage(`{"target":{"kind":"page","page":"faq"},"list":"zzz"}`))
	if !res.IsError || !strings.Contains(res.Content, `"code":"E_LIST"`) {
		t.Fatalf("bad list → %+v", res)
	}
	res, _ = Dispatch(m, DescribeTarget, nil)
	if !res.IsError {
		t.Fatalf("nil input → %+v", res)
	}
}

func TestDispatchDescribeTarget(t *testing.T) {
	m := snapshot(t)
	res, handled := Dispatch(m, DescribeTarget, json.RawMessage(`{"target":{"kind":"page","page":"products"}}`))
	if !handled || res.IsError {
		t.Fatalf("res = %+v", res)
	}
	var out manifest.DescribeResult
	if err := json.Unmarshal([]byte(res.Content), &out); err != nil {
		t.Fatal(err)
	}
	if out.Pinned == nil || out.Pinned.ID != "__pinned_product_list" || out.Lists[0].Key != "sections" {
		t.Fatalf("out = %+v", out)
	}
	res, _ = Dispatch(m, DescribeTarget, json.RawMessage(`{"container":"globalPanels.ProductCardSection"}`))
	if res.IsError || !strings.Contains(res.Content, `"presets"`) {
		t.Fatalf("container form → %+v", res)
	}
}

func TestDispatchReportUnsupported(t *testing.T) {
	m := snapshot(t)
	res, handled := Dispatch(m, ReportUnsupported, json.RawMessage(`{"reason":"There is no setting for that."}`))
	if !handled || res.IsError || !strings.Contains(res.Content, `"shown_to_merchant":true`) || !strings.Contains(res.Content, "Do not invent contact details") {
		t.Fatalf("res = %+v", res)
	}
	if strings.Contains(res.Content, `"where"`) {
		t.Fatalf("no where in, none out: %s", res.Content)
	}
	res, _ = Dispatch(m, ReportUnsupported, json.RawMessage(`{"reason":"  "}`))
	if !res.IsError {
		t.Fatal("empty reason must be an error")
	}
	// `where` comes back in the result with the place the reply names.
	res, _ = Dispatch(m, ReportUnsupported, json.RawMessage(`{"reason":"Pages are added in the Pages tab.","where":"pages","manual_hint":"Pages › New page"}`))
	var out struct {
		OK    bool   `json:"ok"`
		Where string `json:"where"`
		Next  string `json:"next"`
	}
	if res.IsError || json.Unmarshal([]byte(res.Content), &out) != nil || !out.OK || out.Where != "pages" || !strings.Contains(out.Next, "the Pages tab") || !strings.Contains(out.Next, "one sentence") {
		t.Fatalf("where=pages → %+v", res)
	}
	res, _ = Dispatch(m, ReportUnsupported, json.RawMessage(`{"reason":"x","where":" Domain "}`))
	if !strings.Contains(res.Content, `"where":"domain"`) || !strings.Contains(res.Content, "Site Settings › Domain") {
		t.Fatalf("where is normalised like the browser's: %s", res.Content)
	}
	// A value outside the enum (a non-strict route) is not an error: the
	// browser shows no link for it, so the result names no place.
	for _, bad := range []string{`"nowhere"`, `3`, `null`, `""`} {
		res, _ = Dispatch(m, ReportUnsupported, json.RawMessage(`{"reason":"x","where":`+bad+`}`))
		if res.IsError || strings.Contains(res.Content, `"where"`) || !strings.Contains(res.Content, "contact Breadstack support") {
			t.Fatalf("where=%s → %+v", bad, res)
		}
	}
}

// ValidateBrowserOp refuses the keyless (clear-all) shape of reset /
// restore_default in every spelling and lets everything else through.
func TestValidateBrowserOp(t *testing.T) {
	target := `"target":{"kind":"section","page":"home","sectionId":"hero-1"}`
	refused := []string{
		`{` + target + `}`,
		`{` + target + `,"keys":null}`,
		`{` + target + `,"keys":[]}`,
		`{` + target + `,"keys":"headline"}`,
		`{` + target + `,"keys":[""]}`,
		`{` + target + `,"keys":["headline", 3]}`,
		`{` + target + `,"keys":[" "]}`,
		``,
		`not json`,
	}
	for _, name := range []string{Reset, RestoreDefault} {
		for _, in := range refused {
			res, ok := ValidateBrowserOp(name, json.RawMessage(in))
			if !ok || !res.IsError {
				t.Errorf("%s %s: must be refused, got %+v", name, in, res)
				continue
			}
			var e struct{ Code, Message string }
			if err := json.Unmarshal([]byte(res.Content), &e); err != nil || e.Code != "E_OP" {
				t.Errorf("%s %s: result %s", name, in, res.Content)
			}
			if in != `` && in != `not json` && !strings.Contains(e.Message, name+" needs keys") {
				t.Errorf("%s %s: message %q", name, in, e.Message)
			}
		}
		if res, ok := ValidateBrowserOp(name, json.RawMessage(`{`+target+`,"keys":["headline","subheadline"]}`)); ok {
			t.Errorf("%s with keys must pass, got %+v", name, res)
		}
	}
	// Other tools are the engine's business, keys or not.
	for _, name := range []string{SetFields, EditList, SetPageLayout, ResetPinnedSection, PickAsset, LookupFields, "unknown_tool"} {
		if _, ok := ValidateBrowserOp(name, json.RawMessage(`{}`)); ok {
			t.Errorf("%s must not be refused by ValidateBrowserOp", name)
		}
	}
}

func TestDispatchIgnoresBrowserTools(t *testing.T) {
	m := snapshot(t)
	for _, name := range []string{SetFields, EditList, Reset, RestoreDefault, SetPageLayout, ResetPinnedSection, PickAsset, "unknown_tool"} {
		if _, handled := Dispatch(m, name, json.RawMessage(`{}`)); handled {
			t.Errorf("%s must not be handled server-side", name)
		}
	}
}

// TestDescriptionsMatchBrowserContracts pins the parts of the tool
// descriptions that mirror the browser engine's contracts: a page's
// `sections` add takes item {id: <sectionType>} (ops/validate.js, "sections
// add needs item.id (the section type)"), and srcset is the picker's
// {width: url} object (lib/cdnUrl.js compactSrcsetObject), never a string.
func TestDescriptionsMatchBrowserContracts(t *testing.T) {
	tools := Tools(snapshot(t))
	el := toolByName(t, tools, EditList)
	if !strings.Contains(el.Description, "add takes item {id: <sectionType>}") {
		t.Errorf("edit_list must document sections.add item {id: <sectionType>}: %s", el.Description)
	}
	if strings.Contains(el.Description, "item {type}") {
		t.Errorf("edit_list must not claim sections.add takes item {type}: %s", el.Description)
	}
	sf := toolByName(t, tools, SetFields)
	if !strings.Contains(sf.Description, `srcset is an OBJECT {"<width>": "<url>"}`) {
		t.Errorf("set_fields must say srcset is the {width: url} object: %s", sf.Description)
	}
	assets := props(t, sf.InputSchema)["assets"].(map[string]any)["description"].(string)
	if !strings.Contains(assets, `srcset?: object {"<width>": "<url>"}`) {
		t.Errorf("set_fields.assets must type srcset as the object: %s", assets)
	}
	pa := toolByName(t, tools, PickAsset)
	if !strings.Contains(pa.Description, `srcset is an object {"<width>": "<url>"}`) {
		t.Errorf("pick_asset must describe the srcset object it returns: %s", pa.Description)
	}
}
