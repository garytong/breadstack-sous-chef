package loop

import (
	"encoding/json"
	"regexp"
	"strconv"
	"strings"

	"github.com/breadstack/sous-chef/internal/api"
	"github.com/breadstack/sous-chef/internal/manifest"
	"github.com/breadstack/sous-chef/internal/toolgen"
)

// Hidden writes (shared contract 4; plan Q9). The browser's engine writes a
// gated field (`conditional_on` / `hidden_when`) anyway and says so with the
// warning `W_HIDDEN:<key>` — for EVERY gated field, gate open or closed — and
// the launcher adds a structured `hidden: [{key, by, byValue}]` member: the
// setting that decides the field (list indexes filled in) and that setting's
// value after the op (its manifest default when unset). The model reads the
// member (browserResult forwards it); the loop reads it to tell a write the
// merchant will not see from one that is merely gated, and hands such a turn
// to the fallback route (EscalateHiddenWrite). Without the member there is no
// hand-off: the warning alone does not say whether the gate is closed.

// wHidden is the engine's warning prefix (ops/schema.js hiddenWarning).
const wHidden = "W_HIDDEN:"

// hiddenGate is one wire `hidden` entry.
type hiddenGate struct {
	Key     string          `json:"key"`
	By      string          `json:"by"`
	ByValue json.RawMessage `json:"byValue"`
}

// parseHidden reads a result's `hidden` member leniently: a malformed member
// or entry (no key, no by) is dropped, never an error — the warning strings
// still stand on their own.
func parseHidden(raw json.RawMessage) []hiddenGate {
	if len(raw) == 0 {
		return nil
	}
	var items []json.RawMessage
	if json.Unmarshal(raw, &items) != nil {
		return nil
	}
	var out []hiddenGate
	for _, it := range items {
		var g hiddenGate
		if json.Unmarshal(it, &g) != nil || g.Key == "" || g.By == "" {
			continue
		}
		if len(g.ByValue) == 0 {
			g.ByValue = json.RawMessage("null")
		}
		out = append(out, g)
	}
	return out
}

// hiddenDoc is the `hidden` member as the model reads it in the tool_result
// (browserResult): the entries in order, each as a map so the keys marshal
// sorted like the rest of the result ({by, byValue, key}).
func hiddenDoc(gates []hiddenGate) []map[string]any {
	out := make([]map[string]any, 0, len(gates))
	for _, g := range gates {
		out = append(out, map[string]any{"key": g.Key, "by": g.By, "byValue": g.ByValue})
	}
	return out
}

// writtenKeys are the field keys a browser op set: set_fields `values`,
// edit_list `item`. A reset / restore_default clears rather than writes, and
// the host ops write no field; none of them counts.
func writtenKeys(name string, input json.RawMessage) (keys map[string]bool, target manifest.Target, list string) {
	var in struct {
		Target manifest.Target            `json:"target"`
		Values map[string]json.RawMessage `json:"values"`
		Item   map[string]json.RawMessage `json:"item"`
		List   string                     `json:"list"`
	}
	if json.Unmarshal(input, &in) != nil {
		return nil, manifest.Target{}, ""
	}
	var src map[string]json.RawMessage
	switch name {
	case toolgen.SetFields:
		src = in.Values
	case toolgen.EditList:
		src = in.Item
		list = in.List
	}
	if len(src) == 0 {
		return nil, in.Target, list
	}
	keys = make(map[string]bool, len(src))
	for k := range src {
		keys[k] = true
	}
	return keys, in.Target, list
}

// hiddenWrites returns the keys an applied browser op wrote that the
// merchant will not see: a `W_HIDDEN:<key>` warning on a key the op wrote,
// whose gate is closed after the op. The gate is judged from the result's
// `hidden` entries for that key against the manifest's rule (the browser's
// own changes.js gateMatches). A key with no `hidden` entry does NOT count:
// the engine warns on every gated field, gate open or closed, so the warning
// alone cannot tell the two apart, and treating it as hidden would hand every
// correct gated write of an older browser (no `hidden` member) to the
// fallback. A key whose entries cannot be judged (the target or the rule does
// not resolve) counts as hidden. page is the viewed page, for a section
// target that names none.
func hiddenWrites(m *manifest.Manifest, page string, name string, input json.RawMessage, res api.ToolResult) []string {
	if !res.OK || len(res.Warnings) == 0 {
		return nil
	}
	keys, target, list := writtenKeys(name, input)
	if len(keys) == 0 {
		return nil
	}
	gates := parseHidden(res.Hidden)
	var out []string
	seen := map[string]bool{}
	for _, w := range res.Warnings {
		k, ok := strings.CutPrefix(w, wHidden)
		if !ok || !keys[k] || seen[k] {
			continue
		}
		seen[k] = true
		var mine []hiddenGate
		for _, g := range gates {
			if g.Key == k {
				mine = append(mine, g)
			}
		}
		if len(mine) > 0 && hiddenNow(m, page, target, list, k, mine) {
			out = append(out, k)
		}
	}
	return out
}

// indexRE matches a filled list index ("slides[1].layout" → "slides[].layout").
var indexRE = regexp.MustCompile(`\[\d+\]`)

// hiddenNow reports whether the field key on target (an edit_list row field
// when list is set) is hidden after the op, per its gate rules and the
// browser's post-op gate values. Anything it cannot judge is hidden.
func hiddenNow(m *manifest.Manifest, page string, target manifest.Target, list, key string, gates []hiddenGate) bool {
	if m == nil {
		return true
	}
	if target.Kind == "section" && target.Page == "" {
		target.Page = page
		if target.Page == "" && !manifest.IsPinnedSectionID(target.SectionID) {
			target.Page = "home" // a non-pinned section's fields do not depend on the page
		}
	}
	r, err := m.Resolve(target)
	if err != nil {
		return true
	}
	var spec *manifest.Field
	if list != "" {
		if l, ok := r.Lists[list]; ok {
			items, _ := m.ItemFields(l)
			spec = items[key]
		}
	}
	if spec == nil {
		if ref, ok := m.Field(r, key); ok {
			spec = ref.Spec
		}
	}
	if spec == nil || !spec.Gated() {
		return true
	}
	for _, g := range gates {
		gateKey := indexRE.ReplaceAllString(g.By, "[]")
		judged := false
		if c := spec.ConditionalOn; c != nil && c.Key == gateKey {
			judged = true
			if !gateMatches(c.Value, g.ByValue) {
				return true
			}
		}
		if h := spec.HiddenWhen; h != nil && h.Key == gateKey {
			judged = true
			if gateMatches(h.Value, g.ByValue) {
				return true
			}
		}
		if !judged {
			return true
		}
	}
	return false
}

// gateMatches is the launcher's changes.js gateMatches: does v satisfy a
// gate value — "set" (anything but null, "", false), a list of values, or
// a scalar (equal, or equal as text: 2 matches "2").
func gateMatches(gateValue, v json.RawMessage) bool {
	var gv, val any
	if json.Unmarshal(gateValue, &gv) != nil {
		return false
	}
	if len(v) == 0 || json.Unmarshal(v, &val) != nil {
		val = nil
	}
	if s, ok := gv.(string); ok && s == "set" {
		switch x := val.(type) {
		case nil:
			return false
		case string:
			return x != ""
		case bool:
			return x
		}
		return true
	}
	list, ok := gv.([]any)
	if !ok {
		list = []any{gv}
	}
	for _, g := range list {
		if g == nil && val == nil {
			return true
		}
		if g == nil || val == nil {
			continue
		}
		if gs, ok := scalarText(g); ok {
			if vs, ok := scalarText(val); ok && gs == vs {
				return true
			}
		}
	}
	return false
}

// scalarText is JS String(x) for a JSON scalar; objects and arrays have none.
func scalarText(x any) (string, bool) {
	switch v := x.(type) {
	case string:
		return v, true
	case bool:
		return strconv.FormatBool(v), true
	case float64:
		return strconv.FormatFloat(v, 'f', -1, 64), true
	}
	return "", false
}
