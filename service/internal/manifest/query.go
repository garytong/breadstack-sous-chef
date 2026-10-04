package manifest

import (
	"encoding/json"
	"sort"
	"strings"
	"unicode"
)

// Server-tool queries (README §2 "Compact server-tool outputs"). The model
// never sees the full manifest: it gets the index (index.go) and calls
// lookup_fields before writing unfamiliar keys, describe_target to learn a
// container's lists, stems, mirror and presets.

const (
	// DefaultLookupLimit caps a lookup_fields answer when the model gives no limit.
	DefaultLookupLimit = 40
	// MaxLookupLimit is the hard cap on `limit`.
	MaxLookupLimit = 120
	// LookupByteCap bounds the JSON size of the `fields` array (≈ 3k tokens).
	// A slim field averages ~110 bytes; an unfiltered perksCards reaches 22.9 KB.
	LookupByteCap = 12 * 1024
	// EnumLabelsMax is the enum size above which enum_labels are omitted.
	EnumLabelsMax = 12
)

// LookupRequest is lookup_fields' input. Exactly one of Target / Container
// addresses the container; Query, Keys and List narrow the answer.
type LookupRequest struct {
	Target    *Target       `json:"target,omitempty"`
	Container ContainerPath `json:"container,omitempty"`
	// Query matches a key prefix, a camelCase word of the key, a list key, or
	// a label substring (all case-insensitive).
	Query string `json:"query,omitempty"`
	// Keys are exact field keys (mirror / alias / keyed / seo keys included).
	Keys []string `json:"keys,omitempty"`
	// List restricts the search to one list's item fields.
	List  string `json:"list,omitempty"`
	Limit int    `json:"limit,omitempty"`
	// Help includes each field's `help` text.
	Help bool `json:"help,omitempty"`
}

// FieldOut is the compact per-field shape. `source` and `path` are never
// sent; `help` only on request.
type FieldOut struct {
	Key        string            `json:"key"`
	Type       string            `json:"type"`
	Label      string            `json:"label,omitempty"`
	List       string            `json:"list,omitempty"`
	Enum       []any             `json:"enum,omitempty"`
	EnumLabels map[string]string `json:"enum_labels,omitempty"`
	Min        *float64          `json:"min,omitempty"`
	Max        *float64          `json:"max,omitempty"`
	Step       *float64          `json:"step,omitempty"`
	Default    json.RawMessage   `json:"default,omitempty"`
	Unset      json.RawMessage   `json:"unset_value,omitempty"`
	If         *Gate             `json:"if,omitempty"`
	Hide       *Gate             `json:"hide,omitempty"`
	RO         string            `json:"ro,omitempty"`
	Ref        string            `json:"ref,omitempty"`
	KeyedBy    string            `json:"keyed_by,omitempty"`
	Companions map[string]string `json:"companions,omitempty"`
	Aliases    map[string]string `json:"aliases,omitempty"`
	Suggested  []float64         `json:"suggested,omitempty"`
	Help       string            `json:"help,omitempty"`
	Via        string            `json:"via,omitempty"`
}

// ListOut is the compact per-list shape.
type ListOut struct {
	Key     string   `json:"key"`
	Ops     []string `json:"ops"`
	Orders  string   `json:"orders,omitempty"`
	ItemRef string   `json:"item_ref,omitempty"`
	Depth   int      `json:"depth,omitempty"`
	// ItemKeys lists the row's field keys (describe_target and `list` lookups).
	ItemKeys []string `json:"item_keys,omitempty"`
}

// LookupResult is lookup_fields' output.
type LookupResult struct {
	Container ContainerPath `json:"container"`
	Kind      string        `json:"kind"`
	Known     bool          `json:"known"`
	Fields    []FieldOut    `json:"fields"`
	Lists     []ListOut     `json:"lists,omitempty"`
	// Total is the number of matches before the limit / byte cap.
	Total     int    `json:"total"`
	Truncated bool   `json:"truncated,omitempty"`
	Mirror    string `json:"mirror,omitempty"`
	Note      string `json:"note,omitempty"`
}

// LookupFields answers a lookup_fields call.
func (m *Manifest) LookupFields(req LookupRequest) (*LookupResult, error) {
	r, err := m.resolveRequest(req.Target, req.Container)
	if err != nil {
		return nil, err
	}
	limit := req.Limit
	if limit <= 0 {
		limit = DefaultLookupLimit
	}
	if limit > MaxLookupLimit {
		limit = MaxLookupLimit
	}
	res := &LookupResult{Container: r.Container, Kind: r.Kind, Known: r.Known, Fields: []FieldOut{}}
	if r.Mirror != nil {
		res.Mirror = r.Mirror.From
	}
	if !r.Known {
		res.Note = "page is not in the manifest: only the common page fields apply (the browser validates the slug)"
	}

	var matches []scored
	switch {
	case len(req.Keys) > 0:
		matches = m.lookupKeys(r, req.Keys, req.List)
	case req.List != "":
		l, ok := r.Lists[req.List]
		if !ok {
			if req.List == SectionsListKey && r.Kind == "page" && r.Page != SEOPage {
				res.Lists = []ListOut{{Key: SectionsListKey, Ops: SectionsListOps, Depth: 1, ItemKeys: []string{"id", "type", "enabled"}}}
				res.Note = "the page section list: rows are {id, type, enabled}; add takes item {type}, update only {enabled}"
				return res, nil
			}
			return nil, &QueryError{Code: "E_LIST", Message: "unknown list " + req.List + " for " + r.Container + " (known: " + strings.Join(r.ListKeys(), ", ") + ")"}
		}
		items, panel := m.ItemFields(l)
		res.Lists = []ListOut{listOut(m, req.List, l)}
		for _, k := range SortedFieldKeys(items) {
			if s, ok := scoreField(k, items[k], req.Query); ok {
				matches = append(matches, scored{ref: FieldRef{Key: k, Spec: items[k], List: req.List, Via: "item", Panel: panel}, score: s})
			}
		}
	default:
		for _, k := range SortedFieldKeys(r.Fields) {
			if s, ok := scoreField(k, r.Fields[k], req.Query); ok {
				matches = append(matches, scored{ref: FieldRef{Key: k, Spec: r.Fields[k], Via: "field"}, score: s})
			}
		}
		if req.Query != "" {
			// A query also searches every list's item fields and list keys.
			for _, lk := range sortedKeys(r.Lists) {
				l := r.Lists[lk]
				items, panel := m.ItemFields(l)
				listHit := containsFold(lk, req.Query)
				if listHit {
					res.Lists = append(res.Lists, listOut(m, lk, l))
				}
				for _, k := range SortedFieldKeys(items) {
					s, ok := scoreField(k, items[k], req.Query)
					if !ok && listHit {
						s, ok = 5, true
					}
					if ok {
						matches = append(matches, scored{ref: FieldRef{Key: k, Spec: items[k], List: lk, Via: "item", Panel: panel}, score: s + 1})
					}
				}
			}
		}
	}

	sort.SliceStable(matches, func(i, j int) bool {
		if matches[i].score != matches[j].score {
			return matches[i].score < matches[j].score
		}
		if matches[i].ref.List != matches[j].ref.List {
			return matches[i].ref.List < matches[j].ref.List
		}
		return matches[i].ref.Key < matches[j].ref.Key
	})
	res.Total = len(matches)
	bytes := 0
	for _, s := range matches {
		if len(res.Fields) >= limit {
			res.Truncated = true
			break
		}
		out := fieldOut(s.ref, req.Help)
		b, _ := json.Marshal(out)
		if bytes+len(b) > LookupByteCap && len(res.Fields) > 0 {
			res.Truncated = true
			break
		}
		bytes += len(b)
		res.Fields = append(res.Fields, out)
	}
	if res.Truncated {
		res.Note = strings.TrimSpace(res.Note + " truncated: narrow with query, keys or list")
	}
	return res, nil
}

// QueryError is a lookup / describe failure the loop turns into an
// is_error tool_result with the engine's code vocabulary.
type QueryError struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

func (e *QueryError) Error() string { return e.Code + ": " + e.Message }

func (m *Manifest) resolveRequest(t *Target, container ContainerPath) (*Resolved, error) {
	switch {
	case t != nil && t.Kind != "":
		r, err := m.Resolve(*t)
		if err != nil {
			return nil, &QueryError{Code: "E_TARGET", Message: strings.TrimPrefix(err.Error(), "E_TARGET: ")}
		}
		return r, nil
	case container != "":
		r, err := m.ResolveContainer(container)
		if err != nil {
			return nil, &QueryError{Code: "E_TARGET", Message: strings.TrimPrefix(err.Error(), "E_TARGET: ")}
		}
		return r, nil
	}
	return nil, &QueryError{Code: "E_OP", Message: "lookup needs a target or a container"}
}

type scored struct {
	ref   FieldRef
	score int
}

func (m *Manifest) lookupKeys(r *Resolved, keys []string, list string) []scored {
	var out []scored
	seen := map[string]bool{}
	for _, k := range keys {
		if k == "" || seen[k] {
			continue
		}
		seen[k] = true
		if list != "" {
			if l, ok := r.Lists[list]; ok {
				items, panel := m.ItemFields(l)
				if f, ok := items[k]; ok {
					out = append(out, scored{ref: FieldRef{Key: k, Spec: f, List: list, Via: "item", Panel: panel}})
					continue
				}
			}
		}
		if ref, ok := m.Field(r, k); ok {
			out = append(out, scored{ref: ref})
			continue
		}
		// Fall back to any list whose item has the key.
		for _, lk := range sortedKeys(r.Lists) {
			items, panel := m.ItemFields(r.Lists[lk])
			if f, ok := items[k]; ok {
				out = append(out, scored{ref: FieldRef{Key: k, Spec: f, List: lk, Via: "item", Panel: panel}, score: 1})
				break
			}
		}
	}
	return out
}

// scoreField ranks a field against a query: 0 exact key, 1 key prefix,
// 2 camelCase word prefix, 3 key substring, 4 label substring. ok is false
// when nothing matches; an empty query matches everything at 3.
func scoreField(key string, f *Field, query string) (int, bool) {
	q := strings.ToLower(strings.TrimSpace(query))
	if q == "" {
		return 3, true
	}
	lk := strings.ToLower(key)
	switch {
	case lk == q:
		return 0, true
	case strings.HasPrefix(lk, q):
		return 1, true
	}
	for _, w := range camelWords(key) {
		if strings.HasPrefix(strings.ToLower(w), q) {
			return 2, true
		}
	}
	if strings.Contains(lk, q) {
		return 3, true
	}
	if f != nil && containsFold(f.Label, q) {
		return 4, true
	}
	return 0, false
}

// camelWords splits `headlineFontSize` → [headline Font Size]; dots, brackets
// and underscores also break words. A run of capitals stays one word
// (`cardShowTHC` → [card Show THC], `ABC` → [ABC]).
func camelWords(key string) []string {
	var words []string
	start := 0
	flush := func(end int) {
		if end > start {
			words = append(words, key[start:end])
		}
	}
	var prev rune
	for i, r := range key {
		switch {
		case r == '.' || r == '[' || r == ']' || r == '_' || r == '{' || r == '}':
			flush(i)
			start = i + 1
		case unicode.IsUpper(r) && i > start && !unicode.IsUpper(prev):
			flush(i)
			start = i
		}
		prev = r
	}
	flush(len(key))
	return words
}

// Stem is the first camelCase word of a key (`headlineFontSize` → headline,
// `announcementBar.show` → announcementBar, `slides[].layout` → slides).
func Stem(key string) string {
	for i, r := range key {
		if i == 0 {
			continue
		}
		if r == '.' || r == '[' || r == '{' {
			return key[:i]
		}
	}
	w := camelWords(key)
	if len(w) == 0 {
		return key
	}
	return w[0]
}

func containsFold(s, q string) bool {
	return q != "" && strings.Contains(strings.ToLower(s), strings.ToLower(q))
}

func fieldOut(ref FieldRef, help bool) FieldOut {
	f := ref.Spec
	out := FieldOut{Key: ref.Key, Type: f.Type, Label: f.Label, List: ref.List,
		Enum: f.Enum, Min: f.Min, Max: f.Max, Step: f.Step, Default: f.Default, Unset: f.UnsetValue,
		If: f.ConditionalOn, Hide: f.HiddenWhen, RO: f.DerivedFrom, Ref: f.Ref, KeyedBy: f.KeyedBy,
		Companions: f.Companions, Aliases: f.Aliases, Suggested: f.Suggested}
	if len(f.Enum) <= EnumLabelsMax {
		out.EnumLabels = f.EnumLabels
	}
	if help {
		out.Help = f.Help
	}
	if ref.Via != "field" && ref.Via != "item" {
		out.Via = ref.Via
	}
	return out
}

func listOut(m *Manifest, key string, l *List) ListOut {
	items, _ := m.ItemFields(l)
	return ListOut{Key: key, Ops: l.Ops, Orders: l.Orders, ItemRef: l.ItemRef, Depth: ListDepth(key), ItemKeys: SortedFieldKeys(items)}
}

// DescribeResult is describe_target's output: the container's lists, key
// stems, mirror, presets and (page targets) the pinned-section id.
type DescribeResult struct {
	Container  ContainerPath `json:"container"`
	Kind       string        `json:"kind"`
	Known      bool          `json:"known"`
	TypeID     string        `json:"typeId,omitempty"`
	EditorKey  string        `json:"editorKey,omitempty"`
	Pages      []string      `json:"pages,omitempty"`
	Panels     []string      `json:"panels,omitempty"`
	FieldCount int           `json:"fieldCount"`
	// Keys lists every field key when there are at most KeysInline of them,
	// else Stems groups them by first camelCase word with a count.
	Keys   []string          `json:"keys,omitempty"`
	Stems  []StemCount       `json:"stems,omitempty"`
	Lists  []ListOut         `json:"lists"`
	Mirror *MirrorOut        `json:"mirror,omitempty"`
	Preset []PresetOut       `json:"presets,omitempty"`
	Pinned *Pinned           `json:"pinnedSection,omitempty"`
	Keyed  map[string]string `json:"keyedPrefixes,omitempty"`
	Note   string            `json:"note,omitempty"`
}

// KeysInline is the field count up to which describe_target lists keys verbatim.
const KeysInline = 40

// StemCount is a key stem with how many keys share it.
type StemCount struct {
	Stem  string `json:"stem"`
	Count int    `json:"n"`
}

// MirrorOut is the compact mirror description.
type MirrorOut struct {
	From    string              `json:"from"`
	Keys    int                 `json:"keys"`
	Aliases map[string][]string `json:"aliases,omitempty"`
	Hint    string              `json:"hint"`
}

// PresetOut is a preset without its values.
type PresetOut struct {
	Name  string   `json:"name"`
	Label string   `json:"label,omitempty"`
	Keys  []string `json:"keys"`
}

// DescribeTarget answers a describe_target call (by target or container path).
func (m *Manifest) DescribeTarget(t *Target, container ContainerPath) (*DescribeResult, error) {
	r, err := m.resolveRequest(t, container)
	if err != nil {
		return nil, err
	}
	res := &DescribeResult{Container: r.Container, Kind: r.Kind, Known: r.Known, TypeID: r.TypeID,
		EditorKey: r.EditorKey, Pages: r.Pages, Panels: r.Panels, FieldCount: len(r.Fields), Lists: []ListOut{}}
	keys := SortedFieldKeys(r.Fields)
	if len(keys) <= KeysInline {
		res.Keys = keys
	} else {
		res.Stems = Stems(keys)
	}
	if r.Kind == "page" && r.Page != SEOPage {
		res.Lists = append(res.Lists, ListOut{Key: SectionsListKey, Ops: SectionsListOps, Depth: 1, ItemKeys: []string{"id", "type", "enabled"}})
		if p, ok := PinnedByPage[r.Page]; ok {
			pp := p
			res.Pinned = &pp
		}
	}
	for _, lk := range sortedKeys(r.Lists) {
		res.Lists = append(res.Lists, listOut(m, lk, r.Lists[lk]))
	}
	if r.Mirror != nil {
		res.Mirror = &MirrorOut{From: r.Mirror.From, Keys: len(r.Mirror.Keys), Aliases: r.Mirror.Aliases,
			Hint: "these keys mirror " + r.Mirror.From + "; edit the slide with edit_list update on " + mirrorListKey(r.Mirror) + " index 0"}
	}
	for _, p := range r.Presets {
		res.Preset = append(res.Preset, PresetOut{Name: p.Name, Label: p.Label, Keys: sortedKeys(p.Writes)})
	}
	if table, ok := KeyedPrefixes[r.Kind]; ok {
		res.Keyed = map[string]string{}
		for prefix, panel := range table {
			if _, ok := m.Detail(panel); ok {
				res.Keyed[prefix] = prefix + ".<id>.<field of details." + panel + ">"
			}
		}
	}
	if !r.Known {
		res.Note = "page is not in the manifest: only the common page fields and the sections list apply"
	}
	return res, nil
}

// Stems groups keys by Stem, most frequent first then alphabetical.
func Stems(keys []string) []StemCount {
	counts := map[string]int{}
	for _, k := range keys {
		counts[Stem(k)]++
	}
	out := make([]StemCount, 0, len(counts))
	for s, n := range counts {
		out = append(out, StemCount{Stem: s, Count: n})
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Count != out[j].Count {
			return out[i].Count > out[j].Count
		}
		return out[i].Stem < out[j].Stem
	})
	return out
}
