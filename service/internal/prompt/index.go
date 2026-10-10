package prompt

import (
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strings"

	"github.com/breadstack/sous-chef/internal/fields"
)

// Index is system[1]: a map of one manifest the model can navigate with
// lookup_fields / describe_target. It is rendered once per manifest hash
// and cached with the rules; the full key list stays out (README §3: the
// index is ≈3k tokens, the key list would be another 9–10k).
type Index struct {
	Text string
	// Counts, for the register response and tests.
	Sections, GlobalPanels, Pages, Details, Fields, Lists int
	// Aliases maps a section type that is byte-identical to another onto
	// the canonical one (hero → heroBanner).
	Aliases map[string]string
}

// maxStems caps the key-stem list of one container; the rest is "+N".
const maxStems = 28

// Pinned is the engine's PINNED_BY_PAGE table (cookie-cutter
// launcher/client/src/sousChef/ops/schema.js): the built-in section each
// page carries, and whether it is a reorderable row of the page's section
// list. Pages absent from the manifest are still listed when present in
// the manifest's pages map only.
var Pinned = map[string]struct {
	ID          string
	Reorderable bool
}{
	"faq":                {"__pinned_faq", false},
	"product-details":    {"__pinned_product_detail", false},
	"products":           {"__pinned_product_list", true},
	"blog":               {"__pinned_blog_list", false},
	"blog-detail":        {"__pinned_blog_detail", false},
	"privacy-policy":     {"__pinned_privacy", false},
	"terms-of-service":   {"__pinned_terms", false},
	"return-refund":      {"__pinned_return_refund", false},
	"age-gate":           {"__pinned_age_gate", false},
	"location-selection": {"__pinned_location_selection", false},
	"cart":               {"__pinned_cart", false},
	"checkout":           {"__pinned_checkout", false},
	"account":            {"__pinned_account", false},
	"login":              {"__pinned_login", false},
	"order-confirmation": {"__pinned_order_confirmation", false},
	"order-details":      {"__pinned_order_details", false},
}

// The slice of the manifest the index reads (schema v1, emit.js header).
type manifest struct {
	Version      int                   `json:"version"`
	Sections     map[string]*container `json:"sections"`
	GlobalPanels map[string]*container `json:"globalPanels"`
	Branding     *container            `json:"branding"`
	Pages        map[string]*container `json:"pages"`
	Details      map[string]*container `json:"details"`
}

type container struct {
	EditorKey string           `json:"editorKey"`
	Pages     []string         `json:"pages"`
	Fields    map[string]field `json:"fields"`
	Lists     map[string]list  `json:"lists"`
	Mirror    *mirror          `json:"mirror"`
	Presets   []preset         `json:"presets"`
}

type field struct {
	Type        string          `json:"type"`
	DerivedFrom json.RawMessage `json:"derived_from"`
	KeyedBy     string          `json:"keyed_by"`
}

type list struct {
	ItemRef string           `json:"item_ref"`
	Ops     []string         `json:"ops"`
	Orders  string           `json:"orders"`
	Item    map[string]field `json:"item"`
}

type mirror struct {
	From string   `json:"from"`
	Keys []string `json:"keys"`
}

type preset struct {
	Name  string `json:"name"`
	Label string `json:"label"`
}

// RenderIndex renders the index of one manifest (raw JSON, schema v1).
// Deterministic: the same manifest renders the same bytes whatever its key
// order, so the cached prefix is stable per hash.
func RenderIndex(raw []byte) (Index, error) {
	var m manifest
	if err := json.Unmarshal(raw, &m); err != nil {
		return Index{}, fmt.Errorf("manifest: %w", err)
	}
	if m.Version != 1 {
		return Index{}, fmt.Errorf("manifest: version %d is not 1", m.Version)
	}
	if len(m.Sections) == 0 || len(m.Pages) == 0 {
		return Index{}, errors.New("manifest: sections and pages are required")
	}
	idx := Index{Sections: len(m.Sections), GlobalPanels: len(m.GlobalPanels), Pages: len(m.Pages), Details: len(m.Details), Aliases: map[string]string{}}
	count := func(c *container) {
		if c == nil {
			return
		}
		// Fields counts container fields and list item fields, the way the
		// README's 2,094 was measured (1,924 + 170 on the snapshot).
		idx.Fields += len(c.Fields)
		idx.Lists += len(c.Lists)
		for _, l := range c.Lists {
			idx.Fields += len(l.Item)
		}
	}
	for _, c := range m.Sections {
		count(c)
	}
	for _, c := range m.GlobalPanels {
		count(c)
	}
	count(m.Branding)
	for _, c := range m.Pages {
		count(c)
	}
	for _, c := range m.Details {
		count(c)
	}

	var sb strings.Builder
	sb.WriteString(indexHeader)

	// Sections: byte-identical types collapse onto the longest id.
	canon := aliasSections(m.Sections)
	sb.WriteString("\n## Section types (target kind \"section\"; sectionId = type + \"-<n>\")\n")
	for _, id := range sortedKeys(m.Sections) {
		if canon[id] != id {
			continue
		}
		writeContainer(&sb, "s", id, m.Sections[id], true)
	}
	aliases := make([]string, 0)
	for id, c := range canon {
		if id != c {
			idx.Aliases[id] = c
			aliases = append(aliases, fmt.Sprintf("s:%s = %s", id, c))
		}
	}
	sort.Strings(aliases)
	for _, a := range aliases {
		sb.WriteString(a + "\n")
	}

	sb.WriteString("\n## Global panels (target kind \"global\", optional panel)\n")
	for _, id := range sortedKeys(m.GlobalPanels) {
		writeContainer(&sb, "g", id, m.GlobalPanels[id], false)
	}

	if m.Branding != nil {
		sb.WriteString("\n## Branding (target kind \"branding\")\n")
		writeContainer(&sb, "b", "branding", m.Branding, false)
	}

	sb.WriteString("\n## Pages (target kind \"page\"; p:_common fields apply to every page)\n")
	for _, id := range sortedKeys(m.Pages) {
		writeContainer(&sb, "p", id, m.Pages[id], false)
	}

	sb.WriteString("\n## Details panels (list items; keys shown without the item. prefix; edit with edit_list update on the owning list)\n")
	for _, id := range sortedKeys(m.Details) {
		writeContainer(&sb, "d", id, stripItemPrefix(m.Details[id]), false)
	}

	writeAlsoCalled(&sb, &m, canon)

	sb.WriteString("\n## Pinned (built-in) sections: page → sectionId; * = reorderable row of the page's sections list, others are virtual\n")
	pages := make([]string, 0, len(Pinned))
	for p := range Pinned {
		pages = append(pages, p)
	}
	sort.Strings(pages)
	for _, p := range pages {
		d := Pinned[p]
		star := ""
		if d.Reorderable {
			star = "*"
		}
		fmt.Fprintf(&sb, "%s → %s%s\n", p, d.ID, star)
	}
	idx.Text = sb.String()
	return idx, nil
}

const indexHeader = `# Manifest index
One line per container: kind:id [pages=…] f=<fields> [lists=<list>[ops]{->DetailsPanel|;orders=<what>}…] [mirror=<from>] [presets=…] | keys: <stems> [| ro: <read-only keys>]
A stem "head*(n)" stands for n keys sharing that prefix (headline, headlineColor, …); a bare key is itself. Lists print their item keys as stems after ":" unless a details panel owns the items. Keys are exact as written; use lookup_fields to see types, enums, defaults and gates before writing.
`

// writeContainer prints one container line.
func writeContainer(sb *strings.Builder, kind, id string, c *container, withPages bool) {
	if c == nil {
		fmt.Fprintf(sb, "%s:%s f=0\n", kind, id)
		return
	}
	fmt.Fprintf(sb, "%s:%s", kind, id)
	if withPages && len(c.Pages) > 0 {
		pages := append([]string(nil), c.Pages...)
		sort.Strings(pages)
		fmt.Fprintf(sb, " pages=%s", strings.Join(pages, ","))
	}
	fmt.Fprintf(sb, " f=%d", len(c.Fields))
	if len(c.Lists) > 0 {
		parts := make([]string, 0, len(c.Lists))
		for _, name := range sortedKeys(c.Lists) {
			l := c.Lists[name]
			ops := append([]string(nil), l.Ops...)
			sort.Strings(ops)
			s := name + "[" + strings.Join(ops, "|")
			if l.Orders != "" {
				s += ";orders=" + l.Orders
			}
			s += "]"
			if l.ItemRef != "" {
				s += "->" + strings.TrimPrefix(l.ItemRef, "details.")
			} else if len(l.Item) > 0 {
				s += ":" + stems(keysOf(l.Item), 10)
			}
			parts = append(parts, s)
		}
		fmt.Fprintf(sb, " lists=%s", strings.Join(parts, " "))
	}
	if c.Mirror != nil && c.Mirror.From != "" {
		fmt.Fprintf(sb, " mirror=%s(%d keys)", c.Mirror.From, len(c.Mirror.Keys))
	}
	if len(c.Presets) > 0 {
		names := make([]string, 0, len(c.Presets))
		for _, p := range c.Presets {
			names = append(names, p.Name)
		}
		sort.Strings(names)
		fmt.Fprintf(sb, " presets=%s", strings.Join(names, ","))
	}
	if len(c.Fields) > 0 {
		fmt.Fprintf(sb, " | keys: %s", stems(keysOf(c.Fields), maxStems))
		var ro []string
		for k, f := range c.Fields {
			if len(f.DerivedFrom) > 0 && string(f.DerivedFrom) != "null" {
				ro = append(ro, k)
			}
		}
		if len(ro) > 0 {
			sort.Strings(ro)
			fmt.Fprintf(sb, " | ro: %s", strings.Join(ro, " "))
		}
	}
	sb.WriteString("\n")
}

// stems compresses a key list: keys sharing a camelCase stem (the leading
// lowercase run before the first capital, digit, dot or bracket) print as
// stem*(n) when there are two or more; a keyed field
// (locationOverrides.{locationId}.mode) groups on its keyed prefix. Sorted
// by group size, then name; capped at max groups with "+N".
func stems(keys []string, max int) string {
	groups := map[string][]string{}
	for _, k := range keys {
		groups[stemOf(k)] = append(groups[stemOf(k)], k)
	}
	type g struct {
		stem string
		keys []string
	}
	all := make([]g, 0, len(groups))
	for s, ks := range groups {
		all = append(all, g{s, ks})
	}
	sort.Slice(all, func(i, j int) bool {
		if len(all[i].keys) != len(all[j].keys) {
			return len(all[i].keys) > len(all[j].keys)
		}
		return all[i].stem < all[j].stem
	})
	out := make([]string, 0, len(all))
	for i, grp := range all {
		if i >= max {
			out = append(out, fmt.Sprintf("+%d", len(all)-max))
			break
		}
		if len(grp.keys) == 1 {
			out = append(out, grp.keys[0])
			continue
		}
		out = append(out, fmt.Sprintf("%s*(%d)", grp.stem, len(grp.keys)))
	}
	return strings.Join(out, " ")
}

func stemOf(key string) string {
	if i := strings.Index(key, ".{"); i > 0 {
		if j := strings.Index(key[i:], "}."); j > 0 {
			return key[:i+j+2]
		}
	}
	for i, r := range key {
		if i > 0 && (r >= 'A' && r <= 'Z' || r >= '0' && r <= '9' || r == '.' || r == '[' || r == '_') {
			return key[:i]
		}
	}
	return key
}

// aliasSections maps each section type onto its canonical type: the longest
// id (then first alphabetically) among the types whose fields, lists,
// mirror and presets are byte-identical once canonicalised.
func aliasSections(sections map[string]*container) map[string]string {
	sig := map[string][]string{}
	for id, c := range sections {
		key := "nil"
		if c != nil {
			b, _ := json.Marshal(struct {
				Fields  map[string]field `json:"f"`
				Lists   map[string]list  `json:"l"`
				Mirror  *mirror          `json:"m"`
				Presets []preset         `json:"p"`
			}{c.Fields, c.Lists, c.Mirror, c.Presets})
			key = string(b)
		}
		sig[key] = append(sig[key], id)
	}
	canon := map[string]string{}
	for _, ids := range sig {
		sort.Slice(ids, func(i, j int) bool {
			if len(ids[i]) != len(ids[j]) {
				return len(ids[i]) > len(ids[j])
			}
			return ids[i] < ids[j]
		})
		for _, id := range ids {
			canon[id] = ids[0]
		}
	}
	return canon
}

// stripItemPrefix returns a details panel with the item. prefix removed
// from its keys, as the engine addresses them.
func stripItemPrefix(c *container) *container {
	if c == nil {
		return nil
	}
	out := *c
	out.Fields = make(map[string]field, len(c.Fields))
	for k, f := range c.Fields {
		out.Fields[strings.TrimPrefix(k, "item.")] = f
	}
	return &out
}

func sortedKeys[V any](m map[string]V) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

func keysOf[V any](m map[string]V) []string { return sortedKeys(m) }

// alsoCalledMax caps the aliases printed per container (vocab.txt order is
// the priority): the section costs ≈ +315 tokens (Qwen3.5 tokenizer) on
// the snapshot, 1,274 bytes.
const alsoCalledMax = 3

// writeAlsoCalled prints the merchant vocabulary's container aliases
// (internal/fields vocab.txt) for the containers this manifest has: the
// words merchants use for a panel or section that its id does not say
// ("site name" → g:BrandingSection). The candidate-fields retrieval reads
// the same table, so the index and the block agree. Section aliases print
// under the canonical type only; fields.AlsoCalled picks which (page-view
// entries, phrases the id already says and longer variants stay
// retrieval-only).
func writeAlsoCalled(sb *strings.Builder, m *manifest, canon map[string]string) {
	aliases := fields.ContainerAliases()
	var lines []string
	for _, key := range sortedKeys(aliases) {
		kind, id, _ := strings.Cut(key, ":")
		var ok bool
		switch kind {
		case "s":
			_, ok = m.Sections[id]
			ok = ok && canon[id] == id
		case "g":
			_, ok = m.GlobalPanels[id]
		case "b":
			ok = m.Branding != nil
		case "p":
			_, ok = m.Pages[id]
			if id == "seo" {
				_, ok = m.Details["SeoPanel"]
			}
		}
		if !ok {
			continue
		}
		names := fields.AlsoCalled(key, alsoCalledMax)
		if len(names) > 0 {
			lines = append(lines, key+": "+strings.Join(names, ", "))
		}
	}
	if len(lines) == 0 {
		return
	}
	sb.WriteString("\n## Also called (merchant words → container)\n")
	for _, l := range lines {
		sb.WriteString(l + "\n")
	}
}
