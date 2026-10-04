package manifest

import (
	"fmt"
	"sort"
	"strings"
)

// Index is the compact per-hash manifest summary rendered into system[1]
// (README §3). One line per container with its field count, pages, lists
// (ops, item_ref panel) and mirror, followed by the container's key stems;
// byte-identical containers in the same group collapse to an alias line.
// It carries the target/op grammar, the error codes and the pinned-ids
// table so the model can write ops without ever seeing the full manifest.
//
// Size is measured, not estimated: Index.Bytes, and Index.Tokens = bytes/4
// (the task's token-ish unit); README §10 budgets ≤ 5k tokens at 3 bytes/token.
type Index struct {
	Text  string
	Bytes int
	// Tokens is the bytes/4 estimate.
	Tokens int
	// Aliases maps a collapsed container path to the one it duplicates.
	Aliases map[ContainerPath]ContainerPath
}

// TokensAt returns the token estimate at a given bytes-per-token ratio.
func (ix Index) TokensAt(bytesPerToken float64) int {
	return int(float64(ix.Bytes)/bytesPerToken + 0.5)
}

// BuildIndex renders the index for a manifest. Output is deterministic for
// a given hash (sorted everywhere), which keeps the cached prefix stable.
func BuildIndex(m *Manifest) Index {
	var b strings.Builder
	counts := m.Counts()
	fmt.Fprintf(&b, "# Site editor manifest %s — %d fields, %d lists, %d section types, %d global panels, %d pages, %d detail panels\n",
		shortHash(m.Hash()), counts.Fields, counts.Lists, counts.Sections, counts.GlobalPanels, counts.Pages, counts.Details)

	b.WriteString("## Targets\n")
	b.WriteString(`{kind:"section",page,sectionId,storeId?}  type = sectionId minus trailing -N; families ` + strings.Join(SectionPrefixFamilies, "|") + " also match by prefix (locations-x)\n")
	b.WriteString(`{kind:"global",panel?}  all global panels merged; panel only narrows` + "\n")
	b.WriteString(`{kind:"branding"}` + "\n")
	b.WriteString(`{kind:"page",page}  page fields + "published" on every page; page "seo" = SEO panel (keys pages.<page>.title|description|ogImage, general.*, robots.*, social.*, sitemap.*, schema.*); list "sections" = the page's section rows {id,type,enabled}` + "\n")
	b.WriteString("Omit page to act on the page the merchant is viewing (the host fills it in).\n")

	b.WriteString("## Ops\n")
	b.WriteString("set_fields {target, values{key:val}, assets?{key:{alt,srcset,mobile}}}; edit_list {target, list, action add|remove|reorder|update|duplicate, index int|[i,j], item, order, sectionId}; reset {target, keys?} → unset_value; restore_default {target, keys?} → manifest default; set_page_layout {page, layout}; reset_pinned_section {page, sectionId}; pick_asset {purpose, asset_type} suspends until the merchant picks.\n")
	b.WriteString("Detail panels are list rows: edit_list update on the owning list (→Panel below), item keys = the panel's keys. Nested lists use the dotted key (slides[].bullets) with index [slideIdx, bulletIdx]. `update` is allowed on every list; `duplicate` only on sections.\n")
	b.WriteString("Keyed keys: section locationOverrides.<locId>.<f>; global categories.<slug>.<f>, footerLocationOverrides.<locId>.<f> (f = LocationDetailsPanel / CategoryDetailsPanel key).\n")
	b.WriteString("Field flags in lookup_fields: ro = derived (read-only), if/hide = gated, ref = pick from site context (category|tag|brand|location), keyed_by = dotted id key.\n")

	b.WriteString("## Errors\n")
	b.WriteString(strings.Join(ErrorCodes, " ") + "; warning " + strings.Join(WarningCodes, " ") + " (gated field written anyway)\n")

	b.WriteString("## Pinned (built-in) section ids by page — never add/remove/reorder; reset_pinned_section only\n")
	pages := make([]string, 0, len(PinnedByPage))
	for p := range PinnedByPage {
		pages = append(pages, p)
	}
	sort.Strings(pages)
	parts := make([]string, 0, len(pages))
	for _, p := range pages {
		def := PinnedByPage[p]
		s := p + "=" + def.ID
		if def.Reorderable {
			s += "(reorderable)"
		}
		parts = append(parts, s)
	}
	b.WriteString(strings.Join(parts, " ") + "\n")

	aliases := map[ContainerPath]ContainerPath{}
	writeGroup(&b, m, "## Sections  s:<type> f=<fields> pages=… lists=<key>[ops]→Panel mirror=…", "s:", "sections", m.Sections, aliases)
	writeGroup(&b, m, "## Global panels  g:<panel> (target kind global)", "g:", "globalPanels", m.GlobalPanels, aliases)
	b.WriteString("## Branding  b: (target kind branding)\n")
	writeContainer(&b, m, "b:", "", m.Branding)
	writeGroup(&b, m, "## Pages  p:<page> (target kind page; _common applies to every page)", "p:", "pages", m.Pages, aliases)
	writeGroup(&b, m, "## Detail panels  d:<panel> (rows of the lists marked →Panel; keys are the row's keys)", "d:", "details", m.Details, aliases)

	text := b.String()
	return Index{Text: text, Bytes: len(text), Tokens: len(text) / 4, Aliases: aliases}
}

func shortHash(h string) string {
	if len(h) > 12 {
		return h[:12]
	}
	return h
}

// writeGroup writes one group, collapsing containers whose fields, lists,
// mirror and presets are canonically identical into `x = y` alias lines.
// The canonical member is the one with the most pages, then the longer name.
func writeGroup(b *strings.Builder, m *Manifest, heading, prefix, group string, cs map[string]*Container, aliases map[ContainerPath]ContainerPath) {
	b.WriteString(heading + "\n")
	names := sortedKeys(cs)
	sig := map[string]string{}
	for _, n := range names {
		sig[n] = containerSignature(cs[n])
	}
	canonicalFor := map[string]string{}
	for _, n := range names {
		best := n
		for _, o := range names {
			if o == n || sig[o] != sig[n] {
				continue
			}
			if betterCanonical(cs[o], o, cs[best], best) {
				best = o
			}
		}
		canonicalFor[n] = best
	}
	// Near-aliases: same keys and lists, a handful of attribute differences
	// (hero/heroBanner differ only in slides[].bgColor.default). Rendered as
	// `x ≈ y (differs: …)` so the model knows both the shortcut and the gap.
	nearOf := map[string]string{}     // near-alias → the container it points at
	nearDiff := map[string][]string{} // near-alias → differing attributes
	nearTarget := map[string]bool{}   // containers already pointed at (no chains)
	for _, n := range names {
		if canonicalFor[n] != n || nearTarget[n] || len(cs[n].Fields)+len(cs[n].Lists) == 0 {
			continue
		}
		for _, o := range names {
			if o == n || canonicalFor[o] != o || nearOf[o] != "" || !betterCanonical(cs[o], o, cs[n], n) {
				continue
			}
			if diffs, ok := nearIdentical(cs[n], cs[o]); ok {
				nearOf[n] = o
				nearDiff[n] = diffs
				nearTarget[o] = true
				break
			}
		}
	}
	for _, n := range names {
		if c := canonicalFor[n]; c != n {
			aliases[group+"."+n] = group + "." + c
			fmt.Fprintf(b, "%s%s = %s\n", prefix, n, c)
			continue
		}
		if o, ok := nearOf[n]; ok {
			fmt.Fprintf(b, "%s%s ≈ %s (differs: %s)\n", prefix, n, o, strings.Join(nearDiff[n], ", "))
			continue
		}
		writeContainer(b, m, prefix, n, cs[n])
	}
}

// nearAliasMaxDiffs bounds how many attribute differences a near-alias may carry.
const nearAliasMaxDiffs = 6

// nearIdentical reports whether two containers have the same field keys,
// list keys and list specs, with at most nearAliasMaxDiffs differing field
// attributes (source ignored); diffs are `key.attr` strings.
func nearIdentical(a, b *Container) ([]string, bool) {
	if len(a.Fields) != len(b.Fields) || len(a.Lists) != len(b.Lists) || !sameJSON(a.Mirror, b.Mirror) || !sameJSON(a.Presets, b.Presets) {
		return nil, false
	}
	var diffs []string
	collect := func(prefix string, fa, fb map[string]*Field) bool {
		for k, f := range fa {
			g, ok := fb[k]
			if !ok {
				return false
			}
			for _, attr := range FieldAttrsChanged(f, g) {
				diffs = append(diffs, prefix+k+"."+attr)
			}
		}
		return true
	}
	if !collect("", a.Fields, b.Fields) {
		return nil, false
	}
	for k, la := range a.Lists {
		lb, ok := b.Lists[k]
		if !ok || len(listAttrs(la, lb)) > 0 || len(la.Item) != len(lb.Item) {
			return nil, false
		}
		if !collect(k+"[].", la.Item, lb.Item) {
			return nil, false
		}
	}
	if len(diffs) > nearAliasMaxDiffs {
		return nil, false
	}
	sort.Strings(diffs)
	return diffs, true
}

func betterCanonical(a *Container, an string, b *Container, bn string) bool {
	if len(a.Pages) != len(b.Pages) {
		return len(a.Pages) > len(b.Pages)
	}
	if len(an) != len(bn) {
		return len(an) > len(bn)
	}
	return an < bn
}

// containerSignature is the canonical JSON of everything that affects ops
// (fields, lists, mirror, presets) — not editorKey/pages/structural, and not
// the fields' `source` (probe file:line, which differs between two sections
// that share an editor). An empty container (no fields, no lists) gets a
// unique signature so two empty pages never alias each other.
func containerSignature(c *Container) string {
	if len(c.Fields) == 0 && len(c.Lists) == 0 {
		return fmt.Sprintf("empty:%p", c)
	}
	sig, err := Canonical(map[string]any{
		"fields": stripSources(c.Fields), "lists": stripListSources(c.Lists), "mirror": c.Mirror, "presets": c.Presets,
	})
	if err != nil {
		return ""
	}
	return string(sig)
}

func stripSources(fields map[string]*Field) map[string]*Field {
	out := make(map[string]*Field, len(fields))
	for k, f := range fields {
		c := *f
		c.Source = nil
		out[k] = &c
	}
	return out
}

func stripListSources(lists map[string]*List) map[string]*List {
	out := make(map[string]*List, len(lists))
	for k, l := range lists {
		c := *l
		c.Item = stripSources(l.Item)
		out[k] = &c
	}
	return out
}

func writeContainer(b *strings.Builder, m *Manifest, prefix, name string, c *Container) {
	fmt.Fprintf(b, "%s%s f=%d", prefix, name, len(c.Fields))
	if len(c.Pages) > 0 {
		fmt.Fprintf(b, " pages=%s", strings.Join(c.Pages, ","))
	}
	if len(c.Lists) > 0 {
		var ls []string
		for _, lk := range sortedKeys(c.Lists) {
			l := c.Lists[lk]
			s := lk + "[" + strings.Join(l.Ops, "|") + "]"
			if l.ItemRef != "" {
				s += "→" + strings.TrimPrefix(l.ItemRef, "details.")
			}
			if l.Orders != "" {
				s += "~" + l.Orders
			}
			ls = append(ls, s)
		}
		fmt.Fprintf(b, " lists=%s", strings.Join(ls, " "))
	}
	if c.Mirror != nil {
		fmt.Fprintf(b, " mirror=%s", c.Mirror.From)
		if len(c.Mirror.Aliases) > 0 {
			var as []string
			for _, k := range sortedKeys(c.Mirror.Aliases) {
				as = append(as, k+"="+strings.Join(c.Mirror.Aliases[k], "="))
			}
			fmt.Fprintf(b, " aliases=%s", strings.Join(as, ","))
		}
	}
	if len(c.Presets) > 0 {
		var ps []string
		for _, p := range c.Presets {
			ps = append(ps, p.Name)
		}
		fmt.Fprintf(b, " presets=%s(no apply op: set the keys)", strings.Join(ps, "|"))
	}
	b.WriteString("\n")
	if len(c.Fields) > 0 {
		fmt.Fprintf(b, "  keys: %s\n", stemLine(SortedFieldKeys(c.Fields)))
	}
	for _, lk := range sortedKeys(c.Lists) {
		l := c.Lists[lk]
		if l.ItemRef != "" {
			continue // the panel's own d: line carries the keys
		}
		if len(l.Item) == 0 {
			continue
		}
		fmt.Fprintf(b, "  %s[]: %s\n", lk, stemLine(SortedFieldKeys(l.Item)))
	}
}

// stemLine renders keys as stems with counts: `headline(5) tagline(4) layout`.
// Keys with a `{…}` id segment keep their prefix so keyed fields stay visible.
func stemLine(keys []string) string {
	stems := Stems(keys)
	parts := make([]string, 0, len(stems))
	for _, s := range stems {
		if s.Count > 1 {
			parts = append(parts, fmt.Sprintf("%s(%d)", s.Stem, s.Count))
		} else {
			parts = append(parts, s.Stem)
		}
	}
	return strings.Join(parts, " ")
}
