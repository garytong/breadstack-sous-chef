// Package fields picks the manifest fields a merchant's request most likely
// means and renders them as a compact block for the model's user turn.
//
// A small model cannot map "Body Text colour" to bodyColor from the manifest
// index alone: it invents a key and the browser engine refuses the op
// (E_FIELD). Candidates hands it the keys: a deterministic, dependency-free
// lexical retrieval over the registered manifest. It ranks the targets the
// turn can address (the viewed page's sections, the page's own options, the
// global panels, branding, other pages the text names) by how the text names
// them, ranks each target's fields by how the text matches their labels and
// keys, and lists the best under the op target to copy.
//
// The output is a pure function of (manifest, context, text, options): no
// clock, no map-order dependence, no randomness — the lab renders training
// rows with the same code (cmd/sous-chef-fields), so a served turn and a
// training row carry byte-identical blocks.
package fields

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
	"unicode"

	"github.com/breadstack/sous-chef/internal/manifest"
)

// Context is the part of a turn's `context` the retrieval reads: the viewed
// page and its sections. (prompt.TurnContext carries the same members; this
// package keeps its own copy so it depends on the manifest package only.)
type Context struct {
	Page     string    `json:"page"`
	Sections []Section `json:"sections,omitempty"`
	// Categories (slug, name) and Locations (id, name) instantiate keyed
	// fields (categories.<slug>.<field>, locationOverrides.<id>.<field>)
	// when the text mentions one.
	Categories []Named `json:"categories,omitempty"`
	Locations  []Named `json:"locations,omitempty"`
	// Focus is what the merchant has open in the editor: a section of the
	// viewed page, or a Global Settings panel. The block ranks it first
	// among the targets the text does not name (Candidates).
	Focus Focus `json:"focus"`
	// Outline lists, besides the viewed page's own lists, the other pages
	// the merchant's message names: an entry {page, other: true, sections}
	// whose sections are addressable as {kind:"section", page, sectionId}.
	Outline []OutlineEntry `json:"outline,omitempty"`
}

// Focus is the context's `focus` member: area is the editor tab ("editor",
// "global" for Global Settings, …), sectionId the section whose editor is
// open, panel the open Global Settings panel (a manifest panel name).
type Focus struct {
	Area      string `json:"area,omitempty"`
	SectionID string `json:"sectionId,omitempty"`
	Panel     string `json:"panel,omitempty"`
}

// OutlineEntry is one entry of the context's `outline`; the retrieval
// reads only the other-page entries ({page, other: true, sections}).
type OutlineEntry struct {
	Page     string    `json:"page,omitempty"`
	Other    bool      `json:"other,omitempty"`
	Sections []Section `json:"sections,omitempty"`
}

// focusSection is the id of the focused section, "" when the merchant is in
// Global Settings or has no section open.
func (c Context) focusSection() string {
	if c.Focus.Area == "global" {
		return ""
	}
	return c.Focus.SectionID
}

// focusPanel is the open Global Settings panel, "" outside Global Settings.
func (c Context) focusPanel() string {
	if c.Focus.Area != "global" {
		return ""
	}
	return c.Focus.Panel
}

// otherPages are the pages the browser outlined because the message names
// them (slug → their sections), the viewed page excluded.
func (c Context) otherPages() map[string][]Section {
	var out map[string][]Section
	for _, e := range c.Outline {
		if !e.Other || e.Page == "" || e.Page == c.Page {
			continue
		}
		if out == nil {
			out = map[string][]Section{}
		}
		out[e.Page] = append(out[e.Page], e.Sections...)
	}
	return out
}

// Named is a category (slug + name) or a location (id + name).
type Named struct {
	ID   string `json:"id,omitempty"`
	Slug string `json:"slug,omitempty"`
	Name string `json:"name,omitempty"`
}

func (n Named) key() string {
	if n.Slug != "" {
		return n.Slug
	}
	return n.ID
}

// Section is one section of the viewed page.
type Section struct {
	ID    string `json:"id"`
	Type  string `json:"type,omitempty"`
	Title string `json:"title,omitempty"`
}

// ParseContext reads a turn's raw context object. Empty or null is the zero Context.
func ParseContext(raw json.RawMessage) (Context, error) {
	var c Context
	raw = bytes.TrimSpace(raw)
	if len(raw) == 0 || string(raw) == "null" {
		return c, nil
	}
	if err := json.Unmarshal(raw, &c); err != nil {
		return c, fmt.Errorf("fields: context: %w", err)
	}
	return c, nil
}

// Options bound the block. The zero value of a member means its default.
type Options struct {
	// MaxChars caps the rendered block (default 5200 ≈ 1.6k tokens).
	MaxChars int
	// MaxTargets caps the field lists (one per target, or per run of
	// same-type sections that share a list). Default 6.
	MaxTargets int
	// MaxFields caps the fields listed under one target. Default 40.
	MaxFields int
	// ListAll is the container size up to which every field is listed
	// instead of the matching ones. Default 14.
	ListAll int
}

// DefaultOptions are the options the service and the lab use.
func DefaultOptions() Options {
	return Options{MaxChars: 5200, MaxTargets: 6, MaxFields: 40, ListAll: 14}
}

func (o Options) withDefaults() Options {
	d := DefaultOptions()
	if o.MaxChars <= 0 {
		o.MaxChars = d.MaxChars
	}
	if o.MaxTargets <= 0 {
		o.MaxTargets = d.MaxTargets
	}
	if o.MaxFields <= 0 {
		o.MaxFields = d.MaxFields
	}
	if o.ListAll <= 0 {
		o.ListAll = d.ListAll
	}
	return o
}

// Result is what Candidates found.
type Result struct {
	// Block is the text to put in the user turn; "" when nothing matched.
	Block string
	// Targets are the listed targets, in block order.
	Targets []Target
}

// Target is one field list of the block: the op targets it applies to
// (several when same-type sections share it) and the fields listed.
type Target struct {
	Targets []manifest.Target `json:"targets"`
	// Container is the manifest container behind the targets.
	Container manifest.ContainerPath `json:"container"`
	// Named is true when the text names the target (by id, title, type,
	// panel or page), false when only its fields matched.
	Named bool `json:"named"`
	// Total is the number of writable fields the container has.
	Total  int     `json:"total"`
	Fields []Field `json:"fields"`
	// Lists are the lists shown, in block order; a Field with List set
	// belongs to that list's items.
	Lists []string `json:"lists,omitempty"`
}

// Field is one listed field.
type Field struct {
	Key   string  `json:"key"`
	List  string  `json:"list,omitempty"`
	Label string  `json:"label,omitempty"`
	Type  string  `json:"type"`
	Score float64 `json:"score"`
}

const (
	blockOpen  = "<candidate_fields>\n"
	blockIntro = "Fields this request most likely means, per target. Copy the target and the keys exactly as written; for any other key call lookup_fields first.\n"
	blockClose = "</candidate_fields>"

	namedMin     = 0.5 // name score from which a target counts as named
	keepRatio    = 0.3 // a field is listed when it scores this share of its target's best
	minKeep      = 4   // matched fields a target keeps when the block is trimmed to size
	fillFirst    = 12  // matched fields always listed under the best target (when that many match)
	fillRest     = 6   // … and under each other target
	quotedWeight = 0.3 // weight of a quoted word when scoring fields
)

// cand is one scored field of a container.
type cand struct {
	key   string
	list  string // owning list key, "" for a container field
	spec  *manifest.Field
	score float64
	// doc: the field's words.
	label, keyw []string
	extra       map[string]bool // enum labels/values and the type
	desc        string          // overrides describe(spec) (synthesised fields)
	keyed       bool            // instantiated from a keyed prefix
}

// The bonuses for a label or a key the text spells out. Variables so the
// offline recall harness can switch them off to see what is left when the
// merchant paraphrases; nothing else writes them.
var (
	phraseBonus = 1.5 // × the label's idf mass
	keyBonus    = 12.0
)

// span is a run of text words [from, to).
type span struct{ from, to int }

func (s span) within(o span) bool { return s.from >= o.from && s.to <= o.to }

// listInfo is one list of a container.
type listInfo struct {
	key    string
	spec   *manifest.List
	name   []string // distinctive words of the list key
	named  bool     // the text names the list
	mirror bool     // the hero's slides list: its items are the section's fields
	items  int
	note   string // overrides listNote (the synthesised sections list)
}

// group is one candidate field list.
type group struct {
	targets  []manifest.Target
	res      *manifest.Resolved
	name     float64 // how the text names the target, 0..1
	explicit bool    // named as an instance (id, title, ordinal, page)
	prior    float64
	order    int
	// nameAt are the places the text names the target by a type, panel or
	// page word; labelAt the places it spells out one of the fields' labels.
	nameAt  []span
	labelAt []span
	// words are the words of the target's id (nameByWord).
	words []string
	// keyedOnly marks the panel-less global target, which exists only to
	// carry keyed fields (categories.<slug>.<field>).
	keyedOnly bool
	// Explicit on weak evidence, so that a target the text names outranks
	// it (selectGroups): byView, the viewed page, explicit only because it
	// has no sections of its own (the text does not say "page"); soft, a
	// section whose title the text spells unquoted with its stop words
	// ("in our story, …"), or the section list only such a title asks for.
	byView, soft bool
	// viaView: named because the merchant views a page the vocabulary ties
	// it to (vocab.txt "page:<slug>").
	viaView bool
	// focused: the section or Global Settings panel the merchant has open
	// (Context.Focus): unless the text names it, listed right after the
	// targets the text names when any of its fields matches (selectGroups).
	focused bool
	// byTerm: a panel (or branding) named only through a vocabulary term's
	// reading of text words (termWords: "font" reads as "typography").
	// When those words are common field words of other targets too, the
	// reading is a hint, not a name (selectGroups: hintOnly).
	byTerm    bool
	termWords []string
	// byOutline: another page named because the browser outlined it for
	// this message (Context.Outline, other: true).
	byOutline bool
	// listAt: the viewed page is explicit for a section-list edit only
	// because the text names a section type here (no word "section").
	listAt []span
	// extra are synthesised fields and lists added before scoring.
	extraFields []cand
	extraLists  []extraList
	cands       []cand
	lists       []listInfo
	top         float64
	// ownTop is the best score of a field that a word of the text itself
	// matches (not only a vocabulary reading of one).
	ownTop float64
	total  int
	// chosen output
	fields []int // indexes into cands, in output order
}

// extraList is a synthesised list (the page's `sections` list).
type extraList struct {
	info  listInfo
	items []cand
}

// Candidates ranks the targets and fields the text most likely means and
// renders the block. A nil manifest or an empty text gives the zero Result.
func Candidates(m *manifest.Manifest, ctx Context, text string, opts Options) Result {
	opts = opts.withDefaults()
	toks, nameText, all := scanTextAll(text)
	if m == nil || len(toks) == 0 {
		return Result{}
	}
	// Targets are named by the text without the value being set
	// (tokens.go): fields are still scored against all of it.
	groups := buildGroups(m, ctx, nameText, toks, all)
	for i := range groups {
		scoreGroup(m, &groups[i], text, toks)
	}
	groups = selectGroups(groups, opts)
	for i := range groups {
		chooseFields(&groups[i], i == 0, opts)
	}
	kept := groups[:0]
	for _, g := range groups {
		if len(g.fields) > 0 || len(shownLists(&g)) > 0 {
			kept = append(kept, g)
		}
	}
	groups = kept
	if len(groups) == 0 {
		return Result{}
	}
	block := render(groups)
	for len(block) > opts.MaxChars {
		if !trim(&groups) {
			break
		}
		block = render(groups)
	}
	res := Result{Block: block}
	for i := range groups {
		res.Targets = append(res.Targets, export(&groups[i]))
	}
	return res
}

// ── candidate targets ───────────────────────────────────────────────────────

// phrasesFor are the word sequences that name a section type (kind "s") or
// a global panel ("g"): the words of its id, then its aliases in the
// merchant vocabulary (vocab.txt).
func phrasesFor(kind, id, trimSuffix string) [][]string {
	return append([][]string{words(strings.TrimSuffix(id, trimSuffix))}, aliasesOf(kind+":"+id)...)
}

// findAll lists every contiguous occurrence of seq in the text, word for
// word or through the vocabulary's readings (vocab.go).
func findAll(toks []tok, seq []string) []span {
	var out []span
	if len(seq) == 0 {
		return nil
	}
	for i := range toks {
		if e := matchFrom(toks, i, seq); e > i {
			out = append(out, span{i, e})
		}
	}
	return out
}

// wordsAt are the text words at the spans.
func wordsAt(toks []tok, at []span) []string {
	var out []string
	for _, a := range at {
		for i := a.from; i < a.to; i++ {
			out = append(out, toks[i].w)
		}
	}
	return out
}

// hintOnly reports whether g, named only through a term's reading, is a
// hint: every text word the reading starts from is a field word (label or
// key) of at least two other targets — "font" is every panel's, so "the
// font" hints at Typography while "the Button Font on the buttons" stays
// the Buttons panel's; "facebook" is the social panel's own word, and
// "our facebook page" names it.
func hintOnly(g *group, groups []group) bool {
	if !g.byTerm || g.viaView || len(g.termWords) == 0 {
		return false
	}
	for _, w := range g.termWords {
		n := 0
		for i := range groups {
			h := &groups[i]
			if h.order == g.order {
				continue
			}
			for _, c := range h.cands {
				if containsString(c.label, w) || containsString(c.keyw, w) {
					n++
					break
				}
			}
		}
		if n < 2 {
			return false
		}
	}
	return true
}

func containsString(xs []string, x string) bool {
	for _, y := range xs {
		if y == x {
			return true
		}
	}
	return false
}

// namedDirectly reports whether one of the phrases occurs in the text word
// for word (as a name, findName), not only through a vocabulary reading.
func namedDirectly(toks []tok, phrases [][]string) bool {
	for _, p := range phrases {
	occ:
		for _, at := range findName(toks, p) {
			if at.to-at.from != len(p) {
				continue
			}
			for k, w := range p {
				if toks[at.from+k].w != w {
					continue occ
				}
			}
			return true
		}
	}
	return false
}

// nameMatch is 1 with the places when one of the phrases occurs in the
// text, else a small share for a partly matched multi-word phrase.
func nameMatch(toks []tok, phrases [][]string) (float64, []span) {
	var at []span
	for _, p := range phrases {
		at = append(at, findName(toks, p)...)
	}
	if len(at) > 0 {
		return 1, at
	}
	set := map[string]bool{}
	for _, t := range toks {
		if !t.value {
			set[t.w] = true
			for _, a := range t.alts {
				for _, w := range a.w {
					set[w] = true
				}
			}
		}
	}
	best := 0.0
	for _, p := range phrases {
		if len(p) < 2 {
			continue
		}
		n := 0
		for _, w := range p {
			if set[w] {
				n++
			}
		}
		if s := 0.3 * float64(n) / float64(len(p)); s > best {
			best = s
		}
	}
	return best, nil
}

// containsWord reports whether id occurs in text as a whole token (not
// inside a longer identifier).
func containsWord(text, id string) bool {
	if id == "" {
		return false
	}
	isID := func(r byte) bool {
		return r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r >= '0' && r <= '9' || r == '-' || r == '_' || r >= 0x80
	}
	from := 0
	for {
		i := strings.Index(text[from:], id)
		if i < 0 {
			return false
		}
		i += from
		before, after := byte(' '), byte(' ')
		if i > 0 {
			before = text[i-1]
		}
		if j := i + len(id); j < len(text) {
			after = text[j]
			// A '-' or '_' continues an identifier only when more follows.
			if (after == '-' || after == '_') && (j+1 >= len(text) || !isID(text[j+1])) {
				after = ' '
			}
		}
		if before == '-' || before == '_' {
			if i < 2 || !isID(text[i-2]) {
				before = ' '
			}
		}
		if !isID(before) && !isID(after) {
			return true
		}
		from = i + 1
	}
}

// ordinalBefore reports the ordinals that directly precede an occurrence of
// phrase in the text ("the third content block" → 3; "last" → -1).
func ordinalBefore(toks []tok, phrase []string) []int {
	var out []int
	for _, at := range findName(toks, phrase) {
		i := at.from
		if i < 1 {
			continue
		}
		if n, ok := ordinals[toks[i-1].w]; ok {
			out = append(out, n)
		} else if i >= 2 {
			switch toks[i-1].w {
			case "st", "nd", "rd", "th":
				if n, err := strconv.Atoi(toks[i-2].w); err == nil && n > 0 {
					out = append(out, n)
				}
			}
		}
	}
	return out
}

// quotedPhrase reports whether seq is, on its own, a quoted run of the text
// that is not the value being set.
func quotedPhrase(toks []tok, seq []string) bool {
	for _, at := range findName(toks, seq) {
		if toks[at.from].quoted {
			return true
		}
	}
	return false
}

// The words of a request that changes the page's section list rather than
// a field: restructuring verbs; position words, which count only beside the
// word "section" ("make the blog section the first section"); and the verbs
// that ask for a new section.
var (
	restructuring = wordSet("add another insert remove delete rid drop move reorder rearrange duplicate copy clone hide show enable disable off turn")
	positions     = wordSet("first last top bottom above below before after up down")
	adding        = wordSet("add another insert duplicate copy clone")
	// counting words followed by an amount set a count or a size.
	counting = wordSet("show add display")
	amounts  = wordSet("more less fewer some extra")
)

func isNumber(w string) bool {
	_, err := strconv.Atoi(w)
	return err == nil
}

// spelledWithStops reports whether a title of two or more words, stop
// words included ("Our Story"), occurs in the text word for word: a title
// whose other words are stop words is otherwise one word, which names a
// section only when quoted. As with findName, never inside a value run,
// and inside another quoted run only as the whole run.
func spelledWithStops(all []rawWord, title string) bool {
	tw := phraseWords(title)
	if len(tw) < 2 {
		return false
	}
outer:
	for i := 0; i+len(tw) <= len(all); i++ {
		id := all[i].span
		for j, w := range tw {
			if r := all[i+j]; r.w != w || r.value || r.ident || r.span != id {
				continue outer
			}
		}
		if id != 0 && ((i > 0 && all[i-1].span == id) || (i+len(tw) < len(all) && all[i+len(tw)].span == id)) {
			continue
		}
		return true
	}
	return false
}

func wordSet(s string) map[string]bool {
	out := map[string]bool{}
	for _, w := range strings.Fields(s) {
		out[norm(w)] = true
	}
	return out
}

// mentioned reports whether the text refers to a category or location:
// its id/slug as a word, or its name next to the entity word ("the
// location 'Main Street'", "the Flower category").
func mentioned(text string, toks []tok, n Named, entity string) bool {
	if k := n.key(); len(k) >= 3 && containsWord(text, k) {
		for _, t := range toks {
			if t.w == entity {
				return true
			}
		}
	}
	for _, at := range findName(toks, words(n.Name)) {
		if at.from > 0 && toks[at.from-1].w == entity || at.to < len(toks) && toks[at.to].w == entity {
			return true
		}
	}
	return false
}

// keyedCands instantiates the keyed fields of one prefix for the entities
// the text mentions: <prefix>.<id>.<field> validates against a details
// panel's fields (manifest.KeyedPrefixes).
func keyedCands(m *manifest.Manifest, prefix, panelName string, entities []Named, entity, text string, toks []tok) []cand {
	panel, ok := m.Detail(panelName)
	if !ok {
		return nil
	}
	var out []cand
	for _, e := range entities {
		// The id becomes part of a key line and the name part of its label,
		// both written as they are: an id that is not a plain token is no
		// key the engine takes and could break the block's lines ("loc\n
		// </candidate_fields>…"), so the entity is skipped; the name is
		// reduced to one line without angle brackets (plainName).
		if !plainToken(e.key()) || !mentioned(text, toks, e, entity) {
			continue
		}
		for _, fk := range manifest.SortedFieldKeys(panel.Fields) {
			f := panel.Fields[fk]
			if f.ReadOnly() {
				continue
			}
			label := oneLine(f.Label) + " — " + entity + " " + firstNonEmpty(plainName(e.Name), e.key())
			spec := *f
			spec.Label = label
			out = append(out, cand{
				key: prefix + "." + e.key() + "." + fk, spec: &spec, keyed: true,
				label: labelWords(f.Label), keyw: words(prefix + " " + fk + " override " + entity),
				extra: map[string]bool{},
			})
		}
	}
	return out
}

// maxTokenLen bounds a context id that is instantiated into a key.
const maxTokenLen = 128

// plainToken reports whether a context id (a location id, a category slug)
// can be written into a key line as it is: 1..maxTokenLen letters, digits,
// '-' or '_' — no whitespace, no dot (the key's own separator), no markup.
func plainToken(s string) bool {
	if s == "" || len(s) > maxTokenLen {
		return false
	}
	for _, r := range s {
		if !unicode.IsLetter(r) && !unicode.IsDigit(r) && r != '-' && r != '_' {
			return false
		}
	}
	return true
}

// plainName is a context name fit for a label: one line, no control
// characters, no angle brackets (it cannot open or close a tag).
func plainName(s string) string {
	return oneLine(strings.Map(func(r rune) rune {
		switch {
		case r == '<' || r == '>':
			return -1
		case unicode.IsControl(r) || r == '\u2028' || r == '\u2029':
			return ' '
		}
		return r
	}, s))
}

func buildGroups(m *manifest.Manifest, ctx Context, text string, toks []tok, all []rawWord) []group {
	var groups []group
	add := func(g group) int {
		g.order = len(groups)
		groups = append(groups, g)
		return g.order
	}

	// A section type named only inside a longer name of a panel or of
	// branding ("banner" in "the insider banner") is not named: neither as
	// a target (shadowNames) nor as the subject of a section-list edit
	// ("restore the Show banner of the insider banner" is no hero).
	outer := panelNameSpans(m, toks)
	typeNameOf := func(typeID string) (float64, []span) {
		n, at := nameMatch(toks, phrasesFor("s", typeID, ""))
		if n < 1 {
			return n, at
		}
		at = unshadowed(at, outer)
		if len(at) == 0 {
			return 0, nil
		}
		return n, at
	}

	// Sections of the viewed page, then those of the other pages the
	// message names (the browser outlines them) when the text names them.
	ps := pageSections(m, ctx.Page, ctx.Sections, text, toks, all, typeNameOf, ctx.focusSection(), ctx.Locations, false)
	for _, g := range ps.groups {
		add(g)
	}
	sectionNamed, softNamed, typeSpans, regular := ps.named, ps.softNamed, ps.typeSpans, ps.regular
	others := ctx.otherPages()
	for _, page := range sortedStrings(others) {
		for _, g := range pageSections(m, page, others[page], text, toks, all, typeNameOf, "", ctx.Locations, true).groups {
			add(g)
		}
	}

	// Is the request about the page's section list (add / remove / move /
	// show / hide a section)?
	intent, positioned, addIntent, saysSection := false, false, false, false
	for i, t := range toks {
		if t.quoted {
			continue
		}
		if counting[t.w] && i+1 < len(toks) && (amounts[toks[i+1].w] || isNumber(toks[i+1].w)) {
			continue // "show 4 deals per row", "add more space": a count, not a section
		}
		intent = intent || restructuring[t.w]
		positioned = positioned || positions[t.w]
		addIntent = addIntent || adding[t.w]
		saysSection = saysSection || t.w == "section"
	}
	intent = intent || (positioned && saysSection)
	typeNamed := sectionNamed
	if addIntent && !typeNamed { // a type the page does not have yet can only be added
		for _, typeID := range sortedStrings(m.Sections) {
			if n, at := typeNameOf(typeID); n >= 1 {
				typeNamed = true
				typeSpans = append(typeSpans, at...)
				break
			}
		}
	}
	sectionsList := intent && (saysSection || typeNamed || softNamed)
	// … on the evidence of an unquoted title only ("show Our Story in the
	// footer" may mean the footer link): a named target outranks it.
	softList := sectionsList && !saysSection && !typeNamed

	// The viewed page's own options, then any other page the text names,
	// then the seo pseudo-page.
	pageGroup := func(page string, viewed bool) {
		res, err := m.Resolve(manifest.Target{Kind: "page", Page: page})
		if err != nil {
			return
		}
		// A page is named by its slug or its built-in row's label, word
		// for word ("products" is the page, "product" is not), or by
		// either followed by "page".
		phrases := [][]string{rawWords(page)}
		if p, ok := manifest.PinnedByPage[page]; ok {
			phrases = append(phrases, rawWords(p.Label))
		}
		if page == manifest.SEOPage {
			phrases = append(phrases, []string{"meta"}, []string{"search", "engine"})
		}
		g := group{targets: []manifest.Target{{Kind: "page", Page: page}}, res: res, prior: 1}
		// … or by an alias in the merchant vocabulary, word for word too
		// ("products page" is not the "product page").
		for _, a := range containerAliases["p:"+page] {
			if !strings.HasPrefix(a, viewPrefix) {
				phrases = append(phrases, rawWords(a))
			}
		}
		for _, p := range phrases {
			for _, at := range findRaw(toks, p) {
				paged := at.to < len(toks) && toks[at.to].w == "page"
				if toks[at.from].value || (toks[at.from].quoted && !paged) {
					continue // a value that happens to contain the word
				}
				g.nameAt = append(g.nameAt, at)
				if g.name < 0.7 {
					g.name = 0.7
				}
				if paged {
					g.name, g.explicit = 1, true
				}
			}
		}
		if _, ok := others[page]; ok && !viewed && g.name < 0.7 {
			// The browser outlined it: the message names it (by a title the
			// slug does not spell, "the About Us page").
			g.name, g.byOutline = 0.7, true
		}
		if viewed {
			// A page with no sections of its own: its options are the
			// subject, unless the text names it by a word (then its fields
			// decide) without saying "page".
			if saysPage := len(findName(toks, []string{"page"})) > 0; (g.name < 0.7 || saysPage) && (regular == 0 || saysPage) {
				g.name = max(g.name, 0.7)
				if regular == 0 {
					g.explicit, g.byView = true, !saysPage
				}
			}
			if sectionsList && page != manifest.SEOPage {
				g.name, g.explicit, g.soft = 1, true, softList
				if !saysSection && !softList {
					g.listAt = typeSpans
				}
				g.extraLists = append(g.extraLists, sectionsListOf(m, page, addIntent))
			}
		} else if g.name == 0 {
			return
		}
		add(g)
	}
	if ctx.Page != "" {
		pageGroup(ctx.Page, true)
	}
	for _, p := range sortedStrings(m.Pages) {
		if p != manifest.CommonPageKey && p != ctx.Page {
			pageGroup(p, false)
		}
	}
	if _, ok := m.Detail("SeoPanel"); ok && ctx.Page != manifest.SEOPage {
		pageGroup(manifest.SEOPage, false)
	}

	// Global panels, the keyed global fields, branding. A page's vocabulary
	// target (vocab.txt "page:") stands in for the viewed page unless the
	// text names that page as a page ("the 'age-gate' page").
	viewPage := ctx.Page
	for _, g := range groups {
		if g.targets[0].Kind == "page" && g.targets[0].Page == ctx.Page && g.explicit && !g.byView {
			viewPage = ""
		}
	}
	for _, p := range sortedStrings(m.GlobalPanels) {
		res, err := m.Resolve(manifest.Target{Kind: "global", Panel: p})
		if err != nil {
			continue
		}
		name, at := nameMatch(toks, phrasesFor("g", p, "Section"))
		g := viewed(group{targets: []manifest.Target{{Kind: "global", Panel: p}}, res: res, name: name, nameAt: at, prior: 0.9, words: words(strings.TrimSuffix(p, "Section"))}, "g:"+p, viewPage)
		g.byTerm = name >= 1 && !namedDirectly(toks, phrasesFor("g", p, "Section"))
		if g.byTerm {
			g.termWords = wordsAt(toks, at)
		}
		g.focused = p == ctx.focusPanel()
		add(g)
	}
	var keyed []cand
	for _, prefix := range sortedStrings(manifest.KeyedPrefixes["global"]) {
		panel := manifest.KeyedPrefixes["global"][prefix]
		if panel == "CategoryDetailsPanel" {
			keyed = append(keyed, keyedCands(m, prefix, panel, ctx.Categories, "category", text, toks)...)
		} else {
			keyed = append(keyed, keyedCands(m, prefix, panel, ctx.Locations, "location", text, toks)...)
		}
	}
	if len(keyed) > 0 {
		add(group{
			targets: []manifest.Target{{Kind: "global"}}, keyedOnly: true, extraFields: keyed, name: 0.7, prior: 0.9,
			res: &manifest.Resolved{Kind: "global", Container: "globalPanels", Known: true, Fields: map[string]*manifest.Field{}, Lists: map[string]*manifest.List{}},
		})
	}
	if res, err := m.Resolve(manifest.Target{Kind: "branding"}); err == nil {
		name, at := nameMatch(toks, brandingPhrases())
		g := viewed(group{targets: []manifest.Target{{Kind: "branding"}}, res: res, name: name, nameAt: at, prior: brandingPrior, words: []string{"branding"}}, "b:branding", viewPage)
		g.byTerm = name >= 1 && !namedDirectly(toks, brandingPhrases())
		if g.byTerm {
			g.termWords = wordsAt(toks, at)
		}
		add(g)
	}
	nameByWord(groups, toks)
	shadowNames(groups)
	return groups
}

// sectionSet is what pageSections found on one page.
type sectionSet struct {
	groups []group
	// named: the text names a section of the page by type, title, id or
	// ordinal; softNamed: only by a title spelled unquoted with its stop
	// words. typeSpans are where it names a section type, regular the
	// page's sections that are not pinned rows.
	named, softNamed bool
	typeSpans        []span
	regular          int
}

// pageSections are the candidate section targets of one page: same-type
// sections the text names alike share one list. focusID is the section the
// merchant has open (named by the focus unless the text names another of
// its type). With onlyNamed (a page other than the viewed one) only the
// sections the text names are candidates.
func pageSections(m *manifest.Manifest, page string, sections []Section, text string, toks []tok, all []rawWord,
	typeNameOf func(string) (float64, []span), focusID string, locations []Named, onlyNamed bool) sectionSet {
	var out sectionSet
	// Pinned rows edit the page's own options and are covered by the page
	// target.
	type sec struct {
		s        Section
		typeID   string
		res      *manifest.Resolved
		name     float64
		at       []span
		explicit bool
		soft     bool // explicit only by its title spelled unquoted with stop words
		focused  bool
	}
	var secs []sec
	byType := map[string][]int{}
	for _, s := range sections {
		if s.ID == "" || manifest.IsPinnedSectionID(s.ID) {
			continue
		}
		out.regular++
		res, err := m.Resolve(manifest.Target{Kind: "section", Page: page, SectionID: s.ID})
		if err != nil {
			continue
		}
		byType[res.TypeID] = append(byType[res.TypeID], len(secs))
		secs = append(secs, sec{s: s, typeID: res.TypeID, res: res})
	}
	for _, typeID := range sortedStrings(byType) {
		idx := byType[typeID]
		phrases := phrasesFor("s", typeID, "")
		typeName, typeAt := typeNameOf(typeID)
		if typeName >= 1 {
			out.typeSpans = append(out.typeSpans, typeAt...)
		}
		var ords []int
		for _, p := range phrases {
			ords = append(ords, ordinalBefore(toks, p)...)
		}
		anyExplicit := false
		for n, i := range idx {
			sc := &secs[i]
			sc.name, sc.at = typeName, typeAt
			if typeName >= 1 {
				sc.name = 0.7
			}
			title := words(sc.s.Title)
			byID := containsWord(text, sc.s.ID)
			switch {
			case byID && (len(idx) > 1 || sc.s.ID != typeID),
				len(title) >= 2 && len(findName(toks, title)) > 0,
				len(title) == 1 && quotedPhrase(toks, title):
				sc.explicit = true
			case len(title) == 1 && spelledWithStops(all, sc.s.Title):
				sc.explicit, sc.soft = true, true
			}
			for _, o := range ords {
				if len(idx) > 1 && (o == n+1 || (o == -1 && n == len(idx)-1)) {
					sc.explicit = true
				}
			}
			if byID {
				sc.explicit, sc.soft = true, false // the id, spelled out
			}
			if sc.explicit {
				sc.name = 1
				anyExplicit = true
			}
			if sc.name >= 0.7 && !sc.soft {
				out.named = true
			}
			out.softNamed = out.softNamed || sc.soft
		}
		if !anyExplicit && focusID != "" {
			// The open section: the instance a type name means ("make the
			// carousel show 5"), else a target of its own.
			for _, i := range idx {
				if secs[i].s.ID != focusID {
					continue
				}
				sc := &secs[i]
				sc.focused = true
				if typeName >= 1 {
					sc.explicit, sc.name, anyExplicit = true, 1, true
				}
			}
		}
		if anyExplicit {
			// The text picked instances of this type: the others are out.
			for _, i := range idx {
				if !secs[i].explicit {
					secs[i].res = nil
				}
			}
		}
	}
	// Same-type sections named alike share one list.
	type gkey struct {
		typeID                  string
		explicit, soft, focused bool
	}
	at := map[gkey]int{}
	for _, sc := range secs {
		if sc.res == nil || (onlyNamed && sc.name < 0.7) {
			continue
		}
		t := manifest.Target{Kind: "section", Page: page, SectionID: sc.s.ID}
		k := gkey{sc.typeID, sc.explicit, sc.soft, sc.focused}
		if i, ok := at[k]; ok {
			out.groups[i].targets = append(out.groups[i].targets, t)
			continue
		}
		g := group{targets: []manifest.Target{t}, res: sc.res, name: sc.name, nameAt: sc.at, explicit: sc.explicit, soft: sc.soft, prior: 1, words: words(sc.typeID), focused: sc.focused}
		if onlyNamed {
			g.prior = otherPagePrior
		}
		// Keyed per-location fields, when the type has them and the text
		// mentions a location.
		for _, prefix := range sortedStrings(manifest.KeyedPrefixes["section"]) {
			has := false
			for k := range sc.res.Fields {
				has = has || strings.HasPrefix(k, prefix+".")
			}
			if has {
				g.extraFields = append(g.extraFields, keyedCands(m, prefix, manifest.KeyedPrefixes["section"][prefix], locations, "location", text, toks)...)
			}
		}
		at[k] = len(out.groups)
		out.groups = append(out.groups, g)
	}
	return out
}

// otherPagePrior: a section of another page the text names competes a
// little below the viewed page's own.
const otherPagePrior = 0.95

// panelNameSpans are the places the text names a global panel or branding:
// the longer names a section type's alias may lie inside.
func panelNameSpans(m *manifest.Manifest, toks []tok) []span {
	var out []span
	for _, p := range sortedStrings(m.GlobalPanels) {
		if n, at := nameMatch(toks, phrasesFor("g", p, "Section")); n >= 1 {
			out = append(out, at...)
		}
	}
	if n, at := nameMatch(toks, brandingPhrases()); n >= 1 {
		out = append(out, at...)
	}
	return out
}

// brandingPhrases name the branding target (the age gate and the location
// picker).
func brandingPhrases() [][]string {
	return append([][]string{{"branding"}, {"brand"}}, aliasesOf("b:branding")...)
}

// unshadowed drops the spans that lie inside a longer one of outer.
func unshadowed(at, outer []span) []span {
	var out []span
	for _, a := range at {
		inside := false
		for _, o := range outer {
			inside = inside || (a.within(o) && o.to-o.from > a.to-a.from)
		}
		if !inside {
			out = append(out, a)
		}
	}
	return out
}

// brandingPrior: the branding target holds the age gate's and the location
// picker's options (and the certification badges), keyed ageGate* /
// locationPicker* but often labelled without the qualifier ("Logo Size
// (Mobile)"). Unless the text names it (an alias:
// "age gate", "store picker" …, or the page being viewed), its fields
// should not outrank the panel a site-wide word means: "make the logo
// smaller on mobile" is the Branding panel's logo, not the age gate's.
const brandingPrior = 0.6

// viewed names g when the merchant views a page the vocabulary ties it to
// (vocab.txt "page:<slug>"): on the age-gate page, branding holds the age
// gate's options.
func viewed(g group, key, page string) group {
	if g.name < 0.7 && viewedOn(key, page) {
		g.name, g.viaView = 0.7, true
	}
	return g
}

// shadowNames un-names a target whose every name lies inside a longer name
// of another target: "the insider banner" names the Insider Banner panel,
// not the hero (whose alias is "banner").
func shadowNames(groups []group) {
	shadowed := make([]bool, len(groups))
	for i := range groups {
		g := &groups[i]
		if (g.explicit && !g.byView) || g.name < namedMin || len(g.nameAt) == 0 {
			continue
		}
		shadowed[i] = true
		for _, at := range g.nameAt {
			inside := false
			for j := range groups {
				if j == i || groups[j].name < namedMin {
					continue
				}
				for _, o := range groups[j].nameAt {
					inside = inside || (at.within(o) && o.to-o.from > at.to-at.from)
				}
			}
			shadowed[i] = shadowed[i] && inside
		}
	}
	for i := range groups {
		if shadowed[i] {
			groups[i].nameAt = nil
			if !groups[i].byView { // the viewed page stays its default subject
				groups[i].name = 0
			}
		}
	}
}

// nameByWord names a target by a single word of its id when no other
// candidate target shares that word: "the featured section" is
// featuredCategories, "the carousel" the product carousel, "the sidebar"
// the sidebar cart. Weaker than a full name (namedMin), and subject to the
// same in-passing check.
func nameByWord(groups []group, toks []tok) {
	owners := map[string]map[string]bool{} // word → containers whose id has it
	for _, g := range groups {
		for _, w := range g.words {
			if owners[w] == nil {
				owners[w] = map[string]bool{}
			}
			owners[w][g.res.Container] = true
		}
	}
	for i := range groups {
		g := &groups[i]
		if g.name >= namedMin || len(g.words) < 2 {
			continue
		}
		for _, w := range g.words {
			if len(owners[w]) != 1 || genericWords[w] {
				continue
			}
			for j, t := range toks {
				if t.readsAs(w) && !t.quoted {
					g.name = namedMin
					g.nameAt = append(g.nameAt, span{j, j + 1})
				}
			}
		}
	}
}

// genericWords never name a target on their own.
var genericWords = wordSet("section page block content form item list text color image title link button menu")

// sectionsListOf is the page's own section list, which the engine
// synthesises (manifest.SectionsListKey): never a manifest list.
func sectionsListOf(m *manifest.Manifest, page string, withTypes bool) extraList {
	ops := []string{}
	for _, op := range manifest.SectionsListOps {
		if op != "update" {
			ops = append(ops, op)
		}
	}
	typeDesc := "text"
	if withTypes {
		var types []string
		for _, id := range sortedStrings(m.Sections) {
			for _, p := range m.Sections[id].Pages {
				if p == page {
					types = append(types, id)
					break
				}
			}
		}
		if len(types) > 0 {
			typeDesc = "text; types this page takes: " + strings.Join(types, "|")
		}
	}
	mk := func(key, label, desc string) cand {
		return cand{key: key, list: manifest.SectionsListKey, spec: &manifest.Field{Label: label, Type: "text"}, desc: desc,
			label: words(label), keyw: words(key), extra: map[string]bool{}}
	}
	return extraList{
		info: listInfo{
			key: manifest.SectionsListKey, spec: &manifest.List{Ops: ops}, named: true, items: 3,
			note: "edit_list; " + strings.Join(manifest.SectionsListOps, "|") + "; index = the section's position on the page, from 0; order = the section ids",
		},
		items: []cand{
			mk("enabled", "Section shown on the page", "bool"),
			mk("id", "Section id", "text"),
			mk("type", "Section type", typeDesc),
		},
	}
}

func sortedStrings[V any](m map[string]V) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// ── scoring ─────────────────────────────────────────────────────────────────

// listNoise are words of list keys that say nothing about the list.
var listNoise = map[string]bool{"page": true, "visibility": true, "meta": true}

// labelWords are a label's words without its parenthesised asides
// ("Tagline Image (optional)" → tagline, image).
func labelWords(label string) []string {
	var sb strings.Builder
	depth := 0
	for _, r := range label {
		switch {
		case r == '(':
			depth++
		case r == ')' && depth > 0:
			depth--
			sb.WriteRune(' ')
		case depth == 0:
			sb.WriteRune(r)
		}
	}
	if w := words(sb.String()); len(w) > 0 {
		return w
	}
	return words(label)
}

func scoreGroup(m *manifest.Manifest, g *group, text string, toks []tok) {
	res := g.res
	mirrorList := ""
	if res.Mirror != nil {
		mirrorList = strings.SplitN(firstNonEmpty(res.Mirror.From, "slides[0]"), "[", 2)[0]
	}
	addField := func(key, list string, f *manifest.Field) {
		// A read-only field cannot be written; a templated key
		// (locationOverrides.{locationId}.mode) is not a key the engine
		// takes — keyedCands instantiates the ones it does.
		if f == nil || f.ReadOnly() || strings.Contains(key, "{") {
			return
		}
		c := cand{key: key, list: list, spec: f, label: labelWords(f.Label), keyw: words(key), extra: map[string]bool{}}
		for _, w := range words(f.Label) {
			c.extra[w] = true // the label's asides
		}
		for _, w := range words(f.Type) {
			c.extra[w] = true
		}
		for _, v := range f.Enum {
			s := enumValue(v)
			for _, w := range words(s) {
				c.extra[w] = true
			}
			for _, w := range words(f.EnumLabels[s]) {
				c.extra[w] = true
			}
		}
		g.cands = append(g.cands, c)
	}
	for _, k := range manifest.SortedFieldKeys(res.Fields) {
		addField(k, "", res.Fields[k])
	}
	g.cands = append(g.cands, g.extraFields...)
	g.total = len(g.cands) - len(g.extraFields)
	textSet := map[string]bool{}
	for _, t := range toks {
		if !t.quoted {
			textSet[t.w] = true
			for _, a := range t.alts {
				for _, w := range a.w {
					textSet[w] = true
				}
			}
		}
	}
	for _, lk := range sortedStrings(res.Lists) {
		l := res.Lists[lk]
		items, _ := m.ItemFields(l)
		li := listInfo{key: lk, spec: l, mirror: lk == mirrorList && mirrorList != "", items: len(items)}
		for _, w := range words(lk) {
			if !listNoise[w] {
				li.name = append(li.name, w)
			}
		}
		if n := len(li.name); n > 0 {
			hit := 0
			for _, w := range li.name {
				if textSet[w] {
					hit++
				}
			}
			li.named = hit == n || (textSet[li.name[n-1]] && hit*2 >= n)
		}
		if li.mirror {
			g.total += len(items)
		}
		g.lists = append(g.lists, li)
		for _, k := range manifest.SortedFieldKeys(items) {
			addField(k, lk, items[k])
		}
	}
	for _, el := range g.extraLists {
		g.lists = append(g.lists, el.info)
		g.cands = append(g.cands, el.items...)
	}

	// idf over the container's fields: a word most fields share (color)
	// says little, a rare one (tagline) says a lot.
	df := map[string]int{}
	for i := range g.cands {
		seen := map[string]bool{}
		c := &g.cands[i]
		for _, ws := range [][]string{c.label, c.keyw} {
			for _, w := range ws {
				if !seen[w] {
					seen[w] = true
					df[w]++
				}
			}
		}
		for w := range c.extra {
			if !seen[w] {
				seen[w] = true
				df[w]++
			}
		}
	}
	n := float64(len(g.cands))
	idf := func(w string) float64 { return math.Log(1 + n/float64(max(df[w], 1))) }

	// The text's distinct words and the vocabulary's readings of them, in
	// text order.
	weight, order := textWords(toks, quotedWeight)
	written := map[string]bool{} // the text's own words
	for _, t := range toks {
		written[t.w] = true
	}
	for i := range g.cands {
		c := &g.cands[i]
		inLabel := map[string]bool{}
		for _, w := range c.label {
			inLabel[w] = true
		}
		inKey := map[string]bool{}
		for _, w := range c.keyw {
			inKey[w] = true
		}
		score, own := 0.0, false
		for _, w := range order {
			switch {
			case inLabel[w]:
				score += weight[w] * idf(w)
			case inKey[w]:
				score += 0.7 * weight[w] * idf(w)
			case c.extra[w]:
				score += 0.25 * weight[w] * idf(w)
			default:
				continue
			}
			own = own || written[w]
		}
		// How much of the label the text covers, and whether it spells
		// the label out.
		mass, hit := 0.0, 0.0
		counted := map[string]bool{}
		for _, w := range c.label {
			if counted[w] {
				continue
			}
			counted[w] = true
			mass += idf(w)
			if _, ok := weight[w]; ok {
				hit += idf(w)
			}
		}
		if mass > 0 {
			score *= 0.6 + 0.4*hit/mass
			if at := findSpelled(toks, c.label); len(at) > 0 {
				score += phraseBonus * mass
				g.labelAt = append(g.labelAt, at...)
			}
		}
		// The key itself, written out ("(answerUppercase)").
		if len(c.key) >= 4 && containsWord(text, c.key) {
			score += keyBonus
		}
		if c.list != "" && !g.listMirror(c.list) {
			score *= 0.8 // an item field of a list needs a clearer match
		}
		c.score = score
		if score > g.top {
			g.top = score
		}
		if own && score > g.ownTop {
			g.ownTop = score
		}
	}
}

func (g *group) listMirror(key string) bool {
	l := g.list(key)
	return l != nil && l.mirror
}

func (g *group) list(key string) *listInfo {
	for i := range g.lists {
		if g.lists[i].key == key {
			return &g.lists[i]
		}
	}
	return nil
}

func firstNonEmpty(a, b string) string {
	if a != "" {
		return a
	}
	return b
}

// incidental reports whether every place the text names g lies inside a
// field label another named target spells out: "Secondary Button Border
// Color of the content block" names the Buttons panel only in passing.
func incidental(g *group, named []group) bool {
	if g.explicit || len(g.nameAt) == 0 {
		return g.keyedOnly
	}
	for _, at := range g.nameAt {
		covered := false
		for i := range named {
			h := &named[i]
			if h.order == g.order {
				continue
			}
			for _, l := range h.labelAt {
				covered = covered || (at.within(l) && at != l) || (at == l && h.explicit)
			}
		}
		if !covered {
			return false
		}
	}
	return true
}

// listIncidental reports whether the viewed page was made the subject of a
// section-list edit only by section-type names that lie inside a field label
// another named target spells out: "Restore the Show banner on checkout of
// the insider banner" restructures no hero.
func listIncidental(g *group, named []group) bool {
	if len(g.listAt) == 0 {
		return false
	}
	for _, at := range g.listAt {
		covered := false
		for i := range named {
			h := &named[i]
			if h.order == g.order {
				continue
			}
			for _, l := range h.labelAt {
				covered = covered || (at.within(l) && at != l)
			}
		}
		if !covered {
			return false
		}
	}
	return true
}

func byMatch(gs []group) func(i, j int) bool {
	return func(i, j int) bool {
		a, b := gs[i].top*gs[i].prior, gs[j].top*gs[j].prior
		if a != b {
			return a > b
		}
		return gs[i].order < gs[j].order
	}
}

// selectGroups keeps the targets worth listing, best first.
func selectGroups(groups []group, opts Options) []group {
	var named, rest, hinted, focused []group
	bestNamed := 0.0
	for _, g := range groups {
		if g.name >= namedMin && hintOnly(&g, groups) {
			// Named only through a term's reading: placed below, when one
			// of its fields matches.
			if g.top > 0 {
				hinted = append(hinted, g)
			}
			continue
		}
		if g.focused && g.name < namedMin {
			// The open section or panel the text does not name: placed
			// below, when one of its fields matches.
			if g.top > 0 {
				focused = append(focused, g)
			}
			continue
		}
		if g.name >= namedMin {
			named = append(named, g)
			if g.top > bestNamed {
				bestNamed = g.top
			}
		} else {
			rest = append(rest, g)
		}
	}
	sort.SliceStable(rest, byMatch(rest))
	var out []group
	if len(named) > 0 {
		for i := range named {
			g := named[i]
			// A target named only in passing must match on fields about
			// as well as the best named one to stay.
			if (incidental(&g, named) || listIncidental(&g, named)) && (g.top <= 0 || g.top < 0.5*bestNamed) {
				continue
			}
			out = append(out, g)
		}
		// The viewed page is the subject by default only: a target the text
		// names (or the viewed page's own vocabulary target) comes first.
		weak := func(g *group) bool { return g.soft || (g.byView && len(g.nameAt) == 0) }
		textNamed := false
		for i := range out {
			g := &out[i]
			// … by a name, or as a section instance (title, id, ordinal).
			textNamed = textNamed || (!weak(g) && (len(g.nameAt) > 0 || g.viaView || g.byOutline || (g.explicit && g.targets[0].Kind == "section")))
		}
		for i := range out {
			if (textNamed || len(focused) > 0) && weak(&out[i]) {
				out[i].explicit = false
			}
		}
		sort.SliceStable(out, func(i, j int) bool {
			a, b := out[i], out[j]
			if a.explicit != b.explicit {
				return a.explicit
			}
			if a.top != b.top {
				return a.top > b.top
			}
			return a.order < b.order
		})
		// No named target has a field the text spells out, an unnamed
		// one does: the name match was probably not the subject.
		spelled := false
		for _, g := range out {
			spelled = spelled || len(g.labelAt) > 0
		}
		var extras []group
		for _, g := range rest {
			if !spelled && len(extras) < 2 && len(g.labelAt) > 0 {
				extras = append(extras, g)
			}
		}
		// … and it outranks a target that is named only weakly (the viewed
		// page by default, an unquoted title) and whose fields match only
		// through the vocabulary's readings: "make the indica tag purple" on
		// the products page spells the Product Card's "Indica", while the
		// page has only colour fields that "purple" reads into.
		var strong, weakOnes []group
		for _, g := range out {
			if len(extras) > 0 && weak(&g) && g.ownTop <= 0 {
				weakOnes = append(weakOnes, g)
			} else {
				strong = append(strong, g)
			}
		}
		out = append(append(strong, extras...), weakOnes...)
	} else {
		for _, g := range rest {
			if g.top > 0 && g.top*g.prior >= 0.6*rest[0].top*rest[0].prior {
				out = append(out, g)
			}
		}
	}
	// A hint whose every reading lies inside a field label a listed target
	// spells is that label's word, not a hint ("the Marker Font of the
	// content block" is no Typography request).
	kept := hinted[:0]
	for _, g := range hinted {
		if !spelledElsewhere(&g, out) {
			kept = append(kept, g)
		}
	}
	hinted = kept
	if boosted := append(hinted, focused...); len(boosted) > 0 {
		// After the targets the text names (explicitly, or by a name it
		// spells), before the rest.
		at := 0
		for at < len(out) && strongly(&out[at]) {
			at++
		}
		out = append(out[:at:at], append(boosted, out[at:]...)...)
	}
	if len(out) > opts.MaxTargets {
		out = out[:opts.MaxTargets]
	}
	return out
}

// spelledElsewhere reports whether every place the text names g lies
// inside a longer field label one of the others spells out.
func spelledElsewhere(g *group, others []group) bool {
	if len(g.nameAt) == 0 {
		return false
	}
	for _, at := range g.nameAt {
		covered := false
		for i := range others {
			if others[i].order == g.order {
				continue
			}
			for _, l := range others[i].labelAt {
				covered = covered || (at.within(l) && l.to-l.from > at.to-at.from)
			}
		}
		if !covered {
			return false
		}
	}
	return true
}

// strongly reports whether the text itself names g: an instance (id, title,
// ordinal, a page it says), or a name it spells; not the viewed page by
// default, an unquoted title, or a target named only by its fields.
func strongly(g *group) bool {
	if g.name < namedMin || g.soft || (g.byView && len(g.nameAt) == 0) {
		return false
	}
	return g.explicit || len(g.nameAt) > 0 || g.viaView || g.byOutline
}

// chooseFields picks the fields to list under one target.
func chooseFields(g *group, first bool, opts Options) {
	idx := make([]int, 0, len(g.cands))
	for i := range g.cands {
		idx = append(idx, i)
	}
	sort.SliceStable(idx, func(a, b int) bool {
		ca, cb := g.cands[idx[a]], g.cands[idx[b]]
		if ca.score != cb.score {
			return ca.score > cb.score
		}
		if ca.list != cb.list {
			return ca.list < cb.list
		}
		return ca.key < cb.key
	})
	fill := fillRest
	if first {
		fill = fillFirst
	}
	all := g.total <= opts.ListAll
	var picked []int
	matched := 0
	for _, i := range idx {
		c := g.cands[i]
		li := g.list(c.list)
		if li != nil && li.spec.Orders != "" {
			continue // a list of ids: reordered whole, its rows have no fields to write
		}
		if (c.list == "" && all && !c.keyed) || (li != nil && li.named && li.items <= opts.ListAll) {
			picked = append(picked, i)
			continue
		}
		if c.score <= 0 || matched >= opts.MaxFields {
			continue
		}
		if c.score >= keepRatio*g.top || matched < fill {
			picked = append(picked, i)
			matched++
		}
	}
	g.fields = picked // best first; render sorts by key
}

// shownLists are the lists the block shows for a group: those the text
// names and those with a listed item field.
func shownLists(g *group) []string {
	var out []string
	for _, l := range g.lists {
		show := l.named
		for _, i := range g.fields {
			if g.cands[i].list == l.key {
				show = true
			}
		}
		if show {
			out = append(out, l.key)
		}
	}
	return out
}

// trim drops the weakest listed field of the longest list (never below
// minKeep), then the last target. false when nothing is left to drop.
func trim(groups *[]group) bool {
	gs := *groups
	longest := -1
	for i := range gs {
		if len(gs[i].fields) > minKeep && (longest < 0 || len(gs[i].fields) > len(gs[longest].fields)) {
			longest = i
		}
	}
	if longest >= 0 {
		g := &gs[longest]
		worst := 0
		for j, i := range g.fields {
			if g.cands[i].score <= g.cands[g.fields[worst]].score {
				worst = j
			}
		}
		g.fields = append(g.fields[:worst:worst], g.fields[worst+1:]...)
		return true
	}
	if len(gs) > 1 {
		*groups = gs[:len(gs)-1]
		return true
	}
	return false
}

// ── rendering ───────────────────────────────────────────────────────────────

func render(groups []group) string {
	var sb strings.Builder
	sb.WriteString(blockOpen)
	sb.WriteString(blockIntro)
	for gi := range groups {
		g := &groups[gi]
		for _, t := range g.targets {
			sb.WriteString("target ")
			sb.WriteString(targetJSON(t))
			sb.WriteString("\n")
		}
		writeFields(&sb, g, "")
		for _, lk := range shownLists(g) {
			l := g.list(lk)
			sb.WriteString("list ")
			sb.WriteString(lk)
			sb.WriteString(" (")
			sb.WriteString(firstNonEmpty(l.note, listNote(l)))
			sb.WriteString(")\n")
			writeFields(&sb, g, lk)
		}
	}
	sb.WriteString(blockClose)
	return sb.String()
}

func writeFields(sb *strings.Builder, g *group, list string) {
	var idx []int
	for _, i := range g.fields {
		if g.cands[i].list == list {
			idx = append(idx, i)
		}
	}
	sort.Slice(idx, func(a, b int) bool { return g.cands[idx[a]].key < g.cands[idx[b]].key })
	for _, i := range idx {
		c := g.cands[i]
		sb.WriteString("  ")
		sb.WriteString(c.key)
		if label := clip(oneLine(c.spec.Label), 60); label != "" {
			sb.WriteString(" — ")
			sb.WriteString(label)
		}
		sb.WriteString(" (")
		sb.WriteString(firstNonEmpty(c.desc, describe(c.spec)))
		sb.WriteString(")\n")
	}
}

// targetJSON renders an op target with its members in the engine's order.
func targetJSON(t manifest.Target) string {
	q := func(s string) string { b, _ := json.Marshal(s); return string(b) }
	switch t.Kind {
	case "section":
		return `{"kind":"section","page":` + q(t.Page) + `,"sectionId":` + q(t.SectionID) + `}`
	case "global":
		if t.Panel == "" {
			return `{"kind":"global"}`
		}
		return `{"kind":"global","panel":` + q(t.Panel) + `}`
	case "page":
		return `{"kind":"page","page":` + q(t.Page) + `}`
	}
	return `{"kind":` + q(t.Kind) + `}`
}

func listNote(l *listInfo) string {
	parts := []string{"edit_list"}
	ops := append([]string(nil), l.spec.Ops...)
	sort.Strings(ops)
	ops = append(ops, "update")
	parts = append(parts, strings.Join(ops, "|"))
	if l.spec.Orders != "" {
		parts = append(parts, "order = every id of the list ("+l.spec.Orders+") in the new order")
	}
	if manifest.ListDepth(l.key) > 1 {
		parts = append(parts, "index [parent, child]")
	}
	if l.mirror {
		parts = append(parts, "the section's own fields are slide index 0")
	}
	return strings.Join(parts, "; ")
}

// describe is the parenthesised part of a field line: the type and what a
// value may be.
func describe(f *manifest.Field) string {
	t := f.Type
	if t == "" {
		t = "text"
	}
	switch {
	case len(f.Enum) > 0:
		return t + " " + enumText(f)
	case f.Min != nil || f.Max != nil:
		s := t + " " + num(f.Min) + ".." + num(f.Max)
		if f.Step != nil && *f.Step != 1 {
			s += " step " + num(f.Step)
		}
		return s
	case f.Ref != "":
		return t + " " + f.Ref
	}
	return t
}

const (
	enumShown    = 8
	enumValMax   = 32
	enumLabelMax = 24
	enumTextMax  = 160
)

func enumText(f *manifest.Field) string {
	parts := make([]string, 0, enumShown)
	size := 0
	for _, v := range f.Enum {
		s := enumValue(v)
		if len(s) > enumValMax {
			continue
		}
		if lab := oneLine(f.EnumLabels[s]); lab != "" && len(lab) <= enumLabelMax && strings.Join(words(lab), " ") != strings.Join(words(s), " ") {
			s += "=" + lab
		}
		if len(parts) >= enumShown || size+len(s) > enumTextMax {
			break
		}
		parts = append(parts, s)
		size += len(s) + 1
	}
	out := strings.Join(parts, "|")
	if rest := len(f.Enum) - len(parts); rest > 0 {
		if out != "" {
			out += "|"
		}
		out += "…+" + strconv.Itoa(rest)
	}
	return out
}

// enumValue prints one enum member the way the op takes it.
func enumValue(v any) string {
	switch x := v.(type) {
	case nil:
		return "null"
	case string:
		return x
	case float64:
		return strconv.FormatFloat(x, 'f', -1, 64)
	}
	return fmt.Sprint(v)
}

func num(p *float64) string {
	if p == nil {
		return ""
	}
	return strconv.FormatFloat(*p, 'f', -1, 64)
}

func oneLine(s string) string { return strings.Join(strings.Fields(s), " ") }

func clip(s string, n int) string {
	rs := []rune(s)
	if len(rs) <= n {
		return s
	}
	return string(rs[:n-1]) + "…"
}

func export(g *group) Target {
	t := Target{Targets: g.targets, Container: g.res.Container, Named: g.name >= namedMin, Total: g.total, Lists: shownLists(g)}
	emit := func(list string) {
		var idx []int
		for _, i := range g.fields {
			if g.cands[i].list == list {
				idx = append(idx, i)
			}
		}
		sort.Slice(idx, func(a, b int) bool { return g.cands[idx[a]].key < g.cands[idx[b]].key })
		for _, i := range idx {
			c := g.cands[i]
			t.Fields = append(t.Fields, Field{Key: c.key, List: c.list, Label: c.spec.Label, Type: c.spec.Type, Score: math.Round(c.score*1000) / 1000})
		}
	}
	emit("")
	for _, lk := range t.Lists {
		emit(lk)
	}
	return t
}
