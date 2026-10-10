package fields

import (
	_ "embed"
	"fmt"
	"strings"
)

// The merchant vocabulary (vocab.txt): what a merchant calls things that
// the editor's labels and keys call something else. "Change the site name"
// means BrandingSection.storeName ("Store Name"), "rounder corners" a
// *Radius field, "hide the THC pill" cardShowTHC.
//
// The manifest is derived from the editor code by CI and grows as
// developers add fields, so the vocabulary is mostly TERMS — merchant word
// → label/key word — applied to the text at query time: a new field whose
// label says "Border Radius" is found by "rounder" without anyone touching
// this file. A small table of CONTAINER ALIASES names sections, panels,
// branding and pages ("site name" → the Branding panel); the prompt
// index prints it too (prompt.RenderIndex, "Also called"). How to add a
// line: docs/VOCABULARY.md.
//
// A term never replaces what the merchant wrote: the text's words keep
// their weight and the term's words are read beside them at synWeight, so
// a label the text spells out still outranks one only a synonym reaches.

//go:embed vocab.txt
var vocabText string

// synWeight is the weight of a word the vocabulary adds, against 1 for a
// word of the text (quotedWeight scales both alike).
const synWeight = 0.8

// term is one vocabulary line "from = to[, to…]": from are the normalised
// words of the merchant phrase (stop words kept: "turn on" ≠ "turn"), each
// of to the normalised, stop-word-free words it also reads as.
type term struct {
	from []string
	to   [][]string
}

// alt is a reading a term adds to the text: starting at a word of the text,
// n words (stop words not counted) also read as w.
type alt struct {
	n int
	w []string
}

var (
	// termsByFirst indexes the terms by the first word of their merchant
	// phrase.
	termsByFirst map[string][]term
	// containerAliases maps "<kind>:<id>" (s:heroBanner, g:BrandingSection,
	// b:branding, p:seo) to the phrases that name it, as written.
	containerAliases map[string][]string
)

func init() {
	terms, aliases, err := parseVocab(vocabText)
	if err != nil {
		panic(err)
	}
	termsByFirst = map[string][]term{}
	for _, t := range terms {
		termsByFirst[t.from[0]] = append(termsByFirst[t.from[0]], t)
	}
	containerAliases = aliases
}

// parseVocab reads vocab.txt. A malformed line is an error (the package
// panics at init: a bad edit fails every test, not a turn).
func parseVocab(s string) ([]term, map[string][]string, error) {
	var terms []term
	aliases := map[string][]string{}
	for n, line := range strings.Split(s, "\n") {
		if i := strings.Index(line, "#"); i >= 0 {
			line = line[:i]
		}
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		lhs, rhs, ok := strings.Cut(line, "=")
		lhs, rhs = strings.TrimSpace(lhs), strings.TrimSpace(rhs)
		if !ok || lhs == "" || rhs == "" {
			return nil, nil, fmt.Errorf("fields: vocab.txt:%d: want \"merchant words = editor words\" or \"@kind:id = alias, …\"", n+1)
		}
		var rights []string
		for _, r := range strings.Split(rhs, ",") {
			if r = strings.TrimSpace(r); r != "" {
				rights = append(rights, r)
			}
		}
		if strings.HasPrefix(lhs, "@") {
			key := lhs[1:]
			kind, id, _ := strings.Cut(key, ":")
			if id == "" || !(kind == "s" || kind == "g" || kind == "b" || kind == "p") {
				return nil, nil, fmt.Errorf("fields: vocab.txt:%d: container %q is not s:<type>, g:<panel>, b:branding or p:<page>", n+1, lhs)
			}
			aliases[key] = append(aliases[key], rights...)
			continue
		}
		t := term{from: phraseWords(lhs)}
		if len(t.from) == 0 {
			return nil, nil, fmt.Errorf("fields: vocab.txt:%d: %q has no words", n+1, lhs)
		}
		for _, r := range rights {
			w := words(r)
			if len(w) == 0 {
				return nil, nil, fmt.Errorf("fields: vocab.txt:%d: %q has no words once stop words are dropped", n+1, r)
			}
			t.to = append(t.to, w)
		}
		terms = append(terms, t)
	}
	return terms, aliases, nil
}

// phraseWords are the normalised words of a merchant phrase with its stop
// words kept: what a term's left side is matched against.
func phraseWords(s string) []string {
	var out []string
	for _, w := range split(s) {
		if n := norm(w); n != "" {
			out = append(out, n)
		}
	}
	return out
}

// ContainerAliases are the merchant names of the manifest's containers,
// keyed "<kind>:<id>" the way the prompt index writes them (s:heroBanner,
// g:BrandingSection, b:branding, p:seo), in file order; an entry
// "page:<slug>" is a page whose viewing names the container. The caller
// must not modify them.
func ContainerAliases() map[string][]string { return containerAliases }

// viewPrefix marks an alias that is a page: the container counts as named
// while the merchant views that page ("page:age-gate").
const viewPrefix = "page:"

// AlsoCalled are the aliases of one container ("<kind>:<id>") worth
// printing in the prompt index, at most max, in file order: not the
// page-view entries, not those the id already says ("hero" for heroBanner),
// not a phrase that only extends one already printed ("top banner" after
// "banner"), not a second spelling of one ("member banner").
func AlsoCalled(key string, max int) []string {
	_, id, _ := strings.Cut(key, ":")
	idWords := map[string]bool{}
	for _, w := range words(id) {
		idWords[w] = true
	}
	var out []string
	var printed [][]string
next:
	for _, a := range containerAliases[key] {
		if len(out) >= max {
			break
		}
		if strings.HasPrefix(a, viewPrefix) {
			continue
		}
		w := words(a)
		said := true
		for _, x := range w {
			said = said && idWords[x]
		}
		if said {
			printed = append(printed, w) // implied by the id: so are its extensions
			continue
		}
		for _, p := range printed {
			if containsSeq(w, p) {
				continue next
			}
		}
		out = append(out, a)
		printed = append(printed, w)
	}
	return out
}

// containsSeq reports whether seq occurs contiguously in w.
func containsSeq(w, seq []string) bool {
outer:
	for i := 0; i+len(seq) <= len(w); i++ {
		for j, x := range seq {
			if w[i+j] != x {
				continue outer
			}
		}
		return true
	}
	return false
}

// aliasesOf are the alias phrases of one container, as word sequences.
func aliasesOf(key string) [][]string {
	var out [][]string
	for _, a := range containerAliases[key] {
		if !strings.HasPrefix(a, viewPrefix) {
			out = append(out, words(a))
		}
	}
	return out
}

// viewedOn reports whether the container is named while page is viewed.
func viewedOn(key, page string) bool {
	if page == "" {
		return false
	}
	for _, a := range containerAliases[key] {
		if a == viewPrefix+page {
			return true
		}
	}
	return false
}

// rawWord is one word of the merchant's text, stop words included: kept is
// its index among the tokens, -1 for a stop word.
type rawWord struct {
	w     string
	kept  int
	span  int  // the quoted run, as tok.span
	value bool // inside a value run
	ident bool // part of an identifier (our-story, contentBlock-2)
}

// expand attaches to toks the readings the vocabulary adds. A term's words
// match the text's in order, never inside a value run; a stop word of the
// term must be the text's next word ("turn on", "above and below"), while
// before any other word of the term the text's stop words are skipped
// ("rename site" matches "rename my site"). The reading attaches to the
// first non-stop word the match covers.
func expand(all []rawWord, toks []tok) {
	for i := range all {
		for _, t := range termsByFirst[all[i].w] {
			first, n, ok := -1, 0, true
			k := i
			for j, w := range t.from {
				if j > 0 && !stop[w] {
					for k < len(all) && all[k].kept < 0 {
						k++
					}
				}
				if k >= len(all) || all[k].w != w || all[k].value {
					ok = false
					break
				}
				if all[k].kept >= 0 {
					if first < 0 {
						first = all[k].kept
					}
					n = all[k].kept - first + 1
				}
				k++
			}
			if !ok || first < 0 {
				continue
			}
			for _, to := range t.to {
				toks[first].addAlt(alt{n: n, w: to})
			}
		}
	}
}

func (t *tok) addAlt(a alt) {
	if a.n == 1 && len(a.w) == 1 && a.w[0] == t.w {
		return
	}
	for _, b := range t.alts {
		if b.n == a.n && strings.Join(b.w, " ") == strings.Join(a.w, " ") {
			return
		}
	}
	t.alts = append(t.alts, a)
}

// readsAs reports whether the text word t reads as w, itself or through a
// one-word term.
func (t tok) readsAs(w string) bool {
	if t.w == w {
		return true
	}
	for _, a := range t.alts {
		if a.n == 1 && len(a.w) == 1 && a.w[0] == w {
			return true
		}
	}
	return false
}

// matchFrom is the end of an occurrence of seq starting at toks[i] — word
// for word, or through a term's reading — or -1.
func matchFrom(toks []tok, i int, seq []string) int {
	if len(seq) == 0 {
		return i
	}
	if i >= len(toks) {
		return -1
	}
	if toks[i].w == seq[0] {
		if e := matchFrom(toks, i+1, seq[1:]); e >= 0 {
			return e
		}
	}
	for _, a := range toks[i].alts {
		if len(a.w) > len(seq) || i+a.n > len(toks) {
			continue
		}
		same := true
		for k, w := range a.w {
			same = same && seq[k] == w
		}
		if same {
			if e := matchFrom(toks, i+a.n, seq[len(a.w):]); e >= 0 {
				return e
			}
		}
	}
	return -1
}

// textWords are the words the text reads as — its own and the terms'
// readings — each with the larger of its weights, in text order (float
// sums must not depend on map order).
func textWords(toks []tok, quotedW float64) (map[string]float64, []string) {
	weight := map[string]float64{}
	var order []string
	put := func(w string, x float64) {
		if old, ok := weight[w]; !ok {
			order = append(order, w)
			weight[w] = x
		} else if x > old {
			weight[w] = x
		}
	}
	for _, t := range toks {
		w := 1.0
		if t.quoted {
			w = quotedW
		}
		put(t.w, w)
		for _, a := range t.alts {
			for _, aw := range a.w {
				put(aw, synWeight*w)
			}
		}
	}
	return weight, order
}

// QueryReadings are the phrases the vocabulary reads a lookup query as
// ("site name" → "store name", "rounder corners" → "radius"), each once, in
// query order: lookup_fields matches them after the query itself
// (manifest.LookupRequest.Readings).
func QueryReadings(query string) []string {
	toks := textToks(query)
	var out []string
	seen := map[string]bool{}
	for _, t := range toks {
		for _, a := range t.alts {
			r := strings.Join(a.w, " ")
			if !seen[r] {
				seen[r] = true
				out = append(out, r)
			}
		}
	}
	return out
}
