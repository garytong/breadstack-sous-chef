package fields

import (
	"strings"
	"sync"
	"sync/atomic"
	"unicode"
)

// A tok is one normalised word of the merchant's text. Words inside quotes
// are usually a value ("… to 'Fresh picks every week'"), so they count for
// less when fields are scored; they still take part in phrase matches (a
// quoted section title or a quoted label).
//
// A quoted run is one of two things. It is the VALUE being set when the
// words before it say so ("… to 'X'", "should be 'X'", "→ 'X'", ": 'X'"):
// its words name nothing — "make the Body Text 'Visit us on Main Street'"
// is not about the section titled "Visit Us", nor about a location — so
// they are kept out of target matching altogether (findName) and out of
// the text ids are looked up in (scanText's second result). Any other
// quoted run may name a target only as a whole ("the block titled 'Our
// Story'", "the 'Follow along' section"), never by a part of it.
type tok struct {
	w      string // normalised
	raw    string // lower-cased, as written
	quoted bool
	span   int  // which quoted run the word is in, from 1; 0 = not quoted
	value  bool // the quoted run is introduced as the value being set
}

// endsWithMake: the quoted run closing at rs[close] ends the request, the
// text up to rs[before] (the last non-space before the run) has the word
// "make", and rs[before] does not end a nameCue word.
func endsWithMake(rs []rune, before, close int) bool {
	for _, r := range rs[close+1:] {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			return false
		}
	}
	j := before
	for j >= 0 && unicode.IsLetter(rs[j]) {
		j--
	}
	if nameCue[strings.ToLower(string(rs[j+1:before+1]))] {
		return false
	}
	for _, w := range strings.FieldsFunc(strings.ToLower(string(rs[:before+1])), func(r rune) bool { return !unicode.IsLetter(r) }) {
		if w == "make" {
			return true
		}
	}
	return false
}

// valueIntro are the words that, directly before an opening quote, make the
// quoted run the value being set.
var valueIntro = map[string]bool{"to": true, "be": true, "say": true, "says": true, "saying": true, "read": true, "reads": true}

// nameCue are the words a quoted NAME follows ("the 'Follow along' section",
// "titled 'Our Story'", "in 'Our Story'"); see introducesValue.
var nameCue = wordList("the a an titled called named in on of for from at under inside within section block page location category image and or")

func wordList(s string) map[string]bool {
	out := map[string]bool{}
	for _, w := range strings.Fields(s) {
		out[w] = true
	}
	return out
}

// introducesValue reports whether the quoted run rs[open..close] (the two
// quote marks) is the value being set. It is when the text before it says
// so: a value word, an arrow, a colon or an equals sign ("next to 'Our
// Story'" is a place, not a value). It also is in "make the Card button
// text 'Fresh picks every week'": the run ends a request that says "make",
// and does not follow a word a name would follow.
func introducesValue(rs []rune, open, close int) bool {
	i := open - 1
	for i >= 0 && unicode.IsSpace(rs[i]) {
		i--
	}
	if i < 0 {
		return false
	}
	if endsWithMake(rs, i, close) {
		return true
	}
	switch rs[i] {
	case '→', '=', ':':
		return true
	case '>':
		return i > 0 && (rs[i-1] == '-' || rs[i-1] == '=')
	}
	word := func(end int) (string, int) { // the word ending at rs[end], and the index before it
		j := end
		for j >= 0 && unicode.IsLetter(rs[j]) {
			j--
		}
		return strings.ToLower(string(rs[j+1 : end+1])), j
	}
	w, j := word(i)
	if !valueIntro[w] {
		return false
	}
	if w == "to" {
		for j >= 0 && unicode.IsSpace(rs[j]) {
			j--
		}
		if prev, _ := word(j); prev == "next" {
			return false
		}
	}
	return true
}

// stop are function words: dropped everywhere, so "Text size in pixels" and
// "text size pixels" are the same phrase.
var stop = map[string]bool{
	"a": true, "an": true, "the": true, "of": true, "to": true, "on": true, "in": true,
	"for": true, "and": true, "or": true, "is": true, "be": true, "it": true, "its": true,
	"this": true, "that": true, "with": true, "please": true, "can": true, "you": true,
	"my": true, "our": true, "me": true, "i": true, "at": true, "as": true, "by": true,
	"s": true, "should": true, "would": true, "could": true, "want": true, "like": true,
	"px": true,
}

// spell folds spelling variants and the editor's key abbreviations onto one
// word, applied after the plural is stripped.
var spell = map[string]string{
	"colour": "color", "centre": "center", "centred": "centered", "grey": "gray",
	"bg": "background", "btn": "button", "img": "image", "desc": "description",
	"pic": "image", "picture": "image", "photo": "image", "nav": "navigation",
	"dir": "direction", "cat": "category", "config": "setting", "testimonial": "review",
	"col": "column", "faqs": "faq", "hero": "hero",
}

// norm lower-cases one word, strips a plural and folds spelling.
func norm(w string) string {
	w = strings.ToLower(w)
	switch {
	case len(w) > 4 && strings.HasSuffix(w, "ies"):
		w = w[:len(w)-3] + "y"
	case len(w) > 4 && (strings.HasSuffix(w, "sses") || strings.HasSuffix(w, "xes")):
		w = w[:len(w)-2]
	case len(w) > 3 && strings.HasSuffix(w, "s") && !strings.HasSuffix(w, "ss") &&
		!strings.HasSuffix(w, "us") && !strings.HasSuffix(w, "is"):
		w = w[:len(w)-1]
	}
	if v, ok := spell[w]; ok {
		return v
	}
	return w
}

// split cuts a string into raw words: runs of letters or digits, with a
// camelCase boundary (bodyColor → body, Color; SEOPanel → SEO, Panel) and a
// letter/digit boundary (h1 → h, 1) each starting a new word.
func split(s string) []string {
	var out []string
	rs := []rune(s)
	start := -1
	flush := func(end int) {
		if start >= 0 && end > start {
			out = append(out, string(rs[start:end]))
		}
		start = -1
	}
	for i, r := range rs {
		if !unicode.IsLetter(r) && !unicode.IsDigit(r) {
			flush(i)
			continue
		}
		if start < 0 {
			start = i
			continue
		}
		p := rs[i-1]
		switch {
		case unicode.IsDigit(p) != unicode.IsDigit(r):
			flush(i)
			start = i
		case unicode.IsLower(p) && unicode.IsUpper(r):
			flush(i)
			start = i
		case unicode.IsUpper(p) && unicode.IsUpper(r) && i+1 < len(rs) && unicode.IsLower(rs[i+1]):
			flush(i)
			start = i
		}
	}
	flush(len(rs))
	return out
}

// words is the normalised, stop-word-free word sequence of a label, key,
// id or title.
//
// The result is memoised and shared: callers must not modify it. A manifest
// has a few thousand distinct labels and keys and every turn asks for the
// same ones; the memo stops growing at wordsMemoMax entries, so strings that
// vary per turn (section titles) cannot grow it without bound.
func words(s string) []string {
	if v, ok := wordsMemo.Load(s); ok {
		return v.([]string)
	}
	raw := split(s)
	out := make([]string, 0, len(raw))
	for _, w := range raw {
		n := norm(w)
		if n == "" || stop[n] {
			continue
		}
		out = append(out, n)
	}
	out = out[:len(out):len(out)] // an append by a caller copies
	if wordsMemoSize.Load() < wordsMemoMax {
		if _, loaded := wordsMemo.LoadOrStore(s, out); !loaded {
			wordsMemoSize.Add(1)
		}
	}
	return out
}

const wordsMemoMax = 50_000

var (
	wordsMemo     sync.Map // string → []string
	wordsMemoSize atomic.Int64
)

func isOpenQuote(r rune) bool { return r == '\'' || r == '"' || r == '‘' || r == '“' }
func closeQuote(r rune) rune {
	switch r {
	case '‘':
		return '’'
	case '“':
		return '”'
	}
	return r
}

// textToks is scanText's words.
func textToks(text string) []tok {
	toks, _ := scanText(text)
	return toks
}

// scanText tokenises the merchant's text, marking the words inside quotes
// (and which of those are the value being set) and dropping hex colours
// (#c22976) and URLs, which are values. The second result is the text with
// every value run blanked: what a target's id is looked up in.
func scanText(text string) ([]tok, string) {
	rs := []rune(text)
	quoted := make([]bool, len(rs))
	spanOf := make([]int, len(rs))
	isValue := make([]bool, len(rs))
	spans := 0
	drop := make([]bool, len(rs))
	isWord := func(r rune) bool { return unicode.IsLetter(r) || unicode.IsDigit(r) }
	endsValue := func(r rune) bool { return unicode.IsSpace(r) || isOpenQuote(r) || r == '’' || r == '”' }
	// Values that are not words: #c22976, /products, https://…
	for i := 0; i < len(rs); i++ {
		r := rs[i]
		atStart := i == 0 || unicode.IsSpace(rs[i-1]) || isOpenQuote(rs[i-1]) || rs[i-1] == '('
		value := r == '#' && i+1 < len(rs) && isWord(rs[i+1])
		value = value || (r == '/' && atStart && i+1 < len(rs) && unicode.IsLetter(rs[i+1]))
		value = value || (atStart && (strings.HasPrefix(string(rs[i:min(i+8, len(rs))]), "http://") || strings.HasPrefix(string(rs[i:min(i+8, len(rs))]), "https://")))
		if !value {
			continue
		}
		for ; i < len(rs) && !endsValue(rs[i]); i++ {
			drop[i] = true
		}
	}
	// Quoted spans. An opening quote does not follow a letter or digit (so
	// the apostrophe of "block's" opens nothing) and its closing quote is
	// not followed by a letter.
	for i := 0; i < len(rs); i++ {
		if !isOpenQuote(rs[i]) || (i > 0 && isWord(rs[i-1])) {
			continue
		}
		want := closeQuote(rs[i])
		for j := i + 1; j < len(rs); j++ {
			if rs[j] != want || (j+1 < len(rs) && unicode.IsLetter(rs[j+1])) {
				continue
			}
			spans++
			value := introducesValue(rs, i, j)
			for k := i + 1; k < j; k++ {
				quoted[k] = true
				spanOf[k] = spans
				isValue[k] = value
			}
			i = j
			break
		}
	}
	var out []tok
	// Walk the runes once more, cutting words the way split does but keeping
	// each word's quoted flag (that of its first rune).
	i := 0
	for i < len(rs) {
		if drop[i] || !(unicode.IsLetter(rs[i]) || unicode.IsDigit(rs[i])) {
			i++
			continue
		}
		j := i
		for j < len(rs) && !drop[j] && (unicode.IsLetter(rs[j]) || unicode.IsDigit(rs[j])) {
			j++
		}
		for _, w := range split(string(rs[i:j])) {
			n := norm(w)
			if n == "" || stop[n] {
				continue
			}
			out = append(out, tok{w: n, raw: strings.ToLower(w), quoted: quoted[i], span: spanOf[i], value: isValue[i]})
		}
		i = j
	}
	named := make([]rune, len(rs))
	for i, r := range rs {
		if isValue[i] {
			r = ' '
		}
		named[i] = r
	}
	return out, string(named)
}

// findName lists the occurrences of seq that can name a target: none inside
// a value run, and one inside any other quoted run only when it is the whole
// run.
func findName(toks []tok, seq []string) []span {
	var out []span
	for _, at := range findAll(toks, seq) {
		id, ok := toks[at.from].span, !toks[at.from].value
		for i := at.from; i < at.to && ok; i++ {
			ok = toks[i].span == id && !toks[i].value
		}
		if ok && id != 0 {
			ok = (at.from == 0 || toks[at.from-1].span != id) && (at.to == len(toks) || toks[at.to].span != id)
		}
		if ok {
			out = append(out, at)
		}
	}
	return out
}

// findSpelled lists the occurrences of seq outside value runs: the places
// the text spells out a field's label (quoted or not).
func findSpelled(toks []tok, seq []string) []span {
	var out []span
outer:
	for _, at := range findAll(toks, seq) {
		for i := at.from; i < at.to; i++ {
			if toks[i].value {
				continue outer
			}
		}
		out = append(out, at)
	}
	return out
}

// rawWords are the lower-cased words of s, stop words dropped, not
// normalised: what findRaw compares against the text as written.
func rawWords(s string) []string {
	var out []string
	for _, w := range split(s) {
		if !stop[norm(w)] {
			out = append(out, strings.ToLower(w))
		}
	}
	return out
}

// findRaw lists the occurrences of seq in the text as written (no plural
// stripping: "products" does not match "product").
func findRaw(text []tok, seq []string) []span {
	var out []span
	if len(seq) == 0 {
		return nil
	}
outer:
	for i := 0; i+len(seq) <= len(text); i++ {
		for j, w := range seq {
			if text[i+j].raw != w {
				continue outer
			}
		}
		out = append(out, span{i, i + len(seq)})
	}
	return out
}

// hasPhrase reports whether seq occurs contiguously in text.
func hasPhrase(text []tok, seq []string) bool {
	return phraseAt(text, seq) >= 0
}

// phraseAt is the index of the first contiguous occurrence of seq in text, or -1.
func phraseAt(text []tok, seq []string) int {
	if len(seq) == 0 || len(seq) > len(text) {
		return -1
	}
outer:
	for i := 0; i+len(seq) <= len(text); i++ {
		for j, w := range seq {
			if text[i+j].w != w {
				continue outer
			}
		}
		return i
	}
	return -1
}

// ordinals maps an ordinal word to its 1-based position; "last" is -1.
var ordinals = map[string]int{
	"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
	"seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10, "last": -1,
	"1st": 1, "2nd": 2, "3rd": 3, "4th": 4, "5th": 5, "6th": 6, "7th": 7, "8th": 8, "9th": 9, "10th": 10,
}
