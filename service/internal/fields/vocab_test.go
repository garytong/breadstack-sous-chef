package fields

import (
	"reflect"
	"strings"
	"testing"

	"github.com/breadstack/sous-chef/internal/manifest"
)

// The merchant vocabulary (vocab.txt) at work: requests in the merchant's
// words, not the editor's labels.
func TestVocabularyFindsTheField(t *testing.T) {
	m, _ := snapshot(t)
	home := ctxOf(t, homeContext)
	global := func(p string) manifest.Target { return manifest.Target{Kind: "global", Panel: p} }
	sec := func(id string) manifest.Target { return manifest.Target{Kind: "section", Page: "home", SectionID: id} }
	cases := []struct {
		name   string
		text   string
		target manifest.Target
		list   string
		key    string
		first  bool // the target must be the block's first
	}{
		// int, 2026-10-05: the block never listed storeName ("Store Name").
		{"site name", "change the site name to HashiTong", global("BrandingSection"), "", "storeName", true},
		{"shop name", "update the shop name to 'The Herb Shop'", global("BrandingSection"), "", "storeName", true},
		{"business name", "our business name changed, it's now 'Northern Lights Cannabis'", global("BrandingSection"), "", "storeName", true},
		{"rename my site", "rename my site to Green Leaf Co", global("BrandingSection"), "", "storeName", true},
		{"favicon", "change the little icon in the browser tab", global("BrandingSection"), "", "favicon", true},
		// rounder → radius, on a named target.
		{"rounder product cards", "make the product cards rounder", global("ProductCardSection"), "", "productCardRadius", true},
		{"rounder review cards", "rounder corners on the review cards", sec("testimonials"), "", "cardBorderRadius", true},
		// social networks → the social panel.
		{"instagram", "add my instagram: https://instagram.com/hashitong", global("SocialMediaSection"), "", "socialLinks.instagram", true},
		{"tiktok", "link our tiktok, it's tiktok.com/@hashitong", global("SocialMediaSection"), "", "socialLinks.tiktok", true},
		// the announcement bar lives in the header, not the notification panel.
		{"announcement bar", "turn on the announcement bar", global("HeaderSection"), "", "announcementBar.show", true},
		{"bar above the menu", "change the little bar above the menu to say 'Free delivery over $75'", global("HeaderSection"), "", "announcementBar.message", true},
		// "heading font" is labelled "Section Headings".
		{"heading font", "change the heading font to Fraunces", global("TypographyColorsSection"), "", "headingFont", true},
		{"site-wide body text", "make the body text a bit bigger site wide", global("TypographyColorsSection"), "", "bodyFontSize", true},
		// banner → the hero, colour words → a colour field.
		{"top banner button", "make the button on the top banner green", sec("heroBanner"), "slides", "btnColor", true},
		{"small text above the headline", "change the small text above the banner headline to 'Locally grown'", sec("heroBanner"), "slides", "tagline", true},
		{"narrower", "make the hero banner narrower", sec("heroBanner"), "", "maxWidth", true},
		// a title of stop words and one word, unquoted.
		{"our story", "in our story, change the button to say 'Read more'", sec("contentBlock-2"), "", "buttonText", true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			res := Candidates(m, home, tc.text, Options{})
			tg := find(res, tc.target)
			if tg == nil {
				t.Fatalf("target %+v not listed\n%s", tc.target, res.Block)
			}
			if !has(tg, tc.list, tc.key) {
				t.Fatalf("key %q (list %q) not listed under %+v\n%s", tc.key, tc.list, tc.target, res.Block)
			}
			if tc.first && find(Result{Targets: res.Targets[:1]}, tc.target) == nil {
				t.Fatalf("target %+v is not first\n%s", tc.target, res.Block)
			}
		})
	}
}

// Either word finds either toggle: the editor labels some "Show X" and
// some "Hide X"; the value the model writes carries the polarity.
func TestVocabularyVisibilityPolarity(t *testing.T) {
	m, _ := snapshot(t)
	home := ctxOf(t, homeContext)
	header := manifest.Target{Kind: "global", Panel: "HeaderSection"}
	cards := manifest.Target{Kind: "global", Panel: "ProductCardSection"}
	for _, tc := range []struct {
		text   string
		target manifest.Target
		key    string
	}{
		{"hide the cart icon in the header", header, "headerShowCart"},                      // hide → "Show Cart Icon"
		{"turn off the cart icon in the header", header, "headerShowCart"},                  // turn off → show
		{"show the blog link in the header nav", header, "pageVisibility.blog.inHeaderNav"}, // show → "Hide Blog from header nav"
		{"make the THC pill visible on product cards", cards, "cardShowTHC"},                // visible → "Hide THC pill"
		{"hide the announcement bar", header, "announcementBar.show"},                       // hide → "Show Announcement Bar"
	} {
		res := Candidates(m, home, tc.text, Options{})
		if !has(find(res, tc.target), "", tc.key) {
			t.Errorf("%q: %s not listed under %+v\n%s", tc.text, tc.key, tc.target, res.Block)
		}
	}
}

// A synonym must not pull a wrong target to the top.
func TestVocabularyTraps(t *testing.T) {
	m, _ := snapshot(t)
	home := ctxOf(t, homeContext)
	first := func(res Result) manifest.Target {
		if len(res.Targets) == 0 {
			return manifest.Target{}
		}
		return res.Targets[0].Targets[0]
	}
	// "instagram gallery" is the gallery section, not the social panel.
	gallery := ctxOf(t, `{"page":"home","sections":[{"id":"heroBanner","type":"heroBanner"},{"id":"gallery","type":"gallery","title":"Follow along"},{"id":"newsletter","type":"newsletter"}]}`)
	for _, text := range []string{"show 4 pics at once in the instagram gallery", "the follow along button should link to our instagram https://instagram.com/hashitong"} {
		res := Candidates(m, gallery, text, Options{})
		if got := first(res); got.SectionID != "gallery" {
			t.Errorf("%q: first target %+v\n%s", text, got, res.Block)
		}
	}
	var res Result
	// "rename" a heading is not the site name.
	res = Candidates(m, home, "rename the Crafted with Care headline to 'Made by hand'", Options{})
	if got := first(res); got.SectionID != "contentBlock-1" || find(res, manifest.Target{Kind: "global", Panel: "BrandingSection"}) != nil {
		t.Errorf("rename a headline: first target %+v\n%s", got, res.Block)
	}
	// "turn" without "on"/"off" is no visibility toggle.
	for _, tk := range textToks("turn the category circles into cards") {
		for _, a := range tk.alts {
			for _, w := range a.w {
				if w == "show" || w == "hide" {
					t.Errorf("\"turn\" alone reads as %q", w)
				}
			}
		}
	}
	// The insider banner is not the hero (whose alias is "banner").
	res = Candidates(m, home, "change the insider banner heading to 'Join the club'", Options{})
	if got := first(res); got.Panel != "InsiderBannerSection" {
		t.Errorf("insider banner: first target %+v\n%s", got, res.Block)
	}
	// A value never names a target through the vocabulary.
	res = Candidates(m, home, "set the Body Text of the content block titled 'Visit Us' to 'Ask about our site name'", Options{})
	if find(res, manifest.Target{Kind: "global", Panel: "BrandingSection"}) != nil {
		t.Errorf("a value named the Branding panel\n%s", res.Block)
	}
}

func TestVocabularyExpansion(t *testing.T) {
	readings := func(text string) []string {
		var out []string
		for _, tk := range textToks(text) {
			for _, a := range tk.alts {
				out = append(out, tk.w+"→"+strings.Join(a.w, " "))
			}
		}
		return out
	}
	for _, tc := range []struct {
		text string
		want string // one reading the text must have
	}{
		{"change the site name", "site→store name"},        // positional, two words
		{"rename my site", "rename→store name"},            // the text's stop word "my" skipped
		{"turn it off", "turn→show"},                       // "turn off" with "it" between
		{"turn on the bar", "turn→show"},                   // the term's stop word "on", as written
		{"all caps please", "all→uppercase"},               // two words read as one
		{"rounder corners", "rounder→radius"},              // plural stripped before matching
		{"the bar above the menu", "bar→announcement bar"}, // the term's stop word "the" skipped in the text
	} {
		got := readings(tc.text)
		found := false
		for _, r := range got {
			found = found || r == tc.want
		}
		if !found {
			t.Errorf("%q: readings %v lack %q", tc.text, got, tc.want)
		}
	}
	// A value run reads as nothing.
	if got := readings("set the title to 'site name'"); len(got) != 0 && strings.Contains(strings.Join(got, " "), "store") {
		t.Errorf("value run expanded: %v", got)
	}
	// The written word is matched as written too, and a reading spans the
	// words it replaces: "site name" spells the label "Store Name".
	toks := textToks("change the site name")
	if at := findSpelled(toks, words("Store Name")); !reflect.DeepEqual(at, []span{{1, 3}}) {
		t.Errorf("Store Name spelled at %v", at)
	}
	if at := findSpelled(toks, words("Site Name")); !reflect.DeepEqual(at, []span{{1, 3}}) {
		t.Errorf("Site Name spelled at %v", at)
	}
}

func TestParseVocab(t *testing.T) {
	terms, aliases, err := parseVocab("# c\nrounder = radius   # why\nsite name = store name, brand\n@g:BrandingSection = site name, page:age-gate\n")
	if err != nil {
		t.Fatal(err)
	}
	if len(terms) != 2 || !reflect.DeepEqual(terms[1].from, []string{"site", "name"}) || !reflect.DeepEqual(terms[1].to, [][]string{{"store", "name"}, {"brand"}}) {
		t.Fatalf("terms %+v", terms)
	}
	if !reflect.DeepEqual(aliases["g:BrandingSection"], []string{"site name", "page:age-gate"}) {
		t.Fatalf("aliases %+v", aliases)
	}
	for _, bad := range []string{"no equals sign", "= radius", "rounder =", "@x:Foo = bar", "@g: = bar", "rounder = the"} {
		if _, _, err := parseVocab(bad); err == nil {
			t.Errorf("%q: no error", bad)
		}
	}
	// The shipped file parses, and every container alias names a kind the
	// retrieval knows.
	if _, _, err := parseVocab(vocabText); err != nil {
		t.Fatal(err)
	}
}

// Every container the vocabulary names exists in the snapshot manifest: a
// renamed panel or section type must not silently lose its aliases.
func TestVocabularyContainersExist(t *testing.T) {
	m, _ := snapshot(t)
	for key, aliases := range ContainerAliases() {
		kind, id, _ := strings.Cut(key, ":")
		var ok bool
		switch kind {
		case "s":
			_, ok = m.Sections[id]
		case "g":
			_, ok = m.GlobalPanels[id]
		case "b":
			ok = id == "branding"
		case "p":
			_, ok = m.Pages[id]
			ok = ok || id == manifest.SEOPage
		}
		if !ok {
			t.Errorf("vocab.txt: @%s is not in the manifest", key)
		}
		for _, a := range aliases {
			if p, isPage := strings.CutPrefix(a, viewPrefix); isPage {
				if _, ok := m.Pages[p]; !ok {
					t.Errorf("vocab.txt: @%s: %s is not a page", key, a)
				}
			}
		}
	}
}
