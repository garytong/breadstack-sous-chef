package fields

import (
	"encoding/json"
	"flag"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/breadstack/sous-chef/internal/manifest"
)

var update = flag.Bool("update", false, "rewrite testdata/*.golden")

func snapshot(t testing.TB) (*manifest.Manifest, []byte) {
	t.Helper()
	raw, err := os.ReadFile("../../testdata/manifest.snapshot.json")
	if err != nil {
		t.Fatal(err)
	}
	m, err := manifest.Parse(raw)
	if err != nil {
		t.Fatal(err)
	}
	return m, raw
}

// homeContext is a home page the way the editor reports it.
const homeContext = `{"page":"home","template":"editorial","sections":[
 {"id":"heroBanner","type":"heroBanner","title":"Fresh picks every week","enabled":true},
 {"id":"featuredCategories","type":"featuredCategories","title":"","enabled":true},
 {"id":"productCarousel-1","type":"productCarousel","title":"New Arrivals","enabled":true},
 {"id":"productCarousel-2","type":"productCarousel","title":"Customer Favorites","enabled":true},
 {"id":"testimonials","type":"testimonials","title":"","enabled":true},
 {"id":"contentBlock-1","type":"contentBlock","title":"Crafted with Care","enabled":true},
 {"id":"contentBlock-2","type":"contentBlock","title":"Our Story","enabled":true},
 {"id":"contentBlock-features","type":"contentBlock-features","title":"Why Choose Us?","enabled":true},
 {"id":"newsletter","type":"newsletter","title":"","enabled":true}],
 "categories":[{"slug":"flower","name":"Flower"},{"slug":"edibles","name":"Edibles"}],
 "locations":[{"id":"loc-main","name":"Main Street"},{"id":"loc-north","name":"Northgate"}]}`

const faqContext = `{"page":"faq","sections":[{"id":"__pinned_faq","type":"__pinned_faq","title":"","enabled":true}]}`

func ctxOf(t testing.TB, raw string) Context {
	t.Helper()
	c, err := ParseContext(json.RawMessage(raw))
	if err != nil {
		t.Fatal(err)
	}
	return c
}

// find returns the listed target with the given op target, or nil.
func find(res Result, want manifest.Target) *Target {
	for i := range res.Targets {
		for _, t := range res.Targets[i].Targets {
			if t == want {
				return &res.Targets[i]
			}
		}
	}
	return nil
}

func has(t *Target, list, key string) bool {
	if t == nil {
		return false
	}
	for _, f := range t.Fields {
		if f.Key == key && f.List == list {
			return true
		}
	}
	return false
}

func TestCandidatesFindsTheKey(t *testing.T) {
	m, _ := snapshot(t)
	home, faq := ctxOf(t, homeContext), ctxOf(t, faqContext)
	sec := func(id string) manifest.Target { return manifest.Target{Kind: "section", Page: "home", SectionID: id} }
	cases := []struct {
		name   string
		ctx    Context
		text   string
		target manifest.Target
		list   string
		key    string
	}{
		{"label on a named section", home, "Reset the Body Text color of the content block titled 'Our Story'", sec("contentBlock-2"), "", "bodyColor"},
		{"section by id", home, "contentBlock-features: make the Icon Color '#f9dd16'", sec("contentBlock-features"), "", "iconColor"},
		{"section by ordinal", home, "In the second product carousel, set the tagline text size to 20", sec("productCarousel-2"), "", "taglineFontSize"},
		{"section by one word", home, "make the carousel tagline bigger", sec("productCarousel-1"), "", "taglineFontSize"},
		{"alias and british spelling", home, "change the star colour in the customer reviews", sec("testimonials"), "", "starColor"},
		{"hero mirror key lives on slides", home, "Can you set Body Text to '#c22976' for the hero banner?", sec("heroBanner"), "slides", "bodyColor"},
		{"hero container field", home, "Change the Max Width on the hero banner to 740", sec("heroBanner"), "", "maxWidth"},
		{"list item", home, "Set the Star Rating of the second review in the customer reviews section to 5", sec("testimonials"), "reviews", "rating"},
		{"global panel", home, "Please clear the Link Highlight Color in the footer", manifest.Target{Kind: "global", Panel: "FooterSection"}, "", "footerLinkHighlightColor"},
		{"global panel by alias", home, "Reset the Body Text Color of the typography and colours", manifest.Target{Kind: "global", Panel: "TypographyColorsSection"}, "", "bodyTextColor"},
		{"dotted key", home, "Please hide Shop from the header nav (the header)", manifest.Target{Kind: "global", Panel: "HeaderSection"}, "", "pageVisibility.products.inHeaderNav"},
		{"branding", home, "Branding: hide the Location Name", manifest.Target{Kind: "branding"}, "", "locationPickerShowName"},
		{"the viewed page's own options", faq, "Enable uppercase for the answer text", manifest.Target{Kind: "page", Page: "faq"}, "", "answerUppercase"},
		{"another page, named", home, "Reset the Input Border of the 'login' page", manifest.Target{Kind: "page", Page: "login"}, "", "inputBorderColor"},
		{"key spelled out", home, "set answerUppercase on the faq page", manifest.Target{Kind: "page", Page: "faq"}, "", "answerUppercase"},
		{"keyed per-location field", home, "content block contentBlock-2: address override for the location 'Main Street' should be '12 Main St'", sec("contentBlock-2"), "", "locationOverrides.loc-main.address"},
		{"the page's section list", home, "Turn the newsletter off on the home page", manifest.Target{Kind: "page", Page: "home"}, "sections", "enabled"},
		{"unnamed target, label spelled out", home, "make the Link Highlight Color '#112233'", manifest.Target{Kind: "global", Panel: "FooterSection"}, "", "footerLinkHighlightColor"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			res := Candidates(m, tc.ctx, tc.text, Options{})
			tg := find(res, tc.target)
			if tg == nil {
				t.Fatalf("target %+v not listed\n%s", tc.target, res.Block)
			}
			if !has(tg, tc.list, tc.key) {
				t.Fatalf("key %q (list %q) not listed under %+v\n%s", tc.key, tc.list, tc.target, res.Block)
			}
			line := "  " + tc.key + " "
			if !strings.Contains(res.Block, line) || !strings.Contains(res.Block, "target "+targetJSON(tc.target)+"\n") {
				t.Fatalf("block does not print the target and key\n%s", res.Block)
			}
		})
	}
}

func TestExplicitInstanceDropsItsSiblings(t *testing.T) {
	m, _ := snapshot(t)
	res := Candidates(m, ctxOf(t, homeContext), "Clear the Column Border Color on the content block contentBlock-2", Options{})
	for _, id := range []string{"contentBlock-1", "contentBlock-features"} {
		if find(res, manifest.Target{Kind: "section", Page: "home", SectionID: id}) != nil {
			t.Fatalf("%s listed although the text names contentBlock-2\n%s", id, res.Block)
		}
	}
	// A type-level name keeps every instance, under one shared list.
	res = Candidates(m, ctxOf(t, homeContext), "Update the Background Color on the content block section", Options{})
	tg := find(res, manifest.Target{Kind: "section", Page: "home", SectionID: "contentBlock-1"})
	if tg == nil || len(tg.Targets) != 3 {
		t.Fatalf("want the three content blocks sharing one list, got %+v\n%s", tg, res.Block)
	}
	if n := strings.Count(res.Block, "  backgroundColor — "); n != 1 {
		t.Fatalf("shared list printed %d times\n%s", n, res.Block)
	}
}

func TestInPassingNameIsNotATarget(t *testing.T) {
	m, _ := snapshot(t)
	res := Candidates(m, ctxOf(t, homeContext), "Set the Secondary Button Border Color of the content block contentBlock-1 to '#5d904f'", Options{})
	if find(res, manifest.Target{Kind: "global", Panel: "ButtonsSection"}) != nil {
		t.Fatalf("the Buttons panel is listed for a label that merely contains \"Button\"\n%s", res.Block)
	}
	if !has(find(res, manifest.Target{Kind: "section", Page: "home", SectionID: "contentBlock-1"}), "", "secondaryButtonBorderColor") {
		t.Fatalf("key missing\n%s", res.Block)
	}
}

func TestNeverListsUnwritableKeys(t *testing.T) {
	m, _ := snapshot(t)
	for _, text := range []string{
		"content block contentBlock-1: location override mode and address and map location",
		"the product carousel New Arrivals: tag and brand",
		"typography: page title color and section heading color",
	} {
		res := Candidates(m, ctxOf(t, homeContext), text, Options{MaxFields: 200, MaxChars: 1 << 20})
		for _, tg := range res.Targets {
			r, err := m.Resolve(tg.Targets[0])
			if err != nil {
				t.Fatal(err)
			}
			for _, f := range tg.Fields {
				if strings.Contains(f.Key, "{") {
					t.Errorf("%q: templated key %s listed", text, f.Key)
				}
				if f.List != "" {
					continue
				}
				ref, ok := m.Field(r, f.Key)
				if !ok {
					t.Errorf("%q: %s is not a field of %s", text, f.Key, tg.Container)
				} else if ref.Spec.ReadOnly() {
					t.Errorf("%q: read-only %s listed", text, f.Key)
				}
			}
		}
	}
}

func TestDeterministicAndOrderIndependent(t *testing.T) {
	m, raw := snapshot(t)
	// The same manifest re-encoded (Go sorts object keys, the launcher does
	// not) must give the same bytes: nothing may depend on member order.
	var doc any
	if err := json.Unmarshal(raw, &doc); err != nil {
		t.Fatal(err)
	}
	again, err := json.Marshal(doc)
	if err != nil {
		t.Fatal(err)
	}
	m2, err := manifest.Parse(again)
	if err != nil {
		t.Fatal(err)
	}
	ctx := ctxOf(t, homeContext)
	for _, text := range []string{
		"Set the Tagline › Text size in pixels of the content block contentBlock-2 to 10px and set the Body Text color of the product carousel titled 'New Arrivals' to '#e427c2'",
		"make the headline bigger",
		"Please hide Shop from the header nav (the header)",
		"Branding — Primary Button Font › Primary button font family → DM Sans",
	} {
		want := Candidates(m, ctx, text, Options{})
		if want.Block == "" {
			t.Fatalf("%q: empty block", text)
		}
		for i := 0; i < 20; i++ {
			if got := Candidates(m, ctx, text, Options{}); got.Block != want.Block || !reflect.DeepEqual(got.Targets, want.Targets) {
				t.Fatalf("%q: run %d differs", text, i)
			}
		}
		if got := Candidates(m2, ctx, text, Options{}); got.Block != want.Block {
			t.Fatalf("%q: block depends on the manifest's member order", text)
		}
	}
}

func TestCaps(t *testing.T) {
	m, _ := snapshot(t)
	ctx := ctxOf(t, homeContext)
	text := "change the color, the font, the size, the text, the background, the border, the button and the image of the hero banner, the content block contentBlock-1, the newsletter, the footer, the header and the product cards"
	full := Candidates(m, ctx, text, Options{})
	if len(full.Block) > DefaultOptions().MaxChars {
		t.Fatalf("block is %d chars, cap %d", len(full.Block), DefaultOptions().MaxChars)
	}
	if len(full.Targets) > DefaultOptions().MaxTargets {
		t.Fatalf("%d targets, cap %d", len(full.Targets), DefaultOptions().MaxTargets)
	}
	for _, max := range []int{400, 900, 2000} {
		res := Candidates(m, ctx, text, Options{MaxChars: max})
		// A block always keeps one target with minKeep fields, so a very
		// small cap is a floor rather than a guarantee.
		if len(res.Block) > max && (len(res.Targets) > 1 || len(res.Targets[0].Fields) > minKeep) {
			t.Fatalf("MaxChars %d: block is %d chars with %d targets", max, len(res.Block), len(res.Targets))
		}
		if !strings.HasPrefix(res.Block, blockOpen) || !strings.HasSuffix(res.Block, blockClose) {
			t.Fatalf("MaxChars %d: block lost its frame", max)
		}
	}
	two := Candidates(m, ctx, text, Options{MaxTargets: 2})
	if len(two.Targets) != 2 {
		t.Fatalf("MaxTargets 2: %d targets", len(two.Targets))
	}
	few := Candidates(m, ctx, "hero banner: color font size text background border button image headline tagline", Options{MaxFields: 5, ListAll: 1})
	for _, tg := range few.Targets {
		if len(tg.Fields) > 5 {
			t.Fatalf("MaxFields 5: %d fields under %s", len(tg.Fields), tg.Container)
		}
	}
}

func TestEmptyInputs(t *testing.T) {
	m, _ := snapshot(t)
	if res := Candidates(nil, Context{}, "anything", Options{}); res.Block != "" || res.Targets != nil {
		t.Fatal("nil manifest must give the zero result")
	}
	if res := Candidates(m, ctxOf(t, homeContext), "  ", Options{}); res.Block != "" {
		t.Fatal("empty text must give no block")
	}
	if res := Candidates(m, ctxOf(t, homeContext), "zzz qqq xyzzy", Options{}); res.Block != "" {
		t.Fatalf("a text that matches nothing must give no block, got\n%s", res.Block)
	}
	// No context at all: global panels and branding still resolve.
	res := Candidates(m, Context{}, "clear the Link Highlight Color in the footer", Options{})
	if !has(find(res, manifest.Target{Kind: "global", Panel: "FooterSection"}), "", "footerLinkHighlightColor") {
		t.Fatalf("no-context lookup failed\n%s", res.Block)
	}
	if _, err := ParseContext(json.RawMessage(`[1]`)); err == nil {
		t.Fatal("a non-object context must be refused")
	}
	if c, err := ParseContext(json.RawMessage(` null `)); err != nil || c.Page != "" {
		t.Fatal("null context is the zero Context")
	}
}

// TestGolden pins the block's bytes: the lab's training rows carry these
// blocks, so any change here is a retraining event. Run with -update to
// accept one.
func TestGolden(t *testing.T) {
	m, _ := snapshot(t)
	cases := []struct{ name, ctx, text string }{
		{"hero-body-text", homeContext, "Can you set Body Text to '#c22976' for the hero banner?"},
		{"header-nav-toggle", homeContext, "Please hide Shop from the header nav (the header)."},
		{"multi", homeContext, "Set the Tagline › Text size in pixels of the content block contentBlock-2 to 10px and set the Body Text color of the product carousel titled 'New Arrivals' (productCarousel-1) to '#e427c2'."},
		{"clarify-content-block", homeContext, "Update the Background Color on the content block section to '#2a6e88'."},
		{"enum-labels", homeContext, "Please update the 'Fresh picks every week' section: Bullet Item Layout should be Icon Top."},
		{"sections-list-add", homeContext, "I want another loyalty form on the home page."},
		{"faq-page", faqContext, "Enable the Answer Text Style — uppercase (answerUppercase) on the faq page."},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := Candidates(m, ctxOf(t, tc.ctx), tc.text, Options{}).Block + "\n"
			path := filepath.Join("testdata", tc.name+".golden")
			if *update {
				if err := os.MkdirAll("testdata", 0o755); err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(path, []byte(got), 0o644); err != nil {
					t.Fatal(err)
				}
				return
			}
			want, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			if got != string(want) {
				t.Fatalf("block changed (go test ./internal/fields -run TestGolden -update to accept)\n--- got\n%s--- want\n%s", got, want)
			}
		})
	}
}

func TestTokens(t *testing.T) {
	if got := words("bodyColor"); !reflect.DeepEqual(got, []string{"body", "color"}) {
		t.Fatalf("camelCase: %v", got)
	}
	if got := words("Tagline › Text size in pixels"); !reflect.DeepEqual(got, []string{"tagline", "text", "size", "pixel"}) {
		t.Fatalf("label: %v", got)
	}
	if got := words("the Colours of the Categories"); !reflect.DeepEqual(got, []string{"color", "category"}) {
		t.Fatalf("spelling and plural: %v", got)
	}
	if got := labelWords("Tagline Image (optional)"); !reflect.DeepEqual(got, []string{"tagline", "image"}) {
		t.Fatalf("aside: %v", got)
	}
	toks := textToks("Set the block's Body Text to '#c22976' or 'Fresh picks' at https://example.com/x")
	var plain, quoted []string
	for _, tk := range toks {
		if tk.quoted {
			quoted = append(quoted, tk.w)
		} else {
			plain = append(plain, tk.w)
		}
	}
	if !reflect.DeepEqual(plain, []string{"set", "block", "body", "text"}) || !reflect.DeepEqual(quoted, []string{"fresh", "pick"}) {
		t.Fatalf("text tokens: plain %v quoted %v", plain, quoted)
	}
	if !containsWord("the content block contentBlock-2.", "contentBlock-2") || containsWord("contentBlock-21", "contentBlock-2") || containsWord("xheroBanner", "heroBanner") {
		t.Fatal("containsWord")
	}
}

func BenchmarkCandidates(b *testing.B) {
	m, _ := snapshot(b)
	ctx := ctxOf(b, homeContext)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Candidates(m, ctx, "Set the Tagline › Text size in pixels of the content block contentBlock-2 to 10px", Options{})
	}
}

// Candidates is called from concurrent turns and shares a memo: the result
// must not depend on who got there first (run under -race).
func TestConcurrentCallsAgree(t *testing.T) {
	m, _ := snapshot(t)
	ctx := ctxOf(t, homeContext)
	texts := []string{
		"Reset the Body Text color of the content block titled 'Our Story'",
		"Please hide Shop from the header nav (the header)",
		"make the carousel tagline bigger",
	}
	want := make([]string, len(texts))
	for i, text := range texts {
		want[i] = Candidates(m, ctx, text, Options{}).Block
	}
	done := make(chan string, 8)
	for g := 0; g < 8; g++ {
		go func() {
			for n := 0; n < 5; n++ {
				for i, text := range texts {
					if got := Candidates(m, ctx, text, Options{}).Block; got != want[i] {
						done <- "concurrent call differs for: " + text
						return
					}
				}
			}
			done <- ""
		}()
	}
	for g := 0; g < 8; g++ {
		if msg := <-done; msg != "" {
			t.Fatal(msg)
		}
	}
}
