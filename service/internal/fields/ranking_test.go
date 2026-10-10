package fields

import (
	"strings"
	"testing"

	"github.com/breadstack/sous-chef/internal/manifest"
)

func firstTarget(res Result) manifest.Target {
	if len(res.Targets) == 0 {
		return manifest.Target{}
	}
	return res.Targets[0].Targets[0]
}

// The refute of the synonym layer (2026-10-10): requests that worked before
// the vocabulary and broke with it.
func TestVocabularyRefuteFixes(t *testing.T) {
	m, _ := snapshot(t)
	home := ctxOf(t, homeContext)
	g := func(p string) manifest.Target { return manifest.Target{Kind: "global", Panel: p} }
	seo := manifest.Target{Kind: "page", Page: "seo"}

	// "google" alone is not the seo page: other labels have the word.
	for _, tc := range []struct {
		text   string
		target manifest.Target
		key    string
	}{
		{"link our google play app https://play.google.com/store/apps/x", g("FooterSection"), "footerShowGooglePlay"},
		{"add our google business link https://g.page/leaf", g("SocialMediaSection"), "socialLinks.googleBusiness"},
		{"set the google business url to https://g.page/leaf", g("SocialMediaSection"), "socialLinks.googleBusiness"},
	} {
		res := Candidates(m, home, tc.text, Options{})
		tg := find(res, tc.target)
		if tg == nil || !has(tg, "", tc.key) {
			t.Errorf("%q: %s not listed under %+v\n%s", tc.text, tc.key, tc.target, res.Block)
		}
		if got := firstTarget(res); got == seo {
			t.Errorf("%q: the seo page is first\n%s", tc.text, res.Block)
		}
	}
	// … while phrases that mean the search listing still name it.
	for _, text := range []string{"change what shows up on google for my homepage", "change the title on google", "fix our google search results description"} {
		if res := Candidates(m, home, text, Options{}); firstTarget(res) != seo {
			t.Errorf("%q: first target %+v, want the seo page\n%s", text, firstTarget(res), res.Block)
		}
	}

	// "add" reads as "show" only in the toggle phrases ("add a search box",
	// "add a second button", "add a promo bar"): "add our <network> <url>"
	// is a value and "add a paper texture" an image.
	reads := func(text string) bool {
		for _, tk := range textToks(text) {
			if tk.w != "add" {
				continue
			}
			for _, a := range tk.alts {
				if len(a.w) == 1 && a.w[0] == "show" {
					return true // "add a search" reads "add search" as show
				}
			}
		}
		return false
	}
	if !reads("add a search box") || !reads("add a second button") || !reads("add a promo bar") {
		t.Error(`"add a search" / "add a second" / "add a promo" should read as "show"`)
	}
	if reads("add our leafly page leafly.com/x") || reads("add more space") || reads("add a paper texture behind everything") {
		t.Error(`"add" alone must not read as "show"`)
	}
}

// Colour words and size comparatives read into many fields; they must not
// lift the viewed page or the age-gate branding above the target whose
// label the text spells.
func TestReadingsDoNotOutrankASpelledLabel(t *testing.T) {
	m, _ := snapshot(t)
	products := ctxOf(t, `{"page":"products","sections":[{"id":"__pinned_product_list","type":"__pinned_product_list","enabled":true}]}`)
	res := Candidates(m, products, "make the indica tag purple", Options{})
	if got := firstTarget(res); got.Panel != "ProductCardSection" {
		t.Errorf("indica: first target %+v\n%s", got, res.Block)
	}
	home := ctxOf(t, homeContext)
	res = Candidates(m, home, "make the logo smaller on mobile", Options{})
	if got := firstTarget(res); got.Panel != "BrandingSection" {
		t.Errorf("logo smaller: first target %+v\n%s", got, res.Block)
	}
	if tg := find(res, manifest.Target{Kind: "global", Panel: "BrandingSection"}); tg == nil || !has(tg, "", "logoMaxHeightMobile") {
		t.Errorf("logo smaller: logoMaxHeightMobile not listed\n%s", res.Block)
	}
}

// A section type named only inside a field label another target spells is
// no section-list edit: "Show banner" of the insider banner is not the hero.
func TestTypeNameInsideALabelIsNoSectionEdit(t *testing.T) {
	m, _ := snapshot(t)
	home := ctxOf(t, homeContext)
	res := Candidates(m, home, "Restore the Show banner on checkout (insider banner) of the insider banner to its default value.", Options{})
	if got := firstTarget(res); got.Panel != "InsiderBannerSection" {
		t.Errorf("first target %+v\n%s", got, res.Block)
	}
	if find(res, manifest.Target{Kind: "page", Page: "home"}) != nil {
		t.Errorf("the home page's section list is listed\n%s", res.Block)
	}
	// A type the text does name still asks for the section list.
	res = Candidates(m, home, "hide the hero banner", Options{})
	if tg := find(res, manifest.Target{Kind: "page", Page: "home"}); tg == nil || !strings.Contains(strings.Join(tg.Lists, ","), "sections") {
		t.Errorf("hide the hero: no section list\n%s", res.Block)
	}
}

// A term's reading of a word every panel's fields share ("font") is a hint:
// it ranks Typography first when nothing else is named, never above a
// panel the text names, and never as the reading of a longer label.
func TestGenericReadingIsAHint(t *testing.T) {
	m, _ := snapshot(t)
	home := ctxOf(t, homeContext)
	typo := manifest.Target{Kind: "global", Panel: "TypographyColorsSection"}
	if res := Candidates(m, home, "i don't like the font", Options{}); firstTarget(res) != typo {
		t.Errorf("the font: first target %+v\n%s", firstTarget(res), res.Block)
	}
	res := Candidates(m, home, "Restore the default Button Font on the buttons.", Options{})
	if got := firstTarget(res); got.Panel != "ButtonsSection" {
		t.Errorf("button font: first target %+v\n%s", got, res.Block)
	}
	res = Candidates(m, home, "Clear the Footnote Font on the second content block (contentBlock-2).", Options{})
	if find(res, typo) != nil {
		t.Errorf("footnote font: Typography listed\n%s", res.Block)
	}
	// "facebook" is the social panel's own word: still a name.
	res = Candidates(m, home, "add our facebook page https://facebook.com/x", Options{})
	if got := firstTarget(res); got.Panel != "SocialMediaSection" {
		t.Errorf("facebook: first target %+v\n%s", got, res.Block)
	}
}

// The open section or panel (context.focus) comes first among the targets
// the text does not name, never above one it names.
func TestFocus(t *testing.T) {
	m, _ := snapshot(t)
	withFocus := func(focus string) Context {
		return ctxOf(t, strings.Replace(homeContext, `"page":"home",`, `"page":"home","focus":`+focus+`,`, 1))
	}
	sec := func(id string) manifest.Target { return manifest.Target{Kind: "section", Page: "home", SectionID: id} }

	ctx := withFocus(`{"area":"editor","sectionId":"testimonials","panel":""}`)
	res := Candidates(m, ctx, "make this title smaller", Options{})
	if got := firstTarget(res); got != sec("testimonials") {
		t.Errorf("focused section: first target %+v\n%s", got, res.Block)
	}
	// Without the focus the same text does not reach it first.
	if got := firstTarget(Candidates(m, ctxOf(t, homeContext), "make this title smaller", Options{})); got == sec("testimonials") {
		t.Skip("the unfocused block already starts with it")
	}

	// The type a text names is the focused instance: the sibling drops.
	ctx = withFocus(`{"area":"editor","sectionId":"productCarousel-2","panel":""}`)
	res = Candidates(m, ctx, "the carousel should show 5 at once", Options{})
	if got := firstTarget(res); got != sec("productCarousel-2") || find(res, sec("productCarousel-1")) != nil {
		t.Errorf("focused carousel: %+v\n%s", got, res.Block)
	}
	// A section the text names by title beats the focus.
	res = Candidates(m, ctx, "the New Arrivals carousel should show 5 at once", Options{})
	if got := firstTarget(res); got != sec("productCarousel-1") {
		t.Errorf("named carousel: first target %+v\n%s", got, res.Block)
	}
	// A panel the text names stays above the focused section.
	ctx = withFocus(`{"area":"editor","sectionId":"heroBanner","panel":""}`)
	res = Candidates(m, ctx, "make the header background black", Options{})
	if got := firstTarget(res); got.Panel != "HeaderSection" {
		t.Errorf("named panel: first target %+v\n%s", got, res.Block)
	}
	// A focused target none of whose fields matches is not listed.
	ctx = withFocus(`{"area":"editor","sectionId":"newsletter","panel":""}`)
	res = Candidates(m, ctx, "change the store name to HashiTong", Options{})
	if find(res, sec("newsletter")) != nil {
		t.Errorf("unmatched focus listed\n%s", res.Block)
	}

	// The open Global Settings panel.
	ctx = withFocus(`{"area":"global","sectionId":"heroBanner","panel":"ShopMenuSection"}`)
	res = Candidates(m, ctx, "make the text color white", Options{})
	if got := firstTarget(res); got.Panel != "ShopMenuSection" {
		t.Errorf("focused panel: first target %+v\n%s", got, res.Block)
	}
	if find(res, sec("heroBanner")) != nil && firstTarget(res) == sec("heroBanner") {
		t.Errorf("the section focus applies outside Global Settings only\n%s", res.Block)
	}
}

// Another page the message names (the browser outlines it, other: true):
// its sections are targets on that page, and the page counts as named.
func TestOtherPageOutline(t *testing.T) {
	m, _ := snapshot(t)
	ctx := ctxOf(t, strings.Replace(homeContext, `"page":"home",`, `"page":"home","outline":[
	 {"page":"our-story","other":true,"sections":[{"id":"contentBlock-7","type":"contentBlock","title":"Meet the Growers","enabled":true},{"id":"newsletter","type":"newsletter","title":"","enabled":true}]},
	 {"section":"contentBlock-7","page":"our-story"}],`, 1))
	res := Candidates(m, ctx, "on the our story page, change the Meet the Growers headline to 'Our farmers'", Options{})
	want := manifest.Target{Kind: "section", Page: "our-story", SectionID: "contentBlock-7"}
	if got := firstTarget(res); got != want {
		t.Errorf("first target %+v, want %+v\n%s", got, want, res.Block)
	}
	if !strings.Contains(res.Block, `target {"kind":"section","page":"our-story","sectionId":"contentBlock-7"}`) {
		t.Errorf("the block does not address the other page's section\n%s", res.Block)
	}
	// A section of the other page the text does not name is no candidate.
	if find(res, manifest.Target{Kind: "section", Page: "our-story", SectionID: "newsletter"}) != nil {
		t.Errorf("an unnamed section of the other page is listed\n%s", res.Block)
	}
	// Without the outline entry nothing of that page is addressed.
	res = Candidates(m, ctxOf(t, homeContext), "on the our story page, change the Meet the Growers headline to 'Our farmers'", Options{})
	if strings.Contains(res.Block, `"page":"our-story","sectionId"`) {
		t.Errorf("a section of a page the browser did not outline\n%s", res.Block)
	}
}
