package manifest

import (
	"encoding/json"
	"errors"
	"os"
	"strings"
	"sync"
	"testing"
)

// snapshotHash is the hash of testdata/manifest.snapshot.json, cross-checked
// against an independent Python computation (json.dumps sort_keys,
// separators=(',',':'), ensure_ascii=False, generated_from/coverage removed)
// on 2026-10-01.
const snapshotHash = "727d151b5d6b04cd68b73199ab0b3bdf061263f9623f4f6eda00a79137c06f38"

var (
	snapOnce sync.Once
	snapData []byte
	snapM    *Manifest
	snapErr  error
)

func snapshot(t *testing.T) (*Manifest, []byte) {
	t.Helper()
	snapOnce.Do(func() {
		snapData, snapErr = os.ReadFile("../../testdata/manifest.snapshot.json")
		if snapErr != nil {
			return
		}
		snapM, snapErr = Parse(snapData)
	})
	if snapErr != nil {
		t.Fatalf("snapshot: %v", snapErr)
	}
	return snapM, snapData
}

// small is a hand manifest covering the shapes the snapshot tests cannot
// isolate (an enum with more than 12 values, nested lists, an unknown member).
const small = `{
  "version": 1,
  "generated_from": {"branch": "x", "sha": "abc", "at": "2026-10-01T00:00:00Z"},
  "sections": {
    "heroBanner": {
      "editorKey": "hero", "pages": ["home"],
      "fields": {
        "paddingY": {"path": "settings.paddingY", "label": "Vertical Padding", "type": "range", "min": 0, "max": 160, "step": 8, "unset_value": null, "source": {"file": "a.jsx", "line": 1}},
        "mood": {"path": "settings.mood", "label": "Mood", "type": "enum", "enum": ["a","b","c","d","e","f","g","h","i","j","k","l","m"], "enum_labels": {"a":"A"}, "source": "allowlist"},
        "tagId": {"path": "settings.tagId", "label": "", "type": "ref", "ref": "tag", "derived_from": "tagSlug", "source": "allowlist"}
      },
      "lists": {
        "slides": {"path": "settings.slides", "item": {
          "headline": {"path": "settings.slides[].headline", "label": "Headline", "type": "richtext", "default": "", "help": "Main line", "source": "allowlist"},
          "btnText": {"path": "settings.slides[].btnText", "label": "Button text", "type": "text", "source": "allowlist"}
        }, "ops": ["add", "remove"]},
        "slides[].bullets": {"path": "settings.slides[].bullets", "item_ref": "details.HeroBulletDetailsPanel", "item": {}, "ops": ["add", "reorder"]}
      },
      "mirror": {"from": "slides[0]", "keys": ["headline", "btnText"], "aliases": {"btnText": ["ctaText"]}},
      "structural": [],
      "future_member": {"kept": true}
    },
    "locations": {"editorKey": "locations", "pages": ["home"], "fields": {"title": {"path": "settings.title", "label": "Title", "type": "text", "source": "allowlist"}}, "lists": {}, "structural": []}
  },
  "globalPanels": {
    "HeaderSection": {"fields": {"headerBgColor": {"path": "globalSettings.headerBgColor", "label": "Header background", "type": "color", "unset_value": "", "source": "allowlist"},
                                 "showLocationInHeader": {"path": "globalSettings.showLocationInHeader", "label": "Show location", "type": "bool", "source": "allowlist"}}},
    "FooterSection": {"fields": {"showLocationInHeader": {"path": "globalSettings.showLocationInHeader", "label": "Show location", "type": "bool", "source": "allowlist"},
                                 "footerBgColor": {"path": "globalSettings.footerBgColor", "label": "Footer background", "type": "color", "source": "allowlist"}}}
  },
  "branding": {"fields": {"siteName": {"path": "branding.siteName", "label": "Site name", "type": "text", "source": "allowlist"}}},
  "pages": {
    "_common": {"fields": {"published": {"path": "pageVisibility.{page}.published", "label": "Published", "type": "bool", "default": true, "source": "allowlist"}}},
    "faq": {"fields": {"layout": {"path": "pageVisibility.faq.layout", "label": "Layout", "type": "enum", "enum": ["accordion", "flat"], "source": "allowlist"}},
            "lists": {"faqCategories": {"path": "faqCategories", "item": {"name": {"path": "faqCategories[].name", "label": "Category Name", "type": "text", "default": "New Category", "source": "allowlist"}}, "ops": ["add", "remove", "reorder"]}}}
  },
  "details": {
    "HeroBulletDetailsPanel": {"fields": {"title": {"path": "item.title", "label": "Title", "type": "text", "source": "allowlist"}}, "lists": {}},
    "LocationDetailsPanel": {"fields": {"address": {"path": "item.address", "label": "Address", "type": "text", "source": "allowlist"}}, "lists": {}},
    "CategoryDetailsPanel": {"fields": {"name": {"path": "item.name", "label": "Name", "type": "text", "source": "allowlist"}}, "lists": {}},
    "SeoPanel": {"fields": {"pages.home.title": {"path": "item.pages.home.title", "label": "Page Title", "type": "text", "source": "allowlist"},
                            "general.siteTitle": {"path": "item.general.siteTitle", "label": "Site Title", "type": "text", "source": "allowlist"}}, "lists": {}}
  },
  "coverage": {"sites_total": 1}
}`

func parseSmall(t *testing.T) *Manifest {
	t.Helper()
	m, err := Parse([]byte(small))
	if err != nil {
		t.Fatal(err)
	}
	return m
}

// ─── Parse ───────────────────────────────────────────────────────────────────

func TestParseSnapshotCounts(t *testing.T) {
	m, _ := snapshot(t)
	c := m.Counts()
	want := Counts{Sections: 19, GlobalPanels: 12, Pages: 17, Details: 11, Fields: 2094, Lists: 69}
	if c != want {
		t.Fatalf("counts = %+v, want %+v", c, want)
	}
	if m.GeneratedFrom == nil || m.GeneratedFrom.SHA != "0f9fd067303f58871bc212aa045650fb53523fa3" {
		t.Fatalf("generated_from = %+v", m.GeneratedFrom)
	}
	if len(m.Coverage) == 0 {
		t.Fatal("coverage should be kept opaque, not dropped")
	}
}

func TestParseRejectsVersion(t *testing.T) {
	for _, doc := range []string{`{"version": 2}`, `{}`, `{"version": "1"}`} {
		_, err := Parse([]byte(doc))
		if doc == `{"version": "1"}` {
			if err == nil {
				t.Fatalf("%s: expected error", doc)
			}
			continue
		}
		if !errors.Is(err, ErrVersion) {
			t.Fatalf("%s: err = %v, want ErrVersion", doc, err)
		}
	}
}

func TestParseRejectsGarbage(t *testing.T) {
	for _, doc := range []string{`null`, `[]`, `{"version": 1,`, ``} {
		if _, err := Parse([]byte(doc)); err == nil {
			t.Fatalf("%q: expected error", doc)
		}
	}
}

func TestParseToleratesUnknownMembersButHashesThem(t *testing.T) {
	m := parseSmall(t)
	if _, ok := m.Sections["heroBanner"]; !ok {
		t.Fatal("heroBanner missing")
	}
	// Unknown members survive in Raw and change the hash.
	withExtra := strings.Replace(small, `"future_member": {"kept": true}`, `"future_member": {"kept": false}`, 1)
	m2, err := Parse([]byte(withExtra))
	if err != nil {
		t.Fatal(err)
	}
	if m.Hash() == m2.Hash() {
		t.Fatal("unknown member change must change the hash")
	}
	sec := m.Raw()["sections"].(map[string]any)["heroBanner"].(map[string]any)
	if _, ok := sec["future_member"]; !ok {
		t.Fatal("unknown member dropped from Raw")
	}
}

func TestParseNormalisesMissingMaps(t *testing.T) {
	m, err := Parse([]byte(`{"version": 1}`))
	if err != nil {
		t.Fatal(err)
	}
	if m.Sections == nil || m.Pages == nil || m.Branding == nil || m.Branding.Fields == nil || m.Details == nil || m.GlobalPanels == nil {
		t.Fatal("maps must be non-nil after Parse")
	}
	if c := m.Counts(); c.Fields != 0 {
		t.Fatalf("counts = %+v", c)
	}
}

func TestFieldPresenceOfNullValues(t *testing.T) {
	m := parseSmall(t)
	f := m.Sections["heroBanner"].Fields["paddingY"]
	if !f.HasUnsetValue() || string(f.UnsetValue) != "null" {
		t.Fatalf("unset_value: present-null must be kept, got %q", f.UnsetValue)
	}
	if f.HasDefault() {
		t.Fatal("paddingY has no default")
	}
	h := m.Sections["heroBanner"].Lists["slides"].Item["headline"]
	if !h.HasDefault() || string(h.Default) != `""` {
		t.Fatalf("default = %q", h.Default)
	}
	if !m.Sections["heroBanner"].Fields["tagId"].ReadOnly() {
		t.Fatal("derived_from field must be read-only")
	}
}

func TestSnapshotFieldShapes(t *testing.T) {
	m, _ := snapshot(t)
	hb := m.Sections["heroBanner"]
	if hb.EditorKey != "hero" || len(hb.Pages) != 1 || hb.Pages[0] != "home" {
		t.Fatalf("heroBanner meta = %+v", hb)
	}
	if hb.Mirror == nil || hb.Mirror.From != "slides[0]" || len(hb.Mirror.Keys) < 50 || hb.Mirror.Aliases["btnText"][0] != "ctaText" {
		t.Fatalf("mirror = %+v", hb.Mirror)
	}
	slides := hb.Lists["slides"]
	if strings.Join(slides.Ops, "|") != "add|remove" || slides.Path != "settings.slides" {
		t.Fatalf("slides = %+v", slides)
	}
	bullets := hb.Lists["slides[].bullets"]
	if bullets.ItemRef != "details.HeroBulletDetailsPanel" {
		t.Fatalf("bullets item_ref = %q", bullets.ItemRef)
	}
	layout := slides.Item["layout"]
	if layout.Type != "enum" || len(layout.Enum) != 4 || layout.EnumLabels["split"] != "Split" || string(layout.Default) != `"classic"` {
		t.Fatalf("layout = %+v", layout)
	}
	pc := m.GlobalPanels["ProductCardSection"]
	if len(pc.Presets) != 4 || pc.Presets[0].Name != "standard" || len(pc.Presets[0].Writes) == 0 {
		t.Fatalf("presets = %+v", pc.Presets)
	}
	if m.Pages["faq"].Lists["faqCategories[].items"].Item["q"].Type != "richtext" {
		t.Fatal("nested faq list item missing")
	}
	if m.Pages[CommonPageKey].Fields["published"].Path != "pageVisibility.{page}.published" {
		t.Fatal("_common published missing")
	}
	// A keyed field keeps its `{locationId}` key verbatim.
	if f := m.Sections["contentBlock"].Fields["locationOverrides.{locationId}.address"]; f == nil || f.KeyedBy != "location" {
		t.Fatalf("keyed field = %+v", f)
	}
	// Source is a string ("allowlist") on _common and an object elsewhere.
	if !strings.HasPrefix(string(m.Pages[CommonPageKey].Fields["published"].Source), `"`) {
		t.Fatal("string source should be kept")
	}
}

// ─── Hash / Canonical ────────────────────────────────────────────────────────

func TestHashSnapshotMatchesIndependentComputation(t *testing.T) {
	m, data := snapshot(t)
	if m.Hash() != snapshotHash {
		t.Fatalf("hash = %s, want %s", m.Hash(), snapshotHash)
	}
	h, err := HashBytes(data)
	if err != nil || h != snapshotHash {
		t.Fatalf("HashBytes = %s, %v", h, err)
	}
}

func TestHashIgnoresGeneratedFromAndCoverage(t *testing.T) {
	base := parseSmall(t)
	churned := strings.Replace(small, `"at": "2026-10-01T00:00:00Z"`, `"at": "2027-01-01T00:00:00Z"`, 1)
	churned = strings.Replace(churned, `"coverage": {"sites_total": 1}`, `"coverage": {"sites_total": 99, "unreached": []}`, 1)
	m2, err := Parse([]byte(churned))
	if err != nil {
		t.Fatal(err)
	}
	if base.Hash() != m2.Hash() {
		t.Fatal("generated_from / coverage churn must not change the hash")
	}
	noMeta := strings.Replace(small, `"generated_from": {"branch": "x", "sha": "abc", "at": "2026-10-01T00:00:00Z"},`, ``, 1)
	m3, err := Parse([]byte(noMeta))
	if err != nil {
		t.Fatal(err)
	}
	if base.Hash() != m3.Hash() {
		t.Fatal("absent generated_from must hash the same as present")
	}
}

func TestHashStableUnderKeyOrderAndWhitespace(t *testing.T) {
	_, data := snapshot(t)
	// Round-trip through encoding/json: map keys come back sorted and the
	// whitespace is gone, so the document bytes differ while the hash must not.
	var v any
	if err := json.Unmarshal(data, &v); err != nil {
		t.Fatal(err)
	}
	re, _ := json.Marshal(v)
	if string(re) == string(data) {
		t.Fatal("test precondition: re-encoded bytes should differ")
	}
	h, err := HashBytes(re)
	if err != nil || h != snapshotHash {
		t.Fatalf("hash after re-encode = %s, %v", h, err)
	}
	a, _ := HashBytes([]byte(`{"version":1,"b":{"y":1,"x":[1,2]},"a":"s"}`))
	b, _ := HashBytes([]byte(`{ "a" : "s", "version" : 1, "b" : { "x" : [ 1 , 2 ], "y" : 1 } }`))
	if a != b {
		t.Fatal("key order / whitespace changed the hash")
	}
	c, _ := HashBytes([]byte(`{"version":1,"b":{"y":1,"x":[2,1]},"a":"s"}`))
	if a == c {
		t.Fatal("array order is significant")
	}
}

func TestHashBytesDeterministicAcrossParses(t *testing.T) {
	m1 := parseSmall(t)
	m2 := parseSmall(t)
	if m1.Hash() != m2.Hash() || len(m1.Hash()) != 64 {
		t.Fatalf("hashes %s / %s", m1.Hash(), m2.Hash())
	}
}

func TestCanonicalEncoding(t *testing.T) {
	cases := []struct {
		in   any
		want string
	}{
		{nil, `null`},
		{true, `true`},
		{false, `false`},
		{float64(1), `1`},
		{float64(160), `160`},
		{1.5, `1.5`},
		{0.05, `0.05`},
		{float64(0), `0`},
		{-2.5, `-2.5`},
		{1e21, `1e+21`},
		{1e-7, `1e-7`},
		{123456789012345678, `123456789012345678`},
		{json.Number("42"), `42`},
		{"plain", `"plain"`},
		{`q"b\`, `"q\"b\\"`},
		{"\n\r\t\b\f", `"\n\r\t\b\f"`},
		{"\u0001", `"\u0001"`},
		{"<a>&'", `"<a>&'"`},
		{"Tagline › Text", `"Tagline › Text"`},
		// Written without backslash-u sequences in source on purpose.
		{string([]rune{0x2028, 0x2029}), "\"" + "\\" + "u2028" + "\\" + "u2029" + "\""},
		{"bad\xffutf8", "\"bad" + "\\" + "ufffdutf8\""},
		{[]any{}, `[]`},
		{map[string]any{}, `{}`},
		{map[string]any{"b": []any{1.0, "x"}, "a": map[string]any{"z": nil, "y": true}}, `{"a":{"y":true,"z":null},"b":[1,"x"]}`},
		{json.RawMessage(`{ "k" : [ 1 ] }`), `{"k":[1]}`},
		{[]string{"a", "b"}, `["a","b"]`},
		{&Gate{Key: "k", Value: json.RawMessage(`["x","y"]`)}, `{"key":"k","value":["x","y"]}`},
	}
	for _, c := range cases {
		got, err := Canonical(c.in)
		if err != nil {
			t.Fatalf("%v: %v", c.in, err)
		}
		if string(got) != c.want {
			t.Errorf("Canonical(%#v) = %s, want %s", c.in, got, c.want)
		}
	}
}

func TestCanonicalMatchesEncodingJSONForNumbers(t *testing.T) {
	// The number rules are encoding/json's; prove it on a spread of values.
	for _, f := range []float64{0, 1, -1, 0.1, 0.05, 2.5e-7, 1e-6, 123456.789, 1e20, 1e21, 5e300, 4.9e-324} {
		want, _ := json.Marshal(f)
		got, err := Canonical(f)
		if err != nil || string(got) != string(want) {
			t.Errorf("%v: got %s want %s (%v)", f, got, want, err)
		}
	}
}

func TestCanonicalRejectsNaN(t *testing.T) {
	if _, err := Canonical(map[string]any{"x": mathNaN()}); err == nil {
		t.Fatal("NaN must be rejected")
	}
}

func mathNaN() float64 { z := 0.0; return z / z }

// ─── Resolve ─────────────────────────────────────────────────────────────────

func TestLookupSectionTypeID(t *testing.T) {
	m, _ := snapshot(t)
	cases := map[string]string{
		"heroBanner":            "heroBanner",
		"heroBanner-2":          "heroBanner",
		"productCarousel-17":    "productCarousel",
		"productCarousel-deals": "productCarousel",
		"locations-x":           "locations",
		"contentBlock-about-3":  "contentBlock",
		"nope":                  "",
		"newsletter-x":          "", // not a prefix family
		"":                      "",
	}
	for id, want := range cases {
		got, ok := m.LookupSectionTypeID(id)
		if got != want || ok != (want != "") {
			t.Errorf("%q → %q,%v want %q", id, got, ok, want)
		}
	}
}

func TestResolveSection(t *testing.T) {
	m, _ := snapshot(t)
	r, err := m.Resolve(Target{Kind: "section", Page: "home", SectionID: "heroBanner-3"})
	if err != nil {
		t.Fatal(err)
	}
	if r.Container != "sections.heroBanner" || r.TypeID != "heroBanner" || r.EditorKey != "hero" || r.Mirror == nil || len(r.Fields) != 2 || len(r.Lists) != 2 {
		t.Fatalf("resolved = %+v", r)
	}
	for _, bad := range []Target{
		{Kind: "section", SectionID: "heroBanner"},
		{Kind: "section", Page: "home"},
		{Kind: "section", Page: "home", SectionID: "unknownThing"},
		{Kind: "nope"},
		{},
	} {
		if _, err := m.Resolve(bad); !errors.Is(err, ErrTarget) {
			t.Errorf("%+v: err = %v, want ErrTarget", bad, err)
		}
	}
}

func TestResolvePinnedSection(t *testing.T) {
	m, _ := snapshot(t)
	r, err := m.Resolve(Target{Kind: "section", Page: "faq", SectionID: "__pinned_faq"})
	if err != nil {
		t.Fatal(err)
	}
	if r.Pinned == nil || r.Pinned.ID != "__pinned_faq" || r.Container != "pages.faq" || r.Kind != "section" {
		t.Fatalf("resolved = %+v", r)
	}
	if _, ok := r.Fields["published"]; !ok {
		t.Fatal("pinned row edits the page's fields incl. _common published")
	}
	if _, err := m.Resolve(Target{Kind: "section", Page: "home", SectionID: "__pinned_faq"}); !errors.Is(err, ErrTarget) {
		t.Fatal("pinned id on the wrong page must fail")
	}
	// age-gate has a pinned def but no fields in the manifest → still resolves (page entry exists, empty).
	if _, err := m.Resolve(Target{Kind: "section", Page: "age-gate", SectionID: "__pinned_age_gate"}); err != nil {
		t.Fatal(err)
	}
}

func TestResolveGlobalMergesPanels(t *testing.T) {
	m := parseSmall(t)
	r, err := m.Resolve(Target{Kind: "global"})
	if err != nil {
		t.Fatal(err)
	}
	if len(r.Fields) != 3 || len(r.Panels) != 2 || r.Container != "globalPanels" {
		t.Fatalf("merged = %d fields, panels %v", len(r.Fields), r.Panels)
	}
	n, err := m.Resolve(Target{Kind: "global", Panel: "FooterSection"})
	if err != nil || len(n.Fields) != 2 || n.Container != "globalPanels.FooterSection" {
		t.Fatalf("narrowed = %+v, %v", n, err)
	}
	if _, err := m.Resolve(Target{Kind: "global", Panel: "Nope"}); !errors.Is(err, ErrTarget) {
		t.Fatal("unknown panel must fail")
	}
	snap, _ := snapshot(t)
	all, _ := snap.Resolve(Target{Kind: "global"})
	if len(all.Panels) != 12 || len(all.Fields) < 300 {
		t.Fatalf("snapshot global: %d panels, %d fields", len(all.Panels), len(all.Fields))
	}
}

func TestResolvePage(t *testing.T) {
	m := parseSmall(t)
	r, err := m.Resolve(Target{Kind: "page", Page: "faq"})
	if err != nil {
		t.Fatal(err)
	}
	if !r.Known || r.Fields["published"].Path != "pageVisibility.faq.published" || r.Fields["layout"] == nil || len(r.Lists) != 1 {
		t.Fatalf("faq = %+v", r)
	}
	if keys := r.ListKeys(); keys[0] != SectionsListKey || keys[1] != "faqCategories" {
		t.Fatalf("list keys = %v", keys)
	}
	u, err := m.Resolve(Target{Kind: "page", Page: "about-us"})
	if err != nil {
		t.Fatal(err)
	}
	if u.Known || len(u.Fields) != 1 || u.Fields["published"].Path != "pageVisibility.about-us.published" {
		t.Fatalf("unknown page = %+v", u)
	}
	for _, bad := range []string{"", "_common"} {
		if _, err := m.Resolve(Target{Kind: "page", Page: bad}); !errors.Is(err, ErrTarget) {
			t.Errorf("page %q must fail", bad)
		}
	}
	b, err := m.Resolve(Target{Kind: "branding"})
	if err != nil || b.Fields["siteName"] == nil {
		t.Fatalf("branding = %+v, %v", b, err)
	}
}

func TestResolveSeoPage(t *testing.T) {
	m := parseSmall(t)
	r, err := m.Resolve(Target{Kind: "page", Page: SEOPage})
	if err != nil {
		t.Fatal(err)
	}
	if r.Container != "details.SeoPanel" || r.Fields["pages.home.title"].Path != "seo.pages.home.title" {
		t.Fatalf("seo = %+v", r)
	}
	if len(r.ListKeys()) != 0 {
		t.Fatal("seo page has no sections list")
	}
	// The original panel's path must not be mutated.
	if m.Details["SeoPanel"].Fields["pages.home.title"].Path != "item.pages.home.title" {
		t.Fatal("resolvePage mutated the details panel")
	}
}

func TestResolveContainerPaths(t *testing.T) {
	m, _ := snapshot(t)
	for _, p := range m.ContainerPaths() {
		if _, err := m.ResolveContainer(p); err != nil && p != "pages._common" {
			t.Errorf("%s: %v", p, err)
		}
	}
	if len(m.ContainerPaths()) != 19+12+1+17+11 {
		t.Fatalf("paths = %d", len(m.ContainerPaths()))
	}
	r, err := m.ResolveContainer("details.ReviewDetailsPanel")
	if err != nil || r.Kind != "details" || r.Fields["rating"] == nil {
		t.Fatalf("details = %+v, %v", r, err)
	}
	g, err := m.ResolveContainer("globalPanels")
	if err != nil || len(g.Panels) != 12 {
		t.Fatalf("globalPanels = %+v, %v", g, err)
	}
	for _, bad := range []string{"", "sections", "sections.nope", "pages.nope", "x.y", "branding.extra"} {
		if _, err := m.ResolveContainer(bad); !errors.Is(err, ErrTarget) {
			t.Errorf("%q: expected ErrTarget, got %v", bad, err)
		}
	}
	if c, ok := m.Container("branding"); !ok || c != m.Branding {
		t.Fatal("Container(branding)")
	}
	if _, ok := m.Container("sections.zzz"); ok {
		t.Fatal("Container unknown must be !ok")
	}
}

func TestFieldResolutionOrder(t *testing.T) {
	m := parseSmall(t)
	sec, _ := m.Resolve(Target{Kind: "section", Page: "home", SectionID: "heroBanner"})
	cases := []struct {
		key, via, list string
	}{
		{"paddingY", "field", ""},
		{"headline", "mirror", "slides"},
		{"ctaText", "alias", "slides"},
		{"locationOverrides.loc-1.address", "keyed", ""},
	}
	for _, c := range cases {
		ref, ok := m.Field(sec, c.key)
		if !ok || ref.Via != c.via || ref.List != c.list {
			t.Errorf("%s → %+v,%v want via=%s list=%s", c.key, ref, ok, c.via, c.list)
		}
	}
	for _, bad := range []string{"", "nope", "locationOverrides.loc-1.nope", "locationOverrides.address", "categories.x.name"} {
		if _, ok := m.Field(sec, bad); ok {
			t.Errorf("%q must not resolve on a section", bad)
		}
	}
	glob, _ := m.Resolve(Target{Kind: "global"})
	if ref, ok := m.Field(glob, "categories.cookies.name"); !ok || ref.Panel != "CategoryDetailsPanel" {
		t.Fatalf("global keyed = %+v,%v", ref, ok)
	}
	seo, _ := m.Resolve(Target{Kind: "page", Page: SEOPage})
	if ref, ok := m.Field(seo, "pages.faq.title"); !ok || ref.Via != "seo-template" {
		t.Fatalf("seo template = %+v,%v", ref, ok)
	}
	if _, ok := m.Field(seo, "pages.faq.nope"); ok {
		t.Fatal("seo template with unknown tail must fail")
	}
}

func TestItemFieldsFollowsItemRef(t *testing.T) {
	m := parseSmall(t)
	l := m.Sections["heroBanner"].Lists["slides[].bullets"]
	items, panel := m.ItemFields(l)
	if panel != "HeroBulletDetailsPanel" || items["title"] == nil {
		t.Fatalf("items = %v panel = %s", items, panel)
	}
	own, panel := m.ItemFields(m.Sections["heroBanner"].Lists["slides"])
	if panel != "" || own["headline"] == nil {
		t.Fatal("own item fields")
	}
	if ListDepth("slides[].bullets") != 2 || ListDepth("slides") != 1 {
		t.Fatal("ListDepth")
	}
}

// ─── Lookup / Describe ───────────────────────────────────────────────────────

func TestLookupFieldsQueryAcrossLists(t *testing.T) {
	m, _ := snapshot(t)
	res, err := m.LookupFields(LookupRequest{Target: &Target{Kind: "section", Page: "home", SectionID: "heroBanner-1"}, Query: "headline"})
	if err != nil {
		t.Fatal(err)
	}
	if res.Total == 0 || res.Mirror != "slides[0]" {
		t.Fatalf("res = %+v", res)
	}
	seen := map[string]bool{}
	for _, f := range res.Fields {
		seen[f.Key] = true
		if f.List != "slides" {
			t.Errorf("%s should come from the slides list, got %q", f.Key, f.List)
		}
	}
	for _, want := range []string{"headline", "headlineColor", "headlineFontSize"} {
		if !seen[want] {
			t.Errorf("missing %s in %v", want, seen)
		}
	}
	if res.Fields[0].Key != "headline" {
		t.Fatalf("exact key must rank first, got %s", res.Fields[0].Key)
	}
}

func TestLookupFieldsCompactShape(t *testing.T) {
	m, _ := snapshot(t)
	res, err := m.LookupFields(LookupRequest{Container: "sections.heroBanner", Keys: []string{"paddingY", "layout", "ctaText"}})
	if err != nil {
		t.Fatal(err)
	}
	b, _ := json.Marshal(res)
	s := string(b)
	for _, forbidden := range []string{`"path"`, `"source"`, `"help"`} {
		if strings.Contains(s, forbidden) {
			t.Errorf("output must not carry %s: %s", forbidden, s)
		}
	}
	if len(res.Fields) != 3 {
		t.Fatalf("fields = %d: %s", len(res.Fields), s)
	}
	byKey := map[string]FieldOut{}
	for _, f := range res.Fields {
		byKey[f.Key] = f
	}
	p := byKey["paddingY"]
	if p.Type != "range" || p.Min == nil || *p.Max != 160 || string(p.Unset) != "null" {
		t.Fatalf("paddingY = %+v", p)
	}
	l := byKey["layout"]
	if l.List != "slides" || len(l.Enum) != 4 || l.EnumLabels["classic"] != "Classic" || string(l.Default) != `"classic"` {
		t.Fatalf("layout = %+v", l)
	}
	if byKey["ctaText"].Via != "alias" || byKey["ctaText"].List != "slides" {
		t.Fatalf("ctaText = %+v", byKey["ctaText"])
	}
	withHelp, _ := m.LookupFields(LookupRequest{Container: "sections.heroBanner", Keys: []string{"maxWidth"}, Help: true})
	if withHelp.Fields[0].Help == "" {
		t.Fatal("help must be included on request")
	}
}

func TestLookupFieldsEnumLabelsDroppedWhenLarge(t *testing.T) {
	m := parseSmall(t)
	res, err := m.LookupFields(LookupRequest{Container: "sections.heroBanner", Keys: []string{"mood", "tagId"}})
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Fields[0].Enum) != 13 || res.Fields[0].EnumLabels != nil {
		t.Fatalf("mood = %+v", res.Fields[0])
	}
	if res.Fields[1].RO != "tagSlug" || res.Fields[1].Ref != "tag" {
		t.Fatalf("tagId = %+v", res.Fields[1])
	}
}

func TestLookupFieldsListAndItemPrefix(t *testing.T) {
	m, _ := snapshot(t)
	res, err := m.LookupFields(LookupRequest{Target: &Target{Kind: "page", Page: "faq"}, List: "faqCategories[].items"})
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Fields) != 2 || len(res.Lists) != 1 || res.Lists[0].Depth != 2 || strings.Join(res.Lists[0].ItemKeys, ",") != "a,q" {
		t.Fatalf("res = %+v", res)
	}
	// item_ref lists answer with the details panel's keys (already bare of `item.`).
	rev, err := m.LookupFields(LookupRequest{Target: &Target{Kind: "section", Page: "home", SectionID: "testimonials"}, List: "reviews"})
	if err != nil {
		t.Fatal(err)
	}
	keys := map[string]bool{}
	for _, f := range rev.Fields {
		keys[f.Key] = true
		if strings.HasPrefix(f.Key, DetailItemPrefix) {
			t.Fatalf("item. prefix leaked: %s", f.Key)
		}
	}
	if !keys["rating"] || !keys["text"] || rev.Lists[0].ItemRef != "details.ReviewDetailsPanel" {
		t.Fatalf("reviews = %+v", rev)
	}
	// The page `sections` pseudo-list.
	sec, err := m.LookupFields(LookupRequest{Target: &Target{Kind: "page", Page: "faq"}, List: SectionsListKey})
	if err != nil || len(sec.Lists) != 1 || strings.Join(sec.Lists[0].Ops, ",") != strings.Join(SectionsListOps, ",") {
		t.Fatalf("sections = %+v, %v", sec, err)
	}
	// Unknown list → E_LIST naming the known lists.
	_, err = m.LookupFields(LookupRequest{Target: &Target{Kind: "page", Page: "faq"}, List: "nope"})
	var qe *QueryError
	if !errors.As(err, &qe) || qe.Code != "E_LIST" || !strings.Contains(qe.Message, "faqCategories") {
		t.Fatalf("err = %v", err)
	}
}

func TestLookupFieldsCapsAndTruncation(t *testing.T) {
	m, _ := snapshot(t)
	res, err := m.LookupFields(LookupRequest{Container: "sections.perksCards", Limit: 1000})
	if err != nil {
		t.Fatal(err)
	}
	if res.Total != 126 || len(res.Fields) > MaxLookupLimit || !res.Truncated || !strings.Contains(res.Note, "truncated") {
		t.Fatalf("total=%d returned=%d truncated=%v note=%q", res.Total, len(res.Fields), res.Truncated, res.Note)
	}
	b, _ := json.Marshal(res.Fields)
	if len(b) > LookupByteCap+1024 {
		t.Fatalf("fields JSON = %d bytes, cap %d", len(b), LookupByteCap)
	}
	lim, _ := m.LookupFields(LookupRequest{Container: "sections.perksCards", Limit: 5})
	if len(lim.Fields) != 5 || !lim.Truncated || lim.Total != 126 {
		t.Fatalf("limit 5 → %d (total %d)", len(lim.Fields), lim.Total)
	}
	def, _ := m.LookupFields(LookupRequest{Container: "sections.perksCards"})
	if len(def.Fields) != DefaultLookupLimit {
		t.Fatalf("default limit → %d", len(def.Fields))
	}
	small, _ := m.LookupFields(LookupRequest{Container: "sections.sectionDivider"})
	if small.Truncated || small.Total != 14 || len(small.Fields) != 14 {
		t.Fatalf("sectionDivider = %+v", small)
	}
}

func TestLookupFieldsErrors(t *testing.T) {
	m := parseSmall(t)
	_, err := m.LookupFields(LookupRequest{})
	var qe *QueryError
	if !errors.As(err, &qe) || qe.Code != "E_OP" {
		t.Fatalf("empty request err = %v", err)
	}
	_, err = m.LookupFields(LookupRequest{Target: &Target{Kind: "section", Page: "home", SectionID: "zzz"}})
	if !errors.As(err, &qe) || qe.Code != "E_TARGET" || strings.HasPrefix(qe.Message, "E_TARGET") {
		t.Fatalf("bad target err = %v", err)
	}
	if qe.Error() == "" {
		t.Fatal("Error()")
	}
	un, err := m.LookupFields(LookupRequest{Target: &Target{Kind: "page", Page: "custom-slug"}})
	if err != nil || un.Known || !strings.Contains(un.Note, "not in the manifest") || len(un.Fields) != 1 {
		t.Fatalf("unknown page = %+v, %v", un, err)
	}
}

func TestLookupFieldsLabelAndWordMatch(t *testing.T) {
	m, _ := snapshot(t)
	// "vertical" only appears in the label of paddingY on heroBanner.
	res, err := m.LookupFields(LookupRequest{Container: "sections.heroBanner", Query: "vertical"})
	if err != nil || len(res.Fields) == 0 || res.Fields[0].Key != "paddingY" {
		t.Fatalf("label match = %+v, %v", res, err)
	}
	// "color" matches the camelCase word in headlineColor etc.
	res, _ = m.LookupFields(LookupRequest{Container: "sections.heroBanner", Query: "color"})
	if res.Total < 5 {
		t.Fatalf("word match total = %d", res.Total)
	}
	// A list key match lists the list and its item fields.
	res, _ = m.LookupFields(LookupRequest{Container: "pages.faq", Query: "faqCategories"})
	if len(res.Lists) != 2 || res.Total < 3 {
		t.Fatalf("list match = %+v", res)
	}
	// No match → empty, not an error.
	res, _ = m.LookupFields(LookupRequest{Container: "sections.heroBanner", Query: "zzzz"})
	if res.Total != 0 || len(res.Fields) != 0 {
		t.Fatalf("no match = %+v", res)
	}
}

func TestDescribeTarget(t *testing.T) {
	m, _ := snapshot(t)
	d, err := m.DescribeTarget(&Target{Kind: "section", Page: "home", SectionID: "heroBanner-2"}, "")
	if err != nil {
		t.Fatal(err)
	}
	if d.TypeID != "heroBanner" || d.EditorKey != "hero" || len(d.Lists) != 2 || d.Mirror == nil || d.Mirror.Keys < 50 || len(d.Keys) != 2 {
		t.Fatalf("hero describe = %+v", d)
	}
	if d.Lists[1].Key != "slides[].bullets" || d.Lists[1].ItemRef != "details.HeroBulletDetailsPanel" || strings.Join(d.Lists[1].ItemKeys, ",") != "description,icon,title" {
		t.Fatalf("bullets list = %+v", d.Lists[1])
	}
	if d.Keyed["locationOverrides"] == "" {
		t.Fatalf("section keyed prefixes = %v", d.Keyed)
	}

	p, err := m.DescribeTarget(&Target{Kind: "page", Page: "products"}, "")
	if err != nil || p.Pinned == nil || p.Pinned.ID != "__pinned_product_list" || !p.Pinned.Reorderable || p.Lists[0].Key != SectionsListKey {
		t.Fatalf("products describe = %+v, %v", p, err)
	}
	g, err := m.DescribeTarget(nil, "globalPanels.ProductCardSection")
	if err != nil || len(g.Preset) != 4 || len(g.Preset[0].Keys) == 0 || len(g.Stems) == 0 || len(g.Keys) != 0 {
		t.Fatalf("ProductCardSection describe = %+v, %v", g, err)
	}
	if g.Keyed["categories"] == "" {
		t.Fatalf("global keyed prefixes = %v", g.Keyed)
	}
	u, err := m.DescribeTarget(&Target{Kind: "page", Page: "my-custom-page"}, "")
	if err != nil || u.Known || u.Note == "" || len(u.Lists) != 1 {
		t.Fatalf("custom page describe = %+v, %v", u, err)
	}
	if _, err := m.DescribeTarget(nil, "sections.nope"); err == nil {
		t.Fatal("unknown container must fail")
	}
}

func TestStemsAndCamelWords(t *testing.T) {
	cases := map[string]string{
		"headlineFontSize":            "headline",
		"announcementBar.show":        "announcementBar",
		"slides[].layout":             "slides",
		"locationOverrides.{id}.mode": "locationOverrides",
		"a":                           "a",
		"ABC":                         "ABC",
		"pageVisibility._meta.x":      "pageVisibility",
	}
	for in, want := range cases {
		if got := Stem(in); got != want {
			t.Errorf("Stem(%q) = %q want %q", in, got, want)
		}
	}
	if w := camelWords("headlineFontSize"); strings.Join(w, ",") != "headline,Font,Size" {
		t.Fatalf("camelWords = %v", w)
	}
	if w := camelWords("faqCategories[].items.q_x"); strings.Join(w, ",") != "faq,Categories,items,q,x" {
		t.Fatalf("camelWords = %v", w)
	}
	s := Stems([]string{"bodyColor", "bodyFont", "title", "aa"})
	if len(s) != 3 || s[0].Stem != "body" || s[0].Count != 2 || s[1].Stem != "aa" {
		t.Fatalf("Stems = %+v", s)
	}
}

// ─── Index ───────────────────────────────────────────────────────────────────

func TestIndexBudgetAndDeterminism(t *testing.T) {
	m, _ := snapshot(t)
	ix := BuildIndex(m)
	t.Logf("index: %d bytes, %d tokens at 4 B/t, %d tokens at 3 B/t", ix.Bytes, ix.Tokens, ix.TokensAt(3))
	if ix.Bytes != len(ix.Text) || ix.Tokens != ix.Bytes/4 {
		t.Fatal("size bookkeeping")
	}
	if ix.TokensAt(3) > 5000 {
		t.Fatalf("index is %d tokens at 3 B/t, budget 5000 (README §10)", ix.TokensAt(3))
	}
	if ix.Bytes < 8000 {
		t.Fatalf("index suspiciously small: %d bytes", ix.Bytes)
	}
	again := BuildIndex(m)
	if again.Text != ix.Text {
		t.Fatal("index must be deterministic")
	}
	m2, _ := Parse(snapData)
	if BuildIndex(m2).Text != ix.Text {
		t.Fatal("index must be identical across parses")
	}
}

func TestIndexContent(t *testing.T) {
	m, _ := snapshot(t)
	ix := BuildIndex(m)
	for _, want := range []string{
		// reviews/testimonials are byte-identical; hero/heroBanner differ in one
		// default and blog/blogSection in four (measured on the snapshot), so
		// those collapse to near-alias lines that name the gap.
		"s:reviews = testimonials",
		"s:hero ≈ heroBanner (differs: slides[].bgColor.default)",
		"s:blog ≈ blogSection (differs: ctaLink.default, ctaText.default, intro.default, layout.default)",
		"mirror=slides[0]",
		"slides[].bullets[add|remove|reorder]→HeroBulletDetailsPanel",
		"products=__pinned_product_list(reorderable)",
		"E_TARGET E_FIELD E_READONLY",
		"presets=standard|compact|minimal|centered",
		"g:HeaderSection f=69",
		"p:faq f=16",
		"d:ReviewDetailsPanel f=4",
		"b: f=137",
		"faqCategories[].items[]: a q",
		`{kind:"section",page,sectionId,storeId?}`,
	} {
		if !strings.Contains(ix.Text, want) {
			t.Errorf("index missing %q", want)
		}
	}
	if strings.Contains(ix.Text, "age-gate = location-selection") || strings.Contains(ix.Text, "location-selection = age-gate") {
		t.Fatal("empty containers must not alias each other")
	}
	if len(ix.Aliases) != 1 || ix.Aliases["sections.reviews"] != "sections.testimonials" {
		t.Fatalf("aliases = %v", ix.Aliases)
	}
	// A near-aliased container is not rendered in full.
	if strings.Contains(ix.Text, "s:hero f=2") || strings.Contains(ix.Text, "s:blog f=43") {
		t.Fatal("near-aliased containers must not be rendered in full")
	}
	if !strings.Contains(ix.Text, "s:heroBanner f=2") || !strings.Contains(ix.Text, "s:blogSection f=43") || !strings.Contains(ix.Text, "s:testimonials f=29") {
		t.Fatal("canonical containers must be rendered in full")
	}
	// The full key list is excluded: a long camelCase key must not appear verbatim.
	if strings.Contains(ix.Text, "headlineFontSize") {
		t.Fatal("index must carry stems, not the full key list")
	}
	if !strings.Contains(ix.Text, m.Hash()[:12]) {
		t.Fatal("index should name its hash")
	}
}

func TestIndexSmallManifest(t *testing.T) {
	m := parseSmall(t)
	ix := BuildIndex(m)
	if !strings.Contains(ix.Text, "s:heroBanner f=3") || !strings.Contains(ix.Text, "aliases=btnText=ctaText") || !strings.Contains(ix.Text, "slides[]: btn headline") {
		t.Fatalf("index:\n%s", ix.Text)
	}
	if len(ix.Aliases) != 0 {
		t.Fatalf("aliases = %v", ix.Aliases)
	}
}

// ─── Diff ────────────────────────────────────────────────────────────────────

func TestDiffIdenticalIsEmpty(t *testing.T) {
	m, data := snapshot(t)
	m2, _ := Parse(data)
	d := ComputeDiff(m, m2)
	if !d.Empty() {
		b, _ := json.Marshal(d)
		t.Fatalf("diff of identical manifests: %s", b)
	}
	if ComputeDiff(parseSmall(t), parseSmall(t)).Empty() != true {
		t.Fatal("small diff")
	}
}

func mutate(t *testing.T, from, to string) *Manifest {
	t.Helper()
	if !strings.Contains(small, from) {
		t.Fatalf("test fixture lacks %q", from)
	}
	m, err := Parse([]byte(strings.Replace(small, from, to, 1)))
	if err != nil {
		t.Fatal(err)
	}
	return m
}

func TestDiffFieldAddedRemovedChanged(t *testing.T) {
	base := parseSmall(t)
	added := mutate(t, `"siteName": {`, `"tagline": {"path": "branding.tagline", "label": "Tagline", "type": "text", "source": "allowlist"}, "siteName": {`)
	d := ComputeDiff(base, added)
	if strings.Join(d.AddedFields, ",") != "branding.tagline" || len(d.RemovedFields) != 0 || len(d.ChangedFields) != 0 {
		t.Fatalf("added: %+v", d)
	}
	back := ComputeDiff(added, base)
	if strings.Join(back.RemovedFields, ",") != "branding.tagline" || len(back.AddedFields) != 0 {
		t.Fatalf("removed: %+v", back)
	}
	changed := mutate(t, `"label": "Vertical Padding", "type": "range", "min": 0, "max": 160`, `"label": "Vertical Padding", "type": "number", "min": 0, "max": 200`)
	d = ComputeDiff(base, changed)
	if len(d.ChangedFields) != 1 || d.ChangedFields[0].Path != "sections.heroBanner.paddingY" || strings.Join(d.ChangedFields[0].Attrs, ",") != "type,max" {
		t.Fatalf("changed: %+v", d)
	}
}

func TestDiffIgnoresSourceOnly(t *testing.T) {
	base := parseSmall(t)
	moved := mutate(t, `"source": {"file": "a.jsx", "line": 1}`, `"source": {"file": "a.jsx", "line": 99}`)
	if d := ComputeDiff(base, moved); !d.Empty() {
		t.Fatalf("source-only change must not diff: %+v", d)
	}
	if base.Hash() == moved.Hash() {
		t.Fatal("but it does change the hash (the hash covers the whole document)")
	}
}

func TestDiffListsAndItems(t *testing.T) {
	base := parseSmall(t)
	ops := mutate(t, `"ops": ["add", "remove"]}`, `"ops": ["add", "remove", "reorder"]}`)
	d := ComputeDiff(base, ops)
	if len(d.ChangedLists) != 1 || d.ChangedLists[0].Path != "sections.heroBanner.slides" || d.ChangedLists[0].Attrs[0] != "ops" {
		t.Fatalf("list ops: %+v", d)
	}
	item := mutate(t, `"btnText": {"path": "settings.slides[].btnText"`, `"btnLink": {"path": "settings.slides[].btnLink", "label": "Link", "type": "text", "source": "x"}, "btnText": {"path": "settings.slides[].btnText"`)
	d = ComputeDiff(base, item)
	if strings.Join(d.AddedFields, ",") != "sections.heroBanner.slides[].btnLink" {
		t.Fatalf("item added: %+v", d)
	}
	newList := mutate(t, `"lists": {"faqCategories"`, `"lists": {"extras": {"path": "extras", "item": {"x": {"path": "extras[].x", "label": "X", "type": "text", "source": "s"}}, "ops": ["add"]}, "faqCategories"`)
	d = ComputeDiff(base, newList)
	if strings.Join(d.AddedLists, ",") != "pages.faq.extras" || strings.Join(d.AddedFields, ",") != "pages.faq.extras[].x" {
		t.Fatalf("new list: %+v", d)
	}
	if r := ComputeDiff(newList, base); strings.Join(r.RemovedLists, ",") != "pages.faq.extras" || strings.Join(r.RemovedFields, ",") != "pages.faq.extras[].x" {
		t.Fatalf("removed list: %+v", r)
	}
}

func TestDiffNewSectionType(t *testing.T) {
	base := parseSmall(t)
	withNew := mutate(t, `"locations": {"editorKey"`, `"banner": {"editorKey": "banner", "pages": ["home"], "fields": {"text": {"path": "settings.text", "label": "Text", "type": "text", "source": "s"}}, "lists": {"rows": {"path": "settings.rows", "item": {"v": {"path": "settings.rows[].v", "label": "V", "type": "text", "source": "s"}}, "ops": []}}, "structural": []}, "locations": {"editorKey"`)
	d := ComputeDiff(base, withNew)
	if strings.Join(d.AddedContainers, ",") != "sections.banner" || strings.Join(d.AddedLists, ",") != "sections.banner.rows" || strings.Join(d.AddedFields, ",") != "sections.banner.rows[].v,sections.banner.text" {
		t.Fatalf("new section: %+v", d)
	}
	r := ComputeDiff(withNew, base)
	if strings.Join(r.RemovedContainers, ",") != "sections.banner" || len(r.RemovedFields) != 2 {
		t.Fatalf("removed section: %+v", r)
	}
	b, _ := json.Marshal(d)
	for _, k := range []string{"added_containers", "removed_fields", "changed_fields", "added_lists"} {
		if !strings.Contains(string(b), `"`+k+`":[`) {
			t.Fatalf("diff JSON must carry %s as an array: %s", k, b)
		}
	}
}

func TestFieldAttrsChanged(t *testing.T) {
	a := &Field{Type: "text", Label: "A", Enum: []any{"x"}, Source: json.RawMessage(`1`)}
	b := &Field{Type: "text", Label: "A", Enum: []any{"x"}, Source: json.RawMessage(`2`)}
	if attrs := FieldAttrsChanged(a, b); len(attrs) != 0 {
		t.Fatalf("source-only: %v", attrs)
	}
	c := &Field{Type: "bool", Label: "B", Enum: nil, Default: json.RawMessage(`true`), ConditionalOn: &Gate{Key: "k", Value: json.RawMessage(`"v"`)}}
	attrs := FieldAttrsChanged(a, c)
	if strings.Join(attrs, ",") != "label,type,enum,default,conditional_on" {
		t.Fatalf("attrs = %v", attrs)
	}
	// nil vs empty never counts.
	d := &Field{Type: "text", Label: "A", Enum: []any{"x"}, Companions: map[string]string{}, Writes: []string{}}
	if attrs := FieldAttrsChanged(a, d); len(attrs) != 0 {
		t.Fatalf("nil vs empty: %v", attrs)
	}
}

// TestCanonicalHashRecipeMatchesHashBytes pins the README §1.1 recipe that
// the launcher server copies: on the snapshot it must produce the test
// vector 727d151b5d6b04cd68b73199ab0b3bdf061263f9623f4f6eda00a79137c06f38
// and agree with HashBytes on every edge the custom encoder handles.
func TestCanonicalHashRecipeMatchesHashBytes(t *testing.T) {
	_, data := snapshot(t)
	h, err := CanonicalHash(data)
	if err != nil {
		t.Fatal(err)
	}
	if h != snapshotHash {
		t.Fatalf("CanonicalHash(snapshot) = %s, want %s", h, snapshotHash)
	}
	cases := map[string]string{
		"html+separators": `{"version":1,"generated_from":{"at":"x"},"coverage":{"a":1},"s":"<b>&amp;     › \"q\" \\ \t \u0001 \b \f","bad":"` + "\xff" + `"}`,
		"numbers":         `{"version":1,"n":[0,1,-1,1.5,1e21,1e-7,123456789012345,0.1,1e20,-0.0]}`,
		"key order":       `{"z":1,"10":2,"2":3,"a":{"b":null,"A":true,"_":false}}`,
		"empty":           `{}`,
	}
	for name, doc := range cases {
		want, err := HashBytes([]byte(doc))
		if err != nil {
			t.Fatalf("%s: HashBytes: %v", name, err)
		}
		got, err := CanonicalHash([]byte(doc))
		if err != nil {
			t.Fatalf("%s: CanonicalHash: %v", name, err)
		}
		if got != want {
			t.Errorf("%s: CanonicalHash = %s, HashBytes = %s", name, got, want)
		}
	}
	if _, err := CanonicalHash([]byte(`[1]`)); err == nil {
		t.Error("non-object document must fail")
	}
}
