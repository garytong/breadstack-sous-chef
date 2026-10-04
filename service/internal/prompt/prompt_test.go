package prompt

import (
	"encoding/json"
	"os"
	"strings"
	"sync"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/packages/param"

	"github.com/breadstack/sous-chef/internal/model"
)

var (
	snapshotOnce sync.Once
	snapshot     []byte
)

func loadSnapshot(t *testing.T) []byte {
	t.Helper()
	snapshotOnce.Do(func() {
		b, err := os.ReadFile("../../testdata/manifest.snapshot.json")
		if err != nil {
			t.Fatalf("snapshot: %v", err)
		}
		snapshot = b
	})
	return snapshot
}

// --- rules -------------------------------------------------------------------

func TestRules_NameEveryToolAndNoSalvagedLeftovers(t *testing.T) {
	for _, tool := range []string{"set_fields", "edit_list", "reset", "restore_default", "set_page_layout", "reset_pinned_section", "pick_asset", "lookup_fields", "describe_target", "report_unsupported"} {
		if !strings.Contains(Rules, tool) {
			t.Errorf("rules never mention %s", tool)
		}
	}
	for _, gone := range []string{"record_knowledge", "request_asset_selection", "set_global_setting", "set_section_setting", "manage_section", "set_hero_image", "%s"} {
		if strings.Contains(Rules, gone) {
			t.Errorf("rules still carry the old branch's %q", gone)
		}
	}
}

func TestRules_SalvagedBehaviours(t *testing.T) {
	for _, want := range []string{
		"cannot write code, run commands",      // preamble: no fabricated capabilities
		"Stay strictly on the topic",           // stay on topic
		"smallest set of tool calls",           // minimal calls + confirmation
		"never claim a change is permanent",    // live preview, not published
		"wait for the chosen URL",              // pick_asset then wait
		"PAGE CONTEXT",                         // viewed page rule
		"Never fabricate a mailto",             // report_unsupported, no email
		"Never invent setting keys",            // no invented keys
		"one STRUCTURAL action",                // engine: one structural edit_list per call
		"mirror slides[0]",                     // hero mirror
		"alt and srcset go in assets",          // pick_asset → values + assets
		"locationOverrides.<locationId>",       // keyed fields
		"read-only",                            // derived_from
		"lookup_fields before writing any key", // lookup first
		"ask in one sentence",                  // ambiguity
		"retry at most once",                   // failed op
		"E_FIELD",                              // error codes named
	} {
		if !strings.Contains(Rules, want) {
			t.Errorf("rules lack %q", want)
		}
	}
	if strings.Contains(Rules, "{{") || strings.Contains(Rules, "\t") {
		t.Errorf("rules must be literal text")
	}
}

// --- index -------------------------------------------------------------------

func TestRenderIndex_SnapshotCountsAndBudget(t *testing.T) {
	idx, err := RenderIndex(loadSnapshot(t))
	if err != nil {
		t.Fatal(err)
	}
	if idx.Sections != 19 || idx.GlobalPanels != 12 || idx.Pages != 17 || idx.Details != 11 {
		t.Errorf("counts = s%d g%d p%d d%d", idx.Sections, idx.GlobalPanels, idx.Pages, idx.Details)
	}
	if idx.Fields != 2094 || idx.Lists != 69 {
		t.Errorf("fields=%d lists=%d, want 2094/69", idx.Fields, idx.Lists)
	}
	// README §3: index ≤ 5k tokens at 3 bytes/token
	if n := len(idx.Text); n > 15000 || n < 4000 {
		t.Errorf("index is %d bytes; want 4000 < n ≤ 15000", n)
	}
	t.Logf("index bytes=%d (≈%d tokens at 3 B/token)", len(idx.Text), len(idx.Text)/3)
}

func TestRenderIndex_SnapshotContent(t *testing.T) {
	idx, err := RenderIndex(loadSnapshot(t))
	if err != nil {
		t.Fatal(err)
	}
	text := idx.Text
	for _, want := range []string{
		"s:heroBanner pages=home f=2 lists=slides[add|remove]",
		"slides[].bullets[add|remove|reorder]->HeroBulletDetailsPanel",
		"mirror=slides[0](56 keys)",
		"s:hero = heroBanner",
		"s:reviews = testimonials",
		"s:blog = blogSection",
		"s:productCarousel pages=home,product-details",
		"| ro: brandId tagId",
		"g:HeaderSection f=69 lists=",
		"pageVisibility._meta.headerNavOrder[reorder;orders=pages]",
		"g:ProductCardSection f=103",
		"presets=centered,compact,minimal,standard",
		"b:branding f=137 lists=certifications[add|remove|reorder]->CertificationDetailsPanel",
		"p:_common f=1 | keys: published",
		"p:faq f=16 lists=faqCategories[add|remove|reorder]:name faqCategories[].items[add|remove]:a q",
		"d:HeroBulletDetailsPanel f=3 | keys: description icon title",
		"locationOverrides.{locationId}.*(7)",
		"products → __pinned_product_list*",
		"faq → __pinned_faq\n",
		"## Pinned",
	} {
		if !strings.Contains(text, want) {
			t.Errorf("index lacks %q", want)
		}
	}
	// aliases have no line of their own
	for _, alias := range []string{"s:hero pages=", "s:reviews pages=", "s:blog pages="} {
		if strings.Contains(text, alias) {
			t.Errorf("alias rendered as a full container: %q", alias)
		}
	}
	if idx.Aliases["hero"] != "heroBanner" || idx.Aliases["reviews"] != "testimonials" || idx.Aliases["blog"] != "blogSection" {
		t.Errorf("aliases = %v", idx.Aliases)
	}
	// the full key list stays out: a 75-key family prints as one stem
	if !strings.Contains(text, "card*(75)") {
		t.Errorf("ProductCardSection keys should collapse to card*(75)")
	}
	if strings.Count(text, "\n") < 19+12+17+11+16 {
		t.Errorf("too few lines: %d", strings.Count(text, "\n"))
	}
}

func TestRenderIndex_DeterministicUnderKeyOrder(t *testing.T) {
	raw := loadSnapshot(t)
	a, err := RenderIndex(raw)
	if err != nil {
		t.Fatal(err)
	}
	// re-serialise through a map: Go sorts object keys, so the byte layout changes
	var any map[string]any
	if err := json.Unmarshal(raw, &any); err != nil {
		t.Fatal(err)
	}
	reordered, _ := json.Marshal(any)
	if string(reordered) == string(raw) {
		t.Fatal("test needs a differently ordered manifest")
	}
	b, err := RenderIndex(reordered)
	if err != nil {
		t.Fatal(err)
	}
	if a.Text != b.Text {
		t.Errorf("index differs under key order")
	}
	c, _ := RenderIndex(raw)
	if a.Text != c.Text {
		t.Errorf("index differs across calls")
	}
}

const smallManifest = `{
 "version": 1,
 "generated_from": {"branch":"x","sha":"abc","at":"2026-10-01T00:00:00Z"},
 "sections": {
   "heroBanner": {"editorKey":"hero","pages":["home"],
     "fields": {"paddingY":{"path":"settings.paddingY","type":"range"}, "maxWidth":{"path":"settings.maxWidth","type":"range"}, "tagId":{"path":"settings.tagId","type":"text","derived_from":"tagSlug"}},
     "lists": {"slides":{"path":"settings.slides","ops":["remove","add"],"item":{"headline":{"type":"richtext"},"headlineColor":{"type":"color"},"layout":{"type":"enum"}}},
               "slides[].bullets":{"path":"settings.slides[].bullets","ops":["add"],"item_ref":"details.HeroBulletDetailsPanel","item":{}}},
     "mirror": {"from":"slides[0]","keys":["headline","headlineColor"]}},
   "hero": {"editorKey":"hero","pages":["home"],
     "fields": {"paddingY":{"path":"settings.paddingY","type":"range"}, "maxWidth":{"path":"settings.maxWidth","type":"range"}, "tagId":{"path":"settings.tagId","type":"text","derived_from":"tagSlug"}},
     "lists": {"slides":{"path":"settings.slides","ops":["remove","add"],"item":{"headline":{"type":"richtext"},"headlineColor":{"type":"color"},"layout":{"type":"enum"}}},
               "slides[].bullets":{"path":"settings.slides[].bullets","ops":["add"],"item_ref":"details.HeroBulletDetailsPanel","item":{}}},
     "mirror": {"from":"slides[0]","keys":["headline","headlineColor"]}},
   "gallery": {"editorKey":"gallery","pages":["home","products"],"fields":{"imageFit":{"type":"enum"}},"lists":{}}
 },
 "globalPanels": {"HeaderSection": {"fields": {"headerBg":{"type":"color"},"headerHeight":{"type":"number"}}, "lists": {"pageVisibility._meta.headerNavOrder":{"ops":["reorder"],"orders":"pages","item":{"value":{"type":"text"}}}}},
                  "ProductCardSection": {"fields": {"cardLayout":{"type":"enum"}}, "presets":[{"name":"standard","label":"Standard"},{"name":"compact","label":"Compact"}]}},
 "branding": {"fields": {"logo":{"type":"image"}}, "lists": {"certifications":{"ops":["add","remove"],"item_ref":"details.CertificationDetailsPanel","item":{}}}},
 "pages": {"_common": {"fields": {"published":{"type":"bool"}}}, "faq": {"fields": {"layout":{"type":"enum"}}, "lists": {"faqCategories":{"ops":["add","remove","reorder"],"item":{"name":{"type":"text"}}}}}},
 "details": {"HeroBulletDetailsPanel": {"fields": {"item.title":{"type":"text"},"item.icon":{"type":"text"}}, "lists": {}},
             "CertificationDetailsPanel": {"fields": {"item.name":{"type":"text"}}, "lists": {}}}
}`

func TestRenderIndex_SmallManifestExactLines(t *testing.T) {
	idx, err := RenderIndex([]byte(smallManifest))
	if err != nil {
		t.Fatal(err)
	}
	if idx.Sections != 3 || idx.GlobalPanels != 2 || idx.Pages != 2 || idx.Details != 2 {
		t.Errorf("counts = %+v", idx)
	}
	// fields: hero 3 + heroBanner 3 + gallery 1 + header 2 + pcs 1 + branding 1 + pages 2 + details 3 = 16; items: slides 3×2 + navOrder 1 + faq 1 = 8
	if idx.Fields != 24 || idx.Lists != 7 {
		t.Errorf("fields=%d lists=%d", idx.Fields, idx.Lists)
	}
	for _, want := range []string{
		"s:gallery pages=home,products f=1 | keys: imageFit\n",
		"s:heroBanner pages=home f=3 lists=slides[add|remove]:headline*(2) layout slides[].bullets[add]->HeroBulletDetailsPanel mirror=slides[0](2 keys) | keys: maxWidth paddingY tagId | ro: tagId\n",
		"s:hero = heroBanner\n",
		"g:HeaderSection f=2 lists=pageVisibility._meta.headerNavOrder[reorder;orders=pages]:value | keys: header*(2)\n",
		"g:ProductCardSection f=1 presets=compact,standard | keys: cardLayout\n",
		"b:branding f=1 lists=certifications[add|remove]->CertificationDetailsPanel | keys: logo\n",
		"p:_common f=1 | keys: published\n",
		"p:faq f=1 lists=faqCategories[add|remove|reorder]:name | keys: layout\n",
		"d:HeroBulletDetailsPanel f=2 | keys: icon title\n",
	} {
		if !strings.Contains(idx.Text, want) {
			t.Errorf("index lacks line %q\n--- got ---\n%s", want, idx.Text)
		}
	}
}

func TestRenderIndex_Rejects(t *testing.T) {
	for name, raw := range map[string]string{
		"not json":     `{`,
		"no sections":  `{"version":1,"pages":{"home":{}}}`,
		"no pages":     `{"version":1,"sections":{"hero":{}}}`,
		"wrong schema": `{"version":2,"sections":{"hero":{}},"pages":{"home":{}}}`,
	} {
		if _, err := RenderIndex([]byte(raw)); err == nil {
			t.Errorf("%s: expected an error", name)
		}
	}
}

func TestStems(t *testing.T) {
	cases := map[string]string{
		"headline":                            "headline",
		"headlineColor":                       "headline",
		"bgColor":                             "bg",
		"locationOverrides.{locationId}.mode": "locationOverrides.{locationId}.",
		"pageVisibility.products.headerNav":   "page",
		"images[0]":                           "images",
		"card_ribbon":                         "card",
		"Title":                               "Title",
	}
	for key, want := range cases {
		if got := stemOf(key); got != want {
			t.Errorf("stemOf(%q) = %q, want %q", key, got, want)
		}
	}
	got := stems([]string{"headline", "headlineColor", "headlineFont", "bgColor", "bgImage", "layout", "a1", "a2"}, 2)
	if got != "headline*(3) a*(2) +2" {
		t.Errorf("stems = %q", got)
	}
	if got := stems([]string{"one", "two"}, 10); got != "one two" {
		t.Errorf("single keys = %q", got)
	}
	if got := stems(nil, 10); got != "" {
		t.Errorf("empty = %q", got)
	}
}

func TestAliasSections_LongestNameIsCanonical(t *testing.T) {
	same := &container{Fields: map[string]field{"a": {Type: "text"}}}
	canon := aliasSections(map[string]*container{"reviews": same, "testimonials": same, "blog": {Fields: map[string]field{"b": {}}}})
	if canon["reviews"] != "testimonials" || canon["testimonials"] != "testimonials" || canon["blog"] != "blog" {
		t.Errorf("canon = %v", canon)
	}
}

// --- context -----------------------------------------------------------------

func TestContextBlock(t *testing.T) {
	raw := json.RawMessage(`{
	  "page": "home",
	  "sections": [ {"id": "hero-1", "type": "hero", "title": "Hero", "enabled": true} ]
	}`)
	got, err := ContextBlock(raw, "  make the hero title bigger \n")
	if err != nil {
		t.Fatal(err)
	}
	want := "<context>\n{\"page\":\"home\",\"sections\":[{\"id\":\"hero-1\",\"type\":\"hero\",\"title\":\"Hero\",\"enabled\":true}]}\n</context>\nmake the hero title bigger"
	if got != want {
		t.Errorf("block =\n%s\nwant\n%s", got, want)
	}
	if got, _ := ContextBlock(nil, "hi"); got != "hi" {
		t.Errorf("no context → text only, got %q", got)
	}
	if got, _ := ContextBlock(json.RawMessage("null"), "hi"); got != "hi" {
		t.Errorf("null context → text only, got %q", got)
	}
	for name, c := range map[string]struct {
		raw  string
		text string
	}{
		"empty text":   {`{}`, "   "},
		"array ctx":    {`[1]`, "x"},
		"invalid json": {`{"page":`, "x"},
		"too large":    {`{"a":"` + strings.Repeat("x", MaxContextBytes) + `"}`, "x"},
	} {
		if _, err := ContextBlock(json.RawMessage(c.raw), c.text); err == nil {
			t.Errorf("%s: expected an error", name)
		}
	}
}

func TestParseContext(t *testing.T) {
	tc, err := ParseContext(json.RawMessage(`{"page":"products","template":"cozy","layout":"filter-left",
	  "sections":[{"id":"productCarousel-2","type":"productCarousel","title":"Best sellers","enabled":false}],
	  "categories":[{"slug":"cookies","name":"Cookies"}],"customPages":[{"slug":"about","title":"About"}],
	  "nav":[{"label":"Home","href":"/"}],"locations":[{"id":"loc-1","name":"Main St"}],"extra":1}`))
	if err != nil {
		t.Fatal(err)
	}
	if tc.Page != "products" || tc.Template != "cozy" || tc.Layout != "filter-left" {
		t.Errorf("tc = %+v", tc)
	}
	if len(tc.Sections) != 1 || tc.Sections[0].ID != "productCarousel-2" || tc.Sections[0].Enabled == nil || *tc.Sections[0].Enabled {
		t.Errorf("sections = %+v", tc.Sections)
	}
	if tc.Categories[0].Slug != "cookies" || tc.CustomPages[0].Title != "About" || tc.Nav[0].Href != "/" || tc.Locations[0].ID != "loc-1" {
		t.Errorf("refs = %+v", tc)
	}
	if empty, err := ParseContext(nil); err != nil || empty.Page != "" {
		t.Errorf("empty context: %+v %v", empty, err)
	}
	if _, err := ParseContext(json.RawMessage(`"str"`)); err == nil {
		t.Errorf("a string is not a context")
	}
}

// --- build -------------------------------------------------------------------

func sampleTools() []anthropic.BetaToolUnionParam {
	mk := func(name string) anthropic.BetaToolUnionParam {
		return anthropic.BetaToolUnionParam{OfTool: ptr(model.Strict(anthropic.BetaToolParam{
			Name: name, Description: param.NewOpt(name + " op"),
			InputSchema: anthropic.BetaToolInputSchemaParam{Properties: map[string]any{"target": map[string]any{"type": "object"}}, Required: []string{"target"}},
		}))}
	}
	return []anthropic.BetaToolUnionParam{mk("set_fields"), mk("edit_list"), mk("pick_asset")}
}

func ptr[T any](v T) *T { return &v }

func TestBuild_ShapeAndCacheBreakpoints(t *testing.T) {
	idx, err := RenderIndex([]byte(smallManifest))
	if err != nil {
		t.Fatal(err)
	}
	p := NewPrefix(idx, sampleTools())
	history := []anthropic.BetaMessageParam{
		model.UserTextTurn("earlier question"),
		assistantTurn(anthropic.NewBetaTextBlock("earlier answer")),
	}
	req, err := Build(p, history, json.RawMessage(`{"page":"home"}`), "make it pop")
	if err != nil {
		t.Fatal(err)
	}
	if len(req.System) != 2 || req.System[0].Text != Rules || req.System[1].Text != idx.Text {
		t.Fatalf("system = %d blocks", len(req.System))
	}
	for i, s := range req.System {
		if s.CacheControl.Type != "ephemeral" {
			t.Errorf("system[%d] has no cache breakpoint", i)
		}
	}
	if len(req.Tools) != 3 || req.Tools[0].OfTool.Name != "set_fields" || req.Tools[2].OfTool.Name != "pick_asset" {
		t.Errorf("tools reordered or lost")
	}
	if len(req.Messages) != 3 {
		t.Fatalf("messages = %d", len(req.Messages))
	}
	js, _ := json.Marshal(req.Messages[2])
	var last struct {
		Role    string `json:"role"`
		Content []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"content"`
	}
	_ = json.Unmarshal(js, &last)
	if last.Role != "user" || len(last.Content) != 1 || last.Content[0].Type != "text" || !strings.HasPrefix(last.Content[0].Text, "<context>\n{\"page\":\"home\"}\n</context>\nmake it pop") {
		t.Errorf("user turn = %s", js)
	}
	// history is replayed verbatim, first
	h0, _ := json.Marshal(history[0])
	m0, _ := json.Marshal(req.Messages[0])
	if string(h0) != string(m0) {
		t.Errorf("history[0] changed: %s vs %s", h0, m0)
	}
}

func TestBuild_PrefixStableAcrossTurns(t *testing.T) {
	idx, err := RenderIndex(loadSnapshot(t))
	if err != nil {
		t.Fatal(err)
	}
	p := NewPrefix(idx, sampleTools())
	a, err := Build(p, nil, json.RawMessage(`{"page":"home","sections":[{"id":"hero-1"}]}`), "first turn")
	if err != nil {
		t.Fatal(err)
	}
	b, err := Build(p, a.Messages, json.RawMessage(`{"page":"products","categories":[{"slug":"c","name":"C"}]}`), "second turn, other page")
	if err != nil {
		t.Fatal(err)
	}
	pa, _ := json.Marshal(struct {
		T []anthropic.BetaToolUnionParam
		S []anthropic.BetaTextBlockParam
	}{a.Tools, a.System})
	pb, _ := json.Marshal(struct {
		T []anthropic.BetaToolUnionParam
		S []anthropic.BetaTextBlockParam
	}{b.Tools, b.System})
	if string(pa) != string(pb) {
		t.Errorf("prefix bytes differ between turns")
	}
	pbytes, err := p.Bytes()
	if err != nil || len(pbytes) < len(idx.Text) {
		t.Errorf("Prefix.Bytes = %d bytes, %v", len(pbytes), err)
	}
	if !strings.Contains(string(pbytes), `"cache_control":{"type":"ephemeral"}`) {
		t.Errorf("prefix bytes lack the breakpoints")
	}
	// nothing volatile before the user message
	for _, volatile := range []string{"products", "hero-1", "first turn", "2026-"} {
		if strings.Contains(string(pbytes), volatile) && volatile != "products" {
			t.Errorf("prefix carries volatile %q", volatile)
		}
	}
}

func TestBuild_Rejects(t *testing.T) {
	idx, _ := RenderIndex([]byte(smallManifest))
	good := NewPrefix(idx, sampleTools())
	if _, err := Build(good, nil, nil, "   "); err == nil {
		t.Errorf("empty text")
	}
	if _, err := Build(Prefix{Rules: Rules, Index: idx.Text}, nil, nil, "x"); err == nil {
		t.Errorf("no tools")
	}
	if _, err := Build(Prefix{Rules: "", Index: idx.Text, Tools: sampleTools()}, nil, nil, "x"); err == nil {
		t.Errorf("no rules")
	}
	if _, err := Build(Prefix{Rules: Rules, Index: "", Tools: sampleTools()}, nil, nil, "x"); err == nil {
		t.Errorf("no index")
	}
	if _, err := Build(good, nil, json.RawMessage(`[]`), "x"); err == nil {
		t.Errorf("non-object context")
	}
}

func TestContinue_AppendsOneToolResultsMessage(t *testing.T) {
	idx, _ := RenderIndex([]byte(smallManifest))
	p := NewPrefix(idx, sampleTools())
	first, err := Build(p, nil, json.RawMessage(`{"page":"home"}`), "go")
	if err != nil {
		t.Fatal(err)
	}
	history := append(first.Messages, assistantTurn(anthropic.NewBetaToolUseBlock("t1", map[string]any{"target": map[string]any{"kind": "branding"}}, "set_fields")))
	req, err := Continue(p, history, []model.ToolResult{{ID: "t1", Content: `{"ok":true}`}})
	if err != nil {
		t.Fatal(err)
	}
	if len(req.Messages) != 3 {
		t.Fatalf("messages = %d", len(req.Messages))
	}
	js, _ := json.Marshal(req.Messages[2])
	if !strings.Contains(string(js), `"tool_use_id":"t1"`) || !strings.Contains(string(js), `"role":"user"`) {
		t.Errorf("tool results turn = %s", js)
	}
	if len(req.System) != 2 || len(req.Tools) != 3 {
		t.Errorf("prefix not carried")
	}
	if _, err := Continue(p, history, nil); err == nil {
		t.Errorf("no results")
	}
	if _, err := Continue(p, nil, []model.ToolResult{{ID: "t1"}}); err == nil {
		t.Errorf("no history")
	}
}

// assistantTurn is an assistant message of the given blocks (the SDK has no
// NewBetaAssistantMessage helper).
func assistantTurn(blocks ...anthropic.BetaContentBlockParamUnion) anthropic.BetaMessageParam {
	return anthropic.BetaMessageParam{Role: anthropic.BetaMessageParamRoleAssistant, Content: blocks}
}
