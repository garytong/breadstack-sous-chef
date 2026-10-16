package loop

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"strings"
	"testing"

	"github.com/breadstack/sous-chef/internal/api"
	"github.com/breadstack/sous-chef/internal/config"
	"github.com/breadstack/sous-chef/internal/model/modeltest"
	"github.com/breadstack/sous-chef/internal/prompt"
)

// fieldsHomeContext is a home page the way the editor reports it (the
// fixture of internal/fields' golden blocks).
const fieldsHomeContext = `{"page":"home","template":"editorial","sections":[
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

const fieldsHeroText = "Can you set Body Text to '#c22976' for the hero banner?"

func fieldsRequest(h *harness, route, text string) api.TurnRequest {
	req := h.request(text)
	req.Context = json.RawMessage(fieldsHomeContext)
	req.Model = route
	return req
}

// anthropicBodySHA is the SHA-256 of the request an anthropic route sends
// for the fixture turn: tools, system[0], system[1], the user turn and the
// request fields are all inside. History: first computed at 3b15f6b, the
// commit before the candidate-fields context existed (1e937970…); moved on
// purpose when system[1] gained the "Also called" lines
// (prompt.writeAlsoCalled) — the body with the previous index still hashes
// to 1e937970…, so nothing else moved; moved again when the seo page's
// aliases lost the bare "google" (its "Also called" line now reads "meta
// description, search results, google search") — with the previous
// vocab.txt the body still hashes to 6d2dee6f…. Editing a container alias in
// internal/fields/vocab.txt moves system[1] and so this hash
// (docs/VOCABULARY.md); a term does not. Moved again on purpose 2026-10-10
// by system[0] (prompt.Rules: context, the cannot-change list with `where`,
// where things live, FAQ, W_HIDDEN) and the tool list (report_unsupported's
// optional `where` enum and new text, the target.kind description of
// branding): with the previous rules.go, opspec.go and toolgen.go the body
// still hashes to c1c6fe00…; the new rules alone give b9cb1d7b…, the new
// tools alone 1548c986…; system[1] and the user turn did not move. Moved
// once more on purpose 2026-10-10 (plan Q7) because fields_context is now on
// by default for anthropic routes: the user turn gains the candidate-fields
// block (the golden hero-body-text block, the same bytes the student gets);
// tools, system and every request field are unchanged — the same route
// with fields_context "off" still hashes to anthropicBodySHANoBlock.
// Moved again on purpose 2026-10-10 (the verify pass): system[0] names the
// SEO keys, the undo / list-update / FAQ / W_HIDDEN wording; the tools'
// lookup_fields panel note and report_unsupported's "use support"; system[1]'s
// "Also called" gallery line gains "instagram feed" (vocab.txt). With the
// previous rules.go, opspec.go, toolgen.go and vocab.txt the bodies still hash
// to 068c1bdf… / 2a501c2a…; the new vocab.txt alone gives 224101af… /
// 9395c03b…; the user turn's block did not move.
// Moved again on purpose 2026-10-15 (the user's font / vague-request
// decisions): system[0]'s "The font" line names every site font, and a
// "too vague to act on → ask one short question" line follows the
// two-controls rule. Without those two lines the bodies still hash to
// 32b026b7… / 480416c4…; tools, system[1] and the user turn did not move.
// With them alone (the r6 commit): d1605274… / f081b59b….
// Moved again on purpose 2026-10-15 (create_page + dropdown options): the
// tool list gains create_page after set_page_layout, and system[0] gains
// "Creating pages" and "Listing choices" (an empty list = none set up, a
// missing one = not visible), drops "create" from the pages where, reads
// picking a carousel brand as supported and names E_SLUG among the error
// codes. On top of the font / vague-request lines (d1605274… /
// f081b59b…); system[1] and the user turn did not move.
const (
	anthropicBodySHA        = "06b8d4be5590805e90b0e4661a689eb63f404300259203dbabdbd2e16ebf58a0"
	anthropicBodySHANoBlock = "a212b8be08b178ac4d4896ddb2082b5b049576e5f2b6b908be7953be704cd3c4"
)

func TestFieldsContext_AnthropicRequestIsByteIdentical(t *testing.T) {
	for _, tc := range []struct {
		name   string
		routes map[string]config.Route
		want   string
	}{
		{"default (fields_context unset = on)", nil, anthropicBodySHA},
		{"fields_context off", map[string]config.Route{"default": {Kind: config.RouteAnthropic, Auth: config.AuthAPIKey, Model: "claude-sonnet-5-5",
			MaxTokens: 1024, Effort: "medium", FieldsContext: config.FieldsContextOff}}, anthropicBodySHANoBlock},
	} {
		h := newHarnessRoutes(t, tc.routes, nil, modeltest.Reply(modeltest.Message{Content: []modeltest.Block{modeltest.Text("ok")}}))
		h.start(fieldsRequest(h, "", fieldsHeroText))
		reqs := h.fake.Requests()
		if len(reqs) != 1 {
			t.Fatalf("%s: %d requests", tc.name, len(reqs))
		}
		sum := sha256.Sum256(reqs[0].Raw)
		if got := hex.EncodeToString(sum[:]); got != tc.want {
			t.Errorf("%s: anthropic request body changed: sha256 %s, want %s (%d bytes)", tc.name, got, tc.want, len(reqs[0].Raw))
		}
	}
}

var studentRoute = map[string]config.Route{
	"student": {Kind: config.RouteMessagesCompatible, Auth: config.AuthNone, Model: "sous-chef-student", MaxTokens: 512},
}

func okReplies(n int) []modeltest.Step {
	var steps []modeltest.Step
	for i := 0; i < n; i++ {
		steps = append(steps, modeltest.Reply(modeltest.Message{Content: []modeltest.Block{modeltest.Text("ok")}}))
	}
	return steps
}

// userTexts is the text of every user message of a captured request that is
// a plain text turn (tool_result turns are skipped), in order.
func userTexts(t *testing.T, c modeltest.Captured) []string {
	t.Helper()
	var body struct {
		Messages []struct {
			Role    string          `json:"role"`
			Content json.RawMessage `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(c.Raw, &body); err != nil {
		t.Fatal(err)
	}
	var out []string
	for _, m := range body.Messages {
		if m.Role != "user" {
			continue
		}
		var s string
		if json.Unmarshal(m.Content, &s) == nil {
			out = append(out, s)
			continue
		}
		var blocks []struct{ Type, Text string }
		if err := json.Unmarshal(m.Content, &blocks); err != nil {
			t.Fatal(err)
		}
		if len(blocks) == 1 && blocks[0].Type == "text" {
			out = append(out, blocks[0].Text)
		}
	}
	return out
}

func compactJSON(t *testing.T, raw string) string {
	t.Helper()
	var b bytes.Buffer
	if err := json.Compact(&b, []byte(raw)); err != nil {
		t.Fatal(err)
	}
	return b.String()
}

// heroUserTurn is the exact user turn of the fixture case on a route with
// the option on: the context, internal/fields' golden block, the text.
func heroUserTurn(t *testing.T) string {
	t.Helper()
	golden, err := os.ReadFile("../fields/testdata/hero-body-text.golden") // the block + "\n"
	if err != nil {
		t.Fatal(err)
	}
	return "<context>\n" + compactJSON(t, fieldsHomeContext) + "\n</context>\n" + string(golden) + fieldsHeroText
}

func plainUserTurn(t *testing.T, text string) string {
	t.Helper()
	return "<context>\n" + compactJSON(t, fieldsHomeContext) + "\n</context>\n" + text
}

func TestFieldsContext_CompatRouteCarriesTheBlock(t *testing.T) {
	h := newHarnessRoutes(t, studentRoute, nil, okReplies(1)...)
	turn, _ := h.start(fieldsRequest(h, "student", "  "+fieldsHeroText+"\n"))
	reqs := h.fake.Requests()
	if len(reqs) != 1 {
		t.Fatalf("%d requests", len(reqs))
	}
	want := heroUserTurn(t)
	got := userTexts(t, reqs[0])
	if len(got) != 1 || got[0] != want {
		t.Fatalf("user turn:\n%s\n--- want\n%s", strings.Join(got, "\n=====\n"), want)
	}
	if !strings.Contains(want, "\n</context>\n<candidate_fields>\n") || !strings.Contains(want, "\n  bodyColor — Body Text (color)\n") ||
		!strings.HasSuffix(want, "\n</candidate_fields>\n"+fieldsHeroText) {
		t.Fatalf("fixture is not context → block → text:\n%s", want)
	}
	// The rule travels in the block's header; system[0] is the same frozen
	// text every route gets.
	if !strings.Contains(want, "Copy the target and the keys exactly as written; for any other key call lookup_fields first.") {
		t.Error("block header lost its rule")
	}
	if !strings.HasPrefix(reqs[0].SystemText(), prompt.Rules) || strings.Contains(reqs[0].SystemText(), "candidate_fields") {
		t.Error("system prompt differs on the fields_context route")
	}
	// Stored as sent.
	msgs, _ := h.st.ListMessages(context.Background(), h.st.turn(turn.ID()).ConversationID, 0, 0)
	var blocks []struct{ Text string }
	if err := json.Unmarshal(msgs[0].Content, &blocks); err != nil || len(blocks) != 1 || blocks[0].Text != want {
		t.Errorf("stored user message = %s (%v)", msgs[0].Content, err)
	}
	// The trace keeps the merchant's own text.
	if tr, ok := h.st.trace(turn.ID()); !ok || strings.Contains(string(tr.Ops), "candidate_fields") {
		t.Errorf("trace ops carry the block: %v", ok)
	}
}

// Since 2026-10-10 (plan Q7) an anthropic route that does not say
// otherwise carries the block too: the same bytes the student gets.
func TestFieldsContext_AnthropicUserTurnCarriesTheBlockByDefault(t *testing.T) {
	h := newHarnessRoutes(t, studentRoute, nil, okReplies(1)...)
	h.start(fieldsRequest(h, "default", fieldsHeroText))
	got := userTexts(t, h.fake.Requests()[0])
	if len(got) != 1 || got[0] != heroUserTurn(t) {
		t.Fatalf("user turn = %q", got)
	}
}

func TestFieldsContext_Options(t *testing.T) {
	routes := map[string]config.Route{
		"student-off": {Kind: config.RouteMessagesCompatible, Auth: config.AuthNone, Model: "m", MaxTokens: 512, FieldsContext: config.FieldsContextOff},
		"student-on":  {Kind: config.RouteMessagesCompatible, Auth: config.AuthNone, Model: "m", MaxTokens: 512, FieldsContext: config.FieldsContextOn},
		"claude-on":   {Kind: config.RouteAnthropic, Auth: config.AuthAPIKey, Model: "claude-sonnet-5-5", MaxTokens: 1024, FieldsContext: config.FieldsContextOn},
		"claude-off":  {Kind: config.RouteAnthropic, Auth: config.AuthAPIKey, Model: "claude-sonnet-5-5", MaxTokens: 1024, FieldsContext: config.FieldsContextOff},
	}
	for route, on := range map[string]bool{"student-off": false, "student-on": true, "claude-on": true, "claude-off": false, "opus": true} {
		h := newHarnessRoutes(t, routes, nil, okReplies(1)...)
		h.start(fieldsRequest(h, route, fieldsHeroText))
		want := plainUserTurn(t, fieldsHeroText)
		if on {
			want = heroUserTurn(t)
		}
		if got := userTexts(t, h.fake.Requests()[0]); len(got) != 1 || got[0] != want {
			t.Errorf("%s: user turn = %q", route, got)
		}
	}
}

// Every new user text gets its own block. Storage keeps each user message
// as it was built (context + block + text); what a later turn REPLAYS depends
// on its route: a messages-compatible route sends the earlier turns' request
// text only (prompt.StableHistory), an anthropic route the stored history
// byte for byte (a history edit would break Claude's preserved thinking and
// its cache prefix — turn.sendable).
func TestFieldsContext_ConversationContinuation(t *testing.T) {
	h := newHarnessRoutes(t, studentRoute, nil, okReplies(3)...)
	const second = "Please hide Shop from the header nav (the header)."
	const third = "Branding: hide the Location Name"
	h.start(fieldsRequest(h, "student", fieldsHeroText))
	h.start(fieldsRequest(h, "student", second))
	turn3, _ := h.start(fieldsRequest(h, "default", third))
	reqs := h.fake.Requests()
	if len(reqs) != 3 {
		t.Fatalf("%d requests", len(reqs))
	}
	first := heroUserTurn(t)
	golden, err := os.ReadFile("../fields/testdata/header-nav-toggle.golden")
	if err != nil {
		t.Fatal(err)
	}
	wantSecond := "<context>\n" + compactJSON(t, fieldsHomeContext) + "\n</context>\n" + string(golden) + second

	// Second turn on the student: the first turn replays as its text alone.
	two := userTexts(t, reqs[1])
	if len(two) != 2 || two[0] != fieldsHeroText || two[1] != wantSecond {
		t.Fatalf("second turn's user texts = %q", two)
	}
	if strings.Contains(two[1], "bodyColor") || strings.Count(two[1], "<candidate_fields>") != 1 {
		t.Error("second turn's block is not its own")
	}
	// Third turn on an anthropic route: history verbatim (blocks included),
	// and — fields_context on by default — the new text with its own block.
	three := userTexts(t, reqs[2])
	wantThird := "<context>\n" + compactJSON(t, fieldsHomeContext) + "\n</context>\n<candidate_fields>\n"
	if len(three) != 3 || three[0] != first || three[1] != wantSecond || !strings.HasPrefix(three[2], wantThird) ||
		!strings.HasSuffix(three[2], "\n</candidate_fields>\n"+third) {
		t.Fatalf("third turn's user texts = %q", three)
	}
	// Storage holds every user message as it was built, in order.
	conv := h.st.turn(turn3.ID()).ConversationID
	msgs, _ := h.st.ListMessages(context.Background(), conv, 0, 0)
	if len(msgs) != 6 {
		t.Fatalf("%d stored messages", len(msgs))
	}
	for i, want := range three {
		var blocks []struct{ Text string }
		if err := json.Unmarshal(msgs[2*i].Content, &blocks); err != nil || len(blocks) != 1 || blocks[0].Text != want {
			t.Errorf("stored message %d = %s", 2*i, msgs[2*i].Content)
		}
	}
}

// Nothing matched, or a context the retrieval cannot read: the turn runs
// with the plain user turn.
func TestFieldsContext_NoBlock(t *testing.T) {
	h := newHarnessRoutes(t, studentRoute, nil, okReplies(3)...)
	h.start(fieldsRequest(h, "student", "zzz qqq"))
	if got := userTexts(t, h.fake.Requests()[0]); len(got) != 1 || got[0] != plainUserTurn(t, "zzz qqq") {
		t.Errorf("no match: user turn = %q", got)
	}
	req := fieldsRequest(h, "student", fieldsHeroText)
	req.Context = json.RawMessage(`{"page":"home","sections":"all of them"}`)
	_, evs := h.start(req)
	if last(evs).Event != "turn.done" {
		t.Fatalf("unreadable context: events = %s", kinds(evs))
	}
	if got := userTexts(t, h.fake.Requests()[1]); len(got) != 2 || got[1] != "<context>\n"+string(req.Context)+"\n</context>\n"+fieldsHeroText {
		t.Errorf("unreadable context: user turn = %q", got)
	}
	// No context at all: the block opens the message.
	req = fieldsRequest(h, "student", "set answerUppercase on the faq page")
	req.Context = nil
	h.start(req)
	got := userTexts(t, h.fake.Requests()[2])
	if len(got) != 3 || !strings.HasPrefix(got[2], "<candidate_fields>\n") || !strings.HasSuffix(got[2], "\n</candidate_fields>\nset answerUppercase on the faq page") {
		t.Errorf("no context: user turn = %q", got)
	}
}
