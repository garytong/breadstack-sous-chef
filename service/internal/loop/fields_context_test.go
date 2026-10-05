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

// anthropicBodySHA is the SHA-256 of the request an anthropic route sent for
// the fixture turn at 3b15f6b, the commit before the candidate-fields
// context existed (computed by running this test in a worktree of it). The
// option must not move a byte of it: tools, system[0], system[1], the user
// turn and the request fields are all inside.
const anthropicBodySHA = "1e9379703358fa202cbd3972f004e64a26b8446f0a222a79e7d928045cb1a49e"

func TestFieldsContext_AnthropicRequestIsByteIdentical(t *testing.T) {
	h := newHarness(t, nil, modeltest.Reply(modeltest.Message{Content: []modeltest.Block{modeltest.Text("ok")}}))
	h.start(fieldsRequest(h, "", fieldsHeroText))
	reqs := h.fake.Requests()
	if len(reqs) != 1 {
		t.Fatalf("%d requests", len(reqs))
	}
	sum := sha256.Sum256(reqs[0].Raw)
	if got := hex.EncodeToString(sum[:]); got != anthropicBodySHA {
		t.Fatalf("anthropic request body changed: sha256 %s, want %s (%d bytes)", got, anthropicBodySHA, len(reqs[0].Raw))
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

func TestFieldsContext_AnthropicUserTurnHasNoBlock(t *testing.T) {
	h := newHarnessRoutes(t, studentRoute, nil, okReplies(1)...)
	h.start(fieldsRequest(h, "default", fieldsHeroText))
	got := userTexts(t, h.fake.Requests()[0])
	if len(got) != 1 || got[0] != plainUserTurn(t, fieldsHeroText) {
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
	for route, on := range map[string]bool{"student-off": false, "student-on": true, "claude-on": true, "claude-off": false, "opus": false} {
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

// Every new user text gets its own block; the turns already stored are
// replayed as they were sent, whichever route the later turn runs on.
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

	two := userTexts(t, reqs[1])
	if len(two) != 2 || two[0] != first || two[1] != wantSecond {
		t.Fatalf("second turn's user texts = %q", two)
	}
	if strings.Contains(two[1], "bodyColor") || strings.Count(two[1], "<candidate_fields>") != 1 {
		t.Error("second turn's block is not its own")
	}
	// Third turn on an anthropic route: history verbatim (blocks included),
	// the new text without one.
	three := userTexts(t, reqs[2])
	if len(three) != 3 || three[0] != first || three[1] != wantSecond || three[2] != plainUserTurn(t, third) {
		t.Fatalf("third turn's user texts = %q", three)
	}
	// Storage holds exactly what was sent, in order.
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
