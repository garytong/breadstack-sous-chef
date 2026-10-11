package prompt

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"

	"github.com/breadstack/sous-chef/internal/model"
)

func TestPrimeFiller(t *testing.T) {
	f := PrimeFiller(1015)
	if n := len(strings.Fields(f)); n != 1015 || strings.Trim(f, "x ") != "" || strings.Contains(f, "  ") || strings.HasSuffix(f, " ") {
		t.Errorf("PrimeFiller(1015) has %d words: %.20q…", n, f)
	}
	if PrimeFiller(0) != "x" {
		t.Errorf("PrimeFiller(0) = %q", PrimeFiller(0))
	}
}

func messagesJSON(t *testing.T, msgs []anthropic.BetaMessageParam) []string {
	t.Helper()
	out := make([]string, len(msgs))
	for i, m := range msgs {
		b, err := json.Marshal(m)
		if err != nil {
			t.Fatal(err)
		}
		out[i] = string(b)
	}
	return out
}

// The prime is the next turn's request up to its user message: every
// message before it byte-identical to what turn.sendable sends for the real
// next turn on a messages-compatible route (StableHistory, MergeUserRuns),
// and the same system and tools.
func TestPrimeRequest_MatchesTheNextTurnUpToItsUserMessage(t *testing.T) {
	idx, err := RenderIndex([]byte(smallManifest))
	if err != nil {
		t.Fatal(err)
	}
	p := NewPrefix(idx, sampleTools())
	ops := assistantMsg(
		anthropic.NewBetaTextBlock("Updating the title."),
		anthropic.NewBetaToolUseBlock("t1", map[string]any{"target": map[string]any{"kind": "section", "sectionId": "hero-1"}, "values": map[string]any{"headline": "Fresh"}}, "set_fields"),
	)
	results := model.ToolResultsTurn([]model.ToolResult{{ID: "t1", Content: `{"ok":true}`}})
	closed := []anthropic.BetaMessageParam{
		model.UserTextTurn(mustUserText(t, stableCtx, "set the headline to Fresh", stableBlock)),
		ops, results,
		assistantMsg(anthropic.NewBetaTextBlock("Done.")),
	}
	fastFinished := closed[:3] // a turn that ended on its results (no closing reply)

	for name, history := range map[string][]anthropic.BetaMessageParam{"floor": nil, "closing reply": closed, "fast finish": fastFinished} {
		prime, err := PrimeRequest(p, history, 1015)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		next, err := BuildWithFields(p, history, json.RawMessage(`{"page":"products"}`), "make the buttons rounded", stableBlock)
		if err != nil {
			t.Fatal(err)
		}
		real := MergeUserRuns(StableHistory(next.Messages))
		pm, rm := messagesJSON(t, prime.Messages), messagesJSON(t, real)
		if len(pm) != len(rm) {
			t.Fatalf("%s: prime has %d messages, the real request %d", name, len(pm), len(rm))
		}
		for i := 0; i < len(pm)-1; i++ {
			if pm[i] != rm[i] {
				t.Errorf("%s: message %d differs:\nprime %s\nreal  %s", name, i, pm[i], rm[i])
			}
		}
		// The last message: the same blocks before the text, then the filler
		// where the real one has its context and text.
		lp, lr := prime.Messages[len(pm)-1], real[len(rm)-1]
		if len(lp.Content) != len(lr.Content) {
			t.Fatalf("%s: last message has %d blocks, the real one %d", name, len(lp.Content), len(lr.Content))
		}
		for i := range lp.Content {
			if lp.Content[i].OfText != nil {
				if lp.Content[i].OfText.Text != PrimeFiller(1015) || !strings.HasPrefix(lr.Content[i].OfText.Text, "<context>") {
					t.Errorf("%s: text block %d is not the filler", name, i)
				}
				continue
			}
			bp, _ := json.Marshal(lp.Content[i])
			br, _ := json.Marshal(lr.Content[i])
			if string(bp) != string(br) {
				t.Errorf("%s: block %d differs: %s vs %s", name, i, bp, br)
			}
		}
		pb, _ := json.Marshal(struct{ S, T any }{prime.System, prime.Tools})
		nb, _ := json.Marshal(struct{ S, T any }{next.System, next.Tools})
		if string(pb) != string(nb) {
			t.Errorf("%s: system/tools differ", name)
		}
	}
	// The history is not modified.
	if !strings.HasPrefix(closed[0].Content[0].OfText.Text, "<context>") {
		t.Error("PrimeRequest stripped the caller's history in place")
	}
	if _, err := PrimeRequest(Prefix{}, nil, 10); err == nil {
		t.Error("an empty prefix was accepted")
	}
}
