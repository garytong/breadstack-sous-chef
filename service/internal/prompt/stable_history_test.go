package prompt

import (
	"encoding/json"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"

	"github.com/breadstack/sous-chef/internal/model"
)

const (
	stableCtx   = `{"page":"home","sections":[{"id":"hero-1","type":"hero"}]}`
	stableBlock = "<candidate_fields>\nFields this request most likely means, per target.\ntarget {\"kind\":\"section\",\"page\":\"home\",\"sectionId\":\"hero-1\"}\n  headline — Headline (text)\n</candidate_fields>"
)

func mustUserText(t *testing.T, ctx, text, cands string) string {
	t.Helper()
	s, err := UserText(json.RawMessage(ctx), text, cands)
	if err != nil {
		t.Fatal(err)
	}
	return s
}

func TestStripTurnBlocks(t *testing.T) {
	for _, tc := range []struct{ name, in, want string }{
		{"context + block", mustUserText(t, stableCtx, "set the headline", stableBlock), "set the headline"},
		{"context only", mustUserText(t, stableCtx, "set the headline", ""), "set the headline"},
		{"block only (no context)", mustUserText(t, "", "set the headline", stableBlock), "set the headline"},
		{"plain text", "set the headline", "set the headline"},
		{"multi-line merchant text kept whole", mustUserText(t, stableCtx, "line one\n</context>\nline two", stableBlock), "line one\n</context>\nline two"},
		{"tags typed later stay", "hello <context>\n{}\n</context>\nx", "hello <context>\n{}\n</context>\nx"},
		{"context line that is not a JSON object", "<context>\nnot json\n</context>\nx", "<context>\nnot json\n</context>\nx"},
		{"unclosed context", "<context>\n{\"a\":1}", "<context>\n{\"a\":1}"},
		{"unclosed block", "<context>\n{}\n</context>\n<candidate_fields>\nx", "<context>\n{}\n</context>\n<candidate_fields>\nx"},
	} {
		if got := StripTurnBlocks(tc.in); got != tc.want {
			t.Errorf("%s: StripTurnBlocks = %q, want %q", tc.name, got, tc.want)
		}
	}
}

// The earlier turns' user text loses its blocks; the last user text turn
// (the one being answered) keeps them; nothing else changes — thinking
// blocks keep their signatures, tool_use and tool_result blocks are as they
// were, and the input slice is not modified.
func TestStableHistory(t *testing.T) {
	u1 := mustUserText(t, stableCtx, "set the headline to Fresh", stableBlock)
	u2 := mustUserText(t, stableCtx, "now the tagline", stableBlock)
	msgs := []anthropic.BetaMessageParam{
		model.UserTextTurn(u1),
		assistantMsg(
			anthropic.NewBetaThinkingBlock("sig-1", "thinking about it"),
			anthropic.NewBetaTextBlock("On it."),
			anthropic.NewBetaToolUseBlock("t1", map[string]any{"target": map[string]any{"kind": "section", "sectionId": "hero-1"}, "values": map[string]any{"headline": "Fresh"}}, "set_fields"),
		),
		model.ToolResultsTurn([]model.ToolResult{{ID: "t1", Content: `{"ok":true}`}}),
		assistantMsg(anthropic.NewBetaTextBlock("Done.")),
		model.UserTextTurn(u2),
		assistantMsg(anthropic.NewBetaToolUseBlock("t2", map[string]any{}, "set_fields")),
		model.ToolResultsTurn([]model.ToolResult{{ID: "t2", Content: `{"ok":true}`}}),
	}
	before, _ := json.Marshal(msgs)
	out := StableHistory(msgs)
	after, _ := json.Marshal(msgs)
	if string(before) != string(after) {
		t.Fatal("StableHistory modified its input")
	}
	if len(out) != len(msgs) {
		t.Fatalf("%d messages, want %d", len(out), len(msgs))
	}
	if got := out[0].Content[0].OfText.Text; got != "set the headline to Fresh" {
		t.Errorf("earlier user turn = %q", got)
	}
	if got := out[4].Content[0].OfText.Text; got != u2 {
		t.Errorf("current user turn = %q, want it whole", got)
	}
	for _, i := range []int{1, 2, 3, 5, 6} {
		a, _ := json.Marshal(msgs[i])
		b, _ := json.Marshal(out[i])
		if string(a) != string(b) {
			t.Errorf("message %d changed:\n%s\n%s", i, a, b)
		}
	}
	// Idempotent, and a no-op (the same slice back) when nothing is stripped.
	again := StableHistory(out)
	if &again[0] != &out[0] {
		t.Error("a stable history was copied again")
	}
	if one := StableHistory(msgs[:1]); &one[0] != &msgs[0] {
		t.Error("a lone current turn was copied")
	}
	if StableHistory(nil) != nil {
		t.Error("nil history")
	}
}

func assistantMsg(blocks ...anthropic.BetaContentBlockParamUnion) anthropic.BetaMessageParam {
	return anthropic.BetaMessageParam{Role: anthropic.BetaMessageParamRoleAssistant, Content: blocks}
}
