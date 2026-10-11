package prompt

import (
	"encoding/json"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"

	"github.com/breadstack/sous-chef/internal/model"
)

func wire(t *testing.T, msgs []anthropic.BetaMessageParam) string {
	t.Helper()
	b, err := json.Marshal(msgs)
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

func opsTurn(id string) anthropic.BetaMessageParam {
	return assistant(
		anthropic.BetaContentBlockParamUnion{OfThinking: &anthropic.BetaThinkingBlockParam{Thinking: "t", Signature: "sig"}},
		anthropic.NewBetaTextBlock("Updating."),
		anthropic.BetaContentBlockParamUnion{OfToolUse: &anthropic.BetaToolUseBlockParam{ID: id, Name: "set_fields", Input: map[string]any{"values": map[string]any{"headline": "x"}}}},
	)
}

func assistant(blocks ...anthropic.BetaContentBlockParamUnion) anthropic.BetaMessageParam {
	return anthropic.BetaMessageParam{Role: anthropic.BetaMessageParamRoleAssistant, Content: blocks}
}

// A fast-finished turn's results message and the next turn's text are sent
// as one user message, results first; the assistant message and everything
// else go out byte-identical, and the caller's history is not touched.
func TestMergeUserRuns_ResultsThenText(t *testing.T) {
	results := model.ToolResultsTurn([]model.ToolResult{{ID: "s1", Content: `{"ok":true}`}})
	msgs := []anthropic.BetaMessageParam{
		model.UserTextTurn("set the headline"), opsTurn("s1"), results, model.UserTextTurn("<context>\n{}\n</context>\nthanks"),
	}
	before := wire(t, msgs)
	got := MergeUserRuns(msgs)
	if wire(t, msgs) != before {
		t.Fatal("caller's messages changed")
	}
	if len(got) != 3 {
		t.Fatalf("len = %d", len(got))
	}
	if wire(t, got[:2]) != wire(t, msgs[:2]) {
		t.Fatal("messages before the run moved")
	}
	last := got[2]
	if last.Role != anthropic.BetaMessageParamRoleUser || len(last.Content) != 2 ||
		last.Content[0].OfToolResult == nil || last.Content[0].OfToolResult.ToolUseID != "s1" ||
		last.Content[1].OfText == nil || last.Content[1].OfText.Text != "<context>\n{}\n</context>\nthanks" {
		t.Fatalf("merged = %s", wire(t, got[2:]))
	}
	if len(results.Content) != 1 {
		t.Fatal("results message grew")
	}
	// Deterministic: the same history gives the same bytes.
	if wire(t, MergeUserRuns(msgs)) != wire(t, got) {
		t.Fatal("not deterministic")
	}
	// What sendable does on a messages-compatible route: StableHistory first,
	// then the merge; the current turn keeps its context.
	stable := MergeUserRuns(StableHistory(msgs))
	if txt := stable[2].Content[1].OfText.Text; txt != "<context>\n{}\n</context>\nthanks" {
		t.Fatalf("current turn text = %q", txt)
	}
}

// Nothing to merge: the slice itself comes back. Two text messages in a row
// (a superseded turn's text, then the next turn's) stay apart.
func TestMergeUserRuns_LeavesOtherShapes(t *testing.T) {
	alternating := []anthropic.BetaMessageParam{
		model.UserTextTurn("a"), opsTurn("s1"),
		model.ToolResultsTurn([]model.ToolResult{{ID: "s1", Content: `{"ok":true}`}}),
		assistant(anthropic.NewBetaTextBlock("Done.")),
	}
	if got := MergeUserRuns(alternating); &got[0] != &alternating[0] || len(got) != len(alternating) {
		t.Fatal("alternating history was copied or changed")
	}
	texts := []anthropic.BetaMessageParam{model.UserTextTurn("first"), model.UserTextTurn("second")}
	if got := MergeUserRuns(texts); len(got) != 2 {
		t.Fatalf("text+text merged: %s", wire(t, got))
	}
	if got := MergeUserRuns(nil); got != nil {
		t.Fatal("nil")
	}
	// results, results, text (an abandoned batch answered, then a repair):
	// one message, in order.
	run := []anthropic.BetaMessageParam{
		opsTurn("s1"),
		model.ToolResultsTurn([]model.ToolResult{{ID: "s1", Content: "a", IsError: true}}),
		model.ToolResultsTurn([]model.ToolResult{{ID: "s2", Content: "b", IsError: true}}),
		model.UserTextTurn("next"),
	}
	got := MergeUserRuns(run)
	if len(got) != 2 || len(got[1].Content) != 3 || got[1].Content[0].OfToolResult.ToolUseID != "s1" ||
		got[1].Content[1].OfToolResult.ToolUseID != "s2" || got[1].Content[2].OfText.Text != "next" {
		t.Fatalf("run = %s", wire(t, got))
	}
}
