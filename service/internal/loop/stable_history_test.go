package loop

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"

	"github.com/breadstack/sous-chef/internal/api"
	"github.com/breadstack/sous-chef/internal/fields"
	"github.com/breadstack/sous-chef/internal/model"
	"github.com/breadstack/sous-chef/internal/model/modeltest"
	"github.com/breadstack/sous-chef/internal/prompt"
)

// Plan B3: on a messages-compatible route the replayed history carries the
// earlier turns' request text only — their <context> and <candidate_fields>
// blocks are left out at send time — while the turn being answered keeps
// its own blocks on every call, after a rebuild too. tool_use / tool_result
// blocks and the assistant's text go out as stored, and storage keeps every
// user message whole.
func TestStableHistory_StudentReplaysRequestTextOnly(t *testing.T) {
	for _, rebuilt := range []bool{false, true} {
		hy := newHybrid(t, []modeltest.Step{
			reply(setFields("s1", "headline")), reply(modeltest.Text("Done.")),
			reply(setFields("s2", "tagline")), reply(modeltest.Text("Done again.")),
		}, nil)
		const first, second = "set the hero headline to Fresh cookies", "now set the hero tagline to Baked daily"
		turn1, _ := hy.start(hy.req("solo", first))
		if _, err := hy.cont(turn1.ID(), writtenResult("s1", "headline")); err != nil {
			t.Fatal(err)
		}
		turn2, _ := hy.start(hy.req("solo", second))
		if rebuilt {
			hy.runner.unpark(turn2.ID())
		}
		evs, err := hy.cont(turn2.ID(), writtenResult("s2", "tagline"))
		if err != nil || last(evs).Event != "turn.done" {
			t.Fatalf("rebuilt=%v: turn 2 = %s (%v)", rebuilt, kinds(evs), err)
		}
		reqs := hy.student.Requests()
		if len(reqs) != 4 {
			t.Fatalf("rebuilt=%v: %d requests", rebuilt, len(reqs))
		}
		// Turn 1's own calls carry its blocks.
		for i := 0; i < 2; i++ {
			if u := userTexts(t, reqs[i]); len(u) != 1 || !strings.HasPrefix(u[0], "<context>\n") || !strings.HasSuffix(u[0], first) {
				t.Fatalf("rebuilt=%v: turn 1 call %d user texts = %q", rebuilt, i+1, u)
			}
		}
		// Turn 2, both calls: turn 1 is its text; turn 2 is whole.
		for i := 2; i < 4; i++ {
			u := userTexts(t, reqs[i])
			if len(u) != 2 || u[0] != first || !strings.HasPrefix(u[1], "<context>\n") || !strings.Contains(u[1], "\n<candidate_fields>\n") || !strings.HasSuffix(u[1], second) {
				t.Fatalf("rebuilt=%v: turn 2 call %d user texts = %q", rebuilt, i-1, u)
			}
			if got := msgRoles(reqs[i]); !strings.HasPrefix(got, "user assistant user assistant user") {
				t.Fatalf("rebuilt=%v: roles = %s", rebuilt, got)
			}
		}
		// The replayed tool_use / tool_result of turn 1 are as stored.
		msgs := reqs[3].Body["messages"].([]any)
		use := msgs[1].(map[string]any)["content"].([]any)[0].(map[string]any)
		res := msgs[2].(map[string]any)["content"].([]any)[0].(map[string]any)
		content, _ := json.Marshal(res["content"])
		if use["type"] != "tool_use" || use["id"] != "s1" || res["type"] != "tool_result" || res["tool_use_id"] != "s1" ||
			string(content) != `[{"text":"{\"ok\":true,\"written\":[\"headline\"]}","type":"text"}]` {
			t.Fatalf("rebuilt=%v: replayed batch = %v / %v", rebuilt, use, res)
		}
		// Storage keeps turn 1's message whole.
		stored, _ := hy.st.ListMessages(context.Background(), hy.st.turn(turn2.ID()).ConversationID, 0, 0)
		var blocks []struct{ Text string }
		if err := json.Unmarshal(stored[0].Content, &blocks); err != nil || len(blocks) != 1 || !strings.HasPrefix(blocks[0].Text, "<context>\n") {
			t.Fatalf("rebuilt=%v: stored turn 1 = %s", rebuilt, stored[0].Content)
		}
	}
}

func writtenResult(id, key string) api.ToolResult {
	return api.ToolResult{ToolCallID: id, OK: true, Written: []string{key}}
}

// An anthropic fallback is sent the stored history whole: a history edit
// would break Claude's preserved thinking and its cache prefix.
func TestStableHistory_AnthropicFallbackGetsStoredHistory(t *testing.T) {
	hy := newHybrid(t,
		[]modeltest.Step{reply(modeltest.Text("Hi.")), reply(setFields("s2", "itemDescriptions"))},
		[]modeltest.Step{reply(modeltest.Text("Fixed."))})
	const first = "set the hero headline to Fresh cookies"
	hy.start(hy.req("student", first))
	turn2, _ := hy.start(hy.req("student", "describe the items"))
	if _, err := hy.cont(turn2.ID(), refusedResult("s2")); err != nil {
		t.Fatal(err)
	}
	sreqs, creqs := hy.student.Requests(), hy.claude.Requests()
	if len(sreqs) != 2 || len(creqs) != 1 {
		t.Fatalf("requests: student %d, claude %d", len(sreqs), len(creqs))
	}
	if u := userTexts(t, sreqs[1]); len(u) != 2 || u[0] != first {
		t.Fatalf("student turn 2 user texts = %q", u)
	}
	if u := userTexts(t, creqs[0]); len(u) != 2 || !strings.HasPrefix(u[0], "<context>\n") || !strings.HasSuffix(u[0], first) {
		t.Fatalf("claude user texts = %q", u)
	}
}

// Stage 1's follow-up rule is wired: a short answer that names nothing gets
// the block of the previous request + the assistant's reply + the answer.
func TestFieldsContext_FollowUpReadsThePreviousTurn(t *testing.T) {
	const question = "How many categories should the header show, and which ones should stay?"
	hy := newHybrid(t, []modeltest.Step{reply(modeltest.Text(question)), reply(modeltest.Text("ok")), reply(modeltest.Text("ok"))}, nil)
	hy.start(fieldsRequest(hy.harness, "solo", "there are too many categories in the header"))
	hy.start(fieldsRequest(hy.harness, "solo", "4, you pick"))
	reqs := hy.student.Requests()
	if len(reqs) != 2 {
		t.Fatalf("%d requests", len(reqs))
	}
	m := snapshotCompiled(t).Manifest
	ctx, err := fields.ParseContext(json.RawMessage(fieldsHomeContext))
	if err != nil {
		t.Fatal(err)
	}
	after := fields.CandidatesAfter(m, ctx, "4, you pick", fields.Previous{User: "there are too many categories in the header", Assistant: question}, fields.Options{}).Block
	alone := fields.Candidates(m, ctx, "4, you pick", fields.Options{}).Block
	if after == "" || after == alone {
		t.Fatalf("fixture is not a follow-up: after %q, alone %q", after, alone)
	}
	u := userTexts(t, reqs[1])
	want := "<context>\n" + compactJSON(t, fieldsHomeContext) + "\n</context>\n" + after + "\n4, you pick"
	if len(u) != 2 || u[1] != want {
		t.Fatalf("follow-up user turn = %q\nwant %q", u, want)
	}
	// A request that names its target is read on its own.
	hy.start(fieldsRequest(hy.harness, "solo", fieldsHeroText))
	got := userTexts(t, hy.student.Requests()[2])
	direct := fields.Candidates(m, ctx, fieldsHeroText, fields.Options{}).Block
	if len(got) != 3 || got[2] != "<context>\n"+compactJSON(t, fieldsHomeContext)+"\n</context>\n"+direct+"\n"+fieldsHeroText {
		t.Fatalf("named request = %q", got[2])
	}
}

// previousExchange reads the stored previous turn: its request text without
// blocks, and the last assistant text after it.
func TestPreviousExchange(t *testing.T) {
	if p := previousExchange(nil); p != (fields.Previous{}) {
		t.Fatalf("empty history = %+v", p)
	}
	user, err := prompt.UserText(json.RawMessage(`{"page":"home"}`), "there are too many categories in the header", "<candidate_fields>\nx\n</candidate_fields>")
	if err != nil {
		t.Fatal(err)
	}
	history := []anthropic.BetaMessageParam{
		model.UserTextTurn("an older request"),
		{Role: anthropic.BetaMessageParamRoleAssistant, Content: []anthropic.BetaContentBlockParamUnion{anthropic.NewBetaTextBlock("Older reply.")}},
		model.UserTextTurn(user),
		{Role: anthropic.BetaMessageParamRoleAssistant, Content: []anthropic.BetaContentBlockParamUnion{
			anthropic.NewBetaThinkingBlock("sig", "hmm"), anthropic.NewBetaTextBlock("Let me look."), anthropic.NewBetaToolUseBlock("t1", map[string]any{}, "lookup_fields")}},
		model.ToolResultsTurn([]model.ToolResult{{ID: "t1", Content: "{}"}}),
		{Role: anthropic.BetaMessageParamRoleAssistant, Content: []anthropic.BetaContentBlockParamUnion{
			anthropic.NewBetaThinkingBlock("sig2", "ok"), anthropic.NewBetaTextBlock("How many should it show?")}},
	}
	want := fields.Previous{User: "there are too many categories in the header", Assistant: "How many should it show?"}
	if p := previousExchange(history); p != want {
		t.Fatalf("previousExchange = %+v, want %+v", p, want)
	}
	// A turn that ended in tool calls only: the last assistant text after it.
	if p := previousExchange(history[:5]); p.Assistant != "Let me look." || p.User != want.User {
		t.Fatalf("previousExchange (no final text) = %+v", p)
	}
}
