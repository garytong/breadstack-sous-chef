package loop

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/breadstack/sous-chef/internal/api"
	"github.com/breadstack/sous-chef/internal/model/modeltest"
)

// thinkingBlocks counts the thinking blocks in a captured request's messages.
func thinkingBlocks(t *testing.T, c modeltest.Captured) int {
	t.Helper()
	var body struct {
		Messages []struct {
			Content json.RawMessage `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(c.Raw, &body); err != nil {
		t.Fatal(err)
	}
	n := 0
	for _, m := range body.Messages {
		var blocks []struct{ Type string }
		if json.Unmarshal(m.Content, &blocks) != nil {
			continue // a plain string
		}
		for _, b := range blocks {
			if b.Type == "thinking" || b.Type == "redacted_thinking" {
				n++
			}
		}
	}
	return n
}

// A student reply may carry a thinking block (its route's `thinking` is
// enabled, or the server returned one unasked). Claude rejects a thinking
// block it did not sign, so the history an anthropic fallback is sent has
// none of the blocks written before the escalation — this turn's and an
// earlier turn's — while the blocks Claude itself writes afterwards go back
// to it unchanged. Storage keeps everything as returned.
func TestHybrid_StudentThinkingIsNotReplayedToClaude(t *testing.T) {
	for _, rebuilt := range []bool{false, true} {
		name := "same instance"
		if rebuilt {
			name = "rebuilt from the row"
		}
		t.Run(name, func(t *testing.T) {
			hy := newHybrid(t,
				[]modeltest.Step{
					// turn 1: a healthy student turn that thought first.
					reply(modeltest.Thinking("earlier turn"), modeltest.Text("Hello.")),
					// turn 2: thinking, then an op the engine refuses.
					reply(modeltest.Thinking("student thoughts"), modeltest.Text("On it."), setFields("s1", "itemDescriptions")),
				},
				[]modeltest.Step{
					reply(modeltest.Thinking("claude thoughts"), setFields("c1", "headline")),
					reply(modeltest.Text("Done.")),
				})
			if _, evs := hy.start(hy.req("student", "hello")); last(evs).Event != api.EventTurnDone {
				t.Fatalf("turn 1 = %s", kinds(evs))
			}
			turn, _ := hy.start(hy.req("student", "set the hero headline to Fresh cookies"))
			if rebuilt {
				hy.runner.unpark(turn.ID())
			}
			evs, err := hy.cont(turn.ID(), refusedResult("s1"))
			if err != nil {
				t.Fatal(err)
			}
			if kinds(evs) != "turn.escalated tool.call turn.waiting" {
				t.Fatalf("stream 2 = %s", kinds(evs))
			}
			var pd pendingDoc
			if err := json.Unmarshal(hy.st.turn(turn.ID()).PendingToolCalls, &pd); err != nil {
				t.Fatal(err)
			}
			// user, assistant | user, assistant, results: five messages
			// were written before Claude took the turn.
			if pd.ForeignBefore != 5 {
				t.Fatalf("pending foreign_before = %d, want 5", pd.ForeignBefore)
			}
			if rebuilt {
				hy.runner.unpark(turn.ID())
			}
			evs, err = hy.cont(turn.ID(), api.ToolResult{ToolCallID: "c1", OK: true, Written: []string{"headline"}})
			if err != nil {
				t.Fatal(err)
			}
			if last(evs).Event != api.EventTurnDone || text(evs) != "Done." {
				t.Fatalf("stream 3 = %s %q", kinds(evs), text(evs))
			}

			sreqs, creqs := hy.student.Requests(), hy.claude.Requests()
			if len(sreqs) != 2 || len(creqs) != 2 {
				t.Fatalf("requests: student %d, claude %d", len(sreqs), len(creqs))
			}
			// Claude's first request: no thinking block at all, the
			// student's text and tool_use and the error result intact.
			if n := thinkingBlocks(t, creqs[0]); n != 0 {
				t.Errorf("claude call 1 carries %d foreign thinking block(s)", n)
			}
			raw := string(creqs[0].Raw)
			for _, bad := range []string{"student thoughts", "earlier turn", `"signature"`} {
				if strings.Contains(raw, bad) {
					t.Errorf("claude call 1 contains %s", bad)
				}
			}
			if msgRoles(creqs[0]) != "user assistant user assistant user" || !strings.Contains(raw, `"id":"s1"`) || !strings.Contains(raw, "On it.") || !strings.Contains(raw, "Hello.") {
				t.Errorf("claude call 1 lost history: roles %s", msgRoles(creqs[0]))
			}
			if res := lastToolResults(t, creqs[0]); len(res) != 1 || res[0].ID != "s1" || !res[0].IsError {
				t.Errorf("claude call 1 results = %+v", res)
			}
			// Claude's second request: its own thinking block, and only it.
			raw = string(creqs[1].Raw)
			if n := thinkingBlocks(t, creqs[1]); n != 1 || !strings.Contains(raw, "claude thoughts") || strings.Contains(raw, "student thoughts") || strings.Contains(raw, "earlier turn") {
				t.Errorf("claude call 2 thinking blocks = %d (want its own one)", n)
			}
			// Storage is untouched: all three blocks are there.
			msgs, _ := hy.st.ListMessages(context.Background(), hy.st.turn(turn.ID()).ConversationID, 0, 0)
			var stored strings.Builder
			for _, m := range msgs {
				stored.Write(m.Content)
			}
			for _, want := range []string{"earlier turn", "student thoughts", "claude thoughts"} {
				if !strings.Contains(stored.String(), want) {
					t.Errorf("stored history lost %q", want)
				}
			}
		})
	}
}

// The same on the triggers that re-send the request: a model error on a
// conversation whose earlier student turn left a thinking block.
func TestHybrid_ModelErrorEscalationDropsEarlierThinking(t *testing.T) {
	hy := newHybrid(t,
		[]modeltest.Step{
			reply(modeltest.Thinking("earlier turn"), modeltest.Text("Hello.")),
			modeltest.Error(500, "api_error", "boom", nil),
		},
		[]modeltest.Step{reply(modeltest.Text("Fixed."))})
	hy.start(hy.req("student", "hello"))
	_, evs := hy.start(hy.req("student", "set the hero headline"))
	wantEscalated(t, evs, EscalateModelError)
	creqs := hy.claude.Requests()
	if len(creqs) != 1 || thinkingBlocks(t, creqs[0]) != 0 || msgRoles(creqs[0]) != "user assistant user" {
		t.Fatalf("claude request: %d thinking block(s), roles %s", thinkingBlocks(t, creqs[0]), msgRoles(creqs[0]))
	}
}

// A turn on an anthropic route is sent its history as stored: nothing is
// stripped where nothing escalated (Claude's own blocks must go back).
func TestHybrid_NoEscalationNoStripping(t *testing.T) {
	hy := newHybrid(t, nil, []modeltest.Step{
		reply(modeltest.Thinking("claude thoughts"), setFields("c1", "headline")),
		reply(modeltest.Text("Done.")),
	})
	turn, _ := hy.start(hy.req("default", "set the hero headline to Fresh cookies"))
	if _, err := hy.cont(turn.ID(), api.ToolResult{ToolCallID: "c1", OK: true, Written: []string{"headline"}}); err != nil {
		t.Fatal(err)
	}
	creqs := hy.claude.Requests()
	if len(creqs) != 2 || thinkingBlocks(t, creqs[1]) != 1 {
		t.Fatalf("claude call 2 has %d thinking block(s), want its own", thinkingBlocks(t, creqs[1]))
	}
}

// turn.done states the route the turn started on, whether its user message
// was built with the candidate-fields option on, and the route's fallback
// — on every route, escalated or not, rebuilt or not.
func TestTurnDone_StatesRouteOptions(t *testing.T) {
	routeOf := func(t *testing.T, ev api.CollectedEvent) string {
		t.Helper()
		if ev.Event != api.EventTurnDone {
			t.Fatalf("last event = %s", ev.Event)
		}
		raw, _ := json.Marshal(data(t, ev)["route"])
		return string(raw)
	}
	t.Run("anthropic route", func(t *testing.T) {
		hy := newHybrid(t, nil, []modeltest.Step{reply(modeltest.Text("ok"))})
		_, evs := hy.start(hy.req("default", "hello"))
		if got := routeOf(t, last(evs)); got != `{"fallback":null,"fields_context":true,"model":"claude-sonnet-5-5","name":"default"}` {
			t.Errorf("route = %s", got)
		}
		// No route named: the default route, by name.
		hy.claude.Append(answering("claude-sonnet-5-5-20260901", []modeltest.Step{reply(modeltest.Text("ok"))})...)
		_, evs = hy.start(hy.req("", "hello again"))
		// `model` is the id the turn's calls were reported under: the response's.
		if got := routeOf(t, last(evs)); got != `{"fallback":null,"fields_context":true,"model":"claude-sonnet-5-5-20260901","name":"default"}` {
			t.Errorf("route (none named) = %s", got)
		}
	})
	t.Run("student without fallback", func(t *testing.T) {
		hy := newHybrid(t, []modeltest.Step{reply(modeltest.Text("ok"))}, nil)
		_, evs := hy.start(hy.req("solo", "hello"))
		if got := routeOf(t, last(evs)); got != `{"fallback":null,"fields_context":true,"model":"sous-chef-student","name":"solo"}` {
			t.Errorf("route = %s", got)
		}
	})
	t.Run("hybrid, not escalated", func(t *testing.T) {
		hy := newHybrid(t, []modeltest.Step{reply(modeltest.Text("ok"))}, nil)
		_, evs := hy.start(hy.req("student", "hello"))
		if got := routeOf(t, last(evs)); got != `{"fallback":"backstop","fields_context":true,"model":"sous-chef-student","name":"student"}` {
			t.Errorf("route = %s", got)
		}
	})
	t.Run("hybrid, escalated and rebuilt", func(t *testing.T) {
		hy := newHybrid(t,
			[]modeltest.Step{reply(setFields("s1", "itemDescriptions"))},
			[]modeltest.Step{reply(setFields("c1", "headline")), reply(modeltest.Text("Done."))})
		turn, _ := hy.start(hy.req("student", "set the hero headline to Fresh cookies"))
		if _, err := hy.cont(turn.ID(), refusedResult("s1")); err != nil {
			t.Fatal(err)
		}
		var pd pendingDoc
		if err := json.Unmarshal(hy.st.turn(turn.ID()).PendingToolCalls, &pd); err != nil {
			t.Fatal(err)
		}
		if pd.FieldsContext == nil || !*pd.FieldsContext {
			t.Fatalf("pending fields_context = %v", pd.FieldsContext)
		}
		hy.runner.unpark(turn.ID()) // continue from the row
		evs, err := hy.cont(turn.ID(), api.ToolResult{ToolCallID: "c1", OK: true, Written: []string{"headline"}})
		if err != nil {
			t.Fatal(err)
		}
		// Still the route the turn STARTED on, though Claude finished it.
		if got := routeOf(t, last(evs)); got != `{"fallback":"backstop","fields_context":true,"model":"sous-chef-student","name":"student"}` {
			t.Errorf("route = %s", got)
		}
		if got := routesOf(t, last(evs)); got != `["student","backstop"]` {
			t.Errorf("routes = %s", got)
		}
		// The rebuilt turn still knows what the student's call reported.
		if got, _ := json.Marshal(data(t, last(evs))["models"]); string(got) != `{"backstop":"claude-sonnet-5-5","student":"sous-chef-student"}` {
			t.Errorf("models = %s", got)
		}
	})
}
