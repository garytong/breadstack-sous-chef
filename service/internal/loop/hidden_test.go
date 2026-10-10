package loop

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/breadstack/sous-chef/internal/api"
	"github.com/breadstack/sous-chef/internal/model/modeltest"
)

// gateMatches mirrors the launcher's changes.js gateMatches.
func TestGateMatches(t *testing.T) {
	for _, tc := range []struct {
		gate, v string
		want    bool
	}{
		{`"set"`, `"BREADSTACK_CDN/bg.jpg"`, true},
		{`"set"`, `""`, false},
		{`"set"`, `null`, false},
		{`"set"`, ``, false},
		{`"set"`, `false`, false},
		{`"set"`, `true`, true},
		{`"set"`, `0`, true},
		{`["classic","centered"]`, `"classic"`, true},
		{`["classic","centered"]`, `"split"`, false},
		{`"split"`, `"split"`, true},
		{`true`, `true`, true},
		{`true`, `false`, false},
		{`true`, `null`, false},
		{`false`, `false`, true},
		{`2`, `2`, true},
		{`2`, `"2"`, true},
		{`2`, `3`, false},
		{`null`, `null`, true},
		{`null`, `"x"`, false},
		{`["row","radio"]`, `"radio"`, true},
	} {
		if got := gateMatches(json.RawMessage(tc.gate), json.RawMessage(tc.v)); got != tc.want {
			t.Errorf("gateMatches(%s, %s) = %v, want %v", tc.gate, tc.v, got, tc.want)
		}
	}
}

func TestHiddenWrites(t *testing.T) {
	m := snapshotCompiled(t).Manifest
	setMaxWidth := json.RawMessage(`{"target":{"kind":"section","page":"home","sectionId":"hero-1"},"values":{"maxWidth":900,"paddingY":8}}`)
	noPage := json.RawMessage(`{"target":{"kind":"section","sectionId":"hero-1"},"values":{"maxWidth":900}}`)
	slideBg := json.RawMessage(`{"target":{"kind":"section","page":"home","sectionId":"hero-1"},"list":"slides","action":"update","index":1,"item":{"bgColor":"#ff0000"}}`)
	header := json.RawMessage(`{"target":{"kind":"global","panel":"HeaderSection"},"values":{"headerBgImageFit":"cover"}}`)
	restore := json.RawMessage(`{"target":{"kind":"section","page":"home","sectionId":"hero-1"},"keys":["maxWidth"]}`)
	res := func(warnings []string, hidden string) api.ToolResult {
		return api.ToolResult{ToolCallID: "x", OK: true, Warnings: warnings, Hidden: json.RawMessage(hidden)}
	}
	for _, tc := range []struct {
		name, op string
		input    json.RawMessage
		res      api.ToolResult
		want     string
	}{
		{"conditional_on closed (layout split)", "set_fields", setMaxWidth,
			res([]string{"W_HIDDEN:maxWidth"}, `[{"key":"maxWidth","by":"slides[0].layout","byValue":"split"}]`), "maxWidth"},
		{"conditional_on open (layout classic)", "set_fields", setMaxWidth,
			res([]string{"W_HIDDEN:maxWidth"}, `[{"key":"maxWidth","by":"slides[0].layout","byValue":"classic"}]`), ""},
		// The engine warns on every gated field, gate open or closed: the warning
		// alone (an older browser, or a malformed member) is not a hidden write.
		{"no hidden member: the warning alone", "set_fields", setMaxWidth, res([]string{"W_HIDDEN:maxWidth"}, ``), ""},
		{"malformed hidden member: the warning alone", "set_fields", setMaxWidth, res([]string{"W_HIDDEN:maxWidth"}, `{"maxWidth":1}`), ""},
		{"hidden member for another key only", "set_fields", setMaxWidth,
			res([]string{"W_HIDDEN:maxWidth"}, `[{"key":"paddingY","by":"slides[0].layout","byValue":"split"}]`), ""},
		{"section target without a page", "set_fields", noPage,
			res([]string{"W_HIDDEN:maxWidth"}, `[{"key":"maxWidth","by":"slides[0].layout","byValue":"centered"}]`), ""},
		{"warning on a key the op did not write", "set_fields", setMaxWidth, res([]string{"W_HIDDEN:bgColor"}, ``), ""},
		{"no warning", "set_fields", setMaxWidth, res(nil, ``), ""},
		{"list row: hidden_when set", "edit_list", slideBg,
			res([]string{"W_HIDDEN:bgColor"}, `[{"key":"bgColor","by":"slides[1].backgroundImage","byValue":"BREADSTACK_CDN/bg.jpg"}]`), "bgColor"},
		{"list row: hidden_when not set", "edit_list", slideBg,
			res([]string{"W_HIDDEN:bgColor"}, `[{"key":"bgColor","by":"slides[1].backgroundImage","byValue":""}]`), ""},
		{"global: gate unset reads as its default", "set_fields", header,
			res([]string{"W_HIDDEN:headerBgImageFit"}, `[{"key":"headerBgImageFit","by":"headerBgImage","byValue":""}]`), "headerBgImageFit"},
		{"gate key the rule does not name", "set_fields", setMaxWidth,
			res([]string{"W_HIDDEN:maxWidth"}, `[{"key":"maxWidth","by":"somethingElse","byValue":"classic"}]`), "maxWidth"},
		{"a restore writes nothing", "restore_default", restore, res([]string{"W_HIDDEN:maxWidth"}, ``), ""},
		{"a failed op", "set_fields", setMaxWidth, api.ToolResult{ToolCallID: "x", OK: false, Warnings: []string{"W_HIDDEN:maxWidth"}}, ""},
	} {
		got := strings.Join(hiddenWrites(m, "home", tc.op, tc.input, tc.res), ",")
		if got != tc.want {
			t.Errorf("%s: hiddenWrites = %q, want %q", tc.name, got, tc.want)
		}
	}
}

// The model reads the browser's `hidden` member in the tool_result, keys
// sorted like the rest of the result; a malformed member is dropped.
func TestBrowserResultForwardsHidden(t *testing.T) {
	r := browserResult("set_fields", api.ToolResult{ToolCallID: "t1", OK: true, Written: []string{"maxWidth"},
		Warnings: []string{"W_HIDDEN:maxWidth"}, Hidden: json.RawMessage(`[{"key":"maxWidth","by":"slides[0].layout","byValue": "split"},{"by":"x"}]`)})
	want := `{"hidden":[{"by":"slides[0].layout","byValue":"split","key":"maxWidth"}],"ok":true,"warnings":["W_HIDDEN:maxWidth"],"written":["maxWidth"]}`
	if r.Content != want || r.IsError {
		t.Errorf("content = %s", r.Content)
	}
	r = browserResult("set_fields", api.ToolResult{ToolCallID: "t1", OK: true, Hidden: json.RawMessage(`"nope"`)})
	if r.Content != `{"ok":true}` {
		t.Errorf("malformed hidden: content = %s", r.Content)
	}
	r = browserResult("set_fields", api.ToolResult{ToolCallID: "t1", OK: true, Hidden: json.RawMessage(`[{"key":"k","by":"b"}]`)})
	if r.Content != `{"hidden":[{"by":"b","byValue":null,"key":"k"}],"ok":true}` {
		t.Errorf("missing byValue: content = %s", r.Content)
	}
}

func setMaxWidth(id string) modeltest.Block {
	return modeltest.ToolUse(id, "set_fields", map[string]any{"target": heroTarget, "values": map[string]any{"maxWidth": 900}})
}

func hiddenResult(id, layout string) api.ToolResult {
	return api.ToolResult{ToolCallID: id, OK: true, Written: []string{"maxWidth"}, Warnings: []string{"W_HIDDEN:maxWidth"},
		Hidden: json.RawMessage(`[{"key":"maxWidth","by":"slides[0].layout","byValue":"` + layout + `"}]`)}
}

// Plan Q9: the student's applied write that the merchant will not see hands
// the turn to the fallback, which reads the result with its `hidden` member.
func TestHybrid_HiddenWriteEscalates(t *testing.T) {
	hy := newHybrid(t,
		[]modeltest.Step{reply(modeltest.Text("Setting it."), setMaxWidth("s1"))},
		[]modeltest.Step{reply(modeltest.Text("The width only shows with the Classic or Centered layout; your first slide uses Split."))})
	turn, evs := hy.start(hy.req("student", "make the hero narrower"))
	if kinds(evs) != "text.delta tool.call turn.waiting" {
		t.Fatalf("stream 1 = %s", kinds(evs))
	}
	evs, err := hy.cont(turn.ID(), hiddenResult("s1", "split"))
	if err != nil {
		t.Fatal(err)
	}
	wantEscalated(t, evs, EscalateHiddenWrite)
	if last(evs).Event != api.EventTurnDone || routesOf(t, last(evs)) != `["student","backstop"]` {
		t.Fatalf("stream 2 = %s", kinds(evs))
	}
	creqs := hy.claude.Requests()
	if len(creqs) != 1 {
		t.Fatalf("claude requests = %d", len(creqs))
	}
	got := lastToolResults(t, creqs[0])
	if len(got) != 1 || got[0].IsError ||
		got[0].Content != `{"hidden":[{"by":"slides[0].layout","byValue":"split","key":"maxWidth"}],"ok":true,"warnings":["W_HIDDEN:maxWidth"],"written":["maxWidth"]}` {
		t.Fatalf("claude reads = %+v", got)
	}
	var usage map[string]any
	_ = json.Unmarshal(hy.st.turn(turn.ID()).Usage, &usage)
	if esc, _ := usage["escalated"].(map[string]any); esc["reason"] != EscalateHiddenWrite {
		t.Errorf("turns.usage escalated = %v", usage["escalated"])
	}
}

// A gated field whose gate is open is an ordinary write: no escalation; the
// student finishes the turn. A refusal in the same batch wins the reason.
func TestHybrid_HiddenWriteGateOpenOrRefused(t *testing.T) {
	hy := newHybrid(t,
		[]modeltest.Step{reply(setMaxWidth("s1")), reply(modeltest.Text("Done."))},
		nil)
	turn, _ := hy.start(hy.req("student", "make the hero narrower"))
	evs, err := hy.cont(turn.ID(), hiddenResult("s1", "classic"))
	if err != nil {
		t.Fatal(err)
	}
	if len(events(evs, api.EventTurnEscalated)) != 0 || last(evs).Event != api.EventTurnDone || text(evs) != "Done." {
		t.Fatalf("gate open: stream = %s %q", kinds(evs), text(evs))
	}

	hy = newHybrid(t,
		[]modeltest.Step{reply(setMaxWidth("s1"), setFields("s2", "itemDescriptions"))},
		[]modeltest.Step{reply(modeltest.Text("Fixed."))})
	turn, _ = hy.start(hy.req("student", "make the hero narrower and describe items"))
	evs, err = hy.cont(turn.ID(), hiddenResult("s1", "split"), refusedResult("s2"))
	if err != nil {
		t.Fatal(err)
	}
	wantEscalated(t, evs, EscalateToolError)

	// A route without a fallback never escalates.
	hy = newHybrid(t, []modeltest.Step{reply(setMaxWidth("s1")), reply(modeltest.Text("Done."))}, nil)
	turn, _ = hy.start(hy.req("solo", "make the hero narrower"))
	evs, err = hy.cont(turn.ID(), hiddenResult("s1", "split"))
	if err != nil {
		t.Fatal(err)
	}
	if len(events(evs, api.EventTurnEscalated)) != 0 || last(evs).Event != api.EventTurnDone {
		t.Fatalf("solo: stream = %s", kinds(evs))
	}
}
