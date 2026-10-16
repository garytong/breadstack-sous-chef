package loop

import (
	"strings"
	"testing"

	"github.com/breadstack/sous-chef/internal/api"
	"github.com/breadstack/sous-chef/internal/model/modeltest"
)

func createPage(id string, in map[string]any) modeltest.Block {
	return modeltest.ToolUse(id, "create_page", in)
}

// The student was trained to answer "make a new page" with report_unsupported
// where=pages (r4). On a hybrid route that escalates, Claude creates the
// page with create_page — a browser op shown to the page — and reads the
// final slug from the browser's result to target the new page.
func TestCreatePage_StudentUnsupportedEscalatesAndClaudeCreates(t *testing.T) {
	hy := newHybrid(t,
		[]modeltest.Step{reply(modeltest.Text("Pages are made in the Pages tab."),
			modeltest.ToolUse("s1", "report_unsupported", map[string]any{"reason": "Pages are created in the Pages tab.", "where": "pages"}))},
		[]modeltest.Step{
			reply(createPage("c1", map[string]any{"title": "Deals"})),
			reply(modeltest.ToolUse("c2", "edit_list", map[string]any{"target": map[string]any{"kind": "page", "page": "deals"}, "list": "sections", "action": "add", "item": map[string]any{"id": "productCarousel"}})),
			reply(modeltest.Text("Added an empty Deals page (/deals).")),
		})
	turn, evs := hy.start(hy.req("student", "make a new page called Deals"))
	if kinds(evs) != "turn.escalated tool.call turn.waiting" {
		t.Fatalf("stream 1 = %s", kinds(evs))
	}
	wantEscalated(t, evs, EscalateUnsupported)
	if d := data(t, events(evs, api.EventToolCall)[0]); d["id"] != "c1" || d["name"] != "create_page" {
		t.Fatalf("tool.call = %v (the student's report must not be shown)", d)
	}
	if strings.Contains(text(evs), "Pages tab") {
		t.Errorf("the student's text reached the page: %q", text(evs))
	}

	evs, err := hy.cont(turn.ID(), api.ToolResult{ToolCallID: "c1", OK: true, Written: []string{"customPages"}, Slug: "deals"})
	if err != nil {
		t.Fatal(err)
	}
	if kinds(evs) != "tool.call turn.waiting" {
		t.Fatalf("stream 2 = %s", kinds(evs))
	}
	creqs := hy.claude.Requests()
	if res := lastToolResults(t, creqs[1]); len(res) != 1 || res[0].ID != "c1" || res[0].IsError || !strings.Contains(res[0].Content, `"slug":"deals"`) || !strings.Contains(res[0].Content, `"ok":true`) {
		t.Fatalf("claude read create_page's result as %+v", res)
	}

	evs, err = hy.cont(turn.ID(), okResult("c2"))
	if err != nil {
		t.Fatal(err)
	}
	if last(evs).Event != api.EventTurnDone || text(evs) != "Added an empty Deals page (/deals)." {
		t.Fatalf("stream 3 = %s %q", kinds(evs), text(evs))
	}
	if got := routesOf(t, last(evs)); got != `["student","backstop"]` {
		t.Errorf("routes = %s", got)
	}
	// Claude was sent the create_page tool, as every route is.
	found := false
	for _, tool := range creqs[0].Body["tools"].([]any) {
		found = found || tool.(map[string]any)["name"] == "create_page"
	}
	if !found {
		t.Error("claude's request lacks create_page")
	}
}

// A create_page the engine refuses (duplicate page, reserved slug, premium
// template) is a refused op like any other: the student's turn escalates
// tool_error and Claude reads the engine's error.
func TestCreatePage_EngineRefusalEscalates(t *testing.T) {
	hy := newHybrid(t,
		[]modeltest.Step{reply(createPage("s1", map[string]any{"title": "Cart"}))},
		[]modeltest.Step{reply(modeltest.Text("You already have a Cart page."))})
	turn, evs := hy.start(hy.req("student", "add a page called Cart"))
	if kinds(evs) != "tool.call turn.waiting" {
		t.Fatalf("stream 1 = %s", kinds(evs))
	}
	evs, err := hy.cont(turn.ID(), api.ToolResult{ToolCallID: "s1", OK: false, Error: &api.ToolError{Code: "E_VALUE", Message: `slug "cart" is reserved; try "cart-2"`}})
	if err != nil {
		t.Fatal(err)
	}
	wantEscalated(t, evs, EscalateToolError)
	if res := lastToolResults(t, hy.claude.Requests()[0]); len(res) != 1 || !res[0].IsError || !strings.Contains(res[0].Content, "reserved") {
		t.Fatalf("claude results = %+v", res)
	}
}

// An op the service itself refuses (ValidateBrowserOp) escalates even when
// it is the batch's only call, so nothing waits on the browser: a
// create_page without a title, and the keyless reset. Before 2026-10-15 the
// refusal flag was shadowed in dispatch and such a batch stayed on the
// student.
func TestCreatePage_ServiceRefusalAloneEscalates(t *testing.T) {
	for _, tc := range []struct {
		name string
		call modeltest.Block
		want string
	}{
		{"create_page without a title", createPage("s1", map[string]any{"title": "  "}), "create_page needs a title"},
		{"reset without keys", modeltest.ToolUse("s1", "reset", map[string]any{"target": heroTarget}), "reset needs keys"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			hy := newHybrid(t,
				[]modeltest.Step{reply(tc.call)},
				[]modeltest.Step{reply(modeltest.Text("Fixed."))})
			_, evs := hy.start(hy.req("student", "do it"))
			if kinds(evs) != "turn.escalated text.delta text.delta turn.done" {
				t.Fatalf("events = %s", kinds(evs))
			}
			wantEscalated(t, evs, EscalateToolError)
			if len(events(evs, api.EventToolCall)) != 0 {
				t.Fatal("a refused op reached the page")
			}
			if res := lastToolResults(t, hy.claude.Requests()[0]); len(res) != 1 || !res[0].IsError || !strings.Contains(res[0].Content, tc.want) {
				t.Errorf("results = %+v", res)
			}
		})
	}
}

// A clean create_page never fast-finishes: the follow-up (sections on the
// new page) needs its slug, and the closing reply says where the page lives
// and that it goes live on Save.
func TestCreatePage_NoFastFinish(t *testing.T) {
	hy := newFast(t, []modeltest.Step{
		reply(createPage("cp1", map[string]any{"title": "Deals"})),
		reply(modeltest.Text("Added /deals.")),
	}, nil)
	turn, evs := hy.start(hy.req("solo", "add a Deals page"))
	if last(evs).Event != api.EventTurnWaiting {
		t.Fatalf("stream 1 = %s", kinds(evs))
	}
	evs, err := hy.cont(turn.ID(), api.ToolResult{ToolCallID: "cp1", OK: true, Slug: "deals"})
	if err != nil {
		t.Fatal(err)
	}
	wantClosingCall(t, evs, "Added /deals.")
	if n := len(hy.student.Requests()); n != 2 {
		t.Fatalf("model calls = %d, want 2", n)
	}
	if res := lastToolResults(t, hy.student.Requests()[1]); len(res) != 1 || !strings.Contains(res[0].Content, `"slug":"deals"`) {
		t.Fatalf("results = %+v", res)
	}
}

// browserResult forwards `slug` for create_page only.
func TestBrowserResultSlugOnlyForCreatePage(t *testing.T) {
	if got := browserResult("create_page", api.ToolResult{ToolCallID: "a", OK: true, Slug: "deals"}); got.IsError || got.Content != `{"ok":true,"slug":"deals"}` {
		t.Fatalf("create_page → %+v", got)
	}
	if got := browserResult("set_fields", api.ToolResult{ToolCallID: "a", OK: true, Slug: "deals"}); strings.Contains(got.Content, "slug") {
		t.Fatalf("set_fields → %+v", got)
	}
	if got := browserResult("create_page", api.ToolResult{ToolCallID: "a", OK: true}); got.Content != `{"ok":true}` {
		t.Fatalf("create_page without slug → %+v", got)
	}
}

// A create_page the service refuses (here: a title over MaxPageTitle) stops
// every later op of the same batch aimed at its slug — the model passes the
// slug and targets it in the same reply, and "our-story" already exists, so
// the follow-up would otherwise edit that page. Ops aimed elsewhere still
// reach the browser. (The browser applies the same rule after an engine
// refusal: client/turns.js applyOps.)
func TestCreatePage_ServiceRefusalStopsSameBatchFollowUps(t *testing.T) {
	long := strings.Repeat("x", 81)
	section := func(id, page string) modeltest.Block {
		return modeltest.ToolUse(id, "edit_list", map[string]any{"target": map[string]any{"kind": "page", "page": page}, "list": "sections", "action": "add", "item": map[string]any{"id": "productCarousel"}})
	}
	hy := newFast(t, []modeltest.Step{
		reply(
			createPage("cp1", map[string]any{"title": long, "slug": "our-story"}),
			section("e1", "our-story"),
			modeltest.ToolUse("e2", "set_page_layout", map[string]any{"page": "our-story", "layout": "wide"}),
			section("e3", "faq"),
		),
		reply(modeltest.Text("That title is too long.")),
	}, nil)
	turn, evs := hy.start(hy.req("solo", "add a page"))
	calls := events(evs, api.EventToolCall)
	if len(calls) != 1 || data(t, calls[0])["id"] != "e3" {
		t.Fatalf("tool.calls = %s (only the op aimed elsewhere may reach the page)", kinds(evs))
	}
	if _, err := hy.cont(turn.ID(), okResult("e3")); err != nil {
		t.Fatal(err)
	}
	res := lastToolResults(t, hy.student.Requests()[1])
	if len(res) != 4 {
		t.Fatalf("results = %+v", res)
	}
	if !res[0].IsError || !strings.Contains(res[0].Content, "create_page needs a title") {
		t.Errorf("create result = %+v", res[0])
	}
	for _, r := range res[1:3] {
		if !r.IsError || !strings.Contains(r.Content, `"E_TARGET"`) || !strings.Contains(r.Content, `page \"our-story\" was not created`) {
			t.Errorf("follow-up result = %+v", r)
		}
	}
	if res[3].IsError {
		t.Errorf("unrelated op = %+v", res[3])
	}
}
