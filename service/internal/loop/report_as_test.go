package loop

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/breadstack/sous-chef/internal/api"
	"github.com/breadstack/sous-chef/internal/model/modeltest"
	"github.com/breadstack/sous-chef/internal/store"
)

// A route's `report_as` (README §5) names the model in everything that
// attributes a call: usage rows, the trace's calls, the turn row, turn.done.
// The harness's "muffin" routes send model "sous-chef" to a server that
// echoes "sous-chef-student"; neither id may appear.

func usageModels(rows []store.UsageRow) string {
	var out []string
	for _, r := range rows {
		out = append(out, r.Model)
	}
	return strings.Join(out, ",")
}

func traceModels(t *testing.T, tr store.Trace) string {
	t.Helper()
	var ops struct {
		Calls []struct{ Route, Model string }
	}
	if err := json.Unmarshal(tr.Ops, &ops); err != nil {
		t.Fatal(err)
	}
	var out []string
	for _, c := range ops.Calls {
		out = append(out, c.Route+"="+c.Model)
	}
	return strings.Join(out, ",")
}

func doneJSON(t *testing.T, ev api.CollectedEvent, key string) string {
	t.Helper()
	if ev.Event != api.EventTurnDone {
		t.Fatalf("last event = %s", ev.Event)
	}
	raw, _ := json.Marshal(data(t, ev)[key])
	return string(raw)
}

// Compat route with report_as, no fallback: the echoed tag is ignored on
// every call of the turn.
func TestReportAs_CompatRoute(t *testing.T) {
	hy := newHybrid(t, []modeltest.Step{reply(setFields("s1", "headline")), reply(modeltest.Text("Done."))}, nil)
	turn, _ := hy.start(hy.req("muffin-solo", "set the hero headline to Fresh cookies"))
	evs, err := hy.cont(turn.ID(), api.ToolResult{ToolCallID: "s1", OK: true, Written: []string{"headline"}})
	if err != nil {
		t.Fatal(err)
	}
	if got := usageModels(hy.st.usageRows()); got != "blueberry-muffin,blueberry-muffin" {
		t.Errorf("usage row models = %s", got)
	}
	tr, _ := hy.st.trace(turn.ID())
	if got := traceModels(t, tr); got != "muffin-solo=blueberry-muffin,muffin-solo=blueberry-muffin" {
		t.Errorf("trace calls = %s", got)
	}
	if got := doneJSON(t, last(evs), "route"); got != `{"fallback":null,"fields_context":true,"model":"blueberry-muffin","name":"muffin-solo"}` {
		t.Errorf("route = %s", got)
	}
	if d := data(t, last(evs)); d["routes"] != nil || d["models"] != nil {
		t.Errorf("a route without a fallback reports routes/models: %v", d)
	}
	if row := hy.st.turn(turn.ID()); row.Model != "blueberry-muffin" {
		t.Errorf("turn row model = %q", row.Model)
	}
	// The request itself still names the route's model.
	if m := hy.student.Requests()[0].Body["model"]; m != "sous-chef" {
		t.Errorf("request model = %v", m)
	}
}

// Without report_as nothing changes: the response's id is what is recorded.
func TestReportAs_UnsetIsUnchanged(t *testing.T) {
	hy := newHybrid(t, []modeltest.Step{reply(modeltest.Text("ok"))}, []modeltest.Step{reply(modeltest.Text("ok"))})
	turn, evs := hy.start(hy.req("solo", "hello"))
	turn2, _ := hy.start(hy.req("default", "hello"))
	if got := usageModels(hy.st.usageRows()); got != "sous-chef-student,claude-sonnet-5-5" {
		t.Errorf("usage row models = %s", got)
	}
	tr, _ := hy.st.trace(turn.ID())
	tr2, _ := hy.st.trace(turn2.ID())
	if got := traceModels(t, tr) + "," + traceModels(t, tr2); got != "solo=sous-chef-student,default=claude-sonnet-5-5" {
		t.Errorf("trace calls = %s", got)
	}
	if got := doneJSON(t, last(evs), "route"); got != `{"fallback":null,"fields_context":true,"model":"sous-chef-student","name":"solo"}` {
		t.Errorf("route = %s", got)
	}
	if row := hy.st.turn(turn.ID()); row.Model != "sous-chef-student" {
		t.Errorf("turn row model = %q", row.Model)
	}
}

// Hybrid: attribution is per call. The student's call reports
// blueberry-muffin, the fallback's calls the Claude model id — in the usage
// rows, the trace, and turn.done (also after a rebuild from the row).
func TestReportAs_HybridPerCall(t *testing.T) {
	hy := newHybrid(t,
		[]modeltest.Step{reply(setFields("s1", "itemDescriptions"))},
		[]modeltest.Step{reply(setFields("c1", "headline")), reply(modeltest.Text("Done."))})
	turn, _ := hy.start(hy.req("muffin", "set the hero headline to Fresh cookies"))
	evs, err := hy.cont(turn.ID(), refusedResult("s1"))
	if err != nil {
		t.Fatal(err)
	}
	if len(events(evs, api.EventTurnEscalated)) != 1 {
		t.Fatalf("no escalation in %s", kinds(evs))
	}
	hy.runner.unpark(turn.ID()) // the last call runs on a turn rebuilt from the row
	evs, err = hy.cont(turn.ID(), api.ToolResult{ToolCallID: "c1", OK: true, Written: []string{"headline"}})
	if err != nil {
		t.Fatal(err)
	}
	if got := usageModels(hy.st.usageRows()); got != "blueberry-muffin,claude-sonnet-5-5,claude-sonnet-5-5" {
		t.Errorf("usage row models = %s", got)
	}
	done := last(evs)
	if got := doneJSON(t, done, "route"); got != `{"fallback":"backstop","fields_context":true,"model":"blueberry-muffin","name":"muffin"}` {
		t.Errorf("route = %s", got)
	}
	if got := doneJSON(t, done, "routes"); got != `["muffin","backstop"]` {
		t.Errorf("routes = %s", got)
	}
	if got := doneJSON(t, done, "models"); got != `{"backstop":"claude-sonnet-5-5","muffin":"blueberry-muffin"}` {
		t.Errorf("models = %s", got)
	}
	if u := string(hy.st.turn(turn.ID()).Usage); !strings.Contains(u, `"models":{"backstop":"claude-sonnet-5-5","muffin":"blueberry-muffin"}`) {
		t.Errorf("row usage = %s", u)
	}
	for _, r := range hy.st.usageRows() {
		if strings.Contains(r.Model, "sous-chef") {
			t.Errorf("a usage row carries the echoed tag: %+v", r)
		}
	}
}

// A hybrid turn whose student call FAILS: the error row is still the
// student's, under its reported name; the fallback's row is Claude's.
func TestReportAs_FailedStudentCall(t *testing.T) {
	hy := newHybrid(t, []modeltest.Step{modeltest.Error(500, "api_error", "gpu on fire", nil)}, []modeltest.Step{reply(modeltest.Text("Done."))})
	_, evs := hy.start(hy.req("muffin", "hello"))
	rows := hy.st.usageRows()
	if len(rows) != 2 || rows[0].Model != "blueberry-muffin" || rows[0].Status != "error" || rows[1].Model != "claude-sonnet-5-5" || rows[1].Status != "ok" {
		t.Fatalf("usage rows = %+v", rows)
	}
	if got := doneJSON(t, last(evs), "models"); got != `{"backstop":"claude-sonnet-5-5","muffin":"blueberry-muffin"}` {
		t.Errorf("models = %s", got)
	}
}

// An anthropic route may set it too; its request body is not touched.
func TestReportAs_AnthropicRoute(t *testing.T) {
	hy := newHybrid(t, nil, []modeltest.Step{reply(modeltest.Text("ok")), reply(modeltest.Text("ok"))})
	_, evs := hy.start(hy.req("named", "hello"))
	hy.start(hy.req("default", "hello"))
	if got := usageModels(hy.st.usageRows()); got != "house-claude,claude-sonnet-5-5" {
		t.Errorf("usage row models = %s", got)
	}
	if got := doneJSON(t, last(evs), "route"); got != `{"fallback":null,"fields_context":false,"model":"house-claude","name":"named"}` {
		t.Errorf("route = %s", got)
	}
	reqs := hy.claude.Requests()
	if len(reqs) != 2 || strings.Contains(string(reqs[0].Raw), "house-claude") || strings.Contains(string(reqs[0].Raw), "report_as") || reqs[0].Body["model"] != "claude-sonnet-5-5" {
		t.Errorf("anthropic request carries the reported name: %s", reqs[0].Raw)
	}
}
