package loop

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"

	"github.com/breadstack/sous-chef/internal/api"
	"github.com/breadstack/sous-chef/internal/toolgen"
)

// Fast finish (README §4 "Fast finish"; route option `fast_finish`): a turn
// whose batch of browser ops all came back clean ends without the closing
// model call. Measured on int (2026-10-10 exams, 480 turns per route): the
// rule fired on up to 66% of the in-house model's turns and 74% of Claude's
// in the lab harness — less in production, where the launcher echoes every
// section-targeted op's own sectionId (about 15% fewer for the student) —
// and in none of those turns did the model call another tool after the
// batch: the closing call only wrote the "done" sentence (GPU median
// 1.18 s; Claude about 1.5–3 s). Multi-part requests can still be split
// across two clean batches (1 of 19 multi-part teacher cases), and the
// in-house model's closing reply added a caveat or a question in 7% of
// turns, Claude's in 25%, so the option is per route and off unless the
// route says "on".

// fastFinishReply is the service's own closing line after a clean batch of n
// browser ops (text.delta `source:"service"`). Ops land in the merchant's
// unsaved draft — not necessarily the page in view — so the line does not
// claim the change "shows in the preview".
func fastFinishReply(n int) string {
	if n == 1 {
		return "Done — that change is in your unsaved draft. Review it and save when you're happy."
	}
	return fmt.Sprintf("Done — those %d changes are in your unsaved draft. Review them and save when you're happy.", n)
}

// cleanBatch judges the waiting batch against the results just posted: ops
// is the number of browser ops in it, clean whether the turn may end without
// the closing call as far as the batch goes — every call a browser op the
// service knows and showed (no server tool, no report_unsupported, no
// pick_asset, no op the service refused) other than an edit_list add or
// duplicate (the new row's id or index never reaches the model, and the
// follow-up step — describing or filling the new section — needs it), and
// every one answered ok:true with no warnings, no `hidden` entries, no new
// `sectionId` and no asset. A result's sectionId equal to the op's own
// target.sectionId / sectionId is the launcher's echo of the input, not
// news.
func (t *turn) cleanBatch(answered map[string]api.ToolResult) (ops int, clean bool) {
	if t.pending == nil || len(t.pending.Calls) == 0 {
		return 0, false
	}
	for _, c := range t.pending.Calls {
		if c.Server {
			return 0, false
		}
		if runs, known := toolgen.RunsWhere(c.Name); !known || runs != toolgen.RunsBrowser {
			return 0, false
		}
		in := parseOpInput(c.Input)
		if c.Name == toolgen.EditList && (in.Action == "add" || in.Action == "duplicate") {
			return 0, false
		}
		res, ok := answered[c.ID]
		if !ok || !res.OK || res.Error != nil || len(res.Warnings) > 0 || hasHidden(res.Hidden) || res.Asset != nil {
			return 0, false
		}
		if res.SectionID != "" && res.SectionID != in.Target.SectionID && res.SectionID != in.SectionID {
			return 0, false
		}
		ops++
	}
	return ops, true
}

// opInput is the part of a browser op's input cleanBatch reads.
type opInput struct {
	Action    string `json:"action"`
	SectionID string `json:"sectionId"`
	Target    struct {
		SectionID string `json:"sectionId"`
	} `json:"target"`
}

// parseOpInput reads opInput from a call's input; a malformed input reads
// as empty (the service refused such an op before showing it).
func parseOpInput(raw json.RawMessage) opInput {
	var in opInput
	_ = json.Unmarshal(raw, &in)
	return in
}

// hasHidden reports whether a result's `hidden` member says anything: any
// value but absent, null or an empty array counts (a malformed member too —
// the closing call is the safe side).
func hasHidden(raw json.RawMessage) bool {
	raw = bytes.TrimSpace(raw)
	if len(raw) == 0 || string(raw) == "null" {
		return false
	}
	var items []json.RawMessage
	if json.Unmarshal(raw, &items) == nil && len(items) == 0 {
		return false
	}
	return true
}

// decideFastFinish (resume, once the batch's results are stored) arms the
// fast finish: the route says `fast_finish:"on"`, the batch is clean
// (cleanBatch), no escalation is pending (a refused op or a hidden write in
// this batch) and the turn has not escalated, no earlier results message of
// the turn carried an error (hadErrorBefore), and a further call would have
// been allowed. Otherwise the closing call runs as always.
func (t *turn) decideFastFinish(ops int, clean, hadErrorBefore bool) {
	t.fastFinish = 0
	if !clean || ops == 0 || !t.backend.FastFinish() || t.escalated != nil || t.wantEscalate != "" ||
		hadErrorBefore || t.calls >= t.r.d.MaxCalls {
		return
	}
	t.fastFinish = ops
}

// finishFast ends an armed turn without a model call: a trace-only
// turn.fast_finish event, the service's closing line as a text.delta marked
// `source:"service"` (the lab's collector must not read it as model output),
// then turn.done through the usual path, which adds `fast_finish:true` to
// the event and the stored usage. Nothing is written to the conversation:
// it ends `assistant(ops) → user(tool_results)`, and the next turn's text is
// sent merged into that results message (turn.sendable, MergeUserRuns).
func (t *turn) finishFast(ctx context.Context, em api.Emitter) error {
	n := t.fastFinish
	t.fastFinished = true
	t.seq++ // trace-only, like a tool.result: the stream's seq skips it
	if t.rec != nil {
		raw, _ := json.Marshal(map[string]any{"ops": n, "route": t.backend.Name(), "calls": t.calls})
		t.rec.Event(t.seq, EventFastFinish, raw)
	}
	t.r.d.Logf("loop: turn %s: fast finish after %d clean op(s) on route %s (%d call(s))", t.id, n, t.backend.Name(), t.calls)
	t.emit(em, api.EventTextDelta, map[string]string{"text": fastFinishReply(n), "source": api.TextSourceService})
	return t.done(ctx, em, StopEndTurn)
}

// EventFastFinish is the trace event (traces.events) of a turn that ended
// without its closing model call; it is never streamed.
const EventFastFinish = "turn.fast_finish"
