package loop

import (
	"encoding/json"

	"github.com/breadstack/sous-chef/internal/api"
	"github.com/breadstack/sous-chef/internal/model"
	"github.com/breadstack/sous-chef/internal/toolgen"
)

// pendingDoc is the turns.pending_tool_calls column while a turn waits:
// everything a continuation on another instance needs besides the history
// (which is in `messages`) and the manifest (by hash on the turn row).
type pendingDoc struct {
	Route         string         `json:"route"`
	Calls         []pendingCall  `json:"calls"` // every tool_use of the batch, in order
	ServerResults []serverResult `json:"server_results,omitempty"`
	CallCount     int            `json:"call_count"`
	Usage         model.Usage    `json:"usage"`
	Seq           int            `json:"seq"`
	SuspendedID   string         `json:"suspended_id,omitempty"`
	Rotated       bool           `json:"rotated,omitempty"`
}

// pendingCall is one tool_use block of the waiting batch.
type pendingCall struct {
	ID     string          `json:"id"`
	Name   string          `json:"name"`
	Input  json.RawMessage `json:"input"`
	Server bool            `json:"server,omitempty"` // answered inline; result in ServerResults
	Seq    int             `json:"seq,omitempty"`    // the tool.call (or turn.suspended) event's seq
}

// serverResult is a server tool's answer stashed until the batch completes.
type serverResult struct {
	ID      string `json:"id"`
	Content string `json:"content"`
	IsError bool   `json:"is_error,omitempty"`
}

// browserIDs lists the calls the browser must answer (ops and pick_asset).
func (p *pendingDoc) browserIDs() []string {
	ids := []string{}
	for _, c := range p.Calls {
		if !c.Server {
			ids = append(ids, c.ID)
		}
	}
	return ids
}

// browser finds a browser-answered call by id.
func (p *pendingDoc) browser(id string) (pendingCall, bool) {
	for _, c := range p.Calls {
		if c.ID == id && !c.Server {
			return c, true
		}
	}
	return pendingCall{}, false
}

// samePending reports whether two pending docs describe the same waiting
// batch of the same turn: same seq and call count, same tool_use ids in the
// same order. Used to tell a parked entry from a row another instance has
// moved on (a continuation there parks the turn again with a new batch).
func samePending(a, b *pendingDoc) bool {
	if a == nil || b == nil || a.Seq != b.Seq || a.CallCount != b.CallCount || len(a.Calls) != len(b.Calls) || a.SuspendedID != b.SuspendedID {
		return false
	}
	for i := range a.Calls {
		if a.Calls[i].ID != b.Calls[i].ID {
			return false
		}
	}
	return true
}

func (p *pendingDoc) serverResult(id string) (serverResult, bool) {
	for _, r := range p.ServerResults {
		if r.ID == id {
			return r, true
		}
	}
	return serverResult{}, false
}

// suspendedEvent is the turn.suspended payload (README §1.2).
func (p *pendingDoc) suspendedEvent() map[string]any {
	ev := map[string]any{"reason": "pick_asset", "toolCallId": p.SuspendedID, "assetType": "image"}
	for _, c := range p.Calls {
		if c.ID != p.SuspendedID {
			continue
		}
		var in struct {
			Purpose   string `json:"purpose"`
			AssetType string `json:"asset_type"`
		}
		_ = json.Unmarshal(c.Input, &in)
		ev["purpose"] = in.Purpose
		if in.AssetType != "" {
			ev["assetType"] = in.AssetType
		}
	}
	// Other ops in the same batch must be applied too.
	if ids := p.browserIDs(); len(ids) > 1 {
		ev["toolCallIds"] = ids
	}
	return ev
}

// browserResult turns one browser-reported outcome into the tool_result the
// model reads: the engine's {code,message,key} with is_error on failure; for
// pick_asset the chosen asset {public_url, alt, srcset}; otherwise
// {ok, written, warnings, sectionId}.
func browserResult(name string, res api.ToolResult) model.ToolResult {
	if !res.OK {
		e := res.Error
		if e == nil {
			e = &api.ToolError{Code: "E_OP", Message: "the browser reported a failure without details"}
		}
		b, _ := json.Marshal(e)
		return model.ToolResult{ID: res.ToolCallID, Content: string(b), IsError: true}
	}
	if name == toolgen.PickAsset {
		if res.Asset == nil || res.Asset.PublicURL == "" {
			return model.ToolResult{ID: res.ToolCallID, Content: errJSON("E_ASSET", "no asset was chosen"), IsError: true}
		}
		b, _ := json.Marshal(res.Asset)
		return model.ToolResult{ID: res.ToolCallID, Content: string(b)}
	}
	out := map[string]any{"ok": true}
	if len(res.Written) > 0 {
		out["written"] = res.Written
	}
	if len(res.Warnings) > 0 {
		out["warnings"] = res.Warnings
	}
	if res.SectionID != "" {
		out["sectionId"] = res.SectionID
	}
	b, _ := json.Marshal(out)
	return model.ToolResult{ID: res.ToolCallID, Content: string(b)}
}
