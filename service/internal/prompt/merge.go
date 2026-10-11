package prompt

import "github.com/anthropics/anthropic-sdk-go"

// MergeUserRuns joins every user message into the user message before it
// when that earlier one holds no text — a tool_result-only message: the
// results of a batch the turn ended on without a closing reply (a fast
// finish, README §4), or the is_error answers of an abandoned or repaired
// batch, followed by the next turn's user text. The request then
// alternates roles, and its user message reads "tool_result blocks first,
// then text", the shape the Messages API documents for a reply to tool use.
//
// Deterministic, so the prefix a later call or turn shares with this one is
// the same bytes every time. Storage is untouched (append-only: the results
// and the text stay two rows); the loop applies it at send time
// (turn.sendable), after StableHistory, never in Build — Start persists the
// last message of the built request, which must stay the turn's own text.
//
// Two user messages that both carry text are left apart: Claude merges them
// itself, and a messages-compatible server renders them as two user turns
// (Ollama joins a message's text blocks with no separator, so merging would
// change what the in-house model is shown). On such a server a tool_result
// block becomes a `tool` message ahead of the message's text whether merged
// or not, so this merge does not change that prompt at all.
//
// The caller's slices are not modified; msgs is returned as is when nothing
// merges.
func MergeUserRuns(msgs []anthropic.BetaMessageParam) []anthropic.BetaMessageParam {
	merges := false
	for i := 1; i < len(msgs); i++ {
		if mergesInto(msgs[i-1], msgs[i]) {
			merges = true
			break
		}
	}
	if !merges {
		return msgs
	}
	out := make([]anthropic.BetaMessageParam, 0, len(msgs))
	for _, m := range msgs {
		if n := len(out); n > 0 && mergesInto(out[n-1], m) {
			prev := out[n-1]
			content := make([]anthropic.BetaContentBlockParamUnion, 0, len(prev.Content)+len(m.Content))
			content = append(content, prev.Content...)
			content = append(content, m.Content...)
			prev.Content = content
			out[n-1] = prev
			continue
		}
		out = append(out, m)
	}
	return out
}

// mergesInto reports whether m joins prev: both user messages, prev with no
// text block.
func mergesInto(prev, m anthropic.BetaMessageParam) bool {
	return prev.Role == anthropic.BetaMessageParamRoleUser && m.Role == anthropic.BetaMessageParamRoleUser &&
		!hasText(prev.Content)
}
