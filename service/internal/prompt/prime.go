package prompt

import (
	"errors"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"

	"github.com/breadstack/sous-chef/internal/model"
)

// PrimeFiller is the filler text of a prime's user message: n "x" joined by
// single spaces, one token each on the Qwen tokenizer ("x", then " x").
func PrimeFiller(n int) string {
	if n < 1 {
		n = 1
	}
	return strings.TrimSuffix(strings.Repeat("x ", n), " ")
}

// PrimeRequest is the request that primes the model server's prompt cache
// for the NEXT turn of a conversation (internal/model Primer, README §4
// "Prompt-cache priming"): the prefix, the history as that turn will send
// it, and a user message of `filler` tokens in place of its `<context>` and
// text. history nil = the floor prime (a new conversation's first turn).
//
// It is built exactly as the loop's turn.sendable builds a real request on
// a messages-compatible route — StableHistory, then MergeUserRuns — so the
// rendered prompt is byte-identical to the next real one up to that user
// message's `<|im_start|>user\n`: a history that ends in a tool_result-only
// message (a fast-finished turn) gets the filler merged after the results,
// as the next turn's text will be.
func PrimeRequest(p Prefix, history []anthropic.BetaMessageParam, filler int) (model.Request, error) {
	if strings.TrimSpace(p.Rules) == "" || strings.TrimSpace(p.Index) == "" {
		return model.Request{}, errors.New("prompt: rules and index are required")
	}
	if len(p.Tools) == 0 {
		return model.Request{}, errors.New("prompt: no tools")
	}
	msgs := make([]anthropic.BetaMessageParam, 0, len(history)+1)
	msgs = append(msgs, history...)
	msgs = append(msgs, model.UserTextTurn(PrimeFiller(filler)))
	msgs = MergeUserRuns(StableHistory(msgs))
	return model.Request{System: p.System(), Tools: p.Tools, Messages: msgs}, nil
}
