package prompt

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"

	"github.com/breadstack/sous-chef/internal/model"
)

// TurnContext is the per-turn `context` of POST /v1/turns (README §1.2):
// what the editor shows right now. Unknown members are kept on the wire
// (ContextBlock forwards the raw object), these are the ones the service
// reads itself.
type TurnContext struct {
	Page        string       `json:"page"`
	Template    string       `json:"template,omitempty"`
	Layout      string       `json:"layout,omitempty"`
	Sections    []SectionRef `json:"sections,omitempty"`
	Categories  []NamedRef   `json:"categories,omitempty"`
	CustomPages []NamedRef   `json:"customPages,omitempty"`
	Nav         []NavRef     `json:"nav,omitempty"`
	Locations   []NamedRef   `json:"locations,omitempty"`
}

// SectionRef is one section of the viewed page.
type SectionRef struct {
	ID      string `json:"id"`
	Type    string `json:"type,omitempty"`
	Title   string `json:"title,omitempty"`
	Enabled *bool  `json:"enabled,omitempty"`
}

// NamedRef is a category (slug+name), custom page (slug+title) or location (id+name).
type NamedRef struct {
	ID    string `json:"id,omitempty"`
	Slug  string `json:"slug,omitempty"`
	Name  string `json:"name,omitempty"`
	Title string `json:"title,omitempty"`
}

// NavRef is one navigation link.
type NavRef struct {
	Label string `json:"label"`
	Href  string `json:"href,omitempty"`
}

// MaxContextBytes bounds the context object the browser may send.
const MaxContextBytes = 64 << 10

// ParseContext reads the typed members of a context object. An empty or
// absent context is allowed (zero TurnContext).
func ParseContext(raw json.RawMessage) (TurnContext, error) {
	var tc TurnContext
	if len(bytes.TrimSpace(raw)) == 0 {
		return tc, nil
	}
	if err := json.Unmarshal(raw, &tc); err != nil {
		return tc, fmt.Errorf("context: %w", err)
	}
	return tc, nil
}

// ContextBlock is the first text block of a turn's user message: the
// context object, compacted, inside <context> tags, then the merchant's
// text. It sits after the cached prefix, so it may change every turn. A
// context that is not a JSON object is refused; none at all is allowed.
func ContextBlock(raw json.RawMessage, text string) (string, error) {
	return UserText(raw, text, "")
}

// UserText is ContextBlock with the turn's candidate-fields block
// (internal/fields Result.Block) on its own lines between the context and
// the merchant's text:
//
//	<context>\n{…}\n</context>\n<candidate_fields>\n…\n</candidate_fields>\n<text>
//
// An empty (or all-blank) candidates block gives exactly ContextBlock's
// bytes, so a route without the option — and a turn nothing matched — is
// unchanged. The rule that goes with the block ("copy the keys exactly;
// for any other key call lookup_fields first") is the block's own header
// line, not a system rule: system[0] stays byte-identical on every route.
func UserText(raw json.RawMessage, text, candidates string) (string, error) {
	if candidates = strings.TrimSpace(candidates); candidates != "" {
		candidates += "\n"
	}
	text = strings.TrimSpace(text)
	if text == "" {
		return "", errors.New("turn text is empty")
	}
	raw = bytes.TrimSpace(raw)
	if len(raw) == 0 || string(raw) == "null" {
		return candidates + text, nil
	}
	if len(raw) > MaxContextBytes {
		return "", fmt.Errorf("context exceeds %d bytes", MaxContextBytes)
	}
	if raw[0] != '{' || !json.Valid(raw) {
		return "", errors.New("context must be a JSON object")
	}
	var compact bytes.Buffer
	if err := json.Compact(&compact, raw); err != nil {
		return "", fmt.Errorf("context: %w", err)
	}
	return "<context>\n" + compact.String() + "\n</context>\n" + candidates + text, nil
}

// UserTurn is the user message that starts a turn; candidates is the
// candidate-fields block, "" for none.
func UserTurn(contextRaw json.RawMessage, text, candidates string) (anthropic.BetaMessageParam, error) {
	block, err := UserText(contextRaw, text, candidates)
	if err != nil {
		return anthropic.BetaMessageParam{}, err
	}
	return model.UserTextTurn(block), nil
}

// Prefix is what every call on one manifest shares: the rules, the index
// and the tool list in its fixed order. Build renders it identically every
// time so the cache breakpoints hit.
type Prefix struct {
	Rules string // Rules, unless a test overrides it
	Index string // Index.Text for the turn's manifest hash
	Tools []anthropic.BetaToolUnionParam
}

// NewPrefix pairs the frozen rules with a rendered index and the tools.
func NewPrefix(index Index, tools []anthropic.BetaToolUnionParam) Prefix {
	return Prefix{Rules: Rules, Index: index.Text, Tools: tools}
}

// System renders system[0] (rules) and system[1] (index), each with a
// cache_control breakpoint (README §3).
func (p Prefix) System() []anthropic.BetaTextBlockParam {
	return []anthropic.BetaTextBlockParam{
		{Text: p.Rules, CacheControl: anthropic.NewBetaCacheControlEphemeralParam()},
		{Text: p.Index, CacheControl: anthropic.NewBetaCacheControlEphemeralParam()},
	}
}

// Bytes is the prefix as it goes on the wire (tools + system), for
// stability checks: two turns on the same manifest must produce equal
// bytes.
func (p Prefix) Bytes() ([]byte, error) {
	return json.Marshal(struct {
		Tools  []anthropic.BetaToolUnionParam `json:"tools"`
		System []anthropic.BetaTextBlockParam `json:"system"`
	}{p.Tools, p.System()})
}

// Build assembles a model request: the prefix, the stored history as it
// was (append-only, never edited), and the new user turn with its context
// block. history may be nil for a fresh conversation.
func Build(p Prefix, history []anthropic.BetaMessageParam, contextRaw json.RawMessage, text string) (model.Request, error) {
	return BuildWithFields(p, history, contextRaw, text, "")
}

// BuildWithFields is Build with the turn's candidate-fields block in the
// new user turn (UserText). Only the NEW turn gets one: the history is
// replayed as stored, with whatever block each earlier turn was sent.
func BuildWithFields(p Prefix, history []anthropic.BetaMessageParam, contextRaw json.RawMessage, text, candidates string) (model.Request, error) {
	if strings.TrimSpace(p.Rules) == "" || strings.TrimSpace(p.Index) == "" {
		return model.Request{}, errors.New("prompt: rules and index are required")
	}
	if len(p.Tools) == 0 {
		return model.Request{}, errors.New("prompt: no tools")
	}
	turn, err := UserTurn(contextRaw, text, candidates)
	if err != nil {
		return model.Request{}, err
	}
	msgs := make([]anthropic.BetaMessageParam, 0, len(history)+1)
	msgs = append(msgs, history...)
	msgs = append(msgs, turn)
	return model.Request{System: p.System(), Tools: p.Tools, Messages: msgs}, nil
}

// Continue assembles the request that follows a batch of tool results on
// the same turn: the prefix, the history up to and including the assistant
// turn that asked, and the results as one user message.
func Continue(p Prefix, history []anthropic.BetaMessageParam, results []model.ToolResult) (model.Request, error) {
	if len(results) == 0 {
		return model.Request{}, errors.New("prompt: no tool results")
	}
	if len(history) == 0 {
		return model.Request{}, errors.New("prompt: tool results need a history")
	}
	msgs := make([]anthropic.BetaMessageParam, 0, len(history)+1)
	msgs = append(msgs, history...)
	msgs = append(msgs, model.ToolResultsTurn(results))
	return model.Request{System: p.System(), Tools: p.Tools, Messages: msgs}, nil
}
