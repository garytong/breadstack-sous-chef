package loop

import (
	"context"
	"encoding/json"
	"errors"
	"log"
	"strconv"

	"github.com/breadstack/sous-chef/internal/api"
	"github.com/breadstack/sous-chef/internal/store"
)

// The remaining api ports are thin adapters over the store; they live here
// because conversations, feedback and the opt-out flag are the turn's
// companions and share its storage.

// ConvStore is the slice of *store.Store Conversations needs.
type ConvStore interface {
	FindOpenConversation(ctx context.Context, siteID string) (*store.Conversation, error)
	ListMessages(ctx context.Context, conversationID string, beforeSeq, limit int) ([]store.Message, error)
	ListTurns(ctx context.Context, conversationID string) ([]store.Turn, error)
	// EndSiteConversation supersedes the site's live turns and ends its open
	// conversation in one per-site-serialized transaction (store.Store).
	EndSiteConversation(ctx context.Context, siteID string) (*store.EndedConversation, error)
}

// Conversations implements api.ConversationStore.
type Conversations struct {
	St ConvStore
	// Logf receives one line per turn End supersedes; nil → log.Printf.
	Logf func(string, ...any)
}

// Get returns the site's open conversation with its last 200 messages (or
// those before `before`) and its turns; 404 conversation_not_found when the
// site has none open.
func (c Conversations) Get(ctx context.Context, siteID string, before int) (api.Conversation, error) {
	conv, err := c.St.FindOpenConversation(ctx, siteID)
	if errors.Is(err, store.ErrNotFound) {
		return api.Conversation{}, api.ErrNotFound("conversation_not_found", "the site has no open conversation")
	}
	if err != nil {
		return api.Conversation{}, err
	}
	msgs, err := c.St.ListMessages(ctx, conv.ID, before, 200)
	if err != nil {
		return api.Conversation{}, err
	}
	turns, err := c.St.ListTurns(ctx, conv.ID)
	if err != nil {
		return api.Conversation{}, err
	}
	out := api.Conversation{ConversationID: conv.ID, StoreID: conv.StoreID, Messages: []api.StoredMessage{}, Turns: []json.RawMessage{}}
	for _, m := range msgs {
		out.Messages = append(out.Messages, api.StoredMessage{Seq: m.Seq, Role: m.Role, Content: m.Content})
	}
	for _, t := range turns {
		raw, _ := json.Marshal(map[string]any{
			"id": t.ID, "status": t.Status, "manifestHash": t.ManifestHash, "buildId": t.BuildID,
			"model": t.Model, "userEmail": t.UserEmail, "started": t.Started, "ended": t.Ended, "usage": t.Usage,
		})
		out.Turns = append(out.Turns, raw)
	}
	return out, nil
}

// End implements api.ConversationStore.End: the site's live turns are
// superseded and the open conversation ended in one transaction, serialized
// with the site's Starts (README §4 — a turn still waiting on, or running
// in, the conversation being ended must not continue late and write into
// it; its tool-results are 409 status=superseded, which the launcher
// drops; a Start racing the DELETE either opens a fresh conversation after
// it or is superseded by it).
func (c Conversations) End(ctx context.Context, siteID string) error {
	out, err := c.St.EndSiteConversation(ctx, siteID)
	if err != nil {
		return err
	}
	logf := c.Logf
	if logf == nil {
		logf = log.Printf
	}
	for _, e := range out.Superseded {
		logf("loop: site %s: turn %s (%s, conversation %s) superseded by DELETE /v1/conversations", siteID, e.ID, e.WasStatus, e.ConversationID)
	}
	return nil
}

// FeedbackStore is the slice of *store.Store Feedback needs.
type FeedbackStore interface {
	MergeFeedback(ctx context.Context, turnID string, feedback json.RawMessage) error
	GetTurn(ctx context.Context, id string) (*store.Turn, error)
	GetConversation(ctx context.Context, id string) (*store.Conversation, error)
}

// Feedback implements api.FeedbackSink. Use it by pointer: it keeps the
// once-per-turn set behind the "body carries no siteId" log line.
type Feedback struct {
	St FeedbackStore
	// Logf receives the once-per-turn note for a body without siteId; nil →
	// log.Printf.
	Logf func(string, ...any)

	noSite noSiteLog
}

// Record merges the body (minus turnId, siteId, storeId) into
// traces.feedback. Whenever the body names a siteId or a storeId, the
// turn's conversation must belong to it — 404 turn_not_found otherwise,
// also when the turn does not exist (README §1, §11); a body with neither
// is accepted and logged once per turn, as on tool-results. A turn without
// a trace (opted-out store, pruned) is not an error: the handler answers
// 204 either way. A merge that would push the turn's feedback document past
// store.MaxFeedbackBytes (16 KiB) is a 400 `bad_request`.
func (f *Feedback) Record(ctx context.Context, fb api.Feedback) error {
	if err := f.verifySite(ctx, fb); err != nil {
		return err
	}
	doc := map[string]any{}
	if fb.KeptPaths != nil {
		doc["kept_paths"] = fb.KeptPaths
	}
	if fb.Undone != nil {
		doc["undone"] = *fb.Undone
	}
	if fb.Saved != nil {
		doc["saved"] = *fb.Saved
	}
	if fb.Note != "" {
		doc["note"] = fb.Note
	}
	if len(doc) == 0 {
		return nil
	}
	raw, _ := json.Marshal(doc)
	if len(raw) > store.MaxFeedbackBytes {
		return api.ErrBadRequest("feedback too large: merged feedback per turn is capped at " + strconv.Itoa(store.MaxFeedbackBytes) + " bytes")
	}
	err := f.St.MergeFeedback(ctx, fb.TurnID, raw)
	switch {
	case errors.Is(err, store.ErrNotFound):
		return nil
	case errors.Is(err, store.ErrFeedbackTooLarge):
		return api.ErrBadRequest("feedback too large: merged feedback per turn is capped at " + strconv.Itoa(store.MaxFeedbackBytes) + " bytes")
	}
	return err
}

func (f *Feedback) verifySite(ctx context.Context, fb api.Feedback) error {
	logf := f.Logf
	if logf == nil {
		logf = log.Printf
	}
	if fb.SiteID == "" && fb.StoreID == "" {
		// No row read: the note is the whole check, once per turn.
		if f.noSite.first(fb.TurnID) {
			logf("loop: turn %s: feedback body carries no siteId; accepted unverified (the proxy should send it)", fb.TurnID)
		}
		return nil
	}
	row, err := f.St.GetTurn(ctx, fb.TurnID)
	if errors.Is(err, store.ErrNotFound) {
		return api.ErrTurnNotFound(fb.TurnID)
	}
	if err != nil {
		return err
	}
	return verifyTenant(ctx, f.St, &f.noSite, logf, row, fb.SiteID, fb.StoreID, "feedback")
}

// OptOutStore is the slice of *store.Store OptOut needs.
type OptOutStore interface {
	SetOptOut(ctx context.Context, storeID, byEmail string) error
	ClearOptOut(ctx context.Context, storeID string) (bool, error)
}

// OptOut implements api.OptOutStore.
type OptOut struct{ St OptOutStore }

func (o OptOut) Set(ctx context.Context, storeID, byEmail string) error {
	if storeID == "" {
		return api.ErrBadRequest("storeId is required")
	}
	return o.St.SetOptOut(ctx, storeID, byEmail)
}

func (o OptOut) Clear(ctx context.Context, storeID string) error {
	_, err := o.St.ClearOptOut(ctx, storeID)
	return err
}
