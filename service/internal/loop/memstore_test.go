package loop

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"sync"
	"time"

	"github.com/breadstack/sous-chef/internal/store"
)

// memStore is an in-memory stand-in for *store.Store covering every
// interface the loop, its adapters and the usage recorder need. Semantics
// mirror the SQL (dense seq per conversation, status-guarded transitions,
// ErrNotFound on misses).
type memStore struct {
	mu     sync.Mutex
	now    func() time.Time
	nextID int

	// fault injection
	failAppends int  // the next N AppendMessage calls fail
	failDB      bool // GetTurn and ExpireStaleTurnsWithPending fail
	// failGetTurns fails the next N GetTurn calls (a transient blip at one
	// consulted site); unlike failDB it clears itself.
	failGetTurns int
	// afterAppend runs after a successful append with the lock released: a
	// hook for interleaving another actor (the reaper) between two writes.
	afterAppend func(role string)
	// afterResume runs after a successful ResumeTurn, lock released: the
	// reaper slipping in between the resume and the results write.
	afterResume func()
	// afterGetTurn runs after a successful GetTurn, lock released: the
	// reaper slipping in between the loop's pre-call row check and the
	// model call (a slow model, a reaper on another instance).
	afterGetTurn func(id string)
	// afterStartTurn runs after a successful StartTurn with the lock
	// released: another Start slipping in between this Start's storage block
	// and its history read (the only window left before the user text).
	afterStartTurn func(id string)

	convs    map[string]*store.Conversation
	messages map[string][]store.Message
	turns    map[string]*store.Turn
	order    map[string]int // turn id → creation ordinal (= commit order of the Starts)
	traces   map[string]store.Trace
	optout   map[string]bool
	usage    []store.UsageRow
}

func newMemStore() *memStore {
	return &memStore{
		now:      func() time.Time { return time.Now().UTC() },
		convs:    map[string]*store.Conversation{},
		messages: map[string][]store.Message{},
		turns:    map[string]*store.Turn{},
		order:    map[string]int{},
		traces:   map[string]store.Trace{},
		optout:   map[string]bool{},
	}
}

func (s *memStore) id(prefix string) string {
	s.nextID++
	return fmt.Sprintf("%s-%d", prefix, s.nextID)
}

// --- conversations ---

func (s *memStore) openLocked(siteID string) *store.Conversation {
	for _, c := range s.convs {
		if c.SiteID == siteID && c.EndedAt == nil {
			return c
		}
	}
	return nil
}

func (s *memStore) OpenConversation(_ context.Context, siteID, storeID string) (*store.Conversation, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if c := s.openLocked(siteID); c != nil {
		return c, nil
	}
	c := &store.Conversation{ID: s.id("conv"), SiteID: siteID, StoreID: storeID, Created: s.now(), Updated: s.now()}
	s.convs[c.ID] = c
	return c, nil
}

func (s *memStore) FindOpenConversation(_ context.Context, siteID string) (*store.Conversation, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if c := s.openLocked(siteID); c != nil {
		return c, nil
	}
	return nil, store.ErrNotFound
}

func (s *memStore) GetConversation(_ context.Context, id string) (*store.Conversation, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	c, ok := s.convs[id]
	if !ok {
		return nil, store.ErrNotFound
	}
	return c, nil
}

func (s *memStore) RotateConversation(_ context.Context, siteID, storeID string) (*store.Conversation, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	now := s.now()
	if c := s.openLocked(siteID); c != nil {
		c.EndedAt = &now
	}
	c := &store.Conversation{ID: s.id("conv"), SiteID: siteID, StoreID: storeID, Created: now, Updated: now}
	s.convs[c.ID] = c
	return c, nil
}

func (s *memStore) EndConversation(_ context.Context, siteID string) (bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if c := s.openLocked(siteID); c != nil {
		now := s.now()
		c.EndedAt = &now
		return true, nil
	}
	return false, nil
}

func (s *memStore) CountMessages(_ context.Context, convID string) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.messages[convID]), nil
}

func (s *memStore) ListMessages(_ context.Context, convID string, before, limit int) ([]store.Message, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if limit <= 0 {
		limit = 200
	}
	var out []store.Message
	for _, m := range s.messages[convID] {
		if before <= 0 || m.Seq < before {
			out = append(out, m)
		}
	}
	if len(out) > limit {
		out = out[len(out)-limit:]
	}
	return append([]store.Message{}, out...), nil
}

func (s *memStore) AppendMessage(ctx context.Context, convID, role string, content json.RawMessage) (int, error) {
	return s.AppendMessageAfter(ctx, convID, role, content, -1)
}

// AppendMessageAfter mirrors the SQL: afterSeq < 0 is an unconditional
// append; otherwise the tail must still be afterSeq or ErrConflict.
func (s *memStore) AppendMessageAfter(_ context.Context, convID, role string, content json.RawMessage, afterSeq int) (int, error) {
	s.mu.Lock()
	if s.failAppends > 0 {
		s.failAppends--
		s.mu.Unlock()
		return 0, errors.New("memstore: append failed (injected)")
	}
	c, ok := s.convs[convID]
	if !ok {
		s.mu.Unlock()
		return 0, store.ErrNotFound
	}
	if c.EndedAt != nil {
		s.mu.Unlock()
		return 0, fmt.Errorf("memstore: append message: conversation %s: %w", convID, store.ErrConversationEnded)
	}
	if afterSeq >= 0 && len(s.messages[convID]) != afterSeq {
		s.mu.Unlock()
		return 0, store.ErrConflict
	}
	seq := len(s.messages[convID]) + 1
	s.messages[convID] = append(s.messages[convID], store.Message{ConversationID: convID, Seq: seq, Role: role, Content: append(json.RawMessage{}, content...), Created: s.now()})
	hook := s.afterAppend
	s.mu.Unlock()
	if hook != nil {
		hook(role)
	}
	return seq, nil
}

// --- turns ---

func (s *memStore) CreateTurn(_ context.Context, in store.NewTurn) (*store.Turn, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	t := &store.Turn{ID: s.id("turn"), ConversationID: in.ConversationID, Status: store.TurnRunning, ManifestHash: in.ManifestHash,
		BuildID: in.BuildID, Model: in.Model, UserEmail: in.UserEmail, Started: s.now()}
	s.turns[t.ID] = t
	s.order[t.ID] = len(s.order) + 1
	cp := *t
	return &cp, nil
}

// StartTurn mirrors store.Store.StartTurn: one critical section (the
// memstore's mutex is coarser than the SQL's per-site advisory lock, which
// is the same thing for one site) resolving and rotating the conversation,
// creating the row and superseding every other live turn of the site. The
// afterStartTurn hook fires with the lock released.
func (s *memStore) StartTurn(_ context.Context, in store.StartTurn) (*store.StartedTurn, error) {
	if in.SiteID == "" {
		return nil, errors.New("memstore: siteID required")
	}
	s.mu.Lock()
	if s.failDB {
		s.mu.Unlock()
		return nil, errors.New("memstore: database unavailable")
	}
	now := s.now()
	out := &store.StartedTurn{}
	if in.ConversationID != "" {
		c, ok := s.convs[in.ConversationID]
		switch {
		case !ok || c.SiteID != in.SiteID:
			s.mu.Unlock()
			return nil, store.ErrNotFound
		case c.EndedAt != nil:
			s.mu.Unlock()
			return nil, store.ErrConversationEnded
		}
		out.Conversation = c
	} else {
		c := s.openLocked(in.SiteID)
		if c == nil {
			c = &store.Conversation{ID: s.id("conv"), SiteID: in.SiteID, StoreID: in.StoreID, Created: now, Updated: now}
			s.convs[c.ID] = c
		}
		out.Conversation = c
	}
	if in.RotateAt > 0 && len(s.messages[out.Conversation.ID]) >= in.RotateAt {
		if c := s.openLocked(in.SiteID); c != nil {
			ended := now
			c.EndedAt = &ended
		}
		c := &store.Conversation{ID: s.id("conv"), SiteID: in.SiteID, StoreID: in.StoreID, Created: now, Updated: now}
		s.convs[c.ID] = c
		out.Conversation, out.Rotated = c, true
	}
	t := &store.Turn{ID: s.id("turn"), ConversationID: out.Conversation.ID, Status: store.TurnRunning, ManifestHash: in.ManifestHash,
		BuildID: in.BuildID, Model: in.Model, UserEmail: in.UserEmail, Started: now}
	s.turns[t.ID] = t
	s.order[t.ID] = len(s.order) + 1
	cp := *t
	out.Turn = &cp
	out.Superseded = s.supersedeLocked(func(o *store.Turn) bool {
		c, ok := s.convs[o.ConversationID]
		return ok && c.SiteID == in.SiteID && o.ID != t.ID
	})
	hook := s.afterStartTurn
	s.mu.Unlock()
	if hook != nil {
		hook(t.ID)
	}
	return out, nil
}

// EndSiteConversation mirrors store.Store.EndSiteConversation: supersede
// the site's live turns and end its open conversation, atomically.
func (s *memStore) EndSiteConversation(_ context.Context, siteID string) (*store.EndedConversation, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := &store.EndedConversation{}
	out.Superseded = s.supersedeLocked(func(t *store.Turn) bool {
		c, ok := s.convs[t.ConversationID]
		return ok && c.SiteID == siteID
	})
	if c := s.openLocked(siteID); c != nil {
		now := s.now()
		c.EndedAt = &now
		out.Ended = true
	}
	return out, nil
}

func (s *memStore) GetTurn(_ context.Context, id string) (*store.Turn, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.failDB {
		return nil, errors.New("memstore: database unavailable")
	}
	if s.failGetTurns > 0 {
		s.failGetTurns--
		return nil, errors.New("memstore: transient database error")
	}
	t, ok := s.turns[id]
	if !ok {
		return nil, store.ErrNotFound
	}
	cp := *t
	hook := s.afterGetTurn
	s.mu.Unlock()
	if hook != nil {
		hook(id)
	}
	s.mu.Lock()
	return &cp, nil
}

func (s *memStore) SetTurnWaiting(_ context.Context, id, status string, pending json.RawMessage) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	t, ok := s.turns[id]
	if !ok || t.Status != store.TurnRunning {
		return store.ErrNotFound
	}
	t.Status = status
	t.PendingToolCalls = append(json.RawMessage{}, pending...)
	return nil
}

func (s *memStore) ResumeTurn(_ context.Context, id string) (json.RawMessage, error) {
	s.mu.Lock()
	t, ok := s.turns[id]
	if !ok || (t.Status != store.TurnWaitingTools && t.Status != store.TurnSuspended) {
		s.mu.Unlock()
		return nil, store.ErrNotFound
	}
	p := t.PendingToolCalls
	t.Status = store.TurnRunning
	t.PendingToolCalls = nil
	hook := s.afterResume
	s.mu.Unlock()
	if hook != nil {
		hook()
	}
	return p, nil
}

func (s *memStore) FinishTurn(_ context.Context, id, status string, usage json.RawMessage) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	t, ok := s.turns[id]
	if !ok || (t.Status != store.TurnRunning && t.Status != store.TurnWaitingTools && t.Status != store.TurnSuspended) {
		return store.ErrNotFound
	}
	now := s.now()
	t.Status, t.Ended, t.Usage, t.PendingToolCalls = status, &now, usage, nil
	return nil
}

func (s *memStore) ExpireStaleTurnsWithPending(_ context.Context, ttl time.Duration) ([]store.ExpiredTurn, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.failDB {
		return nil, errors.New("memstore: database unavailable")
	}
	cutoff := s.now().Add(-ttl)
	var out []store.ExpiredTurn
	for _, t := range s.turns {
		live := t.Status == store.TurnRunning || t.Status == store.TurnWaitingTools || t.Status == store.TurnSuspended
		if live && t.Started.Before(cutoff) {
			now := s.now()
			out = append(out, store.ExpiredTurn{ID: t.ID, ConversationID: t.ConversationID, WasStatus: t.Status, PendingToolCalls: t.PendingToolCalls, HasNewer: s.hasNewerLocked(t)})
			t.Status, t.Ended, t.PendingToolCalls = store.TurnExpired, &now, nil
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return out, nil
}

// hasNewerLocked mirrors the SQL EXISTS: a turn of the same conversation
// started later (or at the same instant and created later).
func (s *memStore) hasNewerLocked(t *store.Turn) bool {
	for _, o := range s.turns {
		if o.ConversationID != t.ConversationID || o.ID == t.ID {
			continue
		}
		if o.Started.After(t.Started) || (o.Started.Equal(t.Started) && s.order[o.ID] > s.order[t.ID]) {
			return true
		}
	}
	return false
}

func (s *memStore) SupersedeTurns(ctx context.Context, convID string) ([]store.SupersededTurn, error) {
	return s.supersede(func(t *store.Turn) bool { return t.ConversationID == convID })
}

func (s *memStore) SupersedeTurnsForSite(_ context.Context, siteID string) ([]store.SupersededTurn, error) {
	return s.supersede(func(t *store.Turn) bool {
		c, ok := s.convs[t.ConversationID]
		return ok && c.SiteID == siteID
	})
}

// supersede mirrors the SQL: every live turn (running included) matching
// the predicate becomes superseded, returned once in creation order.
func (s *memStore) supersede(match func(*store.Turn) bool) ([]store.SupersededTurn, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.failDB {
		return nil, errors.New("memstore: database unavailable")
	}
	return s.supersedeLocked(match), nil
}

func (s *memStore) supersedeLocked(match func(*store.Turn) bool) []store.SupersededTurn {
	var out []store.SupersededTurn
	for _, t := range s.turns {
		live := t.Status == store.TurnRunning || t.Status == store.TurnWaitingTools || t.Status == store.TurnSuspended
		if !live || !match(t) {
			continue
		}
		now := s.now()
		out = append(out, store.SupersededTurn{ID: t.ID, ConversationID: t.ConversationID, WasStatus: t.Status, PendingToolCalls: t.PendingToolCalls})
		t.Status, t.Ended, t.PendingToolCalls = store.TurnSuperseded, &now, nil
	}
	sort.Slice(out, func(i, j int) bool { return s.order[out[i].ID] < s.order[out[j].ID] })
	return out
}

func (s *memStore) ListTurns(_ context.Context, convID string) ([]store.Turn, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	var out []store.Turn
	for _, t := range s.turns {
		if t.ConversationID == convID {
			out = append(out, *t)
		}
	}
	// Oldest first, as the SQL's ORDER BY started; the creation ordinal
	// breaks the ties a fixed test clock makes of every row.
	sort.Slice(out, func(i, j int) bool {
		if !out[i].Started.Equal(out[j].Started) {
			return out[i].Started.Before(out[j].Started)
		}
		return s.order[out[i].ID] < s.order[out[j].ID]
	})
	return out, nil
}

// --- traces / optout ---

func (s *memStore) IsOptedOut(_ context.Context, storeID string) (bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.optout[storeID], nil
}

func (s *memStore) InsertTrace(_ context.Context, t store.Trace) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if old, ok := s.traces[t.TurnID]; ok {
		t.Feedback = old.Feedback
	}
	s.traces[t.TurnID] = t
	return nil
}

func (s *memStore) MergeFeedback(_ context.Context, turnID string, fb json.RawMessage) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	t, ok := s.traces[turnID]
	if !ok {
		return store.ErrNotFound
	}
	merged := map[string]any{}
	_ = json.Unmarshal(t.Feedback, &merged)
	var add map[string]any
	_ = json.Unmarshal(fb, &add)
	for k, v := range add {
		merged[k] = v
	}
	t.Feedback, _ = json.Marshal(merged)
	s.traces[turnID] = t
	return nil
}

func (s *memStore) SetOptOut(_ context.Context, storeID, _ string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.optout[storeID] = true
	return nil
}

func (s *memStore) ClearOptOut(_ context.Context, storeID string) (bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	was := s.optout[storeID]
	delete(s.optout, storeID)
	return was, nil
}

// --- usage ---

func (s *memStore) InsertUsage(_ context.Context, r store.UsageRow) (int64, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	r.ID = int64(len(s.usage) + 1)
	s.usage = append(s.usage, r)
	return r.ID, nil
}

func (s *memStore) ListUnshippedUsage(_ context.Context, limit int) ([]store.UsageRow, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	var out []store.UsageRow
	for _, r := range s.usage {
		if !r.Shipped && len(out) < limit {
			out = append(out, r)
		}
	}
	return out, nil
}

func (s *memStore) MarkUsageShipped(_ context.Context, ids []int64) (int64, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	var n int64
	for _, id := range ids {
		if id >= 1 && int(id) <= len(s.usage) && !s.usage[id-1].Shipped {
			s.usage[id-1].Shipped = true
			n++
		}
	}
	return n, nil
}

// snapshots for assertions
func (s *memStore) turn(id string) store.Turn {
	s.mu.Lock()
	defer s.mu.Unlock()
	return *s.turns[id]
}

func (s *memStore) usageRows() []store.UsageRow {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]store.UsageRow{}, s.usage...)
}

func (s *memStore) trace(id string) (store.Trace, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	t, ok := s.traces[id]
	return t, ok
}
