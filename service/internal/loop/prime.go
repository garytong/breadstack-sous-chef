package loop

import (
	"context"
	"fmt"
	"time"

	"github.com/anthropics/anthropic-sdk-go"

	"github.com/breadstack/sous-chef/internal/api"
	"github.com/breadstack/sous-chef/internal/model"
	"github.com/breadstack/sous-chef/internal/prompt"
)

// Prompt-cache priming (README §4 "Prompt-cache priming"): the loop's side
// of model.Primer. The Runner decides WHEN to prime and builds WHAT (the
// next turn's request with a filler user message, prompt.PrimeRequest);
// the Primer sends it off the merchant's path, one at a time per model
// server, and never while a turn is in progress on this instance (holding
// a slot, or parked on its batch less than PrimeParkedBusy ago: primeBusy).
//
// Triggers:
//   - turn.done on a primed route (the route the turn STARTED on, so an
//     escalated turn primes the student's server for the next turn): the
//     conversation's next turn, from storage — exactly the history that
//     turn's Start will read;
//   - PUT/POST /v1/manifests: the floor of the new manifest on the default
//     route;
//   - GET /v1/conversations/{siteId} with X-Manifest-Hash (the panel
//     opening): the open conversation's next turn, or the floor;
//   - DELETE /v1/conversations/{siteId}: the floor.
//
// Every trigger is a no-op when Deps.Primer is nil or the route does not
// say `prime:"on"` (an anthropic route never does). The manifest, panel and
// conversation-end triggers prime the DEFAULT route only: when the default
// is an anthropic route, only turn.done on a primed route primes.

// observe tells the Primer about a real call (the slot now holds its
// prompt; a cache miss means the server lost its checkpoints).
func (t *turn) observe(resp *model.Response) {
	if t.r.d.Primer == nil || resp == nil {
		return
	}
	t.r.d.Primer.Observe(t.backend, t.req.ManifestHash, resp.Usage)
}

// primeAfterDone schedules the prime of the conversation's next turn once
// the turn has released its slot (Run's defer), and waits — at most
// PrimeHold — until its requests are on the wire.
func (t *turn) primeAfterDone() {
	if t.r.d.Primer == nil || t.conv == nil {
		return
	}
	backend := t.origin
	if backend == nil {
		backend = t.backend
	}
	d := t.r.schedulePrime(backend, t.req.ManifestHash, "turn_done", t.req.SiteID, t.conv.ID, 0)
	if d == nil {
		return
	}
	started := time.Now()
	d.Wait(PrimeHold)
	// The stream stays open this long after turn.done (the launcher has
	// stopped reading; a client that reads to EOF, like the lab, counts it).
	t.r.d.Logf("loop: turn %s: held %dms after turn.done for the prime dispatch", t.id, time.Since(started).Milliseconds())
}

// PrimeHold bounds how long a trigger waits for its prime's requests to be
// written (model.Dispatch): Cloud Run with request-based CPU throttles a
// goroutine once its request has ended.
const PrimeHold = model.DefaultHold

// PanelPrimeInterval: a panel_open trigger re-primes a site's tail at most
// this often (two open panels re-reading their conversations would
// otherwise re-prime each other's tails on every read).
const PanelPrimeInterval = time.Minute

// ManifestRegistered implements api.Primer: prime the floor of a manifest
// the launcher just registered, on the default route.
func (r *Runner) ManifestRegistered(hash string) api.PrimeDispatch {
	return api.PrimeDispatch(r.schedulePrime(r.defaultBackend(), hash, "manifest", "", "", 0))
}

// ConversationOpened implements api.Primer: the launcher read the site's
// conversation (the panel opened). conversationID "" = none open.
func (r *Runner) ConversationOpened(siteID, conversationID, manifestHash string) api.PrimeDispatch {
	return api.PrimeDispatch(r.schedulePrime(r.defaultBackend(), manifestHash, "panel_open", siteID, conversationID, PanelPrimeInterval))
}

// ConversationEnded implements api.Primer: the site's conversation was
// ended; its next turn starts a fresh one.
func (r *Runner) ConversationEnded(siteID, manifestHash string) api.PrimeDispatch {
	return api.PrimeDispatch(r.schedulePrime(r.defaultBackend(), manifestHash, "conversation_end", siteID, "", 0))
}

func (r *Runner) defaultBackend() *model.Backend {
	if r.d.Primer == nil {
		return nil
	}
	b, err := r.d.Models.Route("")
	if err != nil {
		return nil
	}
	return b
}

// schedulePrime queues a prime of backend's server for the manifest hash:
// the floor, plus — when conversationID names one — that conversation's
// next turn, read from storage when the prime runs.
func (r *Runner) schedulePrime(backend *model.Backend, hash, reason, siteID, conversationID string, siteInterval time.Duration) model.Dispatch {
	if r.d.Primer == nil || backend == nil || !backend.PrimeEnabled() || hash == "" {
		return nil
	}
	filler := backend.PrimeFiller()
	return r.d.Primer.Schedule(model.PrimeJob{
		Backend: backend, Prefix: hash, Reason: reason, Site: siteID, SiteInterval: siteInterval,
		Build: func(ctx context.Context) (model.PrimeRequests, error) {
			var out model.PrimeRequests
			compiled, err := r.d.Manifests.Load(ctx, hash)
			if err != nil {
				return out, err
			}
			if out.Floor, err = prompt.PrimeRequest(compiled.Prefix, nil, filler); err != nil {
				return out, err
			}
			if conversationID == "" {
				return out, nil
			}
			history, tailSeq, err := r.history(ctx, conversationID)
			if err != nil {
				return out, err
			}
			if !r.primeableHistory(history) {
				return out, nil
			}
			next, err := prompt.PrimeRequest(compiled.Prefix, history, filler)
			if err != nil {
				return out, err
			}
			out.Tail, out.TailKey = &next, fmt.Sprintf("%s:%d", conversationID, tailSeq)
			return out, nil
		},
	})
}

// primeableHistory reports whether the conversation's next turn builds on
// history as it stands: not empty (the floor covers that), not about to
// rotate (the next turn starts a fresh conversation — the floor again), and
// not waiting on a batch (the next request is a continuation; the turn's
// own calls keep its checkpoints).
func (r *Runner) primeableHistory(history []anthropic.BetaMessageParam) bool {
	if len(history) == 0 || len(history) >= r.d.RotateAt {
		return false
	}
	last := history[len(history)-1]
	if last.Role == anthropic.BetaMessageParamRoleAssistant {
		for _, c := range last.Content {
			if c.OfToolUse != nil {
				return false
			}
		}
	}
	return true
}
