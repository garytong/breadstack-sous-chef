// Package auth checks the launcher's static service key (README §1).
//
// There is no JWT verification here by decision: the launcher is the only
// caller, it authenticates the merchant itself, and it forwards the
// merchant's email in X-User-Email for attribution only. Two keys may be
// live at once so a key can be rotated without downtime.
package auth

import (
	"context"
	"crypto/subtle"
	"net/http"
	"strings"
)

// HeaderUserEmail is the attribution header the launcher sets from its
// session cookie. It is never used for authorization.
const HeaderUserEmail = "X-User-Email"

// Keys is the set of accepted bearer keys.
type Keys struct {
	keys [][]byte
}

// NewKeys builds the key set from SERVICE_KEYS entries. Empty entries are
// dropped; an empty set accepts nothing.
func NewKeys(keys []string) *Keys {
	k := &Keys{}
	for _, s := range keys {
		s = strings.TrimSpace(s)
		if s != "" {
			k.keys = append(k.keys, []byte(s))
		}
	}
	return k
}

// Len is the number of accepted keys.
func (k *Keys) Len() int { return len(k.keys) }

// Match reports whether presented equals one of the keys. Every key is
// compared with crypto/subtle.ConstantTimeCompare and the loop never exits
// early, so timing does not reveal which key (if any) matched or how many
// keys exist with a given prefix.
func (k *Keys) Match(presented string) bool {
	if presented == "" || len(k.keys) == 0 {
		return false
	}
	p := []byte(presented)
	matched := 0
	for _, key := range k.keys {
		// ConstantTimeCompare returns 0 immediately on a length mismatch;
		// the length itself is not secret (keys are generated with a fixed
		// length), so this is acceptable.
		matched |= subtle.ConstantTimeCompare(p, key)
	}
	return matched == 1
}

// BearerToken extracts the token from an "Authorization: Bearer <token>"
// header. The scheme is matched case-insensitively per RFC 9110; the token
// is returned as-is (trimmed). Missing or non-Bearer headers yield "".
func BearerToken(r *http.Request) string {
	h := r.Header.Get("Authorization")
	if len(h) < len("Bearer ") || !strings.EqualFold(h[:len("Bearer ")], "Bearer ") {
		return ""
	}
	return strings.TrimSpace(h[len("Bearer "):])
}

// Authorized reports whether the request carries a valid service key.
func (k *Keys) Authorized(r *http.Request) bool {
	return k.Match(BearerToken(r))
}

type ctxKey int

const emailKey ctxKey = iota

// WithUserEmail stores the attribution email on the context.
func WithUserEmail(ctx context.Context, email string) context.Context {
	return context.WithValue(ctx, emailKey, email)
}

// UserEmail returns the attribution email stored by Middleware, or "".
func UserEmail(ctx context.Context) string {
	v, _ := ctx.Value(emailKey).(string)
	return v
}

// UserEmailFromRequest reads and normalises X-User-Email: trimmed,
// lower-cased, capped at 254 bytes (RFC 5321 path limit), and dropped
// entirely when it does not look like an address so a garbage header
// cannot pollute traces.
func UserEmailFromRequest(r *http.Request) string {
	e := strings.ToLower(strings.TrimSpace(r.Header.Get(HeaderUserEmail)))
	if e == "" || len(e) > 254 {
		return ""
	}
	at := strings.IndexByte(e, '@')
	if at < 1 || at == len(e)-1 || strings.ContainsAny(e, " \t\r\n") {
		return ""
	}
	return e
}

// Middleware rejects requests without a valid key (401, via onReject) and
// otherwise passes the request on with the attribution email on its
// context. onReject is called with the ResponseWriter and Request so the
// API package can render its own error envelope.
func (k *Keys) Middleware(onReject http.HandlerFunc) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if !k.Authorized(r) {
				onReject(w, r)
				return
			}
			ctx := WithUserEmail(r.Context(), UserEmailFromRequest(r))
			next.ServeHTTP(w, r.WithContext(ctx))
		})
	}
}
