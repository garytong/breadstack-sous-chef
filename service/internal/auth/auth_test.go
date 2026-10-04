package auth

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

func req(authz, email string) *http.Request {
	r := httptest.NewRequest(http.MethodGet, "/v1/x", nil)
	if authz != "" {
		r.Header.Set("Authorization", authz)
	}
	if email != "" {
		r.Header.Set(HeaderUserEmail, email)
	}
	return r
}

func TestMatchMultiKey(t *testing.T) {
	k := NewKeys([]string{"old-key-0000", " new-key-1111 ", ""})
	if k.Len() != 2 {
		t.Fatalf("Len = %d, want 2 (empty dropped)", k.Len())
	}
	for _, good := range []string{"old-key-0000", "new-key-1111"} {
		if !k.Match(good) {
			t.Errorf("Match(%q) = false", good)
		}
	}
	for _, bad := range []string{"", "old-key-000", "old-key-00000", "new-key-1112", "OLD-KEY-0000", "old-key-0000 "} {
		if k.Match(bad) {
			t.Errorf("Match(%q) = true", bad)
		}
	}
}

func TestMatchEmptySetAcceptsNothing(t *testing.T) {
	k := NewKeys(nil)
	if k.Match("") || k.Match("anything") {
		t.Error("empty key set must reject everything")
	}
}

func TestBearerToken(t *testing.T) {
	cases := map[string]string{
		"Bearer abc":   "abc",
		"bearer abc":   "abc",
		"BEARER  abc ": "abc",
		"Basic abc":    "",
		"Bearer":       "",
		"":             "",
		"abc":          "",
	}
	for h, want := range cases {
		if got := BearerToken(req(h, "")); got != want {
			t.Errorf("BearerToken(%q) = %q, want %q", h, got, want)
		}
	}
}

func TestUserEmailFromRequest(t *testing.T) {
	cases := map[string]string{
		"Merchant@Example.com ": "merchant@example.com",
		"":                      "",
		"not-an-email":          "",
		"@example.com":          "",
		"a@":                    "",
		"has space@example.com": "",
		"ok@example.com":        "ok@example.com",
	}
	for h, want := range cases {
		if got := UserEmailFromRequest(req("", h)); got != want {
			t.Errorf("UserEmailFromRequest(%q) = %q, want %q", h, got, want)
		}
	}
	long := make([]byte, 260)
	for i := range long {
		long[i] = 'a'
	}
	if got := UserEmailFromRequest(req("", string(long[:250])+"@e.io")); got != "" {
		t.Errorf("over-long email should be dropped, got %q", got)
	}
}

func TestMiddleware(t *testing.T) {
	k := NewKeys([]string{"k1", "k2"})
	rejected := 0
	var seenEmail string
	passed := 0
	h := k.Middleware(func(w http.ResponseWriter, r *http.Request) {
		rejected++
		w.WriteHeader(http.StatusUnauthorized)
	})(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		passed++
		seenEmail = UserEmail(r.Context())
		w.WriteHeader(http.StatusNoContent)
	}))

	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req("Bearer k2", "Shop@Example.com"))
	if rr.Code != http.StatusNoContent || passed != 1 || seenEmail != "shop@example.com" {
		t.Errorf("valid key: code=%d passed=%d email=%q", rr.Code, passed, seenEmail)
	}

	rr = httptest.NewRecorder()
	h.ServeHTTP(rr, req("Bearer k3", ""))
	if rr.Code != http.StatusUnauthorized || rejected != 1 || passed != 1 {
		t.Errorf("bad key: code=%d rejected=%d passed=%d", rr.Code, rejected, passed)
	}

	rr = httptest.NewRecorder()
	h.ServeHTTP(rr, req("", ""))
	if rr.Code != http.StatusUnauthorized || rejected != 2 {
		t.Errorf("no header: code=%d rejected=%d", rr.Code, rejected)
	}
}

func TestUserEmailMissingIsEmpty(t *testing.T) {
	if UserEmail(req("", "").Context()) != "" {
		t.Error("UserEmail on a bare context should be empty")
	}
}
