package model

// Copied from breadstack-thermometer server/internal/chat/idtoken.go
// (metadataTokens, jwtExpiry) with the names kept; the thermometer module
// is not a dependency. Only the type that consumers see (TokenSource) was
// added.

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"
)

// TokenSource mints the bearer token a messages-compatible route behind
// Cloud Run IAM is called with (route auth "id-token"). The metadata server
// implements it in production; tests pass a fake.
type TokenSource interface {
	IDToken(ctx context.Context, audience string) (string, error)
}

// metadataURL is where Cloud Run's metadata server answers (the container
// contract), whatever the service's VPC egress.
const metadataURL = "http://metadata.google.internal"

// idTokenMargin: a cached ID token is used until this long before it
// expires (about an hour after the metadata server minted it).
const idTokenMargin = 5 * time.Minute

// maxIDTokenBytes bounds the metadata server's reply (a JWT of about a
// kilobyte).
const maxIDTokenBytes = 16 << 10

// metadataTokens is the TokenSource on Cloud Run: the metadata server's ID
// token for the runtime service account, minted for the audience asked (the
// GPU service's URL) and cached until idTokenMargin before it expires.
// Plain HTTP with the Metadata-Flavor header — no Google client library.
// One token at a time is cached (there is one audience per source);
// callers wait for a fetch in flight rather than start their own.
type metadataTokens struct {
	base   string // metadataURL; a fake in tests
	client *http.Client
	now    func() time.Time

	mu       sync.Mutex
	audience string
	token    string
	until    time.Time // use token before this
}

// NewMetadataTokens returns the Cloud Run metadata-server TokenSource.
func NewMetadataTokens() TokenSource { return newMetadataTokens(metadataURL) }

func newMetadataTokens(base string) *metadataTokens {
	return &metadataTokens{base: base, client: &http.Client{Timeout: 5 * time.Second}, now: time.Now}
}

// IDToken returns an ID token for audience, from the cache while it has
// more than idTokenMargin to run.
func (m *metadataTokens) IDToken(ctx context.Context, audience string) (string, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.token != "" && m.audience == audience && m.now().Before(m.until) {
		return m.token, nil
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet,
		m.base+"/computeMetadata/v1/instance/service-accounts/default/identity?audience="+url.QueryEscape(audience), nil)
	if err != nil {
		return "", err
	}
	req.Header.Set("Metadata-Flavor", "Google")
	resp, err := m.client.Do(req)
	if err != nil {
		return "", fmt.Errorf("metadata server: %w", err)
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(io.LimitReader(resp.Body, maxIDTokenBytes))
	if err != nil {
		return "", fmt.Errorf("metadata server: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("metadata server answered %d: %.200s", resp.StatusCode, strings.TrimSpace(string(body)))
	}
	token := strings.TrimSpace(string(body))
	exp, err := jwtExpiry(token)
	if err != nil {
		return "", fmt.Errorf("metadata server's ID token: %w", err)
	}
	m.audience, m.token, m.until = audience, token, exp.Add(-idTokenMargin)
	return token, nil
}

// jwtExpiry reads a JWT's exp claim without checking its signature: the
// token came from the metadata server, and Cloud Run's front end is what
// verifies it.
func jwtExpiry(token string) (time.Time, error) {
	parts := strings.Split(token, ".")
	if len(parts) != 3 {
		return time.Time{}, errors.New("not a JWT")
	}
	payload, err := base64.RawURLEncoding.DecodeString(strings.TrimRight(parts[1], "="))
	if err != nil {
		return time.Time{}, fmt.Errorf("JWT payload: %w", err)
	}
	var claims struct {
		Exp int64 `json:"exp"`
	}
	if err := json.Unmarshal(payload, &claims); err != nil {
		return time.Time{}, fmt.Errorf("JWT payload: %w", err)
	}
	if claims.Exp <= 0 {
		return time.Time{}, errors.New("JWT without an expiry")
	}
	return time.Unix(claims.Exp, 0), nil
}
