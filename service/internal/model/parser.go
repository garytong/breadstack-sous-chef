package model

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
)

// Parser errors (config.Route.ParserRetries). Ollama turns the text a model
// writes into tool_use blocks with a parser of the model's family; when that
// parser fails, the whole call is answered HTTP 500 and nothing of the reply
// reaches the client. Ollama 0.34.4 (read in its source):
//
//   - model/parsers/ministral.go (PARSER ministral, Ministral 3): a call to a
//     tool the request did not offer — "tool '<name>' not found" — or
//     arguments that do not decode as a JSON object: encoding/json's syntax
//     errors ("invalid character …", "unexpected end of JSON input") or the
//     ordered map's decoder (buger/jsonparser v1.1.1's errors, wk8's
//     "not a valid UTF-8 string …"), or a number too large for a float64
//     anywhere in the arguments (encoding/json decoding the value into any:
//     "json: cannot unmarshal number 1e999 into Go value of type float64").
//   - model/parsers/qwen3coder.go (PARSER qwen3.5's tool calls): an XML
//     call that does not decode — encoding/xml's errors.
//
// server/routes.go sends the parser's error as {"error": msg} with status
// 500; the /v1/messages middleware wraps it as
// {"type":"error","error":{"type":"api_error","message":msg}}.
//
// The SDK retries every 5xx (DefaultMaxRetries), so on a route that leaves
// parser retries on (the default) a parser error is redrawn up to twice —
// a fresh sample at the route's temperature that no log of the service
// shows. A route with parser_retries "off" gets parserGuard: the 500 is
// marked not retryable (the SDK honours `x-should-retry: false`), the call
// fails at once, and Message returns it as a *ParserError.

// numberOverflowPrefix / numberOverflowSuffix frame encoding/json's
// UnmarshalTypeError for a JSON number that does not fit a float64, decoded
// into any (the ordered map's values); the number sits between them.
const (
	numberOverflowPrefix = "json: cannot unmarshal number "
	numberOverflowSuffix = " into Go value of type float64"
)

// parserMessagePrefixes / parserMessages are the error texts above.
var (
	parserMessagePrefixes = []string{
		"invalid character ",        // encoding/json SyntaxError
		"not a valid UTF-8 string",  // wk8/go-ordered-map decodeUTF8 (a key)
		"XML syntax error on line ", // encoding/xml SyntaxError (qwen3.5 tool calls)
		"expected element type <",   // encoding/xml (qwen3.5 tool calls)
		"failed to parse JSON: ",    // qwen3 tool calls
	}
	parserMessages = map[string]bool{
		"unexpected end of JSON input": true, // encoding/json
		// buger/jsonparser v1.1.1 (ObjectEach under the ordered map)
		"Key path not found":   true,
		"Unknown value type":   true,
		"Malformed JSON error": true,
		"Value is string, but can't find closing '\"' symbol":                             true,
		"Value is array, but can't find closing ']' symbol":                               true,
		"Value looks like object, but can't find closing '}' symbol":                      true,
		"Value looks like Number/Boolean/None, but can't find its end: ',' or '}' symbol": true,
		"Value is number, but overflowed while parsing":                                   true,
		"Encountered an invalid escape sequence in a string":                              true,
		"empty function name": true, // qwen3 tool calls
	}
)

// IsParserMessage reports whether an Ollama error message is a tool-call
// parser's (the list above).
func IsParserMessage(msg string) bool {
	msg = strings.TrimSpace(msg)
	if strings.HasPrefix(msg, "tool '") && strings.HasSuffix(msg, "' not found") {
		return true // MinistralParser: a tool name the request did not offer
	}
	if strings.HasPrefix(msg, numberOverflowPrefix) && strings.HasSuffix(msg, numberOverflowSuffix) {
		return true // MinistralParser: an argument number too large for a float64
	}
	if parserMessages[msg] {
		return true
	}
	for _, p := range parserMessagePrefixes {
		if strings.HasPrefix(msg, p) {
			return true
		}
	}
	return false
}

// ParserErrorMessage reads a model server's 500 body and returns its
// message when it is a tool-call parser's: the Messages API envelope
// {"error":{"message":…}} Ollama's /v1/messages answers with, or Ollama's
// native {"error":"…"}. Anything else — another 500, a body that is not
// JSON (a front end's page) — is not.
func ParserErrorMessage(body []byte) (string, bool) {
	var env struct {
		Error json.RawMessage `json:"error"`
	}
	if json.Unmarshal(bytes.TrimSpace(body), &env) != nil || len(env.Error) == 0 {
		return "", false
	}
	var msg string
	var obj struct {
		Message string `json:"message"`
	}
	switch {
	case json.Unmarshal(env.Error, &obj) == nil && obj.Message != "":
		msg = obj.Message
	case json.Unmarshal(env.Error, &msg) == nil:
	default:
		return "", false
	}
	if !IsParserMessage(msg) {
		return "", false
	}
	return strings.TrimSpace(msg), true
}

// ParserError is a call on a route with parser_retries "off" that the model
// server answered 500 because it could not parse the model's own output
// into a tool call. It wraps the SDK's *anthropic.Error, so IsUnavailable
// and StatusCode read it as the 500 it is and the loop handles it as any
// other failed call (README §4 "Hybrid": model_error).
type ParserError struct {
	Route   string // the route's name
	Message string // the server's message, e.g. "tool 'set_field' not found"
	Count   int64  // the route's parser errors on this instance so far, this one included
	Err     error  // the SDK's error
}

func (e *ParserError) Error() string {
	return fmt.Sprintf("model: parser error on route %s (not retried): %s: %v", e.Route, e.Message, e.Err)
}

func (e *ParserError) Unwrap() error { return e.Err }

// AsParserError returns err's *ParserError, if it is one. Only a route with
// parser_retries "off" ever returns one.
func AsParserError(err error) (*ParserError, bool) {
	var pe *ParserError
	if errors.As(err, &pe) {
		return pe, true
	}
	return nil, false
}

// maxErrorBody bounds what parserGuard reads of a 500 body (Ollama's are a
// few hundred bytes).
const maxErrorBody = 64 << 10

// parserGuard is the middleware of a route with parser_retries "off": a 500
// whose body is a tool-call parser's error is marked `x-should-retry:
// false`, which the SDK honours before its status rule, so the call ends on
// that attempt. The body is read and put back unchanged (the SDK reads it
// again for its error); every other response passes untouched, and every
// other retry stays the SDK's.
func (b *Backend) parserGuard(req *http.Request, next option.MiddlewareNext) (*http.Response, error) {
	res, err := next(req)
	if err != nil || res == nil || res.StatusCode != http.StatusInternalServerError || res.Body == nil {
		return res, err
	}
	body, readErr := io.ReadAll(io.LimitReader(res.Body, maxErrorBody))
	rest := res.Body
	switch {
	case readErr != nil:
		// The SDK sees the same failed read it would have (a dropped body).
		res.Body = readCloser{io.MultiReader(bytes.NewReader(body), errReader{readErr}), rest}
		return res, nil
	case len(body) == maxErrorBody:
		res.Body = readCloser{io.MultiReader(bytes.NewReader(body), rest), rest}
		return res, nil // not an error body of Ollama's: leave it to the SDK
	}
	_ = rest.Close()
	res.Body = io.NopCloser(bytes.NewReader(body))
	if _, ok := ParserErrorMessage(body); ok {
		if res.Header == nil {
			res.Header = http.Header{}
		}
		res.Header.Set("X-Should-Retry", "false")
	}
	return res, nil
}

type readCloser struct {
	io.Reader
	c io.Closer
}

func (r readCloser) Close() error { return r.c.Close() }

type errReader struct{ err error }

func (e errReader) Read([]byte) (int, error) { return 0, e.err }

// parserFailure turns a failed call on a route with parser_retries "off"
// into a *ParserError when the server's 500 was a tool-call parser's, and
// counts it; any other error is returned as it is.
func (b *Backend) parserFailure(err error) error {
	if b.route.ParserRetriesEnabled() {
		return err
	}
	var apiErr *anthropic.Error
	if !errors.As(err, &apiErr) || apiErr.StatusCode != http.StatusInternalServerError {
		return err
	}
	msg, ok := ParserErrorMessage([]byte(apiErr.RawJSON()))
	if !ok {
		return err
	}
	return &ParserError{Route: b.name, Message: msg, Count: b.parserErrors.Add(1), Err: err}
}
