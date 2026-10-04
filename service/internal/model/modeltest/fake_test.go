package modeltest

import (
	"bufio"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"
)

func post(t *testing.T, url, body string) (*http.Response, string) {
	t.Helper()
	resp, err := http.Post(url+"/v1/messages", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(resp.Body)
	return resp, string(b)
}

func TestJSONReplyAndScript(t *testing.T) {
	s := New(Reply(Message{Content: []Block{Text("hi"), ToolUse("t1", "f", map[string]any{"a": 1})}, Usage: Usage{InputTokens: 3, OutputTokens: 4}}),
		Error(429, "rate_limit_error", "slow", map[string]string{"Retry-After-Ms": "1"}))
	defer s.Close()
	resp, body := post(t, s.URL, `{"model":"m","messages":[]}`)
	if resp.StatusCode != 200 {
		t.Fatalf("status %d", resp.StatusCode)
	}
	var m map[string]any
	if err := json.Unmarshal([]byte(body), &m); err != nil {
		t.Fatal(err)
	}
	if m["stop_reason"] != "tool_use" || len(m["content"].([]any)) != 2 || m["id"] != "msg_fake" {
		t.Errorf("reply = %s", body)
	}
	resp, body = post(t, s.URL, `{}`)
	if resp.StatusCode != 429 || resp.Header.Get("Retry-After-Ms") != "1" || !strings.Contains(body, "rate_limit_error") {
		t.Errorf("error step: %d %s", resp.StatusCode, body)
	}
	resp, _ = post(t, s.URL, `{}`)
	if resp.StatusCode != 500 {
		t.Errorf("exhausted script should answer 500, got %d", resp.StatusCode)
	}
	if reqs := s.Requests(); len(reqs) != 3 || reqs[0].Body["model"] != "m" {
		t.Errorf("captured = %+v", reqs)
	}
}

func TestSSERendering(t *testing.T) {
	s := New(Reply(Message{StopReason: "refusal", StopCategory: "cyber", Content: []Block{Text("abcd"), ToolUse("t1", "f", map[string]any{"k": "v"}), Thinking("hm")},
		Usage: Usage{InputTokens: 10, OutputTokens: 7, CacheReadInputTokens: 99}}))
	defer s.Close()
	resp, err := http.Post(s.URL+"/v1/messages", "application/json", strings.NewReader(`{"stream":true}`))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if ct := resp.Header.Get("Content-Type"); ct != "text/event-stream" {
		t.Fatalf("content-type %q", ct)
	}
	var events []string
	var datas []map[string]any
	sc := bufio.NewScanner(resp.Body)
	for sc.Scan() {
		line := sc.Text()
		if strings.HasPrefix(line, "event: ") {
			events = append(events, strings.TrimPrefix(line, "event: "))
		}
		if strings.HasPrefix(line, "data: ") {
			var d map[string]any
			if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &d); err != nil {
				t.Fatalf("bad data line %q", line)
			}
			datas = append(datas, d)
		}
	}
	joined := strings.Join(events, " ")
	if !strings.HasPrefix(joined, "message_start content_block_start content_block_delta content_block_delta content_block_stop content_block_start content_block_delta") {
		t.Errorf("event order = %s", joined)
	}
	if !strings.HasSuffix(joined, "message_delta message_stop") {
		t.Errorf("event tail = %s", joined)
	}
	if !s.Requests()[0].Stream {
		t.Errorf("stream flag not captured")
	}
	// text split in two, tool input in three, final usage on message_delta
	var textDeltas, jsonDeltas int
	for _, d := range datas {
		if d["type"] == "content_block_delta" {
			switch d["delta"].(map[string]any)["type"] {
			case "text_delta":
				textDeltas++
			case "input_json_delta":
				jsonDeltas++
			}
		}
		if d["type"] == "message_delta" {
			if d["usage"].(map[string]any)["output_tokens"] != float64(7) || d["delta"].(map[string]any)["stop_reason"] != "refusal" {
				t.Errorf("message_delta = %v", d)
			}
			if d["delta"].(map[string]any)["stop_details"].(map[string]any)["category"] != "cyber" {
				t.Errorf("stop_details = %v", d["delta"])
			}
		}
		if d["type"] == "message_start" {
			u := d["message"].(map[string]any)["usage"].(map[string]any)
			if u["cache_read_input_tokens"] != float64(99) || u["output_tokens"] != float64(1) {
				t.Errorf("message_start usage = %v", u)
			}
		}
	}
	if textDeltas != 2 || jsonDeltas != 3 {
		t.Errorf("deltas text=%d json=%d", textDeltas, jsonDeltas)
	}
}

func TestSplit(t *testing.T) {
	if got := split("", 3); len(got) != 1 || got[0] != "" {
		t.Errorf("empty: %q", got)
	}
	if got := split("ab", 3); strings.Join(got, "|") != "a|b" {
		t.Errorf("short: %q", got)
	}
	if got := split("héllo wörld", 2); strings.Join(got, "") != "héllo wörld" || len(got) != 2 {
		t.Errorf("runes: %q", got)
	}
}
