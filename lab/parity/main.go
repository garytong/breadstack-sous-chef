// Command servingrender renders a Sous Chef model request the way Ollama
// 0.34.4 renders it for the served student, offline and with no model loaded
// (docs/LAB.md §5 "Serving parity"): the lab's parity fixture is this
// program's output, and souschef_lab.serving must give the same bytes.
//
// Per input line — a POST /v1/messages body as the service sends it to a
// messages-compatible route, or a fixtures/prompt/real/*/model-requests.jsonl
// line ({"body": <that body>, …}) — it runs Ollama's own code, in the order
// the server does:
//
//   - middleware.AnthropicMessagesMiddleware: bind the body as
//     anthropic.MessagesRequest, anthropic.FromMessagesRequest with the
//     model's thinking descriptor (a RENDERER qwen3.5 model: the renderer's,
//     {false,true} default true), re-encode as an api.ChatRequest;
//   - server.ChatHandler: renderers.ResolveThinking (the request's
//     {"thinking":{"type":"disabled"}} is think=false; an absent field would be
//     the default, true), the qwen3.5 parser's Init (it hands the tools back
//     unchanged), then chatPrompt → renderPrompt → the qwen3.5 renderer.
//
// What it leaves out, and why it does not matter for the lab's rows: the
// export's Modelfile has no SYSTEM or MESSAGE lines (nothing is prepended);
// filterThinkTags touches only the qwen3 family and deepseek-r1 (the student
// is qwen35); chatPrompt's truncation drops leading messages only when the
// prompt exceeds num_ctx (65,536 in the export; rows stop at 32,768 tokens).
//
// Output: one JSON line per input, {"prompt", "think", "messages", "tools"}.
// The module pins github.com/ollama/ollama at the version the model image
// serves (lab/model-image: 0.34.4). A new Ollama means a new pin here and a
// new `python -m souschef_lab.serving --write`. Lab tooling, a module of its
// own: the service never builds it.
package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"os"

	"github.com/ollama/ollama/anthropic"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/model/parsers"
	"github.com/ollama/ollama/model/renderers"
)

// The export's Modelfile (export.py, lab/model-image): RENDERER qwen3.5, PARSER qwen3.5.
const (
	rendererName = "qwen3.5"
	parserName   = "qwen3.5"
)

type out struct {
	Think    string `json:"think"`
	Prompt   string `json:"prompt"`
	Messages int    `json:"messages"`
	Tools    int    `json:"tools"`
}

func main() {
	in := bufio.NewScanner(os.Stdin)
	in.Buffer(make([]byte, 64<<20), 64<<20)
	w := bufio.NewWriter(os.Stdout)
	defer w.Flush()
	n := 0
	for in.Scan() {
		line := bytes.TrimSpace(in.Bytes())
		if len(line) == 0 {
			continue
		}
		n++
		body := line
		var wrapped struct {
			Body json.RawMessage `json:"body"`
		}
		if err := json.Unmarshal(line, &wrapped); err == nil && len(wrapped.Body) > 0 {
			body = wrapped.Body
		}
		res, err := render(body)
		if err != nil {
			fail("line %d: %v", n, err)
		}
		b, _ := json.Marshal(res)
		w.Write(b)
		w.WriteByte('\n')
	}
	if err := in.Err(); err != nil {
		fail("read: %v", err)
	}
}

func fail(format string, args ...any) {
	fmt.Fprintf(os.Stderr, "servingrender: "+format+"\n", args...)
	os.Exit(1)
}

// render is the server's path from the /v1/messages body to the prompt.
func render(body []byte) (*out, error) {
	// middleware/anthropic.go: bind, convert with the model's thinking descriptor, re-encode.
	var mreq anthropic.MessagesRequest
	if err := json.Unmarshal(body, &mreq); err != nil {
		return nil, fmt.Errorf("bind /v1/messages body: %w", err)
	}
	desc := renderers.ThinkingForRenderer(rendererName) // server.Model.genericThinking for a RENDERER qwen3.5 model
	if desc == nil {
		return nil, fmt.Errorf("renderer %s has no thinking descriptor", rendererName)
	}
	converted, err := anthropic.FromMessagesRequest(mreq, desc)
	if err != nil {
		return nil, fmt.Errorf("FromMessagesRequest: %w", err)
	}
	var cb bytes.Buffer
	if err := json.NewEncoder(&cb).Encode(converted); err != nil {
		return nil, err
	}
	// server/routes.go ChatHandler: bind the chat request (think read raw), resolve thinking, parser Init, render.
	var req api.ChatRequest
	bind := struct {
		*api.ChatRequest
		Think json.RawMessage `json:"think"`
	}{ChatRequest: &req}
	if err := json.Unmarshal(cb.Bytes(), &bind); err != nil {
		return nil, fmt.Errorf("bind chat request: %w", err)
	}
	if len(bind.Think) > 0 {
		if err := json.Unmarshal(bind.Think, &req.Think); err != nil {
			return nil, err
		}
	}
	think := renderers.ResolveThinking(req.Think, desc)
	p := parsers.ParserForName(parserName)
	if p == nil {
		return nil, fmt.Errorf("unknown parser %s", parserName)
	}
	var last *api.Message
	if k := len(req.Messages); k > 0 {
		last = &req.Messages[k-1]
	}
	tools := p.Init(req.Tools, last, think)
	prompt, err := renderers.RenderWithRenderer(rendererName, req.Messages, tools, think)
	if err != nil {
		return nil, fmt.Errorf("render: %w", err)
	}
	return &out{Think: fmt.Sprintf("%v", think.Value), Prompt: prompt, Messages: len(req.Messages), Tools: len(tools)}, nil
}
