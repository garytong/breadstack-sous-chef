package prompt

import (
	"bytes"
	"encoding/json"
	"testing"

	mf "github.com/breadstack/sous-chef/internal/manifest"
	"github.com/breadstack/sous-chef/internal/toolgen"
)

// The SDK params must put exactly toolgen's bytes on the wire: same names,
// order, descriptions, strict flags and schemas (canonicalised to ignore key
// order and whitespace).
func TestSDKTools_WireEqualsToolgen(t *testing.T) {
	m, err := mf.Parse(loadSnapshot(t))
	if err != nil {
		t.Fatal(err)
	}
	tools := toolgen.Tools(m)
	sdk, err := SDKTools(tools)
	if err != nil {
		t.Fatal(err)
	}
	if len(sdk) != len(toolgen.Order) {
		t.Fatalf("got %d tools, want %d", len(sdk), len(toolgen.Order))
	}
	wire, err := json.Marshal(sdk)
	if err != nil {
		t.Fatal(err)
	}
	var decoded any
	if err := json.Unmarshal(wire, &decoded); err != nil {
		t.Fatal(err)
	}
	gotCanon, err := mf.Canonical(decoded)
	if err != nil {
		t.Fatal(err)
	}
	want, err := toolgen.ToolsJSON(m)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(gotCanon, want) {
		t.Fatalf("SDK wire bytes differ from toolgen.ToolsJSON\n got %d bytes\nwant %d bytes\n got: %.400s\nwant: %.400s", len(gotCanon), len(want), gotCanon, want)
	}
	// No cache breakpoint rides on the tools (system[0] covers them).
	if bytes.Contains(wire, []byte(`"cache_control"`)) {
		t.Errorf("tools must not carry cache_control")
	}
	// Strict tools carry strict:true and additionalProperties:false at top.
	var arr []map[string]any
	_ = json.Unmarshal(wire, &arr)
	for i, tool := range arr {
		if tools[i].Strict {
			if tool["strict"] != true {
				t.Errorf("%s: strict missing", tools[i].Name)
			}
			schema := tool["input_schema"].(map[string]any)
			if schema["additionalProperties"] != false {
				t.Errorf("%s: additionalProperties not false: %v", tools[i].Name, schema["additionalProperties"])
			}
		}
	}
}

// The tool list opens the cached prefix: its bytes must not vary between
// compiles of the same manifest or between marshals of the same object.
func TestSDKTools_ByteStable(t *testing.T) {
	m, err := mf.Parse(loadSnapshot(t))
	if err != nil {
		t.Fatal(err)
	}
	first, err := SDKTools(toolgen.Tools(m))
	if err != nil {
		t.Fatal(err)
	}
	want, _ := json.Marshal(first)
	for i := 0; i < 20; i++ {
		again, _ := json.Marshal(first)
		if !bytes.Equal(again, want) {
			t.Fatalf("re-marshal %d of the same params differs", i)
		}
		next, err := SDKTools(toolgen.Tools(m))
		if err != nil {
			t.Fatal(err)
		}
		b, _ := json.Marshal(next)
		if !bytes.Equal(b, want) {
			t.Fatalf("compile %d marshals differently", i)
		}
	}
	// And the full prefix the loop sends.
	idx, err := RenderIndex(loadSnapshot(t))
	if err != nil {
		t.Fatal(err)
	}
	p1, _ := NewPrefix(idx, first).Bytes()
	for i := 0; i < 5; i++ {
		tools, _ := SDKTools(toolgen.Tools(m))
		p2, _ := NewPrefix(idx, tools).Bytes()
		if !bytes.Equal(p1, p2) {
			t.Fatalf("prefix bytes differ on compile %d", i)
		}
	}
	t.Logf("tools wire bytes: %d; prefix (tools+system) bytes: %d", len(want), len(p1))
}

func TestSDKTools_Rejects(t *testing.T) {
	if _, err := SDKTools([]toolgen.Tool{{Name: "", InputSchema: map[string]any{"type": "object"}}}); err == nil {
		t.Error("nameless tool accepted")
	}
	if _, err := SDKTools([]toolgen.Tool{{Name: "x", InputSchema: map[string]any{"type": "array"}}}); err == nil {
		t.Error("non-object schema accepted")
	}
	if _, err := SDKTools([]toolgen.Tool{{Name: "x", InputSchema: map[string]any{"type": "object", "required": []any{1}}}}); err == nil {
		t.Error("non-string required accepted")
	}
	got, err := SDKTools([]toolgen.Tool{{Name: "x", InputSchema: map[string]any{"type": "object", "required": []any{"a"}, "properties": map[string]any{"a": map[string]any{"type": "string"}}}}})
	if err != nil || len(got) != 1 {
		t.Fatalf("[]any required rejected: %v", err)
	}
	wire, _ := json.Marshal(got)
	if !bytes.Contains(wire, []byte(`"input_schema":{"properties":{"a":{"type":"string"}},"required":["a"],"type":"object"}`)) {
		t.Errorf("wire = %s", wire)
	}
}
