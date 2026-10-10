package main

import (
	"bytes"
	"encoding/json"
	"os"
	"testing"

	"github.com/breadstack/sous-chef/internal/prompt"
	"github.com/breadstack/sous-chef/internal/registry"
)

const manifestPath = "../../testdata/manifest.snapshot.json"

// The CLI prints the prefix the registry compiles — the bytes every turn's
// request carries (system[0], system[1], tools).
func TestCLIMatchesRegistry(t *testing.T) {
	raw, err := os.ReadFile(manifestPath)
	if err != nil {
		t.Fatal(err)
	}
	c, err := registry.Compile(raw)
	if err != nil {
		t.Fatal(err)
	}
	var buf bytes.Buffer
	if err := run([]string{"-manifest", manifestPath}, &buf); err != nil {
		t.Fatal(err)
	}
	var got out
	if err := json.Unmarshal(buf.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.ManifestHash != c.Hash || len(got.System) != 2 || got.System[0] != prompt.Rules || got.System[1] != c.Prefix.Index {
		t.Fatalf("prefix differs: hash %s vs %s, %d system blocks", got.ManifestHash, c.Hash, len(got.System))
	}
	wire, err := c.Prefix.Bytes()
	if err != nil {
		t.Fatal(err)
	}
	var w struct {
		Tools json.RawMessage `json:"tools"`
	}
	if err := json.Unmarshal(wire, &w); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got.Tools, w.Tools) {
		t.Fatal("tools differ from the wire prefix")
	}

	for part, want := range map[string]string{"index": c.Prefix.Index, "rules": prompt.Rules, "tools": string(w.Tools)} {
		buf.Reset()
		args := []string{"-part", part}
		if part != "rules" {
			args = append(args, "-manifest", manifestPath)
		}
		if err := run(args, &buf); err != nil {
			t.Fatal(part, err)
		}
		if buf.String() != want {
			t.Fatalf("-part %s differs", part)
		}
	}
	if err := run([]string{"-part", "index"}, &buf); err == nil {
		t.Fatal("-part index without -manifest must fail")
	}
	if err := run([]string{"-part", "bogus", "-manifest", manifestPath}, &buf); err == nil {
		t.Fatal("unknown -part must fail")
	}
}
