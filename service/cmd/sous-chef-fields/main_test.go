package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/breadstack/sous-chef/internal/fields"
	"github.com/breadstack/sous-chef/internal/manifest"
)

const manifestPath = "../../testdata/manifest.snapshot.json"

const ctxJSON = `{"page":"home","sections":[{"id":"heroBanner","type":"heroBanner","title":"Fresh picks every week"},{"id":"newsletter","type":"newsletter","title":""}]}`

// The CLI must print exactly what the package returns: the lab's rows and
// the service's turns come from the two paths.
func TestCLIMatchesPackage(t *testing.T) {
	raw, err := os.ReadFile(manifestPath)
	if err != nil {
		t.Fatal(err)
	}
	m, err := manifest.Parse(raw)
	if err != nil {
		t.Fatal(err)
	}
	ctx, err := fields.ParseContext([]byte(ctxJSON))
	if err != nil {
		t.Fatal(err)
	}
	text := "Change the Max Width on the hero banner to 740."
	want := fields.Candidates(m, ctx, text, fields.Options{})
	if want.Block == "" {
		t.Fatal("empty block")
	}

	ctxFile := filepath.Join(t.TempDir(), "context.json")
	if err := os.WriteFile(ctxFile, []byte(ctxJSON), 0o644); err != nil {
		t.Fatal(err)
	}
	var out bytes.Buffer
	if err := run([]string{"-manifest", manifestPath, "-context", ctxFile, "-text", text}, strings.NewReader(""), &out); err != nil {
		t.Fatal(err)
	}
	if out.String() != want.Block+"\n" {
		t.Fatalf("plain mode differs:\n%s", out.String())
	}

	out.Reset()
	if err := run([]string{"-manifest", manifestPath, "-context", "-", "-text", text, "-json"}, strings.NewReader(ctxJSON), &out); err != nil {
		t.Fatal(err)
	}
	var one struct {
		Block   string          `json:"block"`
		Targets []fields.Target `json:"targets"`
	}
	if err := json.Unmarshal(out.Bytes(), &one); err != nil {
		t.Fatal(err)
	}
	if one.Block != want.Block || len(one.Targets) != len(want.Targets) {
		t.Fatalf("-json differs:\n%s", out.String())
	}

	out.Reset()
	in := `{"id":"a","context":` + ctxJSON + `,"text":` + jsonString(text) + `}` + "\n\n" +
		`{"id":2,"request":"zzz qqq"}` + "\n"
	if err := run([]string{"-manifest", manifestPath, "-jsonl"}, strings.NewReader(in), &out); err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(strings.TrimSpace(out.String()), "\n")
	if len(lines) != 2 {
		t.Fatalf("want 2 lines, got %d:\n%s", len(lines), out.String())
	}
	var a, b struct {
		ID      json.RawMessage `json:"id"`
		Block   string          `json:"block"`
		Targets []fields.Target `json:"targets"`
	}
	if err := json.Unmarshal([]byte(lines[0]), &a); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal([]byte(lines[1]), &b); err != nil {
		t.Fatal(err)
	}
	if string(a.ID) != `"a"` || a.Block != want.Block {
		t.Fatalf("-jsonl row 1 differs: %s", lines[0])
	}
	if string(b.ID) != `2` || b.Block != "" || b.Targets == nil {
		t.Fatalf("-jsonl row 2: want an empty block and [] targets: %s", lines[1])
	}
}

func TestCLIErrors(t *testing.T) {
	var out bytes.Buffer
	if err := run([]string{"-text", "x"}, strings.NewReader(""), &out); err == nil {
		t.Fatal("missing -manifest must fail")
	}
	if err := run([]string{"-manifest", manifestPath, "-jsonl", "-text", "x"}, strings.NewReader(""), &out); err == nil {
		t.Fatal("-jsonl with -text must fail")
	}
	if err := run([]string{"-manifest", manifestPath, "-jsonl"}, strings.NewReader("not json\n"), &out); err == nil {
		t.Fatal("a bad line must fail")
	}
	if err := run([]string{"-manifest", manifestPath, "-context", "-", "-text", "x"}, strings.NewReader("[1]"), &out); err == nil {
		t.Fatal("a non-object context must fail")
	}
}

func jsonString(s string) string {
	b, _ := json.Marshal(s)
	return string(b)
}
