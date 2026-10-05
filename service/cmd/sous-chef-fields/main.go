// Command sous-chef-fields renders the candidate-fields block of a turn
// (internal/fields) outside the service: manifest + context + text in, the
// block out. The lab shells out to it so a training row carries the very
// bytes the service puts in the served turn — one implementation, no port.
//
// One turn:
//
//	sous-chef-fields -manifest manifest.json -context context.json -text "make the hero headline red"
//	sous-chef-fields -manifest manifest.json -context - -text "…" < context.json
//
// prints the block and a newline (nothing at all when no field matches);
// with -json it prints {"block": …, "targets": […]} instead.
//
// Many turns (one process, the manifest parsed once):
//
//	sous-chef-fields -manifest manifest.json -jsonl < rows.jsonl > blocks.jsonl
//
// reads one {"id"?, "context"?, "text"} object per line and writes one
// {"id", "block", "targets"} line for each, in order. "request" is accepted
// for "text" (the lab's case rows).
package main

import (
	"bufio"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/breadstack/sous-chef/internal/fields"
	"github.com/breadstack/sous-chef/internal/manifest"
)

func main() {
	if err := run(os.Args[1:], os.Stdin, os.Stdout); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			os.Exit(2)
		}
		fmt.Fprintln(os.Stderr, "sous-chef-fields:", err)
		os.Exit(1)
	}
}

type row struct {
	ID      json.RawMessage `json:"id,omitempty"`
	Context json.RawMessage `json:"context,omitempty"`
	Text    string          `json:"text"`
	Request string          `json:"request,omitempty"`
}

type out struct {
	ID      json.RawMessage `json:"id,omitempty"`
	Block   string          `json:"block"`
	Targets []fields.Target `json:"targets"`
}

func run(args []string, stdin io.Reader, stdout io.Writer) error {
	fs := flag.NewFlagSet("sous-chef-fields", flag.ContinueOnError)
	def := fields.DefaultOptions()
	var (
		manifestPath = fs.String("manifest", "", "manifest JSON file (schema v1), required")
		contextPath  = fs.String("context", "", "turn context JSON file, or - for stdin (optional)")
		text         = fs.String("text", "", "the merchant's text")
		asJSON       = fs.Bool("json", false, "print {block, targets} instead of the block")
		jsonl        = fs.Bool("jsonl", false, "read {id?, context?, text} lines on stdin, write {id, block, targets} lines")
		opts         fields.Options
	)
	fs.IntVar(&opts.MaxChars, "max-chars", def.MaxChars, "cap on the block's size")
	fs.IntVar(&opts.MaxTargets, "max-targets", def.MaxTargets, "cap on the field lists")
	fs.IntVar(&opts.MaxFields, "max-fields", def.MaxFields, "cap on the fields listed under one target")
	fs.IntVar(&opts.ListAll, "list-all", def.ListAll, "container size up to which every field is listed")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *manifestPath == "" {
		fs.Usage()
		return errors.New("-manifest is required")
	}
	raw, err := os.ReadFile(*manifestPath)
	if err != nil {
		return err
	}
	m, err := manifest.Parse(raw)
	if err != nil {
		return err
	}
	w := bufio.NewWriter(stdout)
	defer w.Flush()
	enc := json.NewEncoder(w)
	enc.SetEscapeHTML(false)

	if *jsonl {
		if *contextPath != "" || *text != "" {
			return errors.New("-jsonl takes its turns from stdin: drop -context and -text")
		}
		in := bufio.NewScanner(stdin)
		in.Buffer(make([]byte, 1<<20), 64<<20)
		for n := 1; in.Scan(); n++ {
			if len(in.Bytes()) == 0 {
				continue
			}
			var r row
			if err := json.Unmarshal(in.Bytes(), &r); err != nil {
				return fmt.Errorf("line %d: %w", n, err)
			}
			if r.Text == "" {
				r.Text = r.Request
			}
			ctx, err := fields.ParseContext(r.Context)
			if err != nil {
				return fmt.Errorf("line %d: %w", n, err)
			}
			res := fields.Candidates(m, ctx, r.Text, opts)
			if err := enc.Encode(out{ID: r.ID, Block: res.Block, Targets: nonNil(res.Targets)}); err != nil {
				return err
			}
		}
		return in.Err()
	}

	var ctxRaw []byte
	switch *contextPath {
	case "":
	case "-":
		if ctxRaw, err = io.ReadAll(stdin); err != nil {
			return err
		}
	default:
		if ctxRaw, err = os.ReadFile(*contextPath); err != nil {
			return err
		}
	}
	ctx, err := fields.ParseContext(ctxRaw)
	if err != nil {
		return err
	}
	res := fields.Candidates(m, ctx, *text, opts)
	if *asJSON {
		return enc.Encode(out{Block: res.Block, Targets: nonNil(res.Targets)})
	}
	if res.Block != "" {
		fmt.Fprintln(w, res.Block)
	}
	return nil
}

func nonNil(t []fields.Target) []fields.Target {
	if t == nil {
		return []fields.Target{}
	}
	return t
}
