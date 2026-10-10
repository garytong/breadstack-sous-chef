// Command sous-chef-prefix prints the cached prompt prefix the service builds
// for one manifest — system[0] (the rules), system[1] (the manifest index)
// and the tools — exactly as a model request carries them. The lab re-renders
// built training rows with it (souschef_lab/fields.py `rerender --system`),
// so a row's system turn is the very text the served student reads; one
// implementation (registry.Compile, the path every turn takes), no port.
//
//	sous-chef-prefix -manifest manifest.json            {"manifest_hash", "system": [rules, index], "tools": [...]}
//	sous-chef-prefix -manifest manifest.json -part index  system[1] as plain text (no trailing newline added)
//	sous-chef-prefix -part rules                         system[0] (needs no manifest)
package main

import (
	"bufio"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/breadstack/sous-chef/internal/prompt"
	"github.com/breadstack/sous-chef/internal/registry"
)

func main() {
	if err := run(os.Args[1:], os.Stdout); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			os.Exit(2)
		}
		fmt.Fprintln(os.Stderr, "sous-chef-prefix:", err)
		os.Exit(1)
	}
}

// out is the default (-part all) output.
type out struct {
	ManifestHash string          `json:"manifest_hash"`
	System       []string        `json:"system"`
	Tools        json.RawMessage `json:"tools"`
}

func run(args []string, stdout io.Writer) error {
	fs := flag.NewFlagSet("sous-chef-prefix", flag.ContinueOnError)
	var (
		manifestPath = fs.String("manifest", "", "manifest JSON file (schema v1); required unless -part rules")
		part         = fs.String("part", "all", "all | rules | index | tools")
	)
	if err := fs.Parse(args); err != nil {
		return err
	}
	w := bufio.NewWriter(stdout)
	defer w.Flush()
	if *part == "rules" {
		_, err := io.WriteString(w, prompt.Rules)
		return err
	}
	switch *part {
	case "all", "index", "tools":
	default:
		return fmt.Errorf("-part %q: want all, rules, index or tools", *part)
	}
	if *manifestPath == "" {
		fs.Usage()
		return errors.New("-manifest is required")
	}
	raw, err := os.ReadFile(*manifestPath)
	if err != nil {
		return err
	}
	c, err := registry.Compile(raw)
	if err != nil {
		return err
	}
	tools, err := json.Marshal(c.Prefix.Tools)
	if err != nil {
		return err
	}
	switch *part {
	case "index":
		_, err = io.WriteString(w, c.Prefix.Index)
		return err
	case "tools":
		_, err = w.Write(tools)
		return err
	}
	enc := json.NewEncoder(w)
	enc.SetEscapeHTML(false)
	return enc.Encode(out{ManifestHash: c.Hash, System: []string{c.Prefix.Rules, c.Prefix.Index}, Tools: tools})
}
