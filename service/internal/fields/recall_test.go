package fields

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strconv"
	"strings"
	"testing"

	"github.com/breadstack/sous-chef/internal/manifest"
)

// TestRecallOffline measures recall@block on a lab cases file. It is an
// offline harness, skipped unless FIELDS_RECALL_CASES names a cases.jsonl
// (rows: request, context, expected.ops, holdout) and FIELDS_RECALL_MANIFEST
// the manifest those cases were generated from. FIELDS_RECALL_OUT, when set,
// receives one JSON line per case. Nothing is written otherwise.
//
//	FIELDS_RECALL_CASES=lab/data/int-run3/cases.jsonl \
//	FIELDS_RECALL_MANIFEST=lab/data/int-run3/manifest.json \
//	go test ./internal/fields -run TestRecallOffline -v
func TestRecallOffline(t *testing.T) {
	casesPath, manPath := os.Getenv("FIELDS_RECALL_CASES"), os.Getenv("FIELDS_RECALL_MANIFEST")
	if casesPath == "" || manPath == "" {
		t.Skip("set FIELDS_RECALL_CASES and FIELDS_RECALL_MANIFEST to run the offline recall harness")
	}
	// Stress knobs: FIELDS_RECALL_ABLATE=1 switches off the bonuses for a
	// label or key spelled out in the text; FIELDS_RECALL_DROP=<n> drops
	// every n-th unquoted word of the request (a terse paraphrase).
	if os.Getenv("FIELDS_RECALL_ABLATE") != "" {
		p, k := phraseBonus, keyBonus
		phraseBonus, keyBonus = 0, 0
		defer func() { phraseBonus, keyBonus = p, k }()
	}
	drop, _ := strconv.Atoi(os.Getenv("FIELDS_RECALL_DROP"))
	raw, err := os.ReadFile(manPath)
	if err != nil {
		t.Fatal(err)
	}
	m, err := manifest.Parse(raw)
	if err != nil {
		t.Fatal(err)
	}
	f, err := os.Open(casesPath)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	var out *bufio.Writer
	if p := os.Getenv("FIELDS_RECALL_OUT"); p != "" {
		of, err := os.Create(p)
		if err != nil {
			t.Fatal(err)
		}
		defer of.Close()
		out = bufio.NewWriter(of)
		defer out.Flush()
	}

	type op struct {
		Op     string                     `json:"op"`
		Target *manifest.Target           `json:"target"`
		Values map[string]json.RawMessage `json:"values"`
		Keys   []string                   `json:"keys"`
		List   string                     `json:"list"`
		Item   json.RawMessage            `json:"item"` // an object, or a scalar for a list of scalars
	}
	type row struct {
		ID       string          `json:"id"`
		Kind     string          `json:"kind"`
		Request  string          `json:"request"`
		Context  json.RawMessage `json:"context"`
		Holdout  bool            `json:"holdout"`
		Expected struct {
			Ops []op `json:"ops"`
		} `json:"expected"`
	}
	type tally struct {
		n, keyed, hit int
		chars         []int
	}
	sets := map[string]*tally{"all": {}, "holdout": {}, "train": {}}
	missKinds := map[string]int{}
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 64<<20), 64<<20)
	for sc.Scan() {
		var r row
		if err := json.Unmarshal(sc.Bytes(), &r); err != nil {
			t.Fatal(err)
		}
		ctx, err := ParseContext(r.Context)
		if err != nil {
			t.Fatal(err)
		}
		if drop > 1 {
			r.Request = dropWords(r.Request, drop)
		}
		res := Candidates(m, ctx, r.Request, Options{})
		if again := Candidates(m, ctx, r.Request, Options{}); again.Block != res.Block {
			t.Fatalf("%s: block is not deterministic", r.ID)
		}
		var missing []string
		keyed := 0
		for _, o := range r.Expected.Ops {
			if o.Target == nil {
				continue // pick_asset, set_page_layout, reset_pinned_section: no field key
			}
			var want []Field
			switch o.Op {
			case "set_fields":
				for k := range o.Values {
					want = append(want, Field{Key: k})
				}
			case "reset", "restore_default":
				for _, k := range o.Keys {
					want = append(want, Field{Key: k})
				}
			case "edit_list":
				var item map[string]json.RawMessage
				_ = json.Unmarshal(o.Item, &item)
				if len(item) == 0 {
					want = append(want, Field{List: o.List})
				}
				for k := range item {
					want = append(want, Field{Key: k, List: o.List})
				}
			}
			for _, w := range want {
				keyed++
				if !listed(m, res, *o.Target, w) {
					missing = append(missing, fmt.Sprintf("%s %s list=%q key=%q", o.Op, targetJSON(*o.Target), w.List, w.Key))
				}
			}
		}
		sort.Strings(missing)
		names := []string{"all", "train"}
		if r.Holdout {
			names = []string{"all", "holdout"}
		}
		for _, n := range names {
			s := sets[n]
			s.n++
			s.chars = append(s.chars, len(res.Block))
			if keyed > 0 {
				s.keyed++
				if len(missing) == 0 {
					s.hit++
				}
			}
		}
		if len(missing) > 0 {
			missKinds[r.Kind]++
		}
		if out != nil {
			b, _ := json.Marshal(map[string]any{
				"id": r.ID, "kind": r.Kind, "holdout": r.Holdout, "request": r.Request, "keyed": keyed,
				"hit": keyed > 0 && len(missing) == 0, "missing": missing, "chars": len(res.Block),
				"targets": res.Targets, "block": res.Block,
			})
			out.Write(b)
			out.WriteByte('\n')
		}
	}
	if err := sc.Err(); err != nil {
		t.Fatal(err)
	}
	for _, n := range []string{"all", "holdout", "train"} {
		s := sets[n]
		sort.Ints(s.chars)
		pct := func(p float64) int {
			if len(s.chars) == 0 {
				return 0
			}
			return s.chars[int(p*float64(len(s.chars)-1))]
		}
		t.Logf("%-8s cases=%d keyed=%d recall@block=%d/%d=%.1f%%  chars p50=%d p95=%d max=%d",
			n, s.n, s.keyed, s.hit, s.keyed, 100*float64(s.hit)/float64(max(s.keyed, 1)), pct(0.5), pct(0.95), pct(1))
	}
	var ks []string
	for k, v := range missKinds {
		ks = append(ks, fmt.Sprintf("%s=%d", k, v))
	}
	sort.Strings(ks)
	t.Logf("misses by kind: %s", strings.Join(ks, " "))
}

// dropWords removes every n-th word that is not quoted, parenthesised or
// an id (a word with a digit, dash, # or /), so the target stays
// addressable: what is stressed is the field match.
func dropWords(s string, n int) string {
	const quotes = "'\"‘’“”"
	var out []string
	inQuote, count := false, 0
	for _, w := range strings.Fields(s) {
		opens := strings.ContainsAny(w[:1], quotes+"(")
		closes := strings.ContainsAny(strings.TrimRight(w, ".,:;!?)")+")", quotes+")") && len(w) > 1 && strings.ContainsAny(w[1:], quotes+")")
		keep := inQuote || opens || strings.ContainsAny(w, "0123456789-#/")
		if opens && !closes {
			inQuote = true
		} else if inQuote && closes {
			inQuote = false
		}
		if !keep {
			count++
			if count%n == 0 {
				continue
			}
		}
		out = append(out, w)
	}
	return strings.Join(out, " ")
}

// listed reports whether the block lists want under target.
func listed(m *manifest.Manifest, res Result, target manifest.Target, want Field) bool {
	for _, tg := range res.Targets {
		match := false
		for _, t := range tg.Targets {
			if t.Kind != target.Kind {
				continue
			}
			switch t.Kind {
			case "section":
				match = match || (t.SectionID == target.SectionID && (target.Page == "" || t.Page == target.Page))
			case "page":
				match = match || t.Page == target.Page
			case "global":
				match = match || target.Panel == "" || t.Panel == target.Panel
			default:
				match = true
			}
		}
		if !match {
			continue
		}
		if want.Key == "" {
			for _, l := range tg.Lists {
				if l == want.List {
					return true
				}
			}
			continue
		}
		for _, f := range tg.Fields {
			if f.Key == want.Key && f.List == want.List {
				return true
			}
		}
	}
	return false
}
