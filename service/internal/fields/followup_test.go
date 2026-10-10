package fields

import (
	"testing"

	"github.com/breadstack/sous-chef/internal/manifest"
)

func TestIsFollowUp(t *testing.T) {
	m, _ := snapshot(t)
	home := ctxOf(t, homeContext)
	for text, want := range map[string]bool{
		// Answers to the assistant's question (the 10-06 real turns).
		"yes please":          true,
		"6":                   true,
		"okay, lora":          true,
		"4, you pick":         true,
		"max":                 true,
		"the second one":      true,
		"yeah make it yellow": true,
		// Names a target or spells a field: read on its own.
		"the name of my site is HashiTong": false, // "site name" → the Branding panel
		"make the hero headline red":       false,
		"hide the newsletter":              false,
		"Store Name":                       false, // a label
		"storeName":                        false, // a key
		"the footer":                       false,
		// More than three content words.
		"Free delivery on orders over $60": false,
		"":                                 false,
	} {
		if got := IsFollowUp(m, home, text); got != want {
			t.Errorf("IsFollowUp(%q) = %v, want %v", text, got, want)
		}
	}
}

func TestCandidatesAfter(t *testing.T) {
	m, _ := snapshot(t)
	home := ctxOf(t, homeContext)
	prev := Previous{
		User:      "there are too many categories in the header",
		Assistant: "How many categories should the header show, and which ones should stay?",
	}
	shop := manifest.Target{Kind: "global", Panel: "ShopMenuSection"}
	res := CandidatesAfter(m, home, "4, you pick", prev, Options{})
	if tg := find(res, shop); tg == nil || !has(tg, "", "shopMenuMaxItems") {
		t.Fatalf("follow-up: shopMenuMaxItems not listed\n%s", res.Block)
	}
	if want := Candidates(m, home, FollowUpText(prev, "4, you pick"), Options{}); want.Block != res.Block {
		t.Fatalf("follow-up block is not the joined text's")
	}
	// Not a follow-up, or no history: exactly Candidates.
	for _, tc := range []struct {
		text string
		prev Previous
	}{
		{"the name of my site is HashiTong", prev},
		{"4, you pick", Previous{}},
	} {
		if got, want := CandidatesAfter(m, home, tc.text, tc.prev, Options{}), Candidates(m, home, tc.text, Options{}); got.Block != want.Block {
			t.Errorf("%q: block differs from Candidates\n%s", tc.text, got.Block)
		}
	}
}
