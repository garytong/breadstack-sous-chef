package fields

import (
	"strings"

	"github.com/breadstack/sous-chef/internal/manifest"
)

// Follow-ups. A merchant answers the assistant's question in a word or two
// ("6", "yes please", "okay, lora", "4, you pick"): the text alone names
// nothing, so the block built from it lists whatever happens to share a
// word ("yes" → the age gate's Yes button). Such a turn is read together
// with the turn before it: the merchant's previous message and the
// assistant's reply to it say what the answer is about.

// Previous is the turn before this one: the merchant's text and the
// assistant's final reply to it.
type Previous struct {
	User      string `json:"user"`
	Assistant string `json:"assistant"`
}

// followUpMaxWords is the most content words (stop words dropped) a
// follow-up has.
const followUpMaxWords = 3

// CandidatesAfter is Candidates for a turn that follows prev in the same
// conversation. When the text is a follow-up (IsFollowUp) the shortlist is
// built from prev.User, prev.Assistant and the text, in that order;
// otherwise, or with an empty prev, it is exactly Candidates(m, ctx, text,
// opts). Pure, like Candidates: the loop passes the stored previous
// exchange, the lab the recorded one.
func CandidatesAfter(m *manifest.Manifest, ctx Context, text string, prev Previous, opts Options) Result {
	if strings.TrimSpace(prev.User) == "" && strings.TrimSpace(prev.Assistant) == "" {
		return Candidates(m, ctx, text, opts)
	}
	if !IsFollowUp(m, ctx, text) {
		return Candidates(m, ctx, text, opts)
	}
	return Candidates(m, ctx, FollowUpText(prev, text), opts)
}

// FollowUpText is the text a follow-up's shortlist is built from.
func FollowUpText(prev Previous, text string) string {
	var parts []string
	for _, s := range []string{prev.User, prev.Assistant, text} {
		if s = strings.TrimSpace(s); s != "" {
			parts = append(parts, s)
		}
	}
	return strings.Join(parts, "\n")
}

// IsFollowUp reports whether text reads as an answer to the turn before
// it: at most followUpMaxWords content words, and it names no target (no
// section, panel, branding or page by a name, title or id) and spells no
// field's label or key. "the name of my site is HashiTong" names the
// Branding panel ("site name") and is not one.
func IsFollowUp(m *manifest.Manifest, ctx Context, text string) bool {
	toks, nameText, all := scanTextAll(text)
	if m == nil || len(toks) == 0 || len(toks) > followUpMaxWords {
		return false
	}
	groups := buildGroups(m, ctx, nameText, toks, all)
	for i := range groups {
		g := &groups[i]
		if g.name >= namedMin && (len(g.nameAt) > 0 || (g.explicit && !g.byView) || g.byOutline) {
			return false
		}
		scoreGroup(m, g, text, toks)
		if len(g.labelAt) > 0 {
			return false
		}
		for _, c := range g.cands {
			if len(c.key) >= 4 && containsWord(text, c.key) {
				return false
			}
		}
	}
	return true
}
