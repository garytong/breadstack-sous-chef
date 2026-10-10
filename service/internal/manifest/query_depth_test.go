package manifest

import (
	"slices"
	"strings"
	"testing"
)

// The depth audit (2026-10-10): what describe_target and lookup_fields hid.

func TestItemFieldsMergesTheListsOwnKeys(t *testing.T) {
	m, _ := snapshot(t)
	l := m.Sections["promotion"].Lists["items"]
	items, panel := m.ItemFields(l)
	if panel != "PromotionDetailsPanel" {
		t.Fatalf("panel = %q", panel)
	}
	d, _ := m.Detail(l.ItemRef)
	for k := range l.Item {
		if items[k] == nil {
			t.Errorf("own key %q dropped", k)
		}
	}
	for k, f := range d.Fields {
		if items[k] != f {
			t.Errorf("panel key %q is not the panel's spec", k)
		}
	}
	// The details panel wins a key both spell; the manifest is untouched.
	if len(l.Item) != 1 {
		t.Fatalf("ItemFields wrote into the list's own item map: %v", l.Item)
	}
}

func TestListsAdvertiseUpdate(t *testing.T) {
	m, _ := snapshot(t)
	for _, c := range []struct {
		target Target
		list   string
	}{
		{Target{Kind: "section", Page: "home", SectionID: "promotion"}, "weekdayItems"}, // manifest ops: []
		{Target{Kind: "section", Page: "home", SectionID: "promotion"}, "items"},
		{Target{Kind: "branding"}, "certifications"},
	} {
		d, err := m.DescribeTarget(&c.target, "")
		if err != nil {
			t.Fatal(err)
		}
		found := false
		for _, l := range d.Lists {
			if l.Key != c.list {
				continue
			}
			found = true
			if !slices.Contains(l.Ops, "update") {
				t.Errorf("%s: ops %v without update", c.list, l.Ops)
			}
			if !slices.Contains(l.ItemKeys, "enabled") {
				t.Errorf("%s: item_keys %v without the list's own enabled", c.list, l.ItemKeys)
			}
		}
		if !found {
			t.Errorf("%s not described", c.list)
		}
	}
}

func lookupKeys(res *LookupResult) []string {
	var out []string
	for _, f := range res.Fields {
		out = append(out, f.Key)
	}
	return out
}

func TestLookupFindsKeyedFieldsByQuery(t *testing.T) {
	m, _ := snapshot(t)
	for _, q := range []string{"category name", "categories", "name"} {
		res, err := m.LookupFields(LookupRequest{Target: &Target{Kind: "global"}, Query: q, Limit: 120})
		if err != nil {
			t.Fatal(err)
		}
		i := slices.Index(lookupKeys(res), "categories.<categorySlug>.name")
		if i < 0 {
			t.Errorf("%q: categories.<categorySlug>.name not found: %v", q, lookupKeys(res))
			continue
		}
		if f := res.Fields[i]; f.Via != "keyed" || f.Panel != "CategoryDetailsPanel" {
			t.Errorf("%q: %+v", q, f)
		}
	}
	// Not without a query: an unfiltered lookup lists the fields only.
	res, _ := m.LookupFields(LookupRequest{Target: &Target{Kind: "global"}, Limit: 120})
	for _, k := range lookupKeys(res) {
		if strings.Contains(k, "<") {
			t.Errorf("unfiltered lookup lists template %q", k)
		}
	}
	// A section type without per-location overrides gets none.
	res, _ = m.LookupFields(LookupRequest{Target: &Target{Kind: "section", Page: "home", SectionID: "newsletter"}, Query: "address"})
	for _, k := range lookupKeys(res) {
		if strings.HasPrefix(k, "locationOverrides.") {
			t.Errorf("newsletter lists %q", k)
		}
	}
}

func TestLookupMatchesWordsHelpAndReadings(t *testing.T) {
	m, _ := snapshot(t)
	// Word by word, in any order, with the owning panel of a merged target.
	res, _ := m.LookupFields(LookupRequest{Container: "globalPanels", Query: "name store"})
	i := slices.Index(lookupKeys(res), "storeName")
	if i < 0 || res.Fields[i].Panel != "BrandingSection" {
		t.Fatalf("name store: %v", res.Fields)
	}
	// The vocabulary's readings, after the query's own matches.
	res, _ = m.LookupFields(LookupRequest{Container: "globalPanels", Query: "site name", Readings: []string{"store name"}})
	if !slices.Contains(lookupKeys(res), "storeName") {
		t.Fatalf("site name → store name: %v", lookupKeys(res))
	}
	res, _ = m.LookupFields(LookupRequest{Container: "globalPanels", Query: "site name"})
	if slices.Contains(lookupKeys(res), "storeName") {
		t.Fatalf("site name found storeName without a reading")
	}
	// Help text.
	var key, help string
	for _, p := range sortedKeys(m.GlobalPanels) {
		for _, k := range SortedFieldKeys(m.GlobalPanels[p].Fields) {
			if f := m.GlobalPanels[p].Fields[k]; key == "" && len(f.Help) > 20 && !containsFold(f.Label, f.Help[:12]) {
				key, help = k, f.Help[:12]
			}
		}
	}
	if key == "" {
		t.Skip("no global field with help text")
	}
	res, _ = m.LookupFields(LookupRequest{Container: "globalPanels", Query: help, Limit: 120})
	if !slices.Contains(lookupKeys(res), key) {
		t.Fatalf("help %q did not find %s: %v", help, key, lookupKeys(res))
	}
	// A single panel says no panel: the target does.
	res, _ = m.LookupFields(LookupRequest{Target: &Target{Kind: "global", Panel: "BrandingSection"}, Query: "store"})
	for _, f := range res.Fields {
		if f.Panel != "" && f.Via != "keyed" {
			t.Errorf("%s: panel %q on a one-panel target", f.Key, f.Panel)
		}
	}
}
