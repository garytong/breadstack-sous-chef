package manifest

import (
	"errors"
	"fmt"
	"regexp"
	"sort"
	"strings"
)

// Target is the engine's op target union (cookie-cutter
// launcher/client/src/sousChef/README.md §1):
//
//	{kind:"section", page, sectionId, storeId?} | {kind:"global", panel?} |
//	{kind:"branding"} | {kind:"page", page}
type Target struct {
	Kind      string `json:"kind"`
	Page      string `json:"page,omitempty"`
	SectionID string `json:"sectionId,omitempty"`
	StoreID   string `json:"storeId,omitempty"`
	Panel     string `json:"panel,omitempty"`
}

// TargetKinds are the accepted `kind` values, in schema order.
var TargetKinds = []string{"section", "global", "branding", "page"}

// Static tables copied from cookie-cutter launcher/client/src/sousChef/ops/schema.js
// so the service addresses sections the way the browser engine does.
const (
	// CommonPageKey holds fields templated over every page (`{page}` in path).
	CommonPageKey = "_common"
	// PageTemplateToken is substituted with the page slug in _common paths.
	PageTemplateToken = "{page}"
	// SEOPage is the pseudo-page that edits details.SeoPanel (`item.` → `seo.`).
	SEOPage = "seo"
	// SectionsListKey is the page's own section list, synthesised by the engine
	// (`edit_list {list:"sections"}` on a page target); never a manifest list.
	SectionsListKey = "sections"
	// DetailItemPrefix prefixes every details-panel field path (`item.name`).
	DetailItemPrefix = "item."
	// PinnedPrefix marks the editor's built-in page rows.
	PinnedPrefix = "__pinned_"
	// SectionPathRoot is the root of section-relative paths (`settings.*`).
	SectionPathRoot = "settings"
)

// SectionsListOps are the actions the synthesised `sections` list accepts.
var SectionsListOps = []string{"add", "remove", "reorder", "update", "duplicate"}

// ListActions are every edit_list action; `update` is implicit on any list
// and `duplicate` only exists on the page `sections` list.
var ListActions = []string{"add", "remove", "reorder", "update", "duplicate"}

// repeatableSuffix strips `-<n>` from a section id (`heroBanner-2` → `heroBanner`).
var repeatableSuffix = regexp.MustCompile(`-\d+$`)

// SectionPrefixFamilies are section families resolveSectionEditor routes by
// `id.startsWith(family)`, so any suffix (`locations-x`) still edits as the family.
var SectionPrefixFamilies = []string{
	"locations", "productCarousel", "contentBlock", "sectionDivider", "gallery", "milestones",
}

// Pinned is a built-in page row (BasicEditor PINNED_BY_PAGE). Only a
// reorderable row persists a `{id, enabled}` placeholder in the page's list.
type Pinned struct {
	ID          string `json:"id"`
	Reorderable bool   `json:"reorderable"`
	Label       string `json:"label"`
}

// PinnedByPage is the built-in row of every page that has one.
var PinnedByPage = map[string]Pinned{
	"faq":                {ID: "__pinned_faq", Label: "FAQ"},
	"product-details":    {ID: "__pinned_product_detail", Label: "Product Detail"},
	"products":           {ID: "__pinned_product_list", Reorderable: true, Label: "Product List"},
	"blog":               {ID: "__pinned_blog_list", Label: "Blog List"},
	"blog-detail":        {ID: "__pinned_blog_detail", Label: "Blog Detail"},
	"privacy-policy":     {ID: "__pinned_privacy", Label: "Privacy Policy"},
	"terms-of-service":   {ID: "__pinned_terms", Label: "Terms of Service"},
	"return-refund":      {ID: "__pinned_return_refund", Label: "Refund & Return Policy"},
	"age-gate":           {ID: "__pinned_age_gate", Label: "Age Gate"},
	"location-selection": {ID: "__pinned_location_selection", Label: "Location Selection"},
	"cart":               {ID: "__pinned_cart", Label: "Cart"},
	"checkout":           {ID: "__pinned_checkout", Label: "Checkout"},
	"account":            {ID: "__pinned_account", Label: "Account"},
	"login":              {ID: "__pinned_login", Label: "Login"},
	"order-confirmation": {ID: "__pinned_order_confirmation", Label: "Order Confirmation"},
	"order-details":      {ID: "__pinned_order_details", Label: "Track Order"},
}

// KeyedPrefixes are dotted field keys whose middle segment is a slug or
// location id; the tail validates against the named details panel.
//
//	section  locationOverrides.<locId>.<f>        → LocationDetailsPanel
//	global   categories.<slug>.<f>                → CategoryDetailsPanel
//	global   footerLocationOverrides.<locId>.<f>  → LocationDetailsPanel
var KeyedPrefixes = map[string]map[string]string{
	"section": {"locationOverrides": "LocationDetailsPanel"},
	"global":  {"categories": "CategoryDetailsPanel", "footerLocationOverrides": "LocationDetailsPanel"},
}

// ErrorCodes are the engine's op error codes (schema.js ERRORS), listed in
// the prompt index so the model can read a browser failure.
var ErrorCodes = []string{
	"E_TARGET", "E_FIELD", "E_READONLY", "E_IMAGE", "E_PINNED", "E_VALUE",
	"E_OP", "E_LIST", "E_ACTION", "E_INDEX", "E_ORDER", "E_LIMIT",
}

// WarningCodes are the engine's warning codes.
var WarningCodes = []string{"W_HIDDEN"}

// ErrTarget is wrapped by every target resolution failure (maps to E_TARGET).
var ErrTarget = errors.New("E_TARGET")

// Resolved is a target (or a container path) turned into the field and list
// specs that apply to it.
type Resolved struct {
	Kind string `json:"kind"`
	// Container is the manifest container that backs the target
	// ("sections.heroBanner", "pages.faq", "branding", "globalPanels" for the
	// merged global view, "details.SeoPanel" for page seo).
	Container ContainerPath `json:"container"`
	// Panels lists the global panels merged into a global target.
	Panels []string `json:"panels,omitempty"`
	// TypeID is the section type for a section target.
	TypeID    string `json:"typeId,omitempty"`
	EditorKey string `json:"editorKey,omitempty"`
	Page      string `json:"page,omitempty"`
	SectionID string `json:"sectionId,omitempty"`
	// Pinned is set when a section target names the page's built-in row,
	// which edits the page's own fields.
	Pinned *Pinned `json:"pinned,omitempty"`
	// Known is false for a page target the manifest has no entry for (a
	// custom page or a template page without page-level fields): only the
	// `_common` fields apply. The browser decides whether the slug exists.
	Known bool `json:"known"`

	Fields  map[string]*Field `json:"-"`
	Lists   map[string]*List  `json:"-"`
	Mirror  *Mirror           `json:"-"`
	Presets []Preset          `json:"-"`
	Pages   []string          `json:"-"`
}

// SectionTypeID strips the repeatable `-<n>` suffix from a section id.
func SectionTypeID(sectionID string) string {
	return repeatableSuffix.ReplaceAllString(sectionID, "")
}

// LookupSectionTypeID maps a section id to a manifest section type, the way
// the editor's resolveSectionEditor does: strip `-<n>`, else match a prefix
// family before the first dash. ok is false for an unknown type.
func (m *Manifest) LookupSectionTypeID(sectionID string) (string, bool) {
	stripped := SectionTypeID(sectionID)
	if stripped == "" {
		return "", false
	}
	if _, ok := m.Sections[stripped]; ok {
		return stripped, true
	}
	if dash := strings.IndexByte(stripped, '-'); dash > 0 {
		head := stripped[:dash]
		for _, fam := range SectionPrefixFamilies {
			if fam == head {
				if _, ok := m.Sections[head]; ok {
					return head, true
				}
			}
		}
	}
	return "", false
}

// IsPinnedSectionID reports whether an id names a built-in page row.
func IsPinnedSectionID(id string) bool { return strings.HasPrefix(id, PinnedPrefix) }

// Resolve turns a Target into the specs that apply to it.
func (m *Manifest) Resolve(t Target) (*Resolved, error) {
	switch t.Kind {
	case "section":
		return m.resolveSection(t)
	case "global":
		return m.resolveGlobal(t.Panel)
	case "branding":
		return &Resolved{
			Kind: "branding", Container: "branding", Known: true,
			Fields: m.Branding.Fields, Lists: m.Branding.Lists, Presets: m.Branding.Presets,
		}, nil
	case "page":
		return m.resolvePage(t.Page)
	case "":
		return nil, fmt.Errorf("%w: target needs a kind (%s)", ErrTarget, strings.Join(TargetKinds, "|"))
	}
	return nil, fmt.Errorf("%w: unknown target kind %q", ErrTarget, t.Kind)
}

func (m *Manifest) resolveSection(t Target) (*Resolved, error) {
	if t.Page == "" {
		return nil, fmt.Errorf("%w: section target needs a page", ErrTarget)
	}
	if t.SectionID == "" {
		return nil, fmt.Errorf("%w: section target needs a sectionId", ErrTarget)
	}
	if IsPinnedSectionID(t.SectionID) {
		def, ok := PinnedByPage[t.Page]
		if !ok || def.ID != t.SectionID {
			return nil, fmt.Errorf("%w: no pinned section %s on page %s", ErrTarget, t.SectionID, t.Page)
		}
		if _, ok := m.Pages[t.Page]; !ok {
			return nil, fmt.Errorf("%w: page %s has no editable fields in the manifest", ErrTarget, t.Page)
		}
		r, err := m.resolvePage(t.Page)
		if err != nil {
			return nil, err
		}
		r.Kind = "section"
		r.SectionID = t.SectionID
		d := def
		r.Pinned = &d
		return r, nil
	}
	typeID, ok := m.LookupSectionTypeID(t.SectionID)
	if !ok {
		return nil, fmt.Errorf("%w: unknown section type for %s (known: %s)", ErrTarget, t.SectionID, strings.Join(sortedKeys(m.Sections), ", "))
	}
	s := m.Sections[typeID]
	return &Resolved{
		Kind: "section", Container: "sections." + typeID, TypeID: typeID, EditorKey: s.EditorKey,
		Page: t.Page, SectionID: t.SectionID, Known: true,
		Fields: s.Fields, Lists: s.Lists, Mirror: s.Mirror, Presets: s.Presets, Pages: s.Pages,
	}, nil
}

// resolveGlobal merges every global panel (or the one named). The same key
// only ever appears in two panels with the same path, so first-wins is safe.
func (m *Manifest) resolveGlobal(panel string) (*Resolved, error) {
	var names []string
	if panel != "" {
		if _, ok := m.GlobalPanels[panel]; !ok {
			return nil, fmt.Errorf("%w: unknown global panel %q (known: %s)", ErrTarget, panel, strings.Join(sortedKeys(m.GlobalPanels), ", "))
		}
		names = []string{panel}
	} else {
		names = sortedKeys(m.GlobalPanels)
	}
	r := &Resolved{Kind: "global", Container: "globalPanels", Panels: names, Known: true,
		Fields: map[string]*Field{}, Lists: map[string]*List{}}
	if panel != "" {
		r.Container = "globalPanels." + panel
	}
	for _, n := range names {
		p := m.GlobalPanels[n]
		for k, f := range p.Fields {
			if _, dup := r.Fields[k]; !dup {
				r.Fields[k] = f
			}
		}
		for k, l := range p.Lists {
			if _, dup := r.Lists[k]; !dup {
				r.Lists[k] = l
			}
		}
		r.Presets = append(r.Presets, p.Presets...)
	}
	return r, nil
}

func (m *Manifest) resolvePage(page string) (*Resolved, error) {
	if page == "" || page == CommonPageKey {
		return nil, fmt.Errorf("%w: page target needs a page slug", ErrTarget)
	}
	if page == SEOPage {
		panel, ok := m.Detail("SeoPanel")
		if !ok {
			return nil, fmt.Errorf("%w: manifest has no SeoPanel", ErrTarget)
		}
		fields := make(map[string]*Field, len(panel.Fields))
		for k, f := range panel.Fields {
			c := *f
			c.Path = SEOPage + "." + strings.TrimPrefix(f.Path, DetailItemPrefix)
			fields[k] = &c
		}
		return &Resolved{Kind: "page", Container: "details.SeoPanel", Page: page, Known: true,
			Fields: fields, Lists: map[string]*List{}}, nil
	}
	fields := map[string]*Field{}
	if common, ok := m.Pages[CommonPageKey]; ok {
		for k, f := range common.Fields {
			c := *f
			c.Path = strings.ReplaceAll(f.Path, PageTemplateToken, page)
			fields[k] = &c
		}
	}
	own, known := m.Pages[page]
	r := &Resolved{Kind: "page", Container: "pages." + page, Page: page, Known: known,
		Fields: fields, Lists: map[string]*List{}}
	if known {
		for k, f := range own.Fields {
			fields[k] = f
		}
		r.Lists = own.Lists
		r.Presets = own.Presets
	}
	return r, nil
}

// ResolveContainer resolves a container path ("sections.heroBanner",
// "globalPanels.HeaderSection", "globalPanels", "branding", "pages.faq",
// "details.ReviewDetailsPanel") the way lookup_fields' `container` argument
// addresses it. Details panels resolve to their own item fields.
func (m *Manifest) ResolveContainer(path ContainerPath) (*Resolved, error) {
	group, name := splitContainerPath(path)
	switch group {
	case "sections":
		if s, ok := m.Sections[name]; ok {
			return &Resolved{Kind: "section", Container: path, TypeID: name, EditorKey: s.EditorKey, Known: true,
				Fields: s.Fields, Lists: s.Lists, Mirror: s.Mirror, Presets: s.Presets, Pages: s.Pages}, nil
		}
	case "globalPanels":
		return m.resolveGlobal(name)
	case "branding":
		if name == "" {
			return m.Resolve(Target{Kind: "branding"})
		}
	case "pages":
		if name != "" {
			if _, ok := m.Pages[name]; ok || name == SEOPage {
				return m.resolvePage(name)
			}
		}
	case "details":
		if d, ok := m.Details[name]; ok {
			return &Resolved{Kind: "details", Container: path, Known: true, Fields: d.Fields, Lists: d.Lists}, nil
		}
	}
	return nil, fmt.Errorf("%w: unknown container %q", ErrTarget, path)
}

// FieldRef is a field found on a resolved target: Key is what an op writes,
// List is the owning list key when the field is a list-item field ("" for a
// container field), Via says how the key resolved (field, mirror, alias,
// keyed, seo-template, item).
type FieldRef struct {
	Key   string
	Spec  *Field
	List  string
	Via   string
	Panel string // details panel for keyed / item_ref fields
}

// mirrorListKey turns `slides[0]` into `slides`.
func mirrorListKey(mirror *Mirror) string {
	from := "slides[0]"
	if mirror != nil && mirror.From != "" {
		from = mirror.From
	}
	if i := strings.IndexByte(from, '['); i >= 0 {
		return from[:i]
	}
	return from
}

// ItemFields returns the per-row fields of a list: the referenced details
// panel's fields when ItemRef is set (their paths keep the `item.` prefix;
// keys are already bare), else the list's own `item` map.
func (m *Manifest) ItemFields(l *List) (map[string]*Field, string) {
	if l == nil {
		return nil, ""
	}
	if l.ItemRef != "" {
		if d, ok := m.Detail(l.ItemRef); ok {
			return d.Fields, strings.TrimPrefix(l.ItemRef, "details.")
		}
	}
	return l.Item, ""
}

// Field resolves a key on a resolved target following the engine's
// resolveField order: own field, hero mirror key or alias, keyed dotted
// prefix, seo `pages.<page>.<rest>` templating. ok is false when the key is
// not a field of the target.
func (m *Manifest) Field(r *Resolved, key string) (FieldRef, bool) {
	if r == nil || key == "" {
		return FieldRef{}, false
	}
	if f, ok := r.Fields[key]; ok {
		return FieldRef{Key: key, Spec: f, Via: "field"}, true
	}
	if r.Mirror != nil {
		canonical := ""
		for _, k := range r.Mirror.Keys {
			if k == key {
				canonical = k
				break
			}
		}
		via := "mirror"
		if canonical == "" {
			for k, aliases := range r.Mirror.Aliases {
				for _, a := range aliases {
					if a == key {
						canonical = k
						via = "alias"
					}
				}
			}
		}
		if canonical != "" {
			listKey := mirrorListKey(r.Mirror)
			if l, ok := r.Lists[listKey]; ok {
				items, panel := m.ItemFields(l)
				if f, ok := items[canonical]; ok {
					return FieldRef{Key: key, Spec: f, List: listKey, Via: via, Panel: panel}, true
				}
			}
			// A mirror key without an item spec (synthetic in the engine).
			return FieldRef{Key: key, Spec: &Field{Path: SectionPathRoot + "." + listKey + "[]." + canonical, Label: canonical}, List: listKey, Via: via}, true
		}
	}
	if table, ok := KeyedPrefixes[r.Kind]; ok {
		parts := strings.Split(key, ".")
		if len(parts) >= 3 {
			if panelName, ok := table[parts[0]]; ok && parts[1] != "" {
				if panel, ok := m.Detail(panelName); ok {
					if f, ok := panel.Fields[strings.Join(parts[2:], ".")]; ok {
						return FieldRef{Key: key, Spec: f, Via: "keyed", Panel: panelName}, true
					}
				}
			}
		}
	}
	if r.Kind == "page" && r.Page == SEOPage && strings.HasPrefix(key, "pages.") {
		rest := strings.TrimPrefix(key, "pages.")
		if dot := strings.IndexByte(rest, '.'); dot > 0 {
			if tpl, ok := r.Fields["pages.home."+rest[dot+1:]]; ok {
				return FieldRef{Key: key, Spec: tpl, Via: "seo-template"}, true
			}
		}
	}
	return FieldRef{}, false
}

// ListKeys returns the target's list keys sorted, with the synthesised
// `sections` list first on a page target (not the seo pseudo-page).
func (r *Resolved) ListKeys() []string {
	keys := sortedKeys(r.Lists)
	if r.Kind == "page" && r.Page != SEOPage {
		keys = append([]string{SectionsListKey}, keys...)
	}
	return keys
}

// ListDepth is the nesting depth of a list key: `slides` → 1, `slides[].bullets` → 2.
func ListDepth(listKey string) int { return strings.Count(listKey, "[]") + 1 }

// SortedFieldKeys returns a field map's keys in byte order.
func SortedFieldKeys(fields map[string]*Field) []string {
	keys := make([]string, 0, len(fields))
	for k := range fields {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}
