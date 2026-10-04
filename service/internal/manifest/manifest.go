// Package manifest parses the launcher's Sous Chef manifest (schema v1),
// hashes it, builds the compact prompt index, answers the model's
// lookup_fields / describe_target queries and diffs two manifests.
//
// The shape is the one launcher/client/scripts/sous-chef-manifest/emit.js
// (cookie-cutter) emits:
//
//	{ version: 1,
//	  generated_from: { branch, sha, at },
//	  sections:     { [typeId]: { editorKey, pages, fields, lists, mirror?, structural, presets? } },
//	  globalPanels: { [panel]:  { fields, lists? } },
//	  branding:     { fields, lists? },
//	  pages:        { [page]:   { fields, lists? } },   // pages._common = fields templated over {page}
//	  details:      { [panel]:  { fields, lists } },
//	  coverage:     { ... } }                             // diagnostics, excluded from the hash
//
//	FIELD = { path, label, help?, type, enum?, enum_labels?, min?, max?, step?, default?,
//	          unset_value?, companions?, aliases?, writes?, conditional_on?, hidden_when?,
//	          derived_from?, keyed_by?, ref?, suggested?, set_by?, ranges_by?, source }
//	LIST  = { path, item_ref?, item: { [key]: FIELD }, ops: ['add'|'remove'|'reorder'], orders? }
//
// `fields` and `item` are maps keyed by the field KEY (what ops address);
// `path` is the storage path the browser engine writes. Unknown members at
// every level are tolerated: the typed structs ignore them, and the raw
// document kept beside them is what the hash covers, so an unknown member
// still changes the hash.
package manifest

import (
	"encoding/json"
	"errors"
	"fmt"
)

// Version is the only manifest schema version this package accepts.
const Version = 1

// ErrVersion is returned for a manifest whose `version` is not 1.
var ErrVersion = errors.New("manifest: unsupported schema version")

// Manifest is a parsed schema-v1 manifest.
type Manifest struct {
	Version       int                   `json:"version"`
	GeneratedFrom *GeneratedFrom        `json:"generated_from,omitempty"`
	Sections      map[string]*Container `json:"sections"`
	GlobalPanels  map[string]*Container `json:"globalPanels"`
	Branding      *Container            `json:"branding"`
	Pages         map[string]*Container `json:"pages"`
	Details       map[string]*Container `json:"details"`
	// Coverage is kept opaque; it never affects behaviour or the hash.
	Coverage json.RawMessage `json:"coverage,omitempty"`

	// raw is the document as decoded into generic values — the hash input
	// (minus generated_from and coverage) and the source of truth for
	// members the structs do not model.
	raw map[string]any
	// hash is computed once by Parse.
	hash string
}

// GeneratedFrom records which launcher build produced the manifest.
type GeneratedFrom struct {
	Branch string `json:"branch,omitempty"`
	SHA    string `json:"sha,omitempty"`
	At     string `json:"at,omitempty"`
}

// Container is a section type, a global panel, the branding slice, a page or
// a details panel — anything with `fields` and `lists`.
type Container struct {
	EditorKey  string            `json:"editorKey,omitempty"`
	Pages      []string          `json:"pages,omitempty"`
	Fields     map[string]*Field `json:"fields"`
	Lists      map[string]*List  `json:"lists,omitempty"`
	Mirror     *Mirror           `json:"mirror,omitempty"`
	Structural []Structural      `json:"structural,omitempty"`
	Presets    []Preset          `json:"presets,omitempty"`
}

// Field is one editable control. Default and UnsetValue are RawMessage so a
// present-but-null value (`unset_value: null`, 1,027 times in the snapshot)
// is distinguishable from an absent one: nil = absent, "null" = present.
type Field struct {
	Path          string            `json:"path"`
	Label         string            `json:"label"`
	Help          string            `json:"help,omitempty"`
	Type          string            `json:"type"`
	Enum          []any             `json:"enum,omitempty"`
	EnumLabels    map[string]string `json:"enum_labels,omitempty"`
	Min           *float64          `json:"min,omitempty"`
	Max           *float64          `json:"max,omitempty"`
	Step          *float64          `json:"step,omitempty"`
	Default       json.RawMessage   `json:"default,omitempty"`
	UnsetValue    json.RawMessage   `json:"unset_value,omitempty"`
	Companions    map[string]string `json:"companions,omitempty"`
	Aliases       map[string]string `json:"aliases,omitempty"`
	Writes        []string          `json:"writes,omitempty"`
	ConditionalOn *Gate             `json:"conditional_on,omitempty"`
	HiddenWhen    *Gate             `json:"hidden_when,omitempty"`
	DerivedFrom   string            `json:"derived_from,omitempty"`
	KeyedBy       string            `json:"keyed_by,omitempty"`
	Ref           string            `json:"ref,omitempty"`
	Suggested     []float64         `json:"suggested,omitempty"`
	SetBy         []string          `json:"set_by,omitempty"`
	RangesBy      json.RawMessage   `json:"ranges_by,omitempty"`
	Source        json.RawMessage   `json:"source,omitempty"`
}

// Gate is a `conditional_on` / `hidden_when` clause. Value is a scalar, an
// array of scalars, or the string "set" (hidden_when).
type Gate struct {
	Key   string          `json:"key"`
	Value json.RawMessage `json:"value"`
}

// List is a repeatable collection. Item holds the per-row fields unless
// ItemRef names a details panel (`details.<Panel>`) whose fields apply with
// their `item.` path prefix.
type List struct {
	Path    string            `json:"path"`
	ItemRef string            `json:"item_ref,omitempty"`
	Item    map[string]*Field `json:"item"`
	Ops     []string          `json:"ops"`
	Orders  string            `json:"orders,omitempty"`
}

// Mirror describes the hero slide-0 mirror: every key in Keys is written
// flat on the section as well as on `From` (`slides[0]`), and Aliases maps a
// canonical key to the legacy flat keys that receive the same value.
type Mirror struct {
	From    string              `json:"from"`
	Keys    []string            `json:"keys"`
	Aliases map[string][]string `json:"aliases,omitempty"`
}

// Structural is one structural control the probe observed on a list.
type Structural struct {
	List  string  `json:"list"`
	Op    string  `json:"op"`
	Label *string `json:"label"`
	Kind  string  `json:"kind,omitempty"`
}

// Preset is a one-click bundle of writes (ProductCardSection in the snapshot).
type Preset struct {
	Name   string                     `json:"name"`
	Label  string                     `json:"label,omitempty"`
	Writes map[string]json.RawMessage `json:"writes"`
}

// Parse decodes a manifest, checks its version and computes its hash. The
// document is decoded twice: once generically (kept for hashing and so that
// unknown members survive) and once into the typed structs.
func Parse(data []byte) (*Manifest, error) {
	var raw map[string]any
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, fmt.Errorf("manifest: %w", err)
	}
	if raw == nil {
		return nil, errors.New("manifest: document is null")
	}
	var m Manifest
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, fmt.Errorf("manifest: %w", err)
	}
	if m.Version != Version {
		return nil, fmt.Errorf("%w: got %d, want %d", ErrVersion, m.Version, Version)
	}
	if m.Sections == nil {
		m.Sections = map[string]*Container{}
	}
	if m.GlobalPanels == nil {
		m.GlobalPanels = map[string]*Container{}
	}
	if m.Pages == nil {
		m.Pages = map[string]*Container{}
	}
	if m.Details == nil {
		m.Details = map[string]*Container{}
	}
	if m.Branding == nil {
		m.Branding = &Container{Fields: map[string]*Field{}}
	}
	for _, group := range []map[string]*Container{m.Sections, m.GlobalPanels, m.Pages, m.Details} {
		for _, c := range group {
			c.normalise()
		}
	}
	m.Branding.normalise()
	m.raw = raw
	h, err := hashRaw(raw)
	if err != nil {
		return nil, err
	}
	m.hash = h
	return &m, nil
}

func (c *Container) normalise() {
	if c.Fields == nil {
		c.Fields = map[string]*Field{}
	}
	if c.Lists == nil {
		c.Lists = map[string]*List{}
	}
	for _, l := range c.Lists {
		if l.Item == nil {
			l.Item = map[string]*Field{}
		}
	}
}

// Hash is hex(sha256(canonical(document minus generated_from and coverage))).
func (m *Manifest) Hash() string { return m.hash }

// Raw returns the generically decoded document (read-only by convention).
func (m *Manifest) Raw() map[string]any { return m.raw }

// Counts are the per-kind totals the register endpoint reports.
type Counts struct {
	Sections     int `json:"sections"`
	GlobalPanels int `json:"globalPanels"`
	Pages        int `json:"pages"`
	Details      int `json:"details"`
	Fields       int `json:"fields"`
	Lists        int `json:"lists"`
}

// Counts totals every field (container fields plus list item fields) and
// every list across sections, global panels, branding, pages and details.
func (m *Manifest) Counts() Counts {
	c := Counts{
		Sections:     len(m.Sections),
		GlobalPanels: len(m.GlobalPanels),
		Pages:        len(m.Pages),
		Details:      len(m.Details),
	}
	m.EachContainer(func(_ string, ct *Container) {
		c.Fields += len(ct.Fields)
		c.Lists += len(ct.Lists)
		for _, l := range ct.Lists {
			c.Fields += len(l.Item)
		}
	})
	return c
}

// ContainerPath is the dotted address of a container in the manifest:
// "sections.heroBanner", "globalPanels.HeaderSection", "branding",
// "pages.faq", "details.SeoPanel".
type ContainerPath = string

// EachContainer visits every container in a deterministic order (sections,
// globalPanels, branding, pages, details; names sorted within each group).
func (m *Manifest) EachContainer(fn func(path ContainerPath, c *Container)) {
	for _, name := range sortedKeys(m.Sections) {
		fn("sections."+name, m.Sections[name])
	}
	for _, name := range sortedKeys(m.GlobalPanels) {
		fn("globalPanels."+name, m.GlobalPanels[name])
	}
	fn("branding", m.Branding)
	for _, name := range sortedKeys(m.Pages) {
		fn("pages."+name, m.Pages[name])
	}
	for _, name := range sortedKeys(m.Details) {
		fn("details."+name, m.Details[name])
	}
}

// Container looks a container up by its ContainerPath. ok is false for an
// unknown path.
func (m *Manifest) Container(path ContainerPath) (*Container, bool) {
	group, name := splitContainerPath(path)
	switch group {
	case "branding":
		return m.Branding, name == ""
	case "sections":
		c, ok := m.Sections[name]
		return c, ok
	case "globalPanels":
		c, ok := m.GlobalPanels[name]
		return c, ok
	case "pages":
		c, ok := m.Pages[name]
		return c, ok
	case "details":
		c, ok := m.Details[name]
		return c, ok
	}
	return nil, false
}

// ContainerPaths lists every container path in EachContainer order.
func (m *Manifest) ContainerPaths() []ContainerPath {
	var out []ContainerPath
	m.EachContainer(func(p ContainerPath, _ *Container) { out = append(out, p) })
	return out
}

func splitContainerPath(path string) (group, name string) {
	for i := 0; i < len(path); i++ {
		if path[i] == '.' {
			return path[:i], path[i+1:]
		}
	}
	return path, ""
}

// Detail returns a details panel by `details.<Panel>` or bare `<Panel>`.
func (m *Manifest) Detail(ref string) (*Container, bool) {
	const prefix = "details."
	if len(ref) > len(prefix) && ref[:len(prefix)] == prefix {
		ref = ref[len(prefix):]
	}
	c, ok := m.Details[ref]
	return c, ok
}

// HasDefault reports whether the field carries a `default` (null included).
func (f *Field) HasDefault() bool { return f.Default != nil }

// HasUnsetValue reports whether the field carries an `unset_value` (null included).
func (f *Field) HasUnsetValue() bool { return f.UnsetValue != nil }

// ReadOnly reports whether the field is `derived_from` another (E_READONLY).
func (f *Field) ReadOnly() bool { return f.DerivedFrom != "" }

// Gated reports whether the field has a conditional_on / hidden_when clause.
func (f *Field) Gated() bool { return f.ConditionalOn != nil || f.HiddenWhen != nil }
