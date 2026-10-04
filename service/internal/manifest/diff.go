package manifest

import (
	"encoding/json"
	"reflect"
	"sort"
)

// Diff compares two manifests (README §1: the register endpoint reports the
// diff against the previous manifest of the same build line). Fields are
// addressed as `<container>.<key>` for container fields and
// `<container>.<list>[].<key>` for list-item fields; lists as
// `<container>.<list>`; containers by their ContainerPath.
//
// `source` (file + line, which moves on every build) is ignored when
// deciding whether a field changed; every other attribute counts.
type Diff struct {
	AddedContainers   []ContainerPath `json:"added_containers"`
	RemovedContainers []ContainerPath `json:"removed_containers"`
	AddedFields       []string        `json:"added_fields"`
	RemovedFields     []string        `json:"removed_fields"`
	ChangedFields     []FieldChange   `json:"changed_fields"`
	AddedLists        []string        `json:"added_lists"`
	RemovedLists      []string        `json:"removed_lists"`
	ChangedLists      []FieldChange   `json:"changed_lists"`
}

// FieldChange names a field (or list) and the attributes that differ.
type FieldChange struct {
	Path  string   `json:"path"`
	Attrs []string `json:"attrs"`
}

// Empty reports whether nothing differs.
func (d Diff) Empty() bool {
	return len(d.AddedContainers) == 0 && len(d.RemovedContainers) == 0 &&
		len(d.AddedFields) == 0 && len(d.RemovedFields) == 0 && len(d.ChangedFields) == 0 &&
		len(d.AddedLists) == 0 && len(d.RemovedLists) == 0 && len(d.ChangedLists) == 0
}

// ComputeDiff diffs `prev` → `next`.
func ComputeDiff(prev, next *Manifest) Diff {
	d := Diff{
		AddedContainers: []ContainerPath{}, RemovedContainers: []ContainerPath{},
		AddedFields: []string{}, RemovedFields: []string{}, ChangedFields: []FieldChange{},
		AddedLists: []string{}, RemovedLists: []string{}, ChangedLists: []FieldChange{},
	}
	pc := containerMap(prev)
	nc := containerMap(next)
	for _, p := range sortedKeys(nc) {
		if _, ok := pc[p]; !ok {
			d.AddedContainers = append(d.AddedContainers, p)
			for _, k := range SortedFieldKeys(nc[p].Fields) {
				d.AddedFields = append(d.AddedFields, p+"."+k)
			}
			for _, lk := range sortedKeys(nc[p].Lists) {
				d.AddedLists = append(d.AddedLists, p+"."+lk)
				for _, k := range SortedFieldKeys(nc[p].Lists[lk].Item) {
					d.AddedFields = append(d.AddedFields, p+"."+lk+"[]."+k)
				}
			}
		}
	}
	for _, p := range sortedKeys(pc) {
		if _, ok := nc[p]; !ok {
			d.RemovedContainers = append(d.RemovedContainers, p)
			for _, k := range SortedFieldKeys(pc[p].Fields) {
				d.RemovedFields = append(d.RemovedFields, p+"."+k)
			}
			for _, lk := range sortedKeys(pc[p].Lists) {
				d.RemovedLists = append(d.RemovedLists, p+"."+lk)
				for _, k := range SortedFieldKeys(pc[p].Lists[lk].Item) {
					d.RemovedFields = append(d.RemovedFields, p+"."+lk+"[]."+k)
				}
			}
		}
	}
	for _, p := range sortedKeys(nc) {
		a, ok := pc[p]
		if !ok {
			continue
		}
		b := nc[p]
		diffFields(&d, p, a.Fields, b.Fields)
		for _, lk := range sortedKeys(b.Lists) {
			la, ok := a.Lists[lk]
			if !ok {
				d.AddedLists = append(d.AddedLists, p+"."+lk)
				for _, k := range SortedFieldKeys(b.Lists[lk].Item) {
					d.AddedFields = append(d.AddedFields, p+"."+lk+"[]."+k)
				}
				continue
			}
			lb := b.Lists[lk]
			if attrs := listAttrs(la, lb); len(attrs) > 0 {
				d.ChangedLists = append(d.ChangedLists, FieldChange{Path: p + "." + lk, Attrs: attrs})
			}
			diffFields(&d, p+"."+lk+"[]", la.Item, lb.Item)
		}
		for _, lk := range sortedKeys(a.Lists) {
			if _, ok := b.Lists[lk]; !ok {
				d.RemovedLists = append(d.RemovedLists, p+"."+lk)
				for _, k := range SortedFieldKeys(a.Lists[lk].Item) {
					d.RemovedFields = append(d.RemovedFields, p+"."+lk+"[]."+k)
				}
			}
		}
	}
	sort.Strings(d.AddedFields)
	sort.Strings(d.RemovedFields)
	sort.Strings(d.AddedLists)
	sort.Strings(d.RemovedLists)
	return d
}

func containerMap(m *Manifest) map[ContainerPath]*Container {
	out := map[ContainerPath]*Container{}
	m.EachContainer(func(p ContainerPath, c *Container) { out[p] = c })
	return out
}

func diffFields(d *Diff, prefix string, a, b map[string]*Field) {
	for _, k := range SortedFieldKeys(b) {
		fa, ok := a[k]
		if !ok {
			d.AddedFields = append(d.AddedFields, prefix+"."+k)
			continue
		}
		if attrs := FieldAttrsChanged(fa, b[k]); len(attrs) > 0 {
			d.ChangedFields = append(d.ChangedFields, FieldChange{Path: prefix + "." + k, Attrs: attrs})
		}
	}
	for _, k := range SortedFieldKeys(a) {
		if _, ok := b[k]; !ok {
			d.RemovedFields = append(d.RemovedFields, prefix+"."+k)
		}
	}
}

// FieldAttrsChanged lists the attribute names (wire names) whose values
// differ between two field specs, ignoring `source`.
func FieldAttrsChanged(a, b *Field) []string {
	var attrs []string
	cmp := func(name string, x, y any) {
		if !sameJSON(x, y) {
			attrs = append(attrs, name)
		}
	}
	cmp("path", a.Path, b.Path)
	cmp("label", a.Label, b.Label)
	cmp("help", a.Help, b.Help)
	cmp("type", a.Type, b.Type)
	cmp("enum", a.Enum, b.Enum)
	cmp("enum_labels", a.EnumLabels, b.EnumLabels)
	cmp("min", a.Min, b.Min)
	cmp("max", a.Max, b.Max)
	cmp("step", a.Step, b.Step)
	cmp("default", a.Default, b.Default)
	cmp("unset_value", a.UnsetValue, b.UnsetValue)
	cmp("companions", a.Companions, b.Companions)
	cmp("aliases", a.Aliases, b.Aliases)
	cmp("writes", a.Writes, b.Writes)
	cmp("conditional_on", a.ConditionalOn, b.ConditionalOn)
	cmp("hidden_when", a.HiddenWhen, b.HiddenWhen)
	cmp("derived_from", a.DerivedFrom, b.DerivedFrom)
	cmp("keyed_by", a.KeyedBy, b.KeyedBy)
	cmp("ref", a.Ref, b.Ref)
	cmp("suggested", a.Suggested, b.Suggested)
	cmp("set_by", a.SetBy, b.SetBy)
	cmp("ranges_by", a.RangesBy, b.RangesBy)
	return attrs
}

func listAttrs(a, b *List) []string {
	var attrs []string
	if a.Path != b.Path {
		attrs = append(attrs, "path")
	}
	if a.ItemRef != b.ItemRef {
		attrs = append(attrs, "item_ref")
	}
	if !sameJSON(a.Ops, b.Ops) {
		attrs = append(attrs, "ops")
	}
	if a.Orders != b.Orders {
		attrs = append(attrs, "orders")
	}
	return attrs
}

// sameJSON compares two values by canonical JSON so nil/empty and key order
// never register as changes; RawMessages compare by decoded value.
func sameJSON(x, y any) bool {
	if isNilish(x) && isNilish(y) {
		return true
	}
	cx, errx := Canonical(x)
	cy, erry := Canonical(y)
	if errx != nil || erry != nil {
		return reflect.DeepEqual(x, y)
	}
	return string(cx) == string(cy)
}

func isNilish(v any) bool {
	switch x := v.(type) {
	case nil:
		return true
	case json.RawMessage:
		return len(x) == 0
	case *float64:
		return x == nil
	case *Gate:
		return x == nil
	case string:
		return x == ""
	case []any:
		return len(x) == 0
	case []string:
		return len(x) == 0
	case []float64:
		return len(x) == 0
	case map[string]string:
		return len(x) == 0
	}
	rv := reflect.ValueOf(v)
	switch rv.Kind() {
	case reflect.Ptr, reflect.Slice, reflect.Map:
		return rv.IsNil() || (rv.Kind() != reflect.Ptr && rv.Len() == 0)
	}
	return false
}
