package toolgen

import (
	"strings"

	"github.com/breadstack/sous-chef/internal/manifest"
)

// opSpec is one tool: its schema is built from the manifest-derived context
// so the per-hash tool list names the real containers, pages and lists.
type opSpec struct {
	runs   Runs
	strict bool
	desc   func(ctx schemaContext) string
	schema func(ctx schemaContext) map[string]any
}

// build renders the tool definition under `name`.
func (s opSpec) build(name string, ctx schemaContext) Tool {
	return Tool{Name: name, Description: s.desc(ctx), InputSchema: s.schema(ctx), Strict: s.strict}
}

// ─── JSON Schema helpers ─────────────────────────────────────────────────────

// obj is a closed object: additionalProperties false and a required array
// (possibly empty) — what strict mode needs. Strictness note: this package
// lists only the mandatory members in `required`; optional members are
// plain properties the model may omit. Every optional member of a strict
// tool — nested objects included — counts toward the API's request-wide
// budget of 24 (TestStrictOptionalParameterBudget), so a new optional member
// on targetSchema costs one per strict tool that embeds it.
func obj(desc string, props map[string]any, required ...string) map[string]any {
	req := []string{}
	req = append(req, required...)
	o := map[string]any{
		"type":                 "object",
		"properties":           props,
		"additionalProperties": false,
		"required":             req,
	}
	if desc != "" {
		o["description"] = desc
	}
	return o
}

// openObj is an object with model-chosen keys (manifest field keys); it can
// only appear in a non-strict tool.
func openObj(desc string) map[string]any {
	return map[string]any{"type": "object", "description": desc}
}

func str(desc string) map[string]any { return map[string]any{"type": "string", "description": desc} }

func enumStr(desc string, values []string) map[string]any {
	vals := make([]any, len(values))
	for i, v := range values {
		vals[i] = v
	}
	return map[string]any{"type": "string", "enum": vals, "description": desc}
}

func integer(desc string) map[string]any {
	return map[string]any{"type": "integer", "description": desc}
}

func boolean(desc string) map[string]any {
	return map[string]any{"type": "boolean", "description": desc}
}

func arr(desc string, items map[string]any) map[string]any {
	return map[string]any{"type": "array", "items": items, "description": desc}
}

// nonEmptyArr is an array with minItems 1 — the only array constraint strict
// mode accepts (minItems 0 or 1; TestStrictSchemasObeyStructuredOutputRules).
func nonEmptyArr(desc string, items map[string]any) map[string]any {
	a := arr(desc, items)
	a["minItems"] = 1
	return a
}

func anyOf(desc string, options ...map[string]any) map[string]any {
	opts := make([]any, len(options))
	for i, o := range options {
		opts[i] = o
	}
	return map[string]any{"anyOf": opts, "description": desc}
}

// targetSchema is the engine's target union as a closed object.
func targetSchema(ctx schemaContext) map[string]any {
	return obj("What is edited: a section on a page, the merged global settings, the branding slice, or a page's own options.",
		map[string]any{
			"kind":      enumStr("section = one section row on a page; global = site-wide settings (header, footer, product cards, typography, Branding panel BrandingSection: store name, logo, favicon…); branding = ONLY age gate, location picker, certifications; page = a page's own options (plus \"published\"). Page \"seo\" is the SEO panel.", manifest.TargetKinds),
			"page":      str("Page slug for kind section|page. Omit to act on the page the merchant is currently viewing (the host fills it in); pass one only when the merchant names a different page. Manifest pages: " + strings.Join(ctx.pages, ", ") + "; custom pages use their slug from the context."),
			"sectionId": str("kind section: the section's id exactly as listed in the turn context sections (e.g. heroBanner, productCarousel-2, contentBlock-3), or the page's built-in __pinned_* id to edit a built-in block's options."),
			"storeId":   str("kind section on the location-details page only: the location id whose per-store section list is edited. Omit everywhere else."),
			"panel":     str("kind global, optional: narrow to one panel (" + strings.Join(ctx.panels, ", ") + "). Omit to address every global setting at once."),
		}, "kind")
}

// ─── The ten tools ───────────────────────────────────────────────────────────

// orderDesc is edit_list's `order` member. It spells out the two engine rules
// (validate.js reorder, engine README rule 7) the model broke in int-run2
// (3 of 300 cases, all E_ORDER): a PARTIAL order — [1, 0] for a 3-row list,
// "order must be a permutation of 0..2" — and INDEXES on an id-keyed list —
// [1, 0] on filterOrder, "unknown id(s) 1, 0; order must be a permutation of
// the list's ids (category, brand, price, …)". Id-keyed lists are the ones
// the manifest marks with `orders` (filterOrder, headerNavOrder,
// paymentMethodsOrder, …; describe_target / lookup_fields show the flag) plus
// a page's `sections` list; the service holds no site state, so the current
// rows themselves are only known from the turn context (sections) or from
// the engine's E_ORDER reply, which names the valid ids / row count.
const orderDesc = "reorder only: the COMPLETE new order — list EVERY current row exactly once (a 3-row list needs 3 entries; to move one row, write the full new order, e.g. [1, 0, 2] — a partial order such as [1, 0] is refused, E_ORDER). Plain lists take 0-based indexes into the current rows. ID-KEYED lists take the row ids, NEVER indexes: a list that describe_target / lookup_fields shows with `orders` (filterOrder, headerNavOrder, footerNavOrder, paymentMethodsOrder, cardChipOrder, sortOptionsOrder, …) e.g. [\"brand\", \"category\", \"price\", …] with every current id, and a page's sections list (section ids). An E_ORDER reply names the valid ids / the row count: resend the full order."

var opSpecs = map[string]opSpec{
	SetFields: {
		runs: RunsBrowser, strict: false,
		desc: func(ctx schemaContext) string {
			return "Write one or more fields on a target. Keys are manifest field keys (never storage paths); values are typed per the field (color hex, number, bool, enum member, text/richtext HTML, image = CDN token or absolute URL, null = clear). Call lookup_fields first for any key you have not confirmed in this conversation; never invent keys. Dotted keys address keyed fields (locationOverrides.<locId>.<f>, categories.<slug>.<f>) and SEO keys (pages.<page>.title). On hero sections the flat keys mirror slides[0] — prefer edit_list update on slides index 0. After pick_asset put the returned public_url in values and its alt/srcset in assets under the same key — srcset is an OBJECT {\"<width>\": \"<url>\"} and must be copied exactly as returned, never rewritten as a string. Fields flagged ro (derived) are read-only (E_READONLY)."
		},
		schema: func(ctx schemaContext) map[string]any {
			return obj("", map[string]any{
				"target": targetSchema(ctx),
				"values": openObj("Field key → new value. Use null to clear a field (writes its unset_value)."),
				"assets": openObj("Optional per-image-key metadata from pick_asset: field key → {alt?: string, srcset?: object {\"<width>\": \"<url>\"} exactly as pick_asset returned it, mobile?}. Only for keys whose value is an image URL."),
			}, "target", "values")
		},
	},
	EditList: {
		runs: RunsBrowser, strict: false,
		desc: func(ctx schemaContext) string {
			return "Edit one list on a target: add a row, remove / duplicate a row, reorder rows, or update one row's fields (how detail panels — reviews, FAQ items, hero bullets, posts, footer columns — are edited). ONE structural change (add/remove/reorder/duplicate) per call; `reorder` writes the COMPLETE new order (see `order`). Nested lists use the dotted key (slides[].bullets) and an index array [parentIdx, childIdx]; `add` on a nested list takes the parent index only. On a page target the list \"sections\" is the page's own section rows {id,type,enabled}: add takes item {id: <sectionType>} (the item's id IS the section type to add; the host assigns the row's instance id), update only {enabled}, duplicate copies a row, and built-in __pinned_* rows cannot be added, removed or reordered (E_PINNED). Lists and their allowed ops come from describe_target or lookup_fields {list}. Known lists — " + ctx.listsByCont + ". Section types for sections.add: " + strings.Join(ctx.sectionTypes, ", ") + "."
		},
		schema: func(ctx schemaContext) map[string]any {
			return obj("", map[string]any{
				"target":    targetSchema(ctx),
				"list":      str("The list key on the target (e.g. slides, slides[].bullets, reviews, faqCategories[].items, sections)."),
				"action":    enumStr("add | remove | reorder | update | duplicate. `update` is allowed on every list; `duplicate` only on a page's sections list.", manifest.ListActions),
				"index":     anyOf("Row index (0-based) for remove/update/duplicate, or the parent index for add on a nested list. Nested lists take [parentIdx, childIdx].", integer(""), arr("", integer(""))),
				"item":      openObj("add: the new row's fields (missing keys take the list's defaults). update: only the fields to change — the row's other fields are kept."),
				"order":     arr(orderDesc, anyOf("", integer(""), str(""))),
				"sectionId": str("sections list only: address the row by its id instead of index (remove/update/duplicate)."),
			}, "target", "list", "action")
		},
	},
	// reset / restore_default REQUIRE keys (README §2): the engine's keyless
	// form clears every field of the target (a whole section, page or
	// panel), and the first int collection showed the model reaching for it
	// by default and then apologising for the damage (289 of 300 cases). The
	// keyless shape is therefore not offered to the model at all, and the
	// loop refuses a call that lacks keys anyway (ValidateBrowserOp).
	Reset: {
		runs: RunsBrowser, strict: true,
		desc: func(ctx schemaContext) string {
			return "Clear SPECIFIC fields back to their unset state (the editor's ↺ button): exactly the keys you list are written as their manifest unset_value — nothing else on the target changes, and NOT the template default (use restore_default for that). Clearing a whole target (every field of a section, page or panel) is not available: list every key you mean, confirmed via lookup_fields, one call per target. A call without keys is refused (E_OP) and nothing is cleared."
		},
		schema: func(ctx schemaContext) map[string]any {
			return obj("", map[string]any{
				"target": targetSchema(ctx),
				"keys":   nonEmptyArr("The field keys to clear — only these are touched. At least one; every key must be a field of the target (dotted keyed keys allowed).", str("")),
			}, "target", "keys")
		},
	},
	RestoreDefault: {
		runs: RunsBrowser, strict: true,
		desc: func(ctx schemaContext) string {
			return "Put SPECIFIC fields back to the template's default value (the manifest `default`), going through the normal write path (companions, mirror, aliases): exactly the keys you list are restored — nothing else on the target changes. A listed key with no default is cleared to its unset_value instead. Restoring a whole target (every field of a section, page or panel) is not available: list every key you mean, confirmed via lookup_fields. A call without keys is refused (E_OP) and nothing is restored."
		},
		schema: func(ctx schemaContext) map[string]any {
			return obj("", map[string]any{
				"target": targetSchema(ctx),
				"keys":   nonEmptyArr("The field keys to restore — only these are touched. At least one; every key must be a field of the target (dotted keyed keys allowed).", str("")),
			}, "target", "keys")
		},
	},
	SetPageLayout: {
		runs: RunsBrowser, strict: true,
		desc: func(ctx schemaContext) string {
			return "Switch a page's layout preset (a different setting from per-page options). Layout ids per page come from the turn context (context.layout is the current one; the host rejects unknown ids and returns the valid ones). Changing the HOME layout also re-seeds the home sections to the preset — say so before doing it."
		},
		schema: func(ctx schemaContext) map[string]any {
			return obj("", map[string]any{
				"page":   str("Page slug. Omit for the page the merchant is viewing. Pages: home, " + strings.Join(ctx.pages, ", ") + "."),
				"layout": str("The layout id for that page (e.g. filter-left, tab, summary-left, accordion, hero-featured)."),
			}, "layout")
		},
	},
	ResetPinnedSection: {
		runs: RunsBrowser, strict: true,
		desc: func(ctx schemaContext) string {
			return "Reset a page's built-in (pinned) block to its template defaults: re-enables it and clears any stored settings for it. Built-in blocks (FAQ, product list, cart, checkout, legal pages…) can only be reset — never added, removed, reordered or disabled."
		},
		schema: func(ctx schemaContext) map[string]any {
			return obj("", map[string]any{
				"page":      enumStr("The page whose built-in block is reset.", ctx.pinnedPages),
				"sectionId": enumStr("The page's pinned id (must belong to that page).", ctx.pinnedIDs),
			}, "page", "sectionId")
		},
	},
	PickAsset: {
		runs: RunsSuspend, strict: true,
		desc: func(ctx schemaContext) string {
			return "Ask the merchant to pick an image from their asset library. Call it ALONE (no other tool calls in the same response) and stop; the turn resumes with the picked asset {public_url, alt?, srcset?} as this call's result, where srcset is an object {\"<width>\": \"<url>\"}. Then write the URL with set_fields (values) and its alt/srcset (assets — copy the srcset object as-is). Never invent image URLs."
		},
		schema: func(ctx schemaContext) map[string]any {
			return obj("", map[string]any{
				"purpose":    str("One short sentence shown to the merchant: what the image is for (e.g. \"Hero banner background\")."),
				"asset_type": enumStr("Only images are supported.", []string{"image"}),
			}, "purpose", "asset_type")
		},
	},
	LookupFields: {
		// Not strict: every member is optional (target OR container, plus the
		// narrowing knobs), 11 optionals with the nested target — it alone
		// would spend almost half the request-wide budget of 24 (live 400 on
		// int, 2026-10-04: 29 across the strict set). Dispatch validates the
		// input and answers E_OP/E_TARGET/E_LIST so the model self-corrects.
		runs: RunsServer, strict: false,
		desc: func(ctx schemaContext) string {
			return "Look up the editable fields of a target (or a manifest container) before writing: returns each matching field's key, type, label, enum (+labels), min/max/step, default, unset_value, gates (if/hide), ro (derived, read-only), ref, keyed_by, companions, the list it belongs to and its panel; plus the container's lists. A field's panel is a global panel name you may pass as target.panel, EXCEPT a details panel (…DetailsPanel, …DetailPanel): that is where the editor shows a list row or keyed key, never a target panel — write it with edit_list update on its list, or set_fields with the dotted key on its own target. Narrow with query (key prefix, camelCase word, label substring or list key), keys (exact keys, aliases and dotted keyed keys included) or list (that list's row fields). Compact and capped — refine rather than paginate."
		},
		schema: func(ctx schemaContext) map[string]any {
			return obj("", map[string]any{
				"target":    targetSchema(ctx),
				"container": enumStr("Instead of target: a manifest container path (globalPanels = every panel merged).", ctx.containers),
				"query":     str("Case-insensitive match on key prefix, camelCase word, label text or list key, e.g. \"headline\", \"color\", \"button\"."),
				"keys":      arr("Exact field keys to describe (mirror aliases such as ctaText and keyed keys such as locationOverrides.<locId>.address resolve too).", str("")),
				"list":      str("Only this list's row fields (e.g. slides, reviews, faqCategories[].items, sections)."),
				"limit":     integer("Max fields to return, 1–120 (default 40)."),
				"help":      boolean("Include each field's help text (longer output)."),
			})
		},
	},
	DescribeTarget: {
		// Not strict, as lookup_fields: both members optional (6 optionals
		// with the nested target); Dispatch validates and self-corrects.
		runs: RunsServer, strict: false,
		desc: func(ctx schemaContext) string {
			return "Describe a target (or container): its lists with allowed ops / orders / detail panel, its field key stems (or every key when few), the hero mirror, presets, keyed-key prefixes and — for a page — its built-in pinned section id. Cheap orientation before lookup_fields."
		},
		schema: func(ctx schemaContext) map[string]any {
			return obj("", map[string]any{
				"target":    targetSchema(ctx),
				"container": enumStr("Instead of target: a manifest container path.", ctx.containers),
			})
		},
	},
	// report_unsupported's `where` is shared contract 1 with the launcher
	// (nav/deepLinks.js WHERE): the browser maps it to a fixed label and
	// link, so the enum is closed and model text never becomes a URL. It
	// is optional and costs one of the strict budget's 24 optionals
	// (TestStrictOptionalParameterBudget: 11 with it).
	ReportUnsupported: {
		runs: RunsServer, strict: true,
		desc: func(ctx schemaContext) string {
			return "Tell the merchant a request cannot be done by editing this site's configuration (no matching field or list after looking it up, or something outside the editor). Use it instead of guessing a key, never together with an edit. Set `where` to the place the change is made; the UI shows the reason with a link there. Then say in one sentence where they can do it. Never fabricate an email address or claim to have sent anything."
		},
		schema: func(ctx schemaContext) map[string]any {
			return obj("", map[string]any{
				"reason":      str("Plain-language reason, one or two sentences, merchant-facing."),
				"manual_hint": str("Optional: where in the editor the merchant could do it themselves, if you know."),
				"where":       enumStr(whereDesc, UnsupportedWhere),
			}, "reason")
		},
	},
}
