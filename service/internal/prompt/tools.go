package prompt

import (
	"encoding/json"
	"fmt"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/packages/param"

	mf "github.com/breadstack/sous-chef/internal/manifest"
	"github.com/breadstack/sous-chef/internal/toolgen"
)

// SDKTools converts toolgen's wire-shaped tool definitions into the SDK's
// param types, which is what model.Request carries.
//
// Each input_schema is set as a param override of its canonical bytes
// (sorted keys, manifest.Canonical) rather than through the typed
// Properties/Required fields plus ExtraFields: the SDK's MarshalWithExtras
// ranges over the extras map and appends keys in iteration order, so a
// schema with two or more extras (additionalProperties, description, an
// empty required) marshalled to different bytes on different calls —
// measured 4/10 compiles and 2/10 re-marshals of the same object on the
// snapshot. The tool list is the first part of the cached prefix, so its
// bytes must be identical on every call (tested: TestSDKTools_ByteStable).
//
// No cache_control is placed on the tools: the breakpoint on system[0]
// already covers everything before it (tools → system → messages is one
// prefix), and the request would otherwise carry four breakpoints, the
// API's maximum.
func SDKTools(tools []toolgen.Tool) ([]anthropic.BetaToolUnionParam, error) {
	out := make([]anthropic.BetaToolUnionParam, 0, len(tools))
	for _, t := range tools {
		if t.Name == "" {
			return nil, fmt.Errorf("prompt: tool without a name")
		}
		if typ, _ := t.InputSchema["type"].(string); typ != "object" {
			return nil, fmt.Errorf("prompt: tool %s: input_schema.type %v is not object", t.Name, t.InputSchema["type"])
		}
		if req, ok := t.InputSchema["required"]; ok {
			if _, err := stringSlice(req); err != nil {
				return nil, fmt.Errorf("prompt: tool %s: required: %w", t.Name, err)
			}
		}
		canon, err := mf.Canonical(t.InputSchema)
		if err != nil {
			return nil, fmt.Errorf("prompt: tool %s: input_schema: %w", t.Name, err)
		}
		p := anthropic.BetaToolParam{
			Name:        t.Name,
			InputSchema: param.Override[anthropic.BetaToolInputSchemaParam](json.RawMessage(canon)),
		}
		if t.Description != "" {
			p.Description = param.NewOpt(t.Description)
		}
		if t.Strict {
			p.Strict = param.NewOpt(true)
		}
		if t.CacheControl != nil {
			p.CacheControl = anthropic.NewBetaCacheControlEphemeralParam()
		}
		out = append(out, anthropic.BetaToolUnionParam{OfTool: &p})
	}
	return out, nil
}

func stringSlice(v any) ([]string, error) {
	switch s := v.(type) {
	case nil:
		return nil, nil
	case []string:
		return append([]string(nil), s...), nil
	case []any:
		out := make([]string, 0, len(s))
		for _, x := range s {
			str, ok := x.(string)
			if !ok {
				return nil, fmt.Errorf("non-string entry %v", x)
			}
			out = append(out, str)
		}
		return out, nil
	}
	return nil, fmt.Errorf("unexpected type %T", v)
}
