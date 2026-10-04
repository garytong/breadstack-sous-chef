package manifest

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"strconv"
	"unicode/utf8"
)

// Canonicalisation (README §1.1). The hash input is the manifest document
// with `generated_from` and `coverage` removed, encoded as:
//
//   - objects with keys sorted by byte order (strings.Compare), no whitespace;
//   - arrays in document order;
//   - strings escaped exactly as Go encoding/json does with HTML escaping
//     OFF: `"` and `\` are backslash-escaped, control characters below 0x20
//     are \n \r \t or \u00XX, U+2028 and U+2029 are   and  ,
//     invalid UTF-8 bytes become �, everything else is written raw
//     (so `<`, `>`, `&` and `›` are NOT escaped). This equals JavaScript
//     JSON.stringify except for U+2028/U+2029, which JSON.stringify leaves
//     raw — the launcher must escape those two to match;
//   - numbers as Go encoding/json formats a float64: integers without an
//     exponent below 1e21, shortest round-trip digits otherwise, 'e' form
//     below 1e-6 or at/above 1e21 with the exponent's leading zero dropped
//     (1e-7 → "1e-7", 1e21 → "1e+21"). This matches JavaScript Number
//     toString for every value a JSON.stringify'd manifest can contain;
//   - true, false, null as the literals.
//
// Decoding via encoding/json turns every number into a float64, so integers
// beyond 2^53 lose precision before hashing; the manifest has none.

// HashExcluded lists the top-level members removed before hashing.
var HashExcluded = []string{"generated_from", "coverage"}

// HashBytes parses `data` as JSON and returns the manifest hash without
// building a Manifest. The register endpoint uses it to check the hash in
// the URL against the body before anything else.
func HashBytes(data []byte) (string, error) {
	var raw map[string]any
	if err := json.Unmarshal(data, &raw); err != nil {
		return "", fmt.Errorf("manifest: %w", err)
	}
	return hashRaw(raw)
}

func hashRaw(raw map[string]any) (string, error) {
	stripped := make(map[string]any, len(raw))
	for k, v := range raw {
		stripped[k] = v
	}
	for _, k := range HashExcluded {
		delete(stripped, k)
	}
	b, err := Canonical(stripped)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:]), nil
}

// Canonical encodes a generically decoded JSON value (map[string]any,
// []any, string, float64, bool, nil — also json.Number and the Go integer /
// float kinds for convenience) in the canonical form described above.
func Canonical(v any) ([]byte, error) {
	buf := make([]byte, 0, 1<<12)
	buf, err := appendCanonical(buf, v)
	if err != nil {
		return nil, err
	}
	return buf, nil
}

func appendCanonical(buf []byte, v any) ([]byte, error) {
	switch x := v.(type) {
	case nil:
		return append(buf, "null"...), nil
	case bool:
		if x {
			return append(buf, "true"...), nil
		}
		return append(buf, "false"...), nil
	case string:
		return appendString(buf, x), nil
	case float64:
		return appendFloat(buf, x)
	case float32:
		return appendFloat(buf, float64(x))
	case int:
		return strconv.AppendInt(buf, int64(x), 10), nil
	case int64:
		return strconv.AppendInt(buf, x, 10), nil
	case int32:
		return strconv.AppendInt(buf, int64(x), 10), nil
	case json.Number:
		f, err := x.Float64()
		if err != nil {
			return nil, fmt.Errorf("canonical: number %q: %w", x, err)
		}
		return appendFloat(buf, f)
	case json.RawMessage:
		var decoded any
		if err := json.Unmarshal(x, &decoded); err != nil {
			return nil, fmt.Errorf("canonical: raw message: %w", err)
		}
		return appendCanonical(buf, decoded)
	case []any:
		buf = append(buf, '[')
		for i, e := range x {
			if i > 0 {
				buf = append(buf, ',')
			}
			var err error
			if buf, err = appendCanonical(buf, e); err != nil {
				return nil, err
			}
		}
		return append(buf, ']'), nil
	case map[string]any:
		keys := make([]string, 0, len(x))
		for k := range x {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		buf = append(buf, '{')
		for i, k := range keys {
			if i > 0 {
				buf = append(buf, ',')
			}
			buf = appendString(buf, k)
			buf = append(buf, ':')
			var err error
			if buf, err = appendCanonical(buf, x[k]); err != nil {
				return nil, err
			}
		}
		return append(buf, '}'), nil
	default:
		// Anything else (typed structs, []string, …) goes through encoding/json
		// and back so the same rules apply.
		b, err := json.Marshal(v)
		if err != nil {
			return nil, fmt.Errorf("canonical: %w", err)
		}
		var decoded any
		if err := json.Unmarshal(b, &decoded); err != nil {
			return nil, fmt.Errorf("canonical: %w", err)
		}
		return appendCanonical(buf, decoded)
	}
}

// appendFloat mirrors encoding/json's floatEncoder for 64-bit values.
func appendFloat(buf []byte, f float64) ([]byte, error) {
	if math.IsInf(f, 0) || math.IsNaN(f) {
		return nil, fmt.Errorf("canonical: unsupported float %v", f)
	}
	abs := math.Abs(f)
	format := byte('f')
	if abs != 0 && (abs < 1e-6 || abs >= 1e21) {
		format = 'e'
	}
	start := len(buf)
	buf = strconv.AppendFloat(buf, f, format, -1, 64)
	if format == 'e' {
		// clean up e-09 to e-9 (encoding/json does the same)
		n := len(buf)
		if n-start >= 4 && buf[n-4] == 'e' && buf[n-3] == '-' && buf[n-2] == '0' {
			buf[n-2] = buf[n-1]
			buf = buf[:n-1]
		}
	}
	return buf, nil
}

const hexDigits = "0123456789abcdef"

// appendString mirrors encoding/json's string encoder with escapeHTML=false.
func appendString(buf []byte, s string) []byte {
	buf = append(buf, '"')
	start := 0
	for i := 0; i < len(s); {
		if b := s[i]; b < utf8.RuneSelf {
			if b >= 0x20 && b != '"' && b != '\\' {
				i++
				continue
			}
			buf = append(buf, s[start:i]...)
			switch b {
			case '\\', '"':
				buf = append(buf, '\\', b)
			case '\b':
				buf = append(buf, '\\', 'b')
			case '\f':
				buf = append(buf, '\\', 'f')
			case '\n':
				buf = append(buf, '\\', 'n')
			case '\r':
				buf = append(buf, '\\', 'r')
			case '\t':
				buf = append(buf, '\\', 't')
			default:
				buf = append(buf, '\\', 'u', '0', '0', hexDigits[b>>4], hexDigits[b&0xF])
			}
			i++
			start = i
			continue
		}
		c, size := utf8.DecodeRuneInString(s[i:])
		if c == utf8.RuneError && size == 1 {
			buf = append(buf, s[start:i]...)
			buf = append(buf, '\\', 'u', 'f', 'f', 'f', 'd')
			i += size
			start = i
			continue
		}
		if c == ' ' || c == ' ' {
			buf = append(buf, s[start:i]...)
			buf = append(buf, '\\', 'u', '2', '0', '2', hexDigits[c&0xF])
			i += size
			start = i
			continue
		}
		i += size
	}
	buf = append(buf, s[start:]...)
	return append(buf, '"')
}

// sortedKeys returns a map's keys in byte order.
func sortedKeys[V any](m map[string]V) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

// CanonicalHash is the reference implementation of the manifest hash
// (README §1.1) written with nothing but encoding/json, so any Go program
// — the launcher server in particular — can copy it verbatim and agree
// with the service byte for byte. The recipe, in order:
//
//  1. json.Unmarshal the manifest into map[string]any (numbers → float64).
//  2. delete "generated_from" and "coverage" (HashExcluded).
//  3. encode with a json.Encoder whose SetEscapeHTML(false) has been called
//     (encoding/json sorts map keys by byte order and emits no whitespace;
//     U+2028/U+2029 and invalid UTF-8 are escaped regardless).
//  4. trim the single trailing '\n' the Encoder appends.
//  5. sha256 → lower-case hex.
//
// Test vector: service/testdata/manifest.snapshot.json hashes to
// 727d151b5d6b04cd68b73199ab0b3bdf061263f9623f4f6eda00a79137c06f38
// (TestCanonicalHashRecipeMatchesHashBytes). It returns the same value as
// HashBytes and Manifest.Hash for every document; HashBytes stays the fast
// path the registry uses.
func CanonicalHash(data []byte) (string, error) {
	var doc map[string]any
	if err := json.Unmarshal(data, &doc); err != nil {
		return "", fmt.Errorf("manifest: %w", err)
	}
	delete(doc, "generated_from")
	delete(doc, "coverage")
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(doc); err != nil {
		return "", fmt.Errorf("manifest: %w", err)
	}
	canonical := bytes.TrimSuffix(buf.Bytes(), []byte("\n"))
	sum := sha256.Sum256(canonical)
	return hex.EncodeToString(sum[:]), nil
}
