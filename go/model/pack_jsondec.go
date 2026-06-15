// SPDX-Licence-Identifier: EUPL-1.2

// Hand-rolled JSON walkers for the small sentence-transformers config
// files the embedding/rerank inspect path reads (sentence_bert_config.json,
// modules.json, the Pooling config.json). Each previously routed through
// core.JSONUnmarshal, whose stdlib reflect path allocates a scanner
// parse-state stack (the checkValid pre-scan) plus a decode-state on every
// call — pure overhead for these one-to-four-field shapes. Inspect fires
// these on every embedding/rerank model it sees; model-picker UIs / HF
// discovery sweeps multiply that floor across dozens of candidates.
//
// The walkers reuse the jsonenc.* primitives and the exact object/array
// skeleton already proven byte-identical against encoding/json in
// config_probe_unmarshal.go (unmarshalQuantBlock / parseArchitectures), and
// each enforces the same strict whole-input rule core.JSONUnmarshal applies
// (only trailing whitespace may follow the top-level value). The
// (value, ok) return contract of the callers is therefore unchanged: any
// input encoding/json would reject yields ok=false, any input it would
// accept yields the identical value.

package model

import (
	"dappco.re/go/inference/jsonenc"
)

// parseSingleIntField walks a flat JSON object and returns the integer
// value of key, mirroring encoding/json unmarshalling that object into a
// struct with one int field. Unknown keys are skipped (matching the
// stdlib default decoder). A missing key returns (0, true) — parse
// succeeded, field absent — exactly as the stdlib leaves the zero value.
// Any malformed body, a non-integer value for key, or trailing
// non-whitespace returns ok=false.
func parseSingleIntField(data []byte, key string) (int, bool) {
	// Top-level null unmarshals into a struct as a no-op (zero value, nil
	// error) under encoding/json — mirror that so the (value, ok) contract
	// matches on a `null` body.
	if isTopLevelNull(data) {
		return 0, true
	}
	value := 0
	end, err := walkFlatObject(data, 0, func(k []byte, vi int) (int, error) {
		if string(k) == key {
			if jsonenc.IsJSONNull(data, vi) {
				return vi + 4, nil
			}
			n, next, err := parseStrictJSONInt(data, vi)
			if err != nil {
				return next, err
			}
			value = int(n)
			return next, nil
		}
		return jsonenc.SkipJSONValue(data, vi)
	})
	if err != nil || jsonenc.SkipJSONWhitespace(data, end) != len(data) {
		return 0, false
	}
	return value, true
}

// parsePoolingFlags walks a sentence-transformers Pooling config.json and
// returns the canonical pooling name from the first set mode flag, in the
// same precedence the reflect-based reader applied (mean → cls → max →
// weighted_mean). Mirrors encoding/json unmarshalling into the four-bool
// struct: unknown keys skipped, a missing flag defaults false. ok=false
// on a malformed body, a non-bool flag value, or trailing non-whitespace;
// ok=true with name="" when the body parses but declares no recognised
// mode (the caller then falls through to the next candidate).
func parsePoolingFlags(data []byte) (cls, mean, max, weightedMean bool, ok bool) {
	// Top-level null → struct no-op under encoding/json (all flags zero,
	// nil error). Mirror it so ok stays true on a `null` body.
	if isTopLevelNull(data) {
		return false, false, false, false, true
	}
	end, err := walkFlatObject(data, 0, func(k []byte, vi int) (int, error) {
		var target *bool
		switch string(k) {
		case "pooling_mode_cls_token":
			target = &cls
		case "pooling_mode_mean_tokens":
			target = &mean
		case "pooling_mode_max_tokens":
			target = &max
		case "pooling_mode_weightedmean_tokens":
			target = &weightedMean
		}
		if target == nil {
			return jsonenc.SkipJSONValue(data, vi)
		}
		if jsonenc.IsJSONNull(data, vi) {
			return vi + 4, nil
		}
		b, next, err := jsonenc.ParseJSONBool(data, vi)
		if err != nil {
			return next, err
		}
		*target = b
		return next, nil
	})
	if err != nil || jsonenc.SkipJSONWhitespace(data, end) != len(data) {
		return false, false, false, false, false
	}
	return cls, mean, max, weightedMean, true
}

// modulesDeclareNormalize walks a sentence-transformers modules.json (an
// array of {type, path, ...} objects) and reports whether any entry's
// type or path contains "normalize" (ASCII-insensitive). Mirrors
// encoding/json unmarshalling into []struct{Type, Path string} then
// scanning it — but folds the membership test into the walk so no slice
// is materialised. ok=false on a malformed body or trailing
// non-whitespace; otherwise (found, true).
func modulesDeclareNormalize(data []byte) (normalize, ok bool) {
	// Top-level null → nil slice, nil error under encoding/json. Mirror it
	// so ok stays true (no normalize module) on a `null` body.
	if isTopLevelNull(data) {
		return false, true
	}
	i := jsonenc.SkipJSONWhitespace(data, 0)
	if i >= len(data) || data[i] != '[' {
		return false, false
	}
	i++
	i = jsonenc.SkipJSONWhitespace(data, i)
	if i < len(data) && data[i] == ']' {
		i++
		if jsonenc.SkipJSONWhitespace(data, i) != len(data) {
			return false, false
		}
		return false, true
	}
	for {
		// rowHasNormalize folds the membership test into the walk: as soon
		// as a "type" or "path" value contains "normalize" the row is
		// flagged, but every field is still consumed so the walk's
		// strict-shape validation matches what encoding/json would have
		// rejected. ParseJSONString (not ...Raw) decodes escapes so the
		// substring test is bit-for-bit what the reflect path produced.
		rowHasNormalize := false
		next, err := walkFlatObject(data, i, func(k []byte, vi int) (int, error) {
			ks := string(k)
			if ks != "type" && ks != "path" {
				return jsonenc.SkipJSONValue(data, vi)
			}
			if jsonenc.IsJSONNull(data, vi) {
				return vi + 4, nil
			}
			s, end, err := jsonenc.ParseJSONString(data, vi)
			if err != nil {
				return end, err
			}
			if containsASCIIInsensitive(s, "normalize") {
				rowHasNormalize = true
			}
			return end, nil
		})
		if err != nil {
			return false, false
		}
		if rowHasNormalize {
			normalize = true
		}
		i = jsonenc.SkipJSONWhitespace(data, next)
		if i >= len(data) {
			return false, false
		}
		switch data[i] {
		case ',':
			i++
		case ']':
			i++
			if jsonenc.SkipJSONWhitespace(data, i) != len(data) {
				return false, false
			}
			return normalize, true
		default:
			return false, false
		}
	}
}

// isTopLevelNull reports whether data is the JSON literal null with only
// surrounding whitespace — the one input encoding/json accepts for every
// target type as a no-op (leaving the zero value, returning a nil error).
// The walkers special-case it so their (value, ok) contract matches the
// reflect path on a `null` body.
func isTopLevelNull(data []byte) bool {
	i := jsonenc.SkipJSONWhitespace(data, 0)
	if !jsonenc.IsJSONNull(data, i) {
		return false
	}
	return jsonenc.SkipJSONWhitespace(data, i+4) == len(data)
}

// walkFlatObject matches a JSON object beginning at data[start], invoking
// field for each key with the index of its value, and returns the index
// one past the closing '}'. field must return the index one past the
// value it consumed (the SkipJSONValue contract). The skeleton is the
// one proven byte-identical against encoding/json in unmarshalQuantBlock.
func walkFlatObject(data []byte, start int, field func(key []byte, valueIndex int) (int, error)) (int, error) {
	i, err := jsonenc.MatchObjectStart(data, start)
	if err != nil {
		return i, err
	}
	i = jsonenc.SkipJSONWhitespace(data, i)
	if i < len(data) && data[i] == '}' {
		return i + 1, nil
	}
	for {
		i = jsonenc.SkipJSONWhitespace(data, i)
		if i >= len(data) || data[i] != '"' {
			return i, jsonenc.ErrInvalidJSON
		}
		key, next, err := jsonenc.ParseJSONStringRaw(data, i)
		if err != nil {
			return next, err
		}
		i = jsonenc.SkipJSONWhitespace(data, next)
		if i >= len(data) || data[i] != ':' {
			return i, jsonenc.ErrInvalidJSON
		}
		i = jsonenc.SkipJSONWhitespace(data, i+1)
		i, err = field(key, i)
		if err != nil {
			return i, err
		}
		i = jsonenc.SkipJSONWhitespace(data, i)
		if i >= len(data) {
			return i, jsonenc.ErrInvalidJSON
		}
		if data[i] == ',' {
			i++
			continue
		}
		if data[i] == '}' {
			return i + 1, nil
		}
		return i, jsonenc.ErrInvalidJSON
	}
}
