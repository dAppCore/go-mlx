// SPDX-Licence-Identifier: EUPL-1.2

// Hand-rolled JSON walker for modelConfigProbe. The encoding/json
// reflect path costs 9-12 allocs per HF config.json parse (encoder
// state machine, per-field reflect.Value boxing, per-string allocation,
// per-pointer-field heap allocation, per-architecture-slice heap copy).
// Inspect fires this once per inspected model — model-picker UIs / HF
// discovery sweeps multiply that floor across dozens of candidates.
//
// The single-pass walker lands at ~4-6 allocs for typical shapes —
// the per-string clones the wire contract already requires (model_type,
// inner text_config model_type, each architectures entry) plus the
// pre-sized slice for architectures and pre-sized struct for nested
// quantization/text_config blocks. Pointer fields skip the per-field
// heap escape by stack-allocating the indirected value and taking
// address.
//
// Lifted W11-B pattern from go-inference/anthropic/jsondec.go; shares
// the same jsonenc.* primitives so error contract + null handling +
// escape-string behaviour match what encoding/json.Unmarshal would
// have produced.

package model

import (
	"dappco.re/go/inference/jsonenc"
)

// UnmarshalJSON walks a HuggingFace config.json shape in a single pass.
// Implements json.Unmarshaler so core.JSONUnmarshal / json.Unmarshal /
// json.Decoder all route through this without further plumbing.
//
// Coverage matches the struct tags in config_probe.go:
//   - model_type, vocab_size, hidden_size, num_hidden_layers,
//     max_position_embeddings, num_labels, architectures, text_config,
//     quantization, quantization_config
//   - Unknown keys SkipJSONValue past — matches encoding/json's
//     default decoder behaviour (silent ignore unless
//     DisallowUnknownFields is set, which this package does not).
//   - quantization / quantization_config / text_config pointer or
//     nested struct fields populate only when present.
//
// Numerical fidelity: bit-exact against encoding/json for every field
// — int parse uses the same digit walk, string parse re-uses the
// jsonenc fast path that returns a string copy of the slice range
// (escape decode for the rare \"-bearing case).
//
//	var probe modelConfigProbe
//	r := core.JSONUnmarshal(data, &probe)
func (probe *modelConfigProbe) UnmarshalJSON(data []byte) error {
	_, err := probe.unmarshalObjectAt(data, 0)
	return err
}

// parseConfigProbeStrict walks data into probe directly, then enforces
// the same trailing-byte rule encoding/json.Unmarshal applies (only
// whitespace may follow the top-level object). It is the allocation-lean
// production entry point: routing a modelConfigProbe through
// json.Unmarshal pays a checkValid pre-scan that allocates a scanner
// parse-state stack on every call (≈3 allocs + 170 B), pure overhead the
// hand-rolled walker was built to avoid but could not while reached
// through the stdlib entry point. Calling the walker directly skips that
// scan; the explicit trailing-whitespace check restores the exact strict
// contract (e.g. `{...} garbage` still fails) so behaviour is identical.
//
//	var probe modelConfigProbe
//	err := parseConfigProbeStrict(data, &probe)
func parseConfigProbeStrict(data []byte, probe *modelConfigProbe) error {
	end, err := probe.unmarshalObjectAt(data, 0)
	if err != nil {
		return err
	}
	if jsonenc.SkipJSONWhitespace(data, end) != len(data) {
		return jsonenc.ErrInvalidJSON
	}
	return nil
}

// unmarshalObjectAt walks the modelConfigProbe object beginning at
// data[i] and returns the index one past its closing '}'. Shared by the
// json.Unmarshaler entry point (which ignores anything after the object,
// matching encoding/json's per-value decode) and parseConfigProbeStrict
// (which additionally rejects trailing non-whitespace).
func (probe *modelConfigProbe) unmarshalObjectAt(data []byte, i int) (int, error) {
	*probe = modelConfigProbe{}
	i, err := jsonenc.MatchObjectStart(data, i)
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
		i, err = probe.unmarshalField(data, i, key)
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

// unmarshalField dispatches one modelConfigProbe field by key. Returns
// the index one past the consumed value (which may itself be an object
// or array). Unknown keys SkipJSONValue past.
func (probe *modelConfigProbe) unmarshalField(data []byte, i int, key []byte) (int, error) {
	switch string(key) {
	case "model_type":
		if jsonenc.IsJSONNull(data, i) {
			return i + 4, nil
		}
		s, next, err := jsonenc.ParseJSONString(data, i)
		if err != nil {
			return next, err
		}
		probe.ModelType = s
		return next, nil
	case "vocab_size":
		if jsonenc.IsJSONNull(data, i) {
			return i + 4, nil
		}
		n, next, err := parseStrictJSONInt(data, i)
		if err != nil {
			return next, err
		}
		probe.VocabSize = int(n)
		return next, nil
	case "hidden_size":
		if jsonenc.IsJSONNull(data, i) {
			return i + 4, nil
		}
		n, next, err := parseStrictJSONInt(data, i)
		if err != nil {
			return next, err
		}
		probe.HiddenSize = int(n)
		return next, nil
	case "num_hidden_layers":
		if jsonenc.IsJSONNull(data, i) {
			return i + 4, nil
		}
		n, next, err := parseStrictJSONInt(data, i)
		if err != nil {
			return next, err
		}
		probe.NumHiddenLayers = int(n)
		return next, nil
	case "num_key_value_heads":
		if jsonenc.IsJSONNull(data, i) {
			return i + 4, nil
		}
		n, next, err := parseStrictJSONInt(data, i)
		if err != nil {
			return next, err
		}
		probe.NumKeyValueHeads = int(n)
		return next, nil
	case "head_dim":
		if jsonenc.IsJSONNull(data, i) {
			return i + 4, nil
		}
		n, next, err := parseStrictJSONInt(data, i)
		if err != nil {
			return next, err
		}
		probe.HeadDim = int(n)
		return next, nil
	case "max_position_embeddings":
		if jsonenc.IsJSONNull(data, i) {
			return i + 4, nil
		}
		n, next, err := parseStrictJSONInt(data, i)
		if err != nil {
			return next, err
		}
		probe.MaxPositionEmbeddings = int(n)
		return next, nil
	case "num_labels":
		if jsonenc.IsJSONNull(data, i) {
			return i + 4, nil
		}
		n, next, err := parseStrictJSONInt(data, i)
		if err != nil {
			return next, err
		}
		probe.NumLabels = int(n)
		return next, nil
	case "architectures":
		if jsonenc.IsJSONNull(data, i) {
			return i + 4, nil
		}
		// Single-pass walk — direct array parse with pre-sized slice
		// via CountJSONArrayElements. Avoids the SkipJSONValue +
		// ParseJSONStringList double-walk plus the append growth
		// pattern (which can cost 1-3 mid-walk slice reallocs for
		// the rare 4+ element HF "architectures" array).
		list, next, err := parseArchitectures(data, i)
		if err != nil {
			return next, err
		}
		probe.Architectures = list
		return next, nil
	case "text_config":
		if jsonenc.IsJSONNull(data, i) {
			return i + 4, nil
		}
		return probe.unmarshalTextConfig(data, i)
	case "quantization":
		if jsonenc.IsJSONNull(data, i) {
			return i + 4, nil
		}
		// Fill the block in place inside the (already heap-allocated)
		// probe rather than allocating a fresh &struct: Present mirrors
		// the old pointer-non-nil signal (set only after the block parses
		// cleanly, so the error path leaves the probe as the old &q
		// assignment would have — untouched). An empty `{}` still reads
		// as declared. Saves one heap allocation per quantized config on
		// the parse path.
		next, err := unmarshalQuantBlock(data, i, &probe.Quantization.Bits, &probe.Quantization.GroupSize)
		if err != nil {
			return next, err
		}
		probe.Quantization.Present = true
		return next, nil
	case "quantization_config":
		if jsonenc.IsJSONNull(data, i) {
			return i + 4, nil
		}
		next, err := unmarshalQuantBlock(data, i, &probe.QuantizationConfig.Bits, &probe.QuantizationConfig.GroupSize)
		if err != nil {
			return next, err
		}
		probe.QuantizationConfig.Present = true
		return next, nil
	}
	return jsonenc.SkipJSONValue(data, i)
}

// unmarshalTextConfig walks the nested text_config object in place.
// The embedded struct has no UnmarshalJSON receiver of its own (the
// anonymous-struct field in modelConfigProbe means it cannot grow
// one) so the walk is inlined here.
func (probe *modelConfigProbe) unmarshalTextConfig(data []byte, i int) (int, error) {
	i, err := jsonenc.MatchObjectStart(data, i)
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
		switch string(key) {
		case "model_type":
			if jsonenc.IsJSONNull(data, i) {
				i += 4
			} else {
				s, n, err := jsonenc.ParseJSONString(data, i)
				if err != nil {
					return n, err
				}
				probe.TextConfig.ModelType = s
				i = n
			}
		case "vocab_size":
			if jsonenc.IsJSONNull(data, i) {
				i += 4
			} else {
				n, next, err := parseStrictJSONInt(data, i)
				if err != nil {
					return next, err
				}
				probe.TextConfig.VocabSize = int(n)
				i = next
			}
		case "hidden_size":
			if jsonenc.IsJSONNull(data, i) {
				i += 4
			} else {
				n, next, err := parseStrictJSONInt(data, i)
				if err != nil {
					return next, err
				}
				probe.TextConfig.HiddenSize = int(n)
				i = next
			}
		case "num_hidden_layers":
			if jsonenc.IsJSONNull(data, i) {
				i += 4
			} else {
				n, next, err := parseStrictJSONInt(data, i)
				if err != nil {
					return next, err
				}
				probe.TextConfig.NumHiddenLayers = int(n)
				i = next
			}
		case "num_key_value_heads":
			if jsonenc.IsJSONNull(data, i) {
				i += 4
			} else {
				n, next, err := parseStrictJSONInt(data, i)
				if err != nil {
					return next, err
				}
				probe.TextConfig.NumKeyValueHeads = int(n)
				i = next
			}
		case "head_dim":
			if jsonenc.IsJSONNull(data, i) {
				i += 4
			} else {
				n, next, err := parseStrictJSONInt(data, i)
				if err != nil {
					return next, err
				}
				probe.TextConfig.HeadDim = int(n)
				i = next
			}
		case "max_position_embeddings":
			if jsonenc.IsJSONNull(data, i) {
				i += 4
			} else {
				n, next, err := parseStrictJSONInt(data, i)
				if err != nil {
					return next, err
				}
				probe.TextConfig.MaxPositionEmbeddings = int(n)
				i = next
			}
		default:
			next, err := jsonenc.SkipJSONValue(data, i)
			if err != nil {
				return next, err
			}
			i = next
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

// parseArchitectures walks the architectures field — either a single
// string ("BertModel") or an array (["BertForCausalLM"]) per the HF
// convention. Pre-sizes the slice via CountJSONArrayElements so the
// rare multi-architecture model (composite vision-text packs) avoids
// the append growth pattern. Returns an empty (non-nil) slice for `[]`
// to match encoding/json's behaviour.
func parseArchitectures(data []byte, i int) ([]string, int, error) {
	i = jsonenc.SkipJSONWhitespace(data, i)
	if i >= len(data) {
		return nil, i, jsonenc.ErrInvalidJSON
	}
	if data[i] == '"' {
		s, next, err := jsonenc.ParseJSONString(data, i)
		if err != nil {
			return nil, next, err
		}
		return []string{s}, next, nil
	}
	if data[i] != '[' {
		return nil, i, jsonenc.ErrInvalidJSON
	}
	bodyStart := i + 1
	// Fast path — empty array.
	j := jsonenc.SkipJSONWhitespace(data, bodyStart)
	if j < len(data) && data[j] == ']' {
		return []string{}, j + 1, nil
	}
	count := jsonenc.CountJSONArrayElements(data, bodyStart)
	out := make([]string, 0, count)
	k := bodyStart
	for {
		k = jsonenc.SkipJSONWhitespace(data, k)
		if k >= len(data) || data[k] != '"' {
			return nil, k, jsonenc.ErrInvalidJSON
		}
		s, next, err := jsonenc.ParseJSONString(data, k)
		if err != nil {
			return nil, next, err
		}
		out = append(out, s)
		k = jsonenc.SkipJSONWhitespace(data, next)
		if k >= len(data) {
			return nil, k, jsonenc.ErrInvalidJSON
		}
		switch data[k] {
		case ',':
			k++
		case ']':
			return out, k + 1, nil
		default:
			return nil, k, jsonenc.ErrInvalidJSON
		}
	}
}

// parseStrictJSONInt parses a JSON integer at data[i] with the exact
// leading-zero strictness of encoding/json: a lone 0 (or -0) is fine, but
// 00 / 01 / -01 are rejected. jsonenc.ParseJSONInt is documented strict
// but implemented lenient (it folds leading zeros silently), so the
// walker uses this wrapper to stay bit-for-bit with encoding/json on the
// production fast path — which skips the checkValid scan that would
// otherwise have caught the malformed number. Reached through
// json.Unmarshal the checkValid scan still runs first, so this only
// tightens the direct path; the json.Unmarshaler contract is unchanged.
func parseStrictJSONInt(data []byte, i int) (int64, int, error) {
	// Reject a leading zero that is followed by another digit. The sign
	// (already validated by ParseJSONInt) may precede it; look past a
	// single '-'. A '0' followed by '.', 'e', '}', ',', whitespace or EOF
	// is a legitimate value and parses normally.
	d := i
	if d < len(data) && data[d] == '-' {
		d++
	}
	if d+1 < len(data) && data[d] == '0' && data[d+1] >= '0' && data[d+1] <= '9' {
		return 0, i, jsonenc.ErrInvalidJSON
	}
	return jsonenc.ParseJSONInt(data, i)
}

// unmarshalQuantBlock walks a {bits, group_size} object and stores the
// values into the supplied targets. Shared by the quantization /
// quantization_config branches (identical wire shape, different parent
// field).
func unmarshalQuantBlock(data []byte, i int, bits, groupSize *int) (int, error) {
	i, err := jsonenc.MatchObjectStart(data, i)
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
		switch string(key) {
		case "bits":
			if jsonenc.IsJSONNull(data, i) {
				i += 4
			} else {
				n, end, err := parseStrictJSONInt(data, i)
				if err != nil {
					return end, err
				}
				*bits = int(n)
				i = end
			}
		case "group_size":
			if jsonenc.IsJSONNull(data, i) {
				i += 4
			} else {
				n, end, err := parseStrictJSONInt(data, i)
				if err != nil {
					return end, err
				}
				*groupSize = int(n)
				i = end
			}
		default:
			next, err := jsonenc.SkipJSONValue(data, i)
			if err != nil {
				return next, err
			}
			i = next
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
