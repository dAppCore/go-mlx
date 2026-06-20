// SPDX-Licence-Identifier: EUPL-1.2

// Branch coverage for the modelConfigProbe hand-rolled UnmarshalJSON
// walker in config_probe_unmarshal.go. The parity suite in
// config_probe_unmarshal_test.go covers the populated happy path and a
// handful of nulls (model_type / vocab_size / architectures / text_config),
// but the walker carries a per-field `IsJSONNull` short-circuit AND a
// malformed-value error return for every scalar key, plus the nested
// text_config and quantization-block walkers each repeat the pattern.
// Those branches the in-the-wild fixtures never reach, so they live here
// against direct UnmarshalJSON calls (no Inspect, no file I/O).

package model

import "testing"

// TestModelConfigProbe_AllScalarNulls drives the `IsJSONNull` true-branch
// for every top-level scalar field at once — each `case "<field>": if
// IsJSONNull { return i + 4 }` arm. A JSON null must leave the field at
// its zero value and the walk must still consume the rest of the object.
func TestModelConfigProbe_AllScalarNulls(t *testing.T) {
	const js = `{"model_type":null,"vocab_size":null,"hidden_size":null,` +
		`"num_hidden_layers":null,"num_key_value_heads":null,"head_dim":null,` +
		`"max_position_embeddings":null,"num_labels":null,"architectures":null,` +
		`"text_config":null,"quantization":null,"quantization_config":null}`
	var probe modelConfigProbe
	if err := probe.UnmarshalJSON([]byte(js)); err != nil {
		t.Fatalf("UnmarshalJSON of all-null fields: %v", err)
	}
	if probe.ModelType != "" || probe.VocabSize != 0 || probe.HiddenSize != 0 ||
		probe.NumHiddenLayers != 0 || probe.NumKeyValueHeads != 0 || probe.HeadDim != 0 ||
		probe.MaxPositionEmbeddings != 0 || probe.NumLabels != 0 {
		t.Fatalf("null scalars should stay zero, got %+v", probe)
	}
	if probe.Architectures != nil {
		t.Errorf("architectures:null should leave nil slice, got %v", probe.Architectures)
	}
	if probe.Quantization != nil || probe.QuantizationConfig != nil {
		t.Errorf("quant:null should leave nil pointers")
	}
}

// TestModelConfigProbe_PopulatedKVHeadAndHeadDim exercises the non-null
// store arms for num_key_value_heads and head_dim — the two scalar fields
// the parity control struct (parallelProbeShape) omits, so they get no
// coverage from the parity suite. Asserted directly against the walker.
func TestModelConfigProbe_PopulatedKVHeadAndHeadDim(t *testing.T) {
	const js = `{"model_type":"qwen3","num_key_value_heads":8,"head_dim":128}`
	var probe modelConfigProbe
	if err := probe.UnmarshalJSON([]byte(js)); err != nil {
		t.Fatalf("UnmarshalJSON: %v", err)
	}
	if probe.NumKeyValueHeads != 8 {
		t.Errorf("NumKeyValueHeads = %d, want 8", probe.NumKeyValueHeads)
	}
	if probe.HeadDim != 128 {
		t.Errorf("HeadDim = %d, want 128", probe.HeadDim)
	}
	// Accessors fall through to the populated top-level fields.
	if got := probe.numKeyValueHeads(); got != 8 {
		t.Errorf("numKeyValueHeads() = %d, want 8", got)
	}
	if got := probe.headDim(); got != 128 {
		t.Errorf("headDim() = %d, want 128", got)
	}
}

// TestModelConfigProbe_PerFieldMalformed drives the error-return arm of
// every scalar field — `n, next, err := parseStrictJSONInt(...); if err
// != nil { return next, err }`. A non-numeric value where an int is
// expected (and a non-string where model_type expects a string) must
// fail the whole walk rather than partially populate the probe.
func TestModelConfigProbe_PerFieldMalformed(t *testing.T) {
	cases := []struct {
		name string
		json string
	}{
		{"model_type", `{"model_type":42}`},
		{"vocab_size", `{"vocab_size":"x"}`},
		{"hidden_size", `{"hidden_size":"x"}`},
		{"num_hidden_layers", `{"num_hidden_layers":"x"}`},
		{"num_key_value_heads", `{"num_key_value_heads":"x"}`},
		{"head_dim", `{"head_dim":"x"}`},
		{"max_position_embeddings", `{"max_position_embeddings":"x"}`},
		{"num_labels", `{"num_labels":"x"}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var probe modelConfigProbe
			if err := probe.UnmarshalJSON([]byte(tc.json)); err == nil {
				t.Fatalf("expected error for malformed %s: %q", tc.name, tc.json)
			}
		})
	}
}

// TestModelConfigProbe_ObjectWalkErrors hits the two remaining
// unmarshalObjectAt error arms the parity error suite doesn't reach:
//   - a key whose string body is malformed (bad escape) so
//     ParseJSONStringRaw returns an error inside the loop (line ~100), and
//   - a structurally-valid key:value followed by neither ',' nor '}'
//     (the trailing `return ErrInvalidJSON`, line ~123).
func TestModelConfigProbe_ObjectWalkErrors(t *testing.T) {
	cases := []struct {
		name string
		json string
	}{
		// Invalid \x escape in the key string — ParseJSONStringRaw rejects.
		{"bad_key_escape", `{"mod\xel":"qwen3"}`},
		// Valid key:value, then a stray ':' instead of ',' or '}'.
		{"junk_after_value", `{"model_type":"qwen3":}`},
		// Valid value, then EOF before the closing brace / comma.
		{"eof_after_value", `{"vocab_size":5`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var probe modelConfigProbe
			if err := probe.UnmarshalJSON([]byte(tc.json)); err == nil {
				t.Fatalf("expected error for %q", tc.json)
			}
		})
	}
}

// TestModelConfigProbe_TextConfigBranches walks the nested text_config
// object through its full per-field branch matrix: an empty object (the
// `data[i] == '}'` early return), every scalar populated, every scalar
// null (the IsJSONNull arms), an unknown field (the SkipJSONValue
// default), and each malformed-value error return.
func TestModelConfigProbe_TextConfigBranches(t *testing.T) {
	t.Run("empty", func(t *testing.T) {
		var probe modelConfigProbe
		if err := probe.UnmarshalJSON([]byte(`{"text_config":{}}`)); err != nil {
			t.Fatalf("empty text_config: %v", err)
		}
		if probe.TextConfig.ModelType != "" || probe.TextConfig.VocabSize != 0 {
			t.Errorf("empty text_config should stay zero, got %+v", probe.TextConfig)
		}
	})

	t.Run("populated", func(t *testing.T) {
		const js = `{"text_config":{"model_type":"gemma3_text","vocab_size":262144,` +
			`"hidden_size":2304,"num_hidden_layers":26,"num_key_value_heads":4,` +
			`"head_dim":256,"max_position_embeddings":131072}}`
		var probe modelConfigProbe
		if err := probe.UnmarshalJSON([]byte(js)); err != nil {
			t.Fatalf("populated text_config: %v", err)
		}
		tc := probe.TextConfig
		if tc.ModelType != "gemma3_text" || tc.VocabSize != 262144 || tc.HiddenSize != 2304 ||
			tc.NumHiddenLayers != 26 || tc.NumKeyValueHeads != 4 || tc.HeadDim != 256 ||
			tc.MaxPositionEmbeddings != 131072 {
			t.Fatalf("text_config = %+v", tc)
		}
	})

	t.Run("all_null", func(t *testing.T) {
		const js = `{"text_config":{"model_type":null,"vocab_size":null,` +
			`"hidden_size":null,"num_hidden_layers":null,"num_key_value_heads":null,` +
			`"head_dim":null,"max_position_embeddings":null}}`
		var probe modelConfigProbe
		if err := probe.UnmarshalJSON([]byte(js)); err != nil {
			t.Fatalf("all-null text_config: %v", err)
		}
		if probe.TextConfig.ModelType != "" || probe.TextConfig.VocabSize != 0 ||
			probe.TextConfig.HeadDim != 0 {
			t.Fatalf("null text_config fields should stay zero, got %+v", probe.TextConfig)
		}
	})

	t.Run("unknown_field", func(t *testing.T) {
		// The default arm SkipJSONValue's past an unrecognised key, then
		// the walk continues to a known one.
		const js = `{"text_config":{"rope_theta":1000000,"model_type":"gemma3_text"}}`
		var probe modelConfigProbe
		if err := probe.UnmarshalJSON([]byte(js)); err != nil {
			t.Fatalf("unknown text_config field: %v", err)
		}
		if probe.TextConfig.ModelType != "gemma3_text" {
			t.Errorf("model_type after unknown field = %q", probe.TextConfig.ModelType)
		}
	})

	malformed := []struct {
		name string
		json string
	}{
		{"model_type", `{"text_config":{"model_type":42}}`},
		{"vocab_size", `{"text_config":{"vocab_size":"x"}}`},
		{"hidden_size", `{"text_config":{"hidden_size":"x"}}`},
		{"num_hidden_layers", `{"text_config":{"num_hidden_layers":"x"}}`},
		{"num_key_value_heads", `{"text_config":{"num_key_value_heads":"x"}}`},
		{"head_dim", `{"text_config":{"head_dim":"x"}}`},
		{"max_position_embeddings", `{"text_config":{"max_position_embeddings":"x"}}`},
		{"bad_unknown", `{"text_config":{"rope_scaling":}}`},
		{"junk_after_value", `{"text_config":{"vocab_size":5:}}`},
		{"eof_after_value", `{"text_config":{"vocab_size":5`},
		{"not_object", `{"text_config":5}`},
		{"bad_key", `{"text_config":{5:6}}`},
	}
	for _, tc := range malformed {
		t.Run("malformed_"+tc.name, func(t *testing.T) {
			var probe modelConfigProbe
			if err := probe.UnmarshalJSON([]byte(tc.json)); err == nil {
				t.Fatalf("expected error for %q", tc.json)
			}
		})
	}
}

// TestModelConfigProbe_QuantBlockBranches walks unmarshalQuantBlock
// through its full matrix via both the "quantization" and
// "quantization_config" parents: empty object, bits/group_size null arms,
// unknown field (SkipJSONValue default), and each malformed error return.
func TestModelConfigProbe_QuantBlockBranches(t *testing.T) {
	t.Run("empty", func(t *testing.T) {
		var probe modelConfigProbe
		if err := probe.UnmarshalJSON([]byte(`{"quantization":{}}`)); err != nil {
			t.Fatalf("empty quant block: %v", err)
		}
		if probe.Quantization == nil {
			t.Fatalf("empty quant block should still allocate the pointer")
		}
		if probe.Quantization.Bits != 0 || probe.Quantization.GroupSize != 0 {
			t.Errorf("empty quant block = %+v", *probe.Quantization)
		}
	})

	t.Run("null_fields", func(t *testing.T) {
		var probe modelConfigProbe
		if err := probe.UnmarshalJSON([]byte(`{"quantization_config":{"bits":null,"group_size":null}}`)); err != nil {
			t.Fatalf("null quant fields: %v", err)
		}
		if probe.QuantizationConfig == nil || probe.QuantizationConfig.Bits != 0 ||
			probe.QuantizationConfig.GroupSize != 0 {
			t.Fatalf("null quant fields should stay zero, got %+v", probe.QuantizationConfig)
		}
	})

	t.Run("unknown_field", func(t *testing.T) {
		const js = `{"quantization":{"quant_method":"awq","bits":4,"group_size":128}}`
		var probe modelConfigProbe
		if err := probe.UnmarshalJSON([]byte(js)); err != nil {
			t.Fatalf("unknown quant field: %v", err)
		}
		if probe.Quantization.Bits != 4 || probe.Quantization.GroupSize != 128 {
			t.Errorf("quant after unknown field = %+v", *probe.Quantization)
		}
	})

	malformed := []struct {
		name string
		json string
	}{
		{"bits", `{"quantization":{"bits":"x"}}`},
		{"group_size", `{"quantization_config":{"group_size":"x"}}`},
		{"bad_unknown", `{"quantization":{"quant_method":}}`},
		{"junk_after_value", `{"quantization":{"bits":4:}}`},
		{"eof_after_value", `{"quantization":{"bits":4`},
		{"not_object", `{"quantization":5}`},
		{"bad_key", `{"quantization":{5:6}}`},
	}
	for _, tc := range malformed {
		t.Run("malformed_"+tc.name, func(t *testing.T) {
			var probe modelConfigProbe
			if err := probe.UnmarshalJSON([]byte(tc.json)); err == nil {
				t.Fatalf("expected error for %q", tc.json)
			}
		})
	}
}

// TestParseArchitectures_BranchErrors drives the three parseArchitectures
// arms the parity error suite doesn't pin precisely:
//   - whitespace then EOF (the `i >= len(data)` guard, line ~399),
//   - a malformed single-string value (bad escape → ParseJSONString
//     error, line ~404), and
//   - a malformed element inside the array (bad escape → error, line ~427).
func TestParseArchitectures_BranchErrors(t *testing.T) {
	cases := []struct {
		name string
		json string
	}{
		// "architectures": then trailing whitespace and EOF.
		{"ws_then_eof", `{"architectures":   `},
		// Single string value with an invalid \x escape.
		{"bad_single_escape", `{"architectures":"Foo\xbar"}`},
		// Array whose element carries an invalid \x escape.
		{"bad_element_escape", `{"architectures":["Foo\xbar"]}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var probe modelConfigProbe
			if err := probe.UnmarshalJSON([]byte(tc.json)); err == nil {
				t.Fatalf("expected error for %q", tc.json)
			}
		})
	}
}
