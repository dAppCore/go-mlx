// SPDX-Licence-Identifier: EUPL-1.2

// Parity tests for the modelConfigProbe hand-rolled UnmarshalJSON
// walker. Each fixture decodes via the walker AND via a control path
// using encoding/json directly with a parallel struct definition —
// the two outputs must match field-for-field, byte-for-byte.

package model

import (
	"encoding/json"
	"reflect"
	"testing"
)

// parallelProbeShape mirrors modelConfigProbe without the walker. Used
// as the control decoder so we compare against pure encoding/json
// reflect behaviour (modelConfigProbe.UnmarshalJSON would otherwise
// intercept). Field tags + types must match modelConfigProbe exactly.
type parallelProbeShape struct {
	ModelType             string   `json:"model_type"`
	VocabSize             int      `json:"vocab_size"`
	HiddenSize            int      `json:"hidden_size"`
	NumHiddenLayers       int      `json:"num_hidden_layers"`
	MaxPositionEmbeddings int      `json:"max_position_embeddings"`
	Architectures         []string `json:"architectures"`
	NumLabels             int      `json:"num_labels"`
	TextConfig            struct {
		ModelType             string `json:"model_type"`
		VocabSize             int    `json:"vocab_size"`
		HiddenSize            int    `json:"hidden_size"`
		NumHiddenLayers       int    `json:"num_hidden_layers"`
		MaxPositionEmbeddings int    `json:"max_position_embeddings"`
	} `json:"text_config"`
	Quantization *struct {
		Bits      int `json:"bits"`
		GroupSize int `json:"group_size"`
	} `json:"quantization"`
	QuantizationConfig *struct {
		Bits      int `json:"bits"`
		GroupSize int `json:"group_size"`
	} `json:"quantization_config"`
}

// probeFixtures covers the architecture shapes Inspect sees in the
// wild: dense LLM, MoE LLM, vision-text composite, cross-encoder,
// long architectures slice, edge cases (empty / null fields).
var probeFixtures = []struct {
	name string
	json string
}{
	{
		name: "Qwen3",
		json: `{"model_type":"qwen3","architectures":["Qwen3ForCausalLM"],"vocab_size":151936,"hidden_size":2048,"num_hidden_layers":28,"max_position_embeddings":40960,"quantization_config":{"bits":4,"group_size":64}}`,
	},
	{
		name: "Gemma3WithTextConfig",
		json: `{"model_type":"gemma3","architectures":["Gemma3ForCausalLM"],"text_config":{"model_type":"gemma3_text","vocab_size":262144,"hidden_size":2304,"num_hidden_layers":26,"max_position_embeddings":131072},"quantization":{"bits":4,"group_size":64}}`,
	},
	{
		name: "Llama",
		json: `{"model_type":"llama","architectures":["LlamaForCausalLM"],"vocab_size":128256,"hidden_size":4096,"num_hidden_layers":32,"max_position_embeddings":8192}`,
	},
	{
		name: "BertCrossEncoder",
		json: `{"model_type":"bert","architectures":["BertForSequenceClassification"],"vocab_size":30522,"hidden_size":768,"num_hidden_layers":12,"max_position_embeddings":512,"num_labels":1}`,
	},
	{
		name: "Qwen3MoE",
		json: `{"model_type":"qwen3_moe","architectures":["Qwen3MoeForCausalLM"],"vocab_size":151936,"hidden_size":4096,"num_hidden_layers":48,"max_position_embeddings":32768,"quantization":{"bits":8,"group_size":128}}`,
	},
	{
		name: "MultiArchitectures",
		json: `{"model_type":"qwen3","architectures":["Qwen3ForCausalLM","Qwen3ForConditionalGeneration"],"vocab_size":151936,"hidden_size":2048}`,
	},
	{
		name: "EmptyObject",
		json: `{}`,
	},
	{
		name: "OnlyModelType",
		json: `{"model_type":"phi3"}`,
	},
	{
		name: "WithUnknownFields",
		json: `{"model_type":"qwen3","vocab_size":151936,"unknown_top_field":"ignored","nested_unknown":{"a":1,"b":[1,2,3]},"hidden_size":2048,"architectures":["Qwen3ForCausalLM"]}`,
	},
	{
		name: "NullPointerFields",
		json: `{"model_type":"qwen3","quantization":null,"quantization_config":null,"vocab_size":151936}`,
	},
	{
		name: "NullScalarFields",
		json: `{"model_type":null,"vocab_size":null,"architectures":null,"text_config":null}`,
	},
	{
		name: "BothQuantBlocks",
		json: `{"model_type":"qwen3","quantization":{"bits":4,"group_size":64},"quantization_config":{"bits":8,"group_size":128}}`,
	},
	{
		name: "Whitespace",
		json: `  {  "model_type" : "qwen3" ,  "architectures" : [  "Qwen3ForCausalLM"  ] ,  "vocab_size" : 151936  ,  "hidden_size":2048  }  `,
	},
	{
		name: "EscapedStringInModelType",
		json: `{"model_type":"qwen3-weird","architectures":["Foo\\bar"]}`,
	},
	{
		name: "NegativeNumbers",
		json: `{"model_type":"qwen3","num_labels":-1,"vocab_size":151936}`,
	},
	{
		name: "ZeroFields",
		json: `{"model_type":"qwen3","vocab_size":0,"hidden_size":0}`,
	},
	{
		name: "EmptyArchitectures",
		json: `{"model_type":"qwen3","architectures":[]}`,
	},
}

func TestModelConfigProbe_UnmarshalParity(t *testing.T) {
	for _, fx := range probeFixtures {
		t.Run(fx.name, func(t *testing.T) {
			var walker modelConfigProbe
			if err := walker.UnmarshalJSON([]byte(fx.json)); err != nil {
				t.Fatalf("walker UnmarshalJSON: %v", err)
			}
			var control parallelProbeShape
			if err := json.Unmarshal([]byte(fx.json), &control); err != nil {
				t.Fatalf("control json.Unmarshal: %v", err)
			}
			assertProbeEqual(t, &walker, &control)
		})
	}
}

// assertProbeEqual checks each field of the walker output against the
// reflect-decoded control. We do per-field compares (not a single
// reflect.DeepEqual on the structs as wholes) so the failure messages
// pinpoint the divergent field without grepping a struct dump.
func assertProbeEqual(t *testing.T, w *modelConfigProbe, c *parallelProbeShape) {
	t.Helper()
	if w.ModelType != c.ModelType {
		t.Errorf("ModelType: walker=%q control=%q", w.ModelType, c.ModelType)
	}
	if w.VocabSize != c.VocabSize {
		t.Errorf("VocabSize: walker=%d control=%d", w.VocabSize, c.VocabSize)
	}
	if w.HiddenSize != c.HiddenSize {
		t.Errorf("HiddenSize: walker=%d control=%d", w.HiddenSize, c.HiddenSize)
	}
	if w.NumHiddenLayers != c.NumHiddenLayers {
		t.Errorf("NumHiddenLayers: walker=%d control=%d", w.NumHiddenLayers, c.NumHiddenLayers)
	}
	if w.MaxPositionEmbeddings != c.MaxPositionEmbeddings {
		t.Errorf("MaxPositionEmbeddings: walker=%d control=%d", w.MaxPositionEmbeddings, c.MaxPositionEmbeddings)
	}
	if w.NumLabels != c.NumLabels {
		t.Errorf("NumLabels: walker=%d control=%d", w.NumLabels, c.NumLabels)
	}
	if !reflect.DeepEqual(w.Architectures, c.Architectures) {
		t.Errorf("Architectures: walker=%v control=%v", w.Architectures, c.Architectures)
	}
	if w.TextConfig.ModelType != c.TextConfig.ModelType {
		t.Errorf("TextConfig.ModelType: walker=%q control=%q", w.TextConfig.ModelType, c.TextConfig.ModelType)
	}
	if w.TextConfig.VocabSize != c.TextConfig.VocabSize {
		t.Errorf("TextConfig.VocabSize: walker=%d control=%d", w.TextConfig.VocabSize, c.TextConfig.VocabSize)
	}
	if w.TextConfig.HiddenSize != c.TextConfig.HiddenSize {
		t.Errorf("TextConfig.HiddenSize: walker=%d control=%d", w.TextConfig.HiddenSize, c.TextConfig.HiddenSize)
	}
	if w.TextConfig.NumHiddenLayers != c.TextConfig.NumHiddenLayers {
		t.Errorf("TextConfig.NumHiddenLayers: walker=%d control=%d", w.TextConfig.NumHiddenLayers, c.TextConfig.NumHiddenLayers)
	}
	if w.TextConfig.MaxPositionEmbeddings != c.TextConfig.MaxPositionEmbeddings {
		t.Errorf("TextConfig.MaxPositionEmbeddings: walker=%d control=%d", w.TextConfig.MaxPositionEmbeddings, c.TextConfig.MaxPositionEmbeddings)
	}
	if (w.Quantization == nil) != (c.Quantization == nil) {
		t.Errorf("Quantization nilness: walker=%v control=%v", w.Quantization == nil, c.Quantization == nil)
	} else if w.Quantization != nil {
		if w.Quantization.Bits != c.Quantization.Bits {
			t.Errorf("Quantization.Bits: walker=%d control=%d", w.Quantization.Bits, c.Quantization.Bits)
		}
		if w.Quantization.GroupSize != c.Quantization.GroupSize {
			t.Errorf("Quantization.GroupSize: walker=%d control=%d", w.Quantization.GroupSize, c.Quantization.GroupSize)
		}
	}
	if (w.QuantizationConfig == nil) != (c.QuantizationConfig == nil) {
		t.Errorf("QuantizationConfig nilness: walker=%v control=%v", w.QuantizationConfig == nil, c.QuantizationConfig == nil)
	} else if w.QuantizationConfig != nil {
		if w.QuantizationConfig.Bits != c.QuantizationConfig.Bits {
			t.Errorf("QuantizationConfig.Bits: walker=%d control=%d", w.QuantizationConfig.Bits, c.QuantizationConfig.Bits)
		}
		if w.QuantizationConfig.GroupSize != c.QuantizationConfig.GroupSize {
			t.Errorf("QuantizationConfig.GroupSize: walker=%d control=%d", w.QuantizationConfig.GroupSize, c.QuantizationConfig.GroupSize)
		}
	}
}

// TestModelConfigProbe_UnmarshalErrors covers the malformed-input
// boundary: bad delimiters, truncated bodies, invalid literals. Each
// should return a non-nil error rather than producing a partial probe.
func TestModelConfigProbe_UnmarshalErrors(t *testing.T) {
	cases := []struct {
		name string
		json string
	}{
		{"empty", ``},
		{"not_object", `"qwen3"`},
		{"truncated_open", `{`},
		{"truncated_after_key", `{"model_type"`},
		{"missing_colon", `{"model_type" "qwen3"}`},
		{"truncated_after_value", `{"model_type":"qwen3"`},
		{"bad_int", `{"vocab_size":"not_a_number"}`},
		{"bad_bool", `{"model_type":maybe}`},
		{"truncated_nested", `{"text_config":{"model_type":"x"`},
		{"truncated_quant", `{"quantization":{"bits":4`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var probe modelConfigProbe
			err := probe.UnmarshalJSON([]byte(tc.json))
			if err == nil {
				t.Fatalf("expected error for %q", tc.json)
			}
		})
	}
}

// TestModelConfigProbe_AccessorsAfterWalker exercises the accessor
// chain (architecture / numLayers / vocabSize / etc) on walker-built
// probes — guards against the walker populating a field shape the
// accessors then mis-read.
func TestModelConfigProbe_AccessorsAfterWalker(t *testing.T) {
	cases := []struct {
		name             string
		json             string
		wantArchitecture string
		wantNumLayers    int
		wantVocabSize    int
		wantHiddenSize   int
		wantContextLen   int
		wantQuantBits    int
		wantQuantGroup   int
	}{
		{
			name:             "Qwen3WithQuantConfig",
			json:             `{"model_type":"qwen3","architectures":["Qwen3ForCausalLM"],"vocab_size":151936,"hidden_size":2048,"num_hidden_layers":28,"max_position_embeddings":40960,"quantization_config":{"bits":4,"group_size":64}}`,
			wantArchitecture: "qwen3",
			wantNumLayers:    28,
			wantVocabSize:    151936,
			wantHiddenSize:   2048,
			wantContextLen:   40960,
			wantQuantBits:    4,
			wantQuantGroup:   64,
		},
		{
			// TextConfig.ModelType takes precedence over Architectures
			// when ModelType itself is empty — the architecture()
			// loop only short-circuits for bert_rerank, so the
			// TextConfig branch wins. "gemma3_text" doesn't hit any
			// normalize switch, returns as-is.
			name:             "Gemma3WithTextConfigFallback",
			json:             `{"architectures":["Gemma3ForCausalLM"],"text_config":{"model_type":"gemma3_text","vocab_size":262144,"hidden_size":2304,"num_hidden_layers":26,"max_position_embeddings":131072},"quantization":{"bits":4,"group_size":64}}`,
			wantArchitecture: "gemma3_text",
			wantNumLayers:    26,
			wantVocabSize:    262144,
			wantHiddenSize:   2304,
			wantContextLen:   131072,
			wantQuantBits:    4,
			wantQuantGroup:   64,
		},
		{
			name:             "BertCrossEncoderShortcut",
			json:             `{"model_type":"bert","architectures":["BertForSequenceClassification"],"vocab_size":30522,"hidden_size":768,"num_hidden_layers":12,"max_position_embeddings":512,"num_labels":1}`,
			wantArchitecture: "bert_rerank",
			wantNumLayers:    12,
			wantVocabSize:    30522,
			wantHiddenSize:   768,
			wantContextLen:   512,
			wantQuantBits:    0,
			wantQuantGroup:   0,
		},
		{
			name:             "Gemma412BUnifiedKeepsMultimodalArchitecture",
			json:             `{"model_type":"gemma4_unified","architectures":["Gemma4UnifiedForConditionalGeneration"],"text_config":{"model_type":"gemma4_unified_text","vocab_size":262144,"hidden_size":3840,"num_hidden_layers":48,"max_position_embeddings":262144},"quantization_config":{"bits":6,"group_size":64}}`,
			wantArchitecture: "gemma4_unified",
			wantNumLayers:    48,
			wantVocabSize:    262144,
			wantHiddenSize:   3840,
			wantContextLen:   262144,
			wantQuantBits:    6,
			wantQuantGroup:   64,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var probe modelConfigProbe
			if err := probe.UnmarshalJSON([]byte(tc.json)); err != nil {
				t.Fatalf("UnmarshalJSON: %v", err)
			}
			if got := probe.architecture(); got != tc.wantArchitecture {
				t.Errorf("architecture(): got %q want %q", got, tc.wantArchitecture)
			}
			if got := probe.numLayers(); got != tc.wantNumLayers {
				t.Errorf("numLayers(): got %d want %d", got, tc.wantNumLayers)
			}
			if got := probe.vocabSize(); got != tc.wantVocabSize {
				t.Errorf("vocabSize(): got %d want %d", got, tc.wantVocabSize)
			}
			if got := probe.hiddenSize(); got != tc.wantHiddenSize {
				t.Errorf("hiddenSize(): got %d want %d", got, tc.wantHiddenSize)
			}
			if got := probe.contextLength(); got != tc.wantContextLen {
				t.Errorf("contextLength(): got %d want %d", got, tc.wantContextLen)
			}
			if got := probe.quantBits(); got != tc.wantQuantBits {
				t.Errorf("quantBits(): got %d want %d", got, tc.wantQuantBits)
			}
			if got := probe.quantGroup(); got != tc.wantQuantGroup {
				t.Errorf("quantGroup(): got %d want %d", got, tc.wantQuantGroup)
			}
		})
	}
}

// TestParseConfigProbeStrict_Parity pins the production fast path
// (parseConfigProbeStrict, used by readModelConfigAt) against the
// encoding/json control decoder it replaced. On every valid fixture the
// parsed result must match field-for-field, AND the accept/reject
// decision must agree with json.Unmarshal — including the trailing-byte
// rule that distinguishes the strict path from the lenient
// json.Unmarshaler interface method. This is the byte-identical guard
// for routing config.json around the stdlib checkValid pre-scan.
func TestParseConfigProbeStrict_Parity(t *testing.T) {
	for _, fx := range probeFixtures {
		t.Run(fx.name, func(t *testing.T) {
			var strict modelConfigProbe
			if err := parseConfigProbeStrict([]byte(fx.json), &strict); err != nil {
				t.Fatalf("parseConfigProbeStrict: %v", err)
			}
			var control parallelProbeShape
			if err := json.Unmarshal([]byte(fx.json), &control); err != nil {
				t.Fatalf("control json.Unmarshal: %v", err)
			}
			assertProbeEqual(t, &strict, &control)
		})
	}
}

// TestModelConfigProbe_ArchitecturesSingleString pins the one deliberate
// divergence from stdlib in parseArchitectures: a bare string value for
// "architectures" is accepted as a one-element list, where encoding/json into
// []string rejects it. HuggingFace configs are almost always an array, but the
// scalar form appears in the wild and the walker chooses to tolerate it rather
// than fail the whole probe — so the architecture() resolver still sees the
// name. This is the `data[i] == '"'` fast path in parseArchitectures.
func TestModelConfigProbe_ArchitecturesSingleString(t *testing.T) {
	var probe modelConfigProbe
	if err := probe.UnmarshalJSON([]byte(`{"architectures":"Qwen3ForCausalLM"}`)); err != nil {
		t.Fatalf("UnmarshalJSON of single-string architectures: %v", err)
	}
	if len(probe.Architectures) != 1 || probe.Architectures[0] != "Qwen3ForCausalLM" {
		t.Fatalf("Architectures = %v, want [Qwen3ForCausalLM]", probe.Architectures)
	}
	// The resolver maps the transformers class name to the canonical id.
	if got := probe.architecture(); got != "qwen3" {
		t.Errorf("architecture() = %q, want qwen3", got)
	}
}

// TestModelConfigProbe_ArchitecturesEmptyArray pins the empty-array fast path:
// "architectures":[] parses clean to a zero-length slice (matching stdlib), and
// architecture() then falls through to model_type rather than panicking on an
// empty list.
func TestModelConfigProbe_ArchitecturesEmptyArray(t *testing.T) {
	var probe modelConfigProbe
	if err := probe.UnmarshalJSON([]byte(`{"model_type":"qwen3","architectures":[]}`)); err != nil {
		t.Fatalf("UnmarshalJSON of empty architectures: %v", err)
	}
	if len(probe.Architectures) != 0 {
		t.Fatalf("Architectures = %v, want empty", probe.Architectures)
	}
	if got := probe.architecture(); got != "qwen3" {
		t.Errorf("architecture() = %q, want qwen3 (model_type fallback)", got)
	}
}

// TestParseConfigProbeStrict_ErrorParity asserts the strict fast path
// makes the same accept/reject call as json.Unmarshal on malformed and
// trailing-byte inputs — the contract readModelConfigAt depends on.
func TestParseConfigProbeStrict_ErrorParity(t *testing.T) {
	cases := []string{
		``,
		`"qwen3"`,
		`{`,
		`{"model_type"`,
		`{"model_type" "qwen3"}`,
		`{"model_type":"qwen3"`,
		`{"vocab_size":"not_a_number"}`,
		`{"model_type":maybe}`,
		`{"text_config":{"model_type":"x"`,
		`{"quantization":{"bits":4`,
		// Malformed architectures arrays — the walker's parseArchitectures
		// must make the same reject call as stdlib's []string decode. A
		// non-array/non-string scalar, a non-string element, a truncated
		// array, a missing comma, a trailing comma, and an unterminated
		// element list are all rejected by both. (A bare string value is
		// deliberately NOT here: the walker accepts "X" as a one-element
		// array where stdlib rejects it — an intentional leniency pinned by
		// TestModelConfigProbe_ArchitecturesSingleString, not a parity case.)
		`{"architectures":5}`,
		`{"architectures":[5]}`,
		`{"architectures":["A"`,
		`{"architectures":["A" "B"]}`,
		`{"architectures":["A",]}`,
		`{"architectures":[}`,
		// Malformed nested text_config / quantization blocks — the nested
		// object walkers reject the same shapes stdlib does.
		`{"text_config":{"vocab_size":"x"}}`,
		`{"text_config":5}`,
		`{"quantization_config":{"bits":"x"}}`,
		`{"quantization":{"group_size":}}`,
		`{"model_type":"qwen3"} trailing`,
		`{"model_type":"qwen3"}{`,
		`{"model_type":"qwen3"}]`,
		`{"model_type":"qwen3"}` + "\n\n",
		`  {"model_type":"qwen3"}  `,
		`{}`,
		`{"model_type":"qwen3"}`,
	}
	for _, in := range cases {
		t.Run(in, func(t *testing.T) {
			var strict modelConfigProbe
			strictErr := parseConfigProbeStrict([]byte(in), &strict)
			var control parallelProbeShape
			controlErr := json.Unmarshal([]byte(in), &control)
			if (strictErr != nil) != (controlErr != nil) {
				t.Fatalf("error-decision mismatch for %q: strict=%v control=%v", in, strictErr, controlErr)
			}
		})
	}
}
