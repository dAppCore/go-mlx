// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package bert

import (
	"math"
	"testing"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
	"dappco.re/go/mlx/pkg/metal"
)

func requireMetalRuntime(t testing.TB) {
	t.Helper()
	if !metal.MetalAvailable() {
		t.Skip("Metal runtime unavailable")
	}
}

func TestBERT_LoadStagedModelEncoder_Good(t *testing.T) {
	dir := t.TempDir()
	config := `{
		"architectures": ["BertModel"],
		"model_type": "bert",
		"hidden_size": 384,
		"num_hidden_layers": 6,
		"num_attention_heads": 12,
		"intermediate_size": 1536,
		"vocab_size": 30522,
		"max_position_embeddings": 512
	}`
	if err := coreio.Local.Write(core.JoinPath(dir, "config.json"), config); err != nil {
		t.Fatalf("write config: %v", err)
	}
	writeMinimalTokenizer(t, dir)

	model, err := loadBERTStagedModel(dir, []byte(config), "bert")
	if err != nil {
		t.Fatalf("loadBERTStagedModel(bert) error = %v", err)
	}
	if model.ModelType() != "bert" || model.NumLayers() != 6 {
		t.Fatalf("model metadata = %s/%d, want bert/6", model.ModelType(), model.NumLayers())
	}
	if caches := model.NewCache(); caches != nil {
		t.Fatalf("NewCache() = %#v, want nil for encoder no-KV staged loader", caches)
	}
	if model.Tokenizer() == nil {
		t.Fatal("Tokenizer() = nil, want staged BERT loader to expose tokenizer metadata")
	}
	info := metal.ModelInfo{Architecture: model.ModelType(), NumLayers: model.NumLayers()}
	model.FillModelInfo(&info)
	if info.VocabSize != 30522 || info.HiddenSize != 384 || info.ContextLength != 512 {
		t.Fatalf("FillModelInfo = %+v, want BERT config metadata", info)
	}
}

func TestBERT_LoadStagedModelRerank_Good(t *testing.T) {
	dir := t.TempDir()
	config := `{
		"architectures": ["BertForSequenceClassification"],
		"model_type": "bert",
		"hidden_size": 768,
		"num_hidden_layers": 12,
		"num_attention_heads": 12,
		"intermediate_size": 3072,
		"vocab_size": 30522,
		"max_position_embeddings": 512,
		"num_labels": 1
	}`
	if err := coreio.Local.Write(core.JoinPath(dir, "config.json"), config); err != nil {
		t.Fatalf("write config: %v", err)
	}
	writeMinimalTokenizer(t, dir)

	model, err := loadBERTStagedModel(dir, []byte(config), "bert_rerank")
	if err != nil {
		t.Fatalf("loadBERTStagedModel(bert_rerank) error = %v", err)
	}
	if model.ModelType() != "bert_rerank" {
		t.Fatalf("ModelType() = %q, want bert_rerank", model.ModelType())
	}
	if model.config.NumLabels != 1 {
		t.Fatalf("NumLabels = %d, want 1", model.config.NumLabels)
	}
	info := metal.ModelInfo{Architecture: model.ModelType(), NumLayers: model.NumLayers()}
	model.FillModelInfo(&info)
	if info.VocabSize != 30522 || info.HiddenSize != 768 || info.ContextLength != 512 {
		t.Fatalf("FillModelInfo = %+v, want BERT rerank config metadata", info)
	}
}

// TestBERT_RegistryDispatch_Good drives the init()-registered loader closures
// through the real loader registry (metal.LoadAndInit → loadModel →
// lookupModelLoader → closure), the path the runtime takes for a checkpoint
// directory. A staged loader reads config + tokenizer only — no weights, no GPU
// allocation — so a synthetic t.TempDir() pack exercises both registered ids
// (bert + bert_rerank) without loading a model.
func TestBERT_RegistryDispatch_Good(t *testing.T) {
	requireMetalRuntime(t)

	cases := []struct {
		name      string
		config    string
		wantType  string
		wantVocab int
	}{
		{
			name: "bert-encoder",
			config: `{
				"architectures": ["BertModel"],
				"model_type": "bert",
				"hidden_size": 384,
				"num_hidden_layers": 6,
				"num_attention_heads": 12,
				"intermediate_size": 1536,
				"vocab_size": 30522,
				"max_position_embeddings": 512
			}`,
			wantType:  "bert",
			wantVocab: 30522,
		},
		{
			name: "bert-rerank",
			config: `{
				"architectures": ["BertForSequenceClassification"],
				"model_type": "bert_rerank",
				"hidden_size": 768,
				"num_hidden_layers": 12,
				"num_attention_heads": 12,
				"intermediate_size": 3072,
				"vocab_size": 30522,
				"max_position_embeddings": 512,
				"num_labels": 1
			}`,
			wantType:  "bert_rerank",
			wantVocab: 30522,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			if err := coreio.Local.Write(core.JoinPath(dir, "config.json"), tc.config); err != nil {
				t.Fatalf("write config: %v", err)
			}
			writeMinimalTokenizer(t, dir)

			model, err := metal.LoadAndInit(dir)
			if err != nil {
				t.Fatalf("LoadAndInit(%s) error = %v", tc.name, err)
			}
			if model.ModelType() != tc.wantType {
				t.Fatalf("ModelType() = %q, want %q", model.ModelType(), tc.wantType)
			}
			info := model.Info()
			if info.VocabSize != tc.wantVocab {
				t.Fatalf("Info().VocabSize = %d, want %d", info.VocabSize, tc.wantVocab)
			}
		})
	}
}

// TestBERT_RegistryDispatch_Bad drives the same registry path with a config that
// passes architecture probing (model_type "bert") but fails staged validation
// (no hidden size / layers / vocab), exercising the init() closure's core.E wrap
// branch — the error path callers see for a malformed BERT checkpoint.
func TestBERT_RegistryDispatch_Bad(t *testing.T) {
	requireMetalRuntime(t)

	dir := t.TempDir()
	config := `{"architectures": ["BertModel"], "model_type": "bert"}`
	if err := coreio.Local.Write(core.JoinPath(dir, "config.json"), config); err != nil {
		t.Fatalf("write config: %v", err)
	}
	writeMinimalTokenizer(t, dir)

	if _, err := metal.LoadAndInit(dir); err == nil {
		t.Fatal("LoadAndInit(invalid bert) error = nil, want validation diagnostic")
	}
}

// TestBERT_ParseStagedConfigAutoDetect_Good covers the parseBERTStagedConfig
// auto-detection branch: with an empty requested model type the loader resolves
// the registry id from the architectures list rather than the explicit field.
func TestBERT_ParseStagedConfigAutoDetect_Good(t *testing.T) {
	cfg, err := parseBERTStagedConfig([]byte(`{
		"architectures": ["BertForSequenceClassification"],
		"hidden_size": 768,
		"num_hidden_layers": 12,
		"vocab_size": 30522,
		"max_position_embeddings": 512,
		"num_labels": 1
	}`), "")
	if err != nil {
		t.Fatalf("parseBERTStagedConfig auto-detect error = %v", err)
	}
	if cfg.ModelType != "bert_rerank" {
		t.Fatalf("auto-detected ModelType = %q, want bert_rerank", cfg.ModelType)
	}
}

// TestBERT_ParseStagedConfigMalformed_Ugly covers the JSON-unmarshal failure
// branch of parseBERTStagedConfig — truncated bytes surface a parse error, never
// a zero-value config silently accepted downstream.
func TestBERT_ParseStagedConfigMalformed_Ugly(t *testing.T) {
	if _, err := parseBERTStagedConfig([]byte(`{"hidden_size": `), "bert"); err == nil {
		t.Fatal("parseBERTStagedConfig(truncated) error = nil, want JSON parse error")
	}
}

// TestBERT_Validate_Bad walks the staged validation branches independently of
// loading: each malformed config must trip its own diagnostic so the loader
// rejects a partial checkpoint with a specific message.
func TestBERT_Validate_Bad(t *testing.T) {
	cases := []struct {
		name string
		cfg  bertStagedConfig
		mt   string
		want string
	}{
		{
			name: "wrong-model-type",
			cfg:  bertStagedConfig{HiddenSize: 768, NumHiddenLayers: 12, VocabSize: 30522, MaxPositionEmbeddings: 512},
			mt:   "llama",
			want: "bert or bert_rerank",
		},
		{
			name: "missing-hidden-size",
			cfg:  bertStagedConfig{NumHiddenLayers: 12, VocabSize: 30522, MaxPositionEmbeddings: 512},
			mt:   "bert",
			want: "hidden size, layer count, and vocab size",
		},
		{
			name: "missing-max-position",
			cfg:  bertStagedConfig{HiddenSize: 768, NumHiddenLayers: 12, VocabSize: 30522},
			mt:   "bert",
			want: "max_position_embeddings",
		},
		{
			name: "rerank-missing-labels",
			cfg:  bertStagedConfig{HiddenSize: 768, NumHiddenLayers: 12, VocabSize: 30522, MaxPositionEmbeddings: 512},
			mt:   "bert_rerank",
			want: "num_labels",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.cfg.validate(tc.mt)
			if err == nil || !core.Contains(err.Error(), tc.want) {
				t.Fatalf("validate(%s) error = %v, want %q", tc.mt, err, tc.want)
			}
		})
	}
}

// TestBERT_Validate_Good confirms a complete encoder config and a complete
// rerank config both validate clean — the success edge the loader depends on.
func TestBERT_Validate_Good(t *testing.T) {
	encoder := bertStagedConfig{HiddenSize: 384, NumHiddenLayers: 6, VocabSize: 30522, MaxPositionEmbeddings: 512}
	if err := encoder.validate("bert"); err != nil {
		t.Fatalf("validate(bert) error = %v, want nil", err)
	}
	rerank := bertStagedConfig{HiddenSize: 768, NumHiddenLayers: 12, VocabSize: 30522, MaxPositionEmbeddings: 512, NumLabels: 1}
	if err := rerank.validate("bert_rerank"); err != nil {
		t.Fatalf("validate(bert_rerank) error = %v, want nil", err)
	}
}

// TestBERT_FirstArchitectureName covers the architecture→registry-id mapping:
// the four sequence-classification heads resolve to bert_rerank, a plain Bert*
// architecture resolves to bert, and an unrelated architecture returns "" so the
// caller falls back to the config's model_type field.
func TestBERT_FirstArchitectureName(t *testing.T) {
	cases := []struct {
		name  string
		archs []string
		want  string
	}{
		{name: "bert-seqcls", archs: []string{"BertForSequenceClassification"}, want: "bert_rerank"},
		{name: "roberta-seqcls", archs: []string{"RobertaForSequenceClassification"}, want: "bert_rerank"},
		{name: "xlm-roberta-seqcls", archs: []string{"XLMRobertaForSequenceClassification"}, want: "bert_rerank"},
		{name: "deberta-seqcls", archs: []string{"DebertaV2ForSequenceClassification"}, want: "bert_rerank"},
		{name: "plain-bert", archs: []string{"BertModel"}, want: "bert"},
		{name: "non-bert", archs: []string{"LlamaForCausalLM"}, want: ""},
		{name: "empty", archs: nil, want: ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := firstBERTArchitectureName(tc.archs); got != tc.want {
				t.Fatalf("firstBERTArchitectureName(%v) = %q, want %q", tc.archs, got, tc.want)
			}
		})
	}
}

// TestBERT_DecodeUnavailableError names the operation and the model type so a
// caller that reaches for native text decode on the staged loader gets a
// diagnostic pointing at the encoder/rerank API instead of a silent nil.
func TestBERT_DecodeUnavailableError(t *testing.T) {
	model := &bertStagedModel{modelType: "bert_rerank"}
	err := model.DecodeUnavailableError("score")
	if err == nil {
		t.Fatal("DecodeUnavailableError = nil, want diagnostic")
	}
	msg := err.Error()
	if !core.Contains(msg, "score") || !core.Contains(msg, "bert_rerank") || !core.Contains(msg, "scorer kernels land") {
		t.Fatalf("DecodeUnavailableError message = %q, want operation + model type + kernel note", msg)
	}
}

func TestBERT_LoadStagedModelRerankMissingLabels_Bad(t *testing.T) {
	config := `{
		"architectures": ["BertForSequenceClassification"],
		"model_type": "bert",
		"hidden_size": 768,
		"num_hidden_layers": 12,
		"vocab_size": 30522,
		"max_position_embeddings": 512
	}`
	_, err := loadBERTStagedModel(t.TempDir(), []byte(config), "bert_rerank")
	if err == nil || !core.Contains(err.Error(), "bert_rerank") || !core.Contains(err.Error(), "num_labels") {
		t.Fatalf("error = %v, want bert_rerank num_labels diagnostic", err)
	}
}

func TestBERTPoolCLS_Good(t *testing.T) {
	requireMetalRuntime(t)

	hidden := metal.FromValues([]float32{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
		10, 11, 12,
	}, 2, 2, 3)
	defer metal.Free(hidden)

	pooled, ok := bertPoolCLS(hidden)
	if !ok {
		t.Fatal("bertPoolCLS ok = false, want true")
	}
	defer metal.Free(pooled)
	metal.Materialize(pooled)

	if gotShape := pooled.Shape(); len(gotShape) != 2 || gotShape[0] != 2 || gotShape[1] != 3 {
		t.Fatalf("shape = %v, want [2 3]", gotShape)
	}
	assertFloat32SliceClose(t, pooled.Floats(), []float32{1, 2, 3, 7, 8, 9}, 1e-5)
}

func TestBERTPoolMean_Masked_Good(t *testing.T) {
	requireMetalRuntime(t)

	hidden := metal.FromValues([]float32{
		1, 2,
		3, 4,
		5, 6,
		10, 20,
		30, 40,
		50, 60,
	}, 2, 3, 2)
	mask := metal.FromValues([]int32{
		1, 1, 0,
		1, 0, 0,
	}, 2, 3)
	defer metal.Free(hidden, mask)

	pooled, ok := bertPoolMean(hidden, mask)
	if !ok {
		t.Fatal("bertPoolMean ok = false, want true")
	}
	defer metal.Free(pooled)
	metal.Materialize(pooled)

	if gotShape := pooled.Shape(); len(gotShape) != 2 || gotShape[0] != 2 || gotShape[1] != 2 {
		t.Fatalf("shape = %v, want [2 2]", gotShape)
	}
	assertFloat32SliceClose(t, pooled.Floats(), []float32{2, 3, 10, 20}, 1e-5)
}

func TestBERTRerankHead_Score_Good(t *testing.T) {
	requireMetalRuntime(t)

	hidden := metal.FromValues([]float32{
		2, 3,
		4, 5,
	}, 1, 2, 2)
	weight := metal.FromValues([]float32{
		1, 2,
		-1, 1,
	}, 2, 2)
	bias := metal.FromValues([]float32{0.5, -0.5}, 2)
	head := bertRerankHead{
		Classifier: metal.NewLinear(weight, bias),
		PoolMode:   bertPoolingCLS,
	}
	defer metal.Free(hidden, weight, bias)

	logits, ok := head.Score(hidden, nil)
	if !ok {
		t.Fatal("Score ok = false, want true")
	}
	defer metal.Free(logits)
	metal.Materialize(logits)

	if gotShape := logits.Shape(); len(gotShape) != 2 || gotShape[0] != 1 || gotShape[1] != 2 {
		t.Fatalf("shape = %v, want [1 2]", gotShape)
	}
	assertFloat32SliceClose(t, logits.Floats(), []float32{8.5, 0.5}, 1e-5)
}

func TestBERTPoolMean_Bad(t *testing.T) {
	requireMetalRuntime(t)

	hidden := metal.FromValues([]float32{1, 2, 3, 4}, 1, 2, 2)
	mask := metal.FromValues([]int32{1, 1, 1}, 1, 3)
	defer metal.Free(hidden, mask)

	if pooled, ok := bertPoolMean(hidden, mask); ok || pooled != nil {
		metal.Free(pooled)
		t.Fatalf("bertPoolMean ok = %v pooled=%v, want false nil for wrong mask shape", ok, pooled)
	}
}

func assertFloat32SliceClose(t *testing.T, got, want []float32, tolerance float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("len = %d, want %d; got=%v want=%v", len(got), len(want), got, want)
	}
	for i := range got {
		if math.Abs(float64(got[i]-want[i])) > tolerance {
			t.Fatalf("value[%d] = %v, want %v within %g; got=%v want=%v", i, got[i], want[i], tolerance, got, want)
		}
	}
}

func writeMinimalTokenizer(t *testing.T, dir string) {
	t.Helper()
	tokenizer := `{
		"model": {"type": "BPE", "vocab": {"hello": 0, "<unk>": 1}, "merges": []},
		"pre_tokenizer": {"type": "ByteLevel"},
		"decoder": {"type": "ByteLevel"}
	}`
	if err := coreio.Local.Write(core.JoinPath(dir, "tokenizer.json"), tokenizer); err != nil {
		t.Fatalf("write tokenizer: %v", err)
	}
}
