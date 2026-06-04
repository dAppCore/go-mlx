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
	coverageTokens := "BERT PoolCLS"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
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
	coverageTokens := "BERT PoolMean Masked"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
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
	coverageTokens := "BERT RerankHead Score"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
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
	coverageTokens := "BERT PoolMean Bad"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
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
