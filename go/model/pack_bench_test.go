// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the model.Inspect / model.Validate path — the entry
// point CLI tools call to decide if a downloaded HF model is ready to
// load. Per AX-11 — Inspect walks the directory, parses config.json,
// reads tokenizer.json, classifies architecture, picks chat template,
// and validates quant + context. It fires once per model-pack and is
// the path users see ("scan local cache, what can I run today?").
//
// Run:    go test -bench=BenchmarkPack -benchmem -run='^$' ./go/model

package model

import (
	"testing"

	core "dappco.re/go"
	mp "dappco.re/go/mlx/pack"
)

// Sinks defeat compiler DCE.
var (
	mpSinkPack mp.ModelPack
	mpSinkErr  error
	mpSinkBool bool
)

// benchTokenizerJSON is the same shape model/pack_test.go uses — keeps
// the parser path realistic without needing a full vocab table.
const benchTokenizerJSON = `{
  "model": {
    "type": "BPE",
    "vocab": {"h": 0, "e": 1, "l": 2, "o": 3},
    "merges": ["h e", "l l"],
    "byte_fallback": false
  },
  "added_tokens": [
    {"id": 100, "content": "<bos>", "special": true},
    {"id": 101, "content": "<eos>", "special": true}
  ]
}`

func writeBenchPackFile(b *testing.B, path string, data string) {
	b.Helper()
	if r := core.WriteFile(path, []byte(data), 0o644); !r.OK {
		b.Fatalf("write %s: %v", path, r.Value)
	}
}

func writeBenchSafetensorsPack(b *testing.B, dir, modelType string) {
	b.Helper()
	writeBenchPackFile(b, core.JoinPath(dir, "config.json"), core.Sprintf(`{
		"model_type": %q,
		"vocab_size": 262208,
		"hidden_size": 2048,
		"num_hidden_layers": 26,
		"max_position_embeddings": 131072,
		"quantization_config": {"bits": 4, "group_size": 64}
	}`, modelType))
	writeBenchPackFile(b, core.JoinPath(dir, "tokenizer.json"), benchTokenizerJSON)
	writeBenchPackFile(b, core.JoinPath(dir, "model-00001-of-00001.safetensors"), "stub")
}

// --- Inspect — safetensors paths ---

func BenchmarkPack_Inspect_SafetensorsGemma4(b *testing.B) {
	dir := b.TempDir()
	writeBenchSafetensorsPack(b, dir, "gemma4_text")
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		mpSinkPack, mpSinkErr = Inspect(dir,
			mp.WithPackQuantization(4),
			mp.WithPackMaxContextLength(131072),
		)
	}
}

func BenchmarkPack_Inspect_SafetensorsQwen3(b *testing.B) {
	dir := b.TempDir()
	writeBenchSafetensorsPack(b, dir, "qwen3")
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		mpSinkPack, mpSinkErr = Inspect(dir)
	}
}

func BenchmarkPack_Inspect_SafetensorsNestedTextConfig(b *testing.B) {
	dir := b.TempDir()
	writeBenchPackFile(b, core.JoinPath(dir, "config.json"), `{
		"architectures": ["Qwen3_5ForConditionalGeneration"],
		"model_type": "qwen3_5",
		"text_config": {
			"model_type": "qwen3_5_text",
			"vocab_size": 248320,
			"hidden_size": 5120,
			"intermediate_size": 17408,
			"num_hidden_layers": 64,
			"max_position_embeddings": 262144
		},
		"quantization": {"bits": 4, "group_size": 64}
	}`)
	writeBenchPackFile(b, core.JoinPath(dir, "tokenizer.json"), benchTokenizerJSON)
	writeBenchPackFile(b, core.JoinPath(dir, "model-00001-of-00001.safetensors"), "stub")
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		mpSinkPack, mpSinkErr = Inspect(dir, mp.WithPackRequireChatTemplate(false))
	}
}

// --- Inspect — encoder + cross-encoder paths (no MoE/quant) ---

func BenchmarkPack_Inspect_BertEmbedding(b *testing.B) {
	dir := b.TempDir()
	writeBenchPackFile(b, core.JoinPath(dir, "config.json"), `{
		"architectures": ["BertModel"],
		"model_type": "bert",
		"vocab_size": 30522,
		"hidden_size": 384,
		"num_hidden_layers": 6,
		"max_position_embeddings": 512
	}`)
	writeBenchPackFile(b, core.JoinPath(dir, "sentence_bert_config.json"), `{"max_seq_length": 256}`)
	writeBenchPackFile(b, core.JoinPath(dir, "modules.json"), `[
		{"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"},
		{"idx": 1, "name": "1", "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
		{"idx": 2, "name": "2", "path": "2_Normalize", "type": "sentence_transformers.models.Normalize"}
	]`)
	poolingDir := core.JoinPath(dir, "1_Pooling")
	if r := core.MkdirAll(poolingDir, 0o755); !r.OK {
		b.Fatalf("MkdirAll: %v", r.Value)
	}
	writeBenchPackFile(b, core.JoinPath(poolingDir, "config.json"), `{
		"pooling_mode_cls_token": false,
		"pooling_mode_mean_tokens": true,
		"pooling_mode_max_tokens": false
	}`)
	writeBenchPackFile(b, core.JoinPath(dir, "tokenizer.json"), benchTokenizerJSON)
	writeBenchPackFile(b, core.JoinPath(dir, "model.safetensors"), "stub")
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		mpSinkPack, mpSinkErr = Inspect(dir)
	}
}

func BenchmarkPack_Inspect_BertRerank(b *testing.B) {
	dir := b.TempDir()
	writeBenchPackFile(b, core.JoinPath(dir, "config.json"), `{
		"architectures": ["BertForSequenceClassification"],
		"model_type": "bert",
		"vocab_size": 30522,
		"hidden_size": 768,
		"num_hidden_layers": 12,
		"max_position_embeddings": 512,
		"num_labels": 1
	}`)
	writeBenchPackFile(b, core.JoinPath(dir, "tokenizer.json"), benchTokenizerJSON)
	writeBenchPackFile(b, core.JoinPath(dir, "model.safetensors"), "stub")
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		mpSinkPack, mpSinkErr = Inspect(dir)
	}
}

// --- Inspect — error/edge paths ---

func BenchmarkPack_Inspect_MissingTokenizer(b *testing.B) {
	dir := b.TempDir()
	writeBenchPackFile(b, core.JoinPath(dir, "config.json"), `{"model_type":"qwen3"}`)
	writeBenchPackFile(b, core.JoinPath(dir, "model.safetensors"), "stub")
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		mpSinkPack, mpSinkErr = Inspect(dir, mp.WithPackRequireChatTemplate(false))
	}
}

func BenchmarkPack_Inspect_MissingWeights(b *testing.B) {
	dir := b.TempDir()
	writeBenchPackFile(b, core.JoinPath(dir, "config.json"), `{"model_type":"qwen3"}`)
	writeBenchPackFile(b, core.JoinPath(dir, "tokenizer.json"), benchTokenizerJSON)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		mpSinkPack, mpSinkErr = Inspect(dir, mp.WithPackRequireChatTemplate(false))
	}
}

// --- Validate — Inspect + IssueSummary path ---

func BenchmarkPack_Validate_Valid(b *testing.B) {
	dir := b.TempDir()
	writeBenchSafetensorsPack(b, dir, "gemma4_text")
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		mpSinkPack, mpSinkErr = Validate(dir, mp.WithPackQuantization(4), mp.WithPackMaxContextLength(131072))
	}
}

func BenchmarkPack_Validate_QuantMismatch(b *testing.B) {
	dir := b.TempDir()
	writeBenchSafetensorsPack(b, dir, "gemma4_text")
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		mpSinkPack, mpSinkErr = Validate(dir, mp.WithPackQuantization(8), mp.WithPackMaxContextLength(8192))
	}
}

// --- SupportsArchitecture — cheap predicate that fires for every candidate ---

func BenchmarkPack_SupportsArchitecture_Hit(b *testing.B) {
	name := "qwen3"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		mpSinkBool = SupportsArchitecture(name)
	}
}

func BenchmarkPack_SupportsArchitecture_Miss(b *testing.B) {
	name := "future_arch"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		mpSinkBool = SupportsArchitecture(name)
	}
}
