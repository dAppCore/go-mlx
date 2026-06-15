// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package bert

import (
	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/metal"
)

// Example_modelType shows the architecture token the loader registry dispatched
// on — "bert" for the encoder loader, "bert_rerank" for the
// sequence-classification reranker. Diagnostics and the registry share this id.
func Example_modelType() {
	model := &bertStagedModel{modelType: "bert_rerank"}

	core.Println(model.ModelType())
	// Output: bert_rerank
}

// Example_numLayers shows the encoder-layer count read from num_hidden_layers;
// the staged loader carries it as metadata even though the native encoder
// kernels have not landed.
func Example_numLayers() {
	model := &bertStagedModel{config: bertStagedConfig{NumHiddenLayers: 6}}

	core.Println(model.NumLayers())
	// Output: 6
}

// Example_newCache documents the staged loader's no-KV contract: BERT is a
// bidirectional encoder with no autoregressive KV cache, so NewCache always
// returns nil — Forward/ForwardMasked stay inert until scorer kernels land.
func Example_newCache() {
	model := &bertStagedModel{}

	core.Println(model.NewCache() == nil)
	// Output: true
}

// Example_fillModelInfo copies the config's vocab, hidden width, and context
// window into the shared metal.ModelInfo the runtime reports to callers.
func Example_fillModelInfo() {
	model := &bertStagedModel{config: bertStagedConfig{
		VocabSize:             30522,
		HiddenSize:            384,
		MaxPositionEmbeddings: 512,
	}}
	info := metal.ModelInfo{}
	model.FillModelInfo(&info)

	core.Println(info.VocabSize, info.HiddenSize, info.ContextLength)
	// Output: 30522 384 512
}

// Example_decodeUnavailableError shows the diagnostic callers see when they ask
// the staged loader for native text decode — the encoder/rerank kernels are not
// landed, so the message names the operation and the model type.
func Example_decodeUnavailableError() {
	model := &bertStagedModel{modelType: "bert"}

	core.Println(model.DecodeUnavailableError("generate").Error())
	// Output: generate: bert staged loader has no native text decode kernels; use the encoder/rerank API once scorer kernels land
}

// Example_parseConfigAutoDetect parses raw config.json bytes. With an empty
// requested model type the loader auto-detects from architectures —
// "BertForSequenceClassification" resolves to the bert_rerank loader.
func Example_parseConfigAutoDetect() {
	cfg, err := parseBERTStagedConfig([]byte(`{
		"architectures": ["BertForSequenceClassification"],
		"hidden_size": 768,
		"num_hidden_layers": 12,
		"vocab_size": 30522,
		"max_position_embeddings": 512,
		"num_labels": 1
	}`), "")
	if err != nil {
		core.Println("parse error:", err)
		return
	}

	core.Println(cfg.ModelType, cfg.NumLabels)
	// Output: bert_rerank 1
}

// Example_architectureNameMapping maps a Hugging Face architecture list to the
// loader registry id: sequence-classification heads become bert_rerank, plain
// encoders become bert, and unrecognised lists return the empty string so the
// caller falls back to other detection.
func Example_architectureNameMapping() {
	core.Println(firstBERTArchitectureName([]string{"XLMRobertaForSequenceClassification"}))
	core.Println(firstBERTArchitectureName([]string{"BertModel"}))
	core.Println(firstBERTArchitectureName([]string{"LlamaForCausalLM"}) == "")
	// Output:
	// bert_rerank
	// bert
	// true
}
