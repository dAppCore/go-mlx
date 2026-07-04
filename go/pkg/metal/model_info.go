// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

// Model metadata reporting (go-mlx #45): each architecture fills a ModelInfo from
// its own config (ModelInfoReporter), so Model.Info dispatches on the capability
// interface instead of a concrete type-switch over every model type. Each method
// is the verbatim body of the old switch arm; extract a model's reporter alongside
// it when that model moves out of package metal.

// GemmaModel's FillModelInfo travels with the model in package metal/model/gemma3.

// Qwen3Model's FillModelInfo travels with the model in package metal/model/qwen3.
// Qwen3MoEModel's FillModelInfo travels with the model in package metal/model/qwen3.
// Qwen3.6 staged FillModelInfo travels with the model in package metal/model/qwen3.

// DeepSeek staged FillModelInfo travels with the model in package metal/model/deepseek.
// BERT staged FillModelInfo travels with the model in package metal/model/bert.
// KimiModel's FillModelInfo travels with the model in package metal/model/kimi.
// MixtralModel's FillModelInfo travels with the model in package metal/model/mixtral.
// GptOssModel's FillModelInfo travels with the model in package metal/model/gptoss.
// MiniMaxM2 FillModelInfo travels with the model in package metal/model/minimaxm2.
