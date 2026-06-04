// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

// Model metadata reporting (go-mlx #45): each architecture fills a ModelInfo from
// its own config (ModelInfoReporter), so Model.Info dispatches on the capability
// interface instead of a concrete type-switch over every model type. Each method
// is the verbatim body of the old switch arm; extract a model's reporter alongside
// it when that model moves out of package metal.

// GemmaModel's FillModelInfo travels with the model in package metal/model/gemma3.

func (v *Qwen3Model) FillModelInfo(info *ModelInfo) {
	info.VocabSize = int(v.Cfg.VocabSize)
	info.HiddenSize = int(v.Cfg.HiddenSize)
	info.ContextLength = int(v.Cfg.MaxPositionEmbeddings)
	if v.Cfg.Quantization != nil {
		info.QuantBits = v.Cfg.Quantization.Bits
		info.QuantGroup = v.Cfg.Quantization.GroupSize
	}
}

func (v *Qwen3MoEModel) FillModelInfo(info *ModelInfo) {
	info.VocabSize = int(v.Cfg.VocabSize)
	info.HiddenSize = int(v.Cfg.HiddenSize)
	info.ContextLength = int(v.Cfg.MaxPositionEmbeddings)
	if v.Cfg.Quantization != nil {
		info.QuantBits = v.Cfg.Quantization.Bits
		info.QuantGroup = v.Cfg.Quantization.GroupSize
	}
}

// KimiModel's FillModelInfo travels with the model in package metal/model/kimi.
// MixtralModel's FillModelInfo travels with the model in package metal/model/mixtral.
// GptOssModel's FillModelInfo travels with the model in package metal/model/gptoss.
// MiniMaxM2 FillModelInfo travels with the model in package metal/model/minimaxm2.

func (v *qwen36StagedModel) FillModelInfo(info *ModelInfo) {
	info.VocabSize = v.config.VocabSize
	info.HiddenSize = v.config.HiddenSize
	info.ContextLength = v.config.MaxPositionEmbeddings
	if info.ContextLength == 0 {
		info.ContextLength = v.config.SlidingWindow
	}
	info.QuantBits = v.config.Quantization.Bits
	info.QuantGroup = v.config.Quantization.GroupSize
}

func (v *bertStagedModel) FillModelInfo(info *ModelInfo) {
	info.VocabSize = v.config.VocabSize
	info.HiddenSize = v.config.HiddenSize
	info.ContextLength = v.config.MaxPositionEmbeddings
}

func (v *moeStagedModel) FillModelInfo(info *ModelInfo) {
	info.VocabSize = v.config.VocabSize
	info.HiddenSize = v.config.HiddenSize
	info.ContextLength = v.config.MaxPositionEmbeddings
	info.QuantBits = v.config.Quantization.Bits
	info.QuantGroup = v.config.Quantization.GroupSize
}

func (v *qwen36MoEStagedModel) FillModelInfo(info *ModelInfo) {
	info.VocabSize = int(v.config.VocabSize)
	info.HiddenSize = int(v.config.HiddenSize)
	info.ContextLength = int(v.config.MaxPositionEmbeddings)
	if v.config.Quantization != nil {
		info.QuantBits = v.config.Quantization.Bits
		info.QuantGroup = v.config.Quantization.GroupSize
	}
}
