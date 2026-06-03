// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

// Model metadata reporting (go-mlx #45): each architecture fills a ModelInfo from
// its own config (modelInfoReporter), so Model.Info dispatches on the capability
// interface instead of a concrete type-switch over every model type. Each method
// is the verbatim body of the old switch arm; extract a model's reporter alongside
// it when that model moves out of package metal.

func (v *GemmaModel) fillModelInfo(info *ModelInfo) {
	info.VocabSize = int(v.Cfg.VocabSize)
	info.HiddenSize = int(v.Cfg.HiddenSize)
	info.ContextLength = int(v.Cfg.MaxPositionEmbeddings)
	if v.Cfg.Quantization != nil {
		info.QuantBits = v.Cfg.Quantization.Bits
		info.QuantGroup = v.Cfg.Quantization.GroupSize
	}
}

func (v *Gemma4Model) fillModelInfo(info *ModelInfo) {
	info.VocabSize = int(v.Cfg.VocabSize)
	info.NumHeads = int(v.Cfg.NumAttentionHeads)
	info.HiddenSize = int(v.Cfg.HiddenSize)
	info.ContextLength = int(v.Cfg.MaxPositionEmbeddings)
	info.Gemma4SlidingWindow = int(v.Cfg.SlidingWindow)
	if v.Cfg.Quantization != nil {
		info.QuantBits = v.Cfg.Quantization.Bits
		info.QuantGroup = v.Cfg.Quantization.GroupSize
	}
}

func (v *Qwen3Model) fillModelInfo(info *ModelInfo) {
	info.VocabSize = int(v.Cfg.VocabSize)
	info.HiddenSize = int(v.Cfg.HiddenSize)
	info.ContextLength = int(v.Cfg.MaxPositionEmbeddings)
	if v.Cfg.Quantization != nil {
		info.QuantBits = v.Cfg.Quantization.Bits
		info.QuantGroup = v.Cfg.Quantization.GroupSize
	}
}

func (v *Qwen3MoEModel) fillModelInfo(info *ModelInfo) {
	info.VocabSize = int(v.Cfg.VocabSize)
	info.HiddenSize = int(v.Cfg.HiddenSize)
	info.ContextLength = int(v.Cfg.MaxPositionEmbeddings)
	if v.Cfg.Quantization != nil {
		info.QuantBits = v.Cfg.Quantization.Bits
		info.QuantGroup = v.Cfg.Quantization.GroupSize
	}
}

func (v *MixtralModel) fillModelInfo(info *ModelInfo) {
	info.VocabSize = int(v.Cfg.VocabSize)
	info.HiddenSize = int(v.Cfg.HiddenSize)
	info.ContextLength = int(v.Cfg.MaxPositionEmbeddings)
	if v.Cfg.Quantization != nil {
		info.QuantBits = v.Cfg.Quantization.Bits
		info.QuantGroup = v.Cfg.Quantization.GroupSize
	}
}

func (v *KimiModel) fillModelInfo(info *ModelInfo) {
	info.VocabSize = int(v.Cfg.VocabSize)
	info.HiddenSize = int(v.Cfg.HiddenSize)
	info.ContextLength = int(v.Cfg.MaxPositionEmbeddings)
	if v.Cfg.Quantization != nil {
		info.QuantBits = v.Cfg.Quantization.Bits
		info.QuantGroup = v.Cfg.Quantization.GroupSize
	}
}

func (v *GptOssModel) fillModelInfo(info *ModelInfo) {
	info.VocabSize = int(v.Cfg.VocabSize)
	info.HiddenSize = int(v.Cfg.HiddenSize)
	info.ContextLength = int(v.Cfg.MaxPositionEmbeddings)
	if v.Cfg.Quantization != nil {
		info.QuantBits = v.Cfg.Quantization.Bits
		info.QuantGroup = v.Cfg.Quantization.GroupSize
	}
}

func (v *miniMaxM2StagedModel) fillModelInfo(info *ModelInfo) {
	info.VocabSize = v.plan.Config.VocabSize
	info.HiddenSize = v.plan.Config.HiddenSize
	info.ContextLength = v.plan.Config.MaxPositionEmbeddings
	if info.ContextLength == 0 {
		info.ContextLength = v.plan.Config.SlidingWindow
	}
	info.QuantBits = v.plan.JANG.MXTQBits.RoutedExpert
	if info.QuantBits == 0 {
		info.QuantBits = v.plan.JANG.Quantization.BitsDefault
	}
	info.QuantGroup = v.plan.JANG.Quantization.GroupSize
}

func (v *qwen36StagedModel) fillModelInfo(info *ModelInfo) {
	info.VocabSize = v.config.VocabSize
	info.HiddenSize = v.config.HiddenSize
	info.ContextLength = v.config.MaxPositionEmbeddings
	if info.ContextLength == 0 {
		info.ContextLength = v.config.SlidingWindow
	}
	info.QuantBits = v.config.Quantization.Bits
	info.QuantGroup = v.config.Quantization.GroupSize
}

func (v *bertStagedModel) fillModelInfo(info *ModelInfo) {
	info.VocabSize = v.config.VocabSize
	info.HiddenSize = v.config.HiddenSize
	info.ContextLength = v.config.MaxPositionEmbeddings
}

func (v *moeStagedModel) fillModelInfo(info *ModelInfo) {
	info.VocabSize = v.config.VocabSize
	info.HiddenSize = v.config.HiddenSize
	info.ContextLength = v.config.MaxPositionEmbeddings
	info.QuantBits = v.config.Quantization.Bits
	info.QuantGroup = v.config.Quantization.GroupSize
}

func (v *qwen36MoEStagedModel) fillModelInfo(info *ModelInfo) {
	info.VocabSize = int(v.config.VocabSize)
	info.HiddenSize = int(v.config.HiddenSize)
	info.ContextLength = int(v.config.MaxPositionEmbeddings)
	if v.config.Quantization != nil {
		info.QuantBits = v.config.Quantization.Bits
		info.QuantGroup = v.config.Quantization.GroupSize
	}
}
