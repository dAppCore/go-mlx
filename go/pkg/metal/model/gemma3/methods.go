// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma3

import (
	"dappco.re/go/mlx/pkg/metal"

	// Registers the gemma chat formatter with the neutral chat dispatcher
	// whenever the gemma3 model package is built in. Pure-Go (cgo-free).
	_ "dappco.re/go/mlx/pkg/metal/model/gemma3/chat"
)

// init registers the Gemma 3 loader for its architecture ids so the metal
// loader registry dispatches to LoadGemma3 without a central switch. Gemma 2
// shares the Gemma 3 loader. A blank import of this package wires it in.
func init() {
	gemma3 := func(modelPath string, _ []byte) (metal.InternalModel, error) {
		return LoadGemma3(modelPath)
	}
	for _, arch := range []string{"gemma3", "gemma3_text", "gemma2"} {
		metal.RegisterModelLoader(arch, gemma3)
	}
}

// FillModelInfo reports vocab/hidden/context sizing and quantization for the
// Gemma 3 model (ModelInfoReporter capability).
func (v *GemmaModel) FillModelInfo(info *metal.ModelInfo) {
	info.VocabSize = int(v.Cfg.VocabSize)
	info.HiddenSize = int(v.Cfg.HiddenSize)
	info.ContextLength = int(v.Cfg.MaxPositionEmbeddings)
	if v.Cfg.Quantization != nil {
		info.QuantBits = v.Cfg.Quantization.Bits
		info.QuantGroup = v.Cfg.Quantization.GroupSize
	}
}
