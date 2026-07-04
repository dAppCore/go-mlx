// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package kimi

import "dappco.re/go/mlx/pkg/metal"

// init registers the Kimi loader for its architecture id so the metal loader
// registry dispatches to LoadKimi without a central switch. A blank import of
// this package wires it in.
func init() {
	metal.RegisterModelLoader("kimi", func(modelPath string, _ []byte) (metal.InternalModel, error) {
		return LoadKimi(modelPath)
	})
}

// FillModelInfo reports vocab/hidden/context sizing and quantization for the
// Kimi model (metal.ModelInfoReporter capability).
func (v *KimiModel) FillModelInfo(info *metal.ModelInfo) {
	info.VocabSize = int(v.Cfg.VocabSize)
	info.HiddenSize = int(v.Cfg.HiddenSize)
	info.ContextLength = int(v.Cfg.MaxPositionEmbeddings)
	if v.Cfg.Quantization != nil {
		info.QuantBits = v.Cfg.Quantization.Bits
		info.QuantGroup = v.Cfg.Quantization.GroupSize
	}
}
