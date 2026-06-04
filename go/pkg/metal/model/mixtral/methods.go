// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package mixtral

import "dappco.re/go/mlx/pkg/metal"

// init registers the Mixtral loader for its architecture id so the metal loader
// registry dispatches to LoadMixtral without a central switch. A blank import
// of this package wires it in.
func init() {
	metal.RegisterModelLoader("mixtral", func(modelPath string, _ []byte) (metal.InternalModel, error) {
		return LoadMixtral(modelPath)
	})
}

// FillModelInfo reports vocab/hidden/context sizing and quantization for the
// Mixtral model (metal.ModelInfoReporter capability).
func (v *MixtralModel) FillModelInfo(info *metal.ModelInfo) {
	info.VocabSize = int(v.Cfg.VocabSize)
	info.HiddenSize = int(v.Cfg.HiddenSize)
	info.ContextLength = int(v.Cfg.MaxPositionEmbeddings)
	if v.Cfg.Quantization != nil {
		info.QuantBits = v.Cfg.Quantization.Bits
		info.QuantGroup = v.Cfg.Quantization.GroupSize
	}
}
