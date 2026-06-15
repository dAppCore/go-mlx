// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package mixtral

import (
	"testing"

	"dappco.re/go/mlx/pkg/metal"
)

// --- FillModelInfo (methods.go) ---

// TestMethods_FillModelInfo_Good copies vocab/hidden/context sizing out of an
// unquantized config; the quant fields stay zero when no QuantizationConfig is
// present.
func TestMethods_FillModelInfo_Good(t *testing.T) {
	model := &MixtralModel{Cfg: &MixtralConfig{
		VocabSize:             32000,
		HiddenSize:            4096,
		MaxPositionEmbeddings: 32768,
	}}
	info := &metal.ModelInfo{}
	model.FillModelInfo(info)
	if info.VocabSize != 32000 || info.HiddenSize != 4096 || info.ContextLength != 32768 {
		t.Fatalf("FillModelInfo = vocab %d hidden %d context %d, want 32000/4096/32768",
			info.VocabSize, info.HiddenSize, info.ContextLength)
	}
	if info.QuantBits != 0 || info.QuantGroup != 0 {
		t.Fatalf("FillModelInfo quant = %d/%d, want 0/0 for unquantized", info.QuantBits, info.QuantGroup)
	}
}

// TestMethods_FillModelInfo_Loaded_Good copies sizing out of a real loaded
// config — the loaded fixture is unquantized so the quant branch stays zero.
func TestMethods_FillModelInfo_Loaded_Good(t *testing.T) {
	model := loadMixtralModel(t)
	info := &metal.ModelInfo{}
	model.FillModelInfo(info)
	if info.VocabSize != mixtralLoadVocab || info.HiddenSize != mixtralLoadHidden {
		t.Fatalf("FillModelInfo = vocab %d hidden %d, want %d/%d", info.VocabSize, info.HiddenSize, mixtralLoadVocab, mixtralLoadHidden)
	}
	if info.ContextLength != 32 {
		t.Fatalf("ContextLength = %d, want 32", info.ContextLength)
	}
}

// TestMethods_FillModelInfo_Quantized_Ugly exercises the quantization branch: a
// config carrying a QuantizationConfig must report its bits/group into ModelInfo
// (the loaded-model fixture is unquantized, so this drives the branch directly
// off a hand-built config).
func TestMethods_FillModelInfo_Quantized_Ugly(t *testing.T) {
	model := &MixtralModel{Cfg: &MixtralConfig{
		VocabSize:             5,
		HiddenSize:            8,
		MaxPositionEmbeddings: 32,
		Quantization:          &metal.QuantizationConfig{Bits: 4, GroupSize: 64},
	}}
	info := &metal.ModelInfo{}
	model.FillModelInfo(info)
	if info.QuantBits != 4 || info.QuantGroup != 64 {
		t.Fatalf("FillModelInfo quant = %d-bit/group %d, want 4/64", info.QuantBits, info.QuantGroup)
	}
}

// --- init loader registration (methods.go) ---
//
// init() registers a "mixtral" loader closure with the metal model registry. The
// registry's lookup is unexported (metal.lookupModelLoader), so the package's own
// test cannot fetch+invoke the registered closure through any exported metal API
// — the closure BODY (which calls LoadMixtral) is exercised only by the
// orchestrator's loader dispatch on a real checkpoint, an honest live-model-only
// floor. What the package CAN assert is that the metal-side registry contract the
// init relies on is the same one LoadMixtral already satisfies: LoadMixtral
// returns a value that satisfies metal.InternalModel (the closure's return type),
// so a successful load is dispatch-compatible. The load path itself is covered by
// the LoadMixtral_* tests.

// TestMethods_LoaderRegistration_Good asserts a loaded model satisfies the
// metal.InternalModel contract the registered loader closure returns — the
// compile-time + run-time guarantee that LoadMixtral is dispatch-compatible.
func TestMethods_LoaderRegistration_Good(t *testing.T) {
	model := loadMixtralModel(t)
	var im metal.InternalModel = model // compile-time: *MixtralModel is an InternalModel
	if im == nil {
		t.Fatal("loaded model is not a metal.InternalModel")
	}
	if im.ModelType() != "mixtral" {
		t.Fatalf("InternalModel.ModelType() = %q, want mixtral", im.ModelType())
	}
}
