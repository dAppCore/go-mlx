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

// TestMethods_RegistryDispatch_Good loads the synthetic mixed dense+MoE fixture
// through metal.LoadAndInit, which probes model_type="mixtral" from config.json
// and dispatches via the metal loader registry — exercising the closure
// registered in init (methods.go 13-15) that bridges the registry to LoadMixtral.
// This drives the closure BODY end-to-end without an import cycle (the kimi
// precedent), so the registration is covered by load, not just a type assertion.
func TestMethods_RegistryDispatch_Good(t *testing.T) {
	requireMetalRuntime(t)
	dir := t.TempDir()
	writeMixtralModel(t, dir)

	model, err := metal.LoadAndInit(dir)
	if err != nil {
		t.Fatalf("LoadAndInit(mixtral) error = %v", err)
	}
	defer model.Close()
	if model.ModelType() != "mixtral" {
		t.Fatalf("dispatched ModelType() = %q, want mixtral", model.ModelType())
	}
}
