// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gptoss

import (
	"testing"

	"dappco.re/go/mlx/pkg/metal"
)

// ── ModelType / NumLayers / Tokenizer accessors ────────────────────────────

func TestGptOss_ModelType_Good(t *testing.T) {
	m := &GptOssModel{modelType: "gpt_oss"}
	if m.ModelType() != "gpt_oss" {
		t.Fatalf("ModelType() = %q, want gpt_oss", m.ModelType())
	}
}

func TestGptOss_NumLayers_Good(t *testing.T) {
	m := &GptOssModel{Layers: []*GptOssDecoderLayer{nil, nil, nil}}
	if m.NumLayers() != 3 {
		t.Fatalf("NumLayers() = %d, want 3", m.NumLayers())
	}
}

func TestGptOss_Tokenizer_Good(t *testing.T) {
	model := loadMixedGptOss(t)
	if model.Tokenizer() == nil {
		t.Fatal("Tokenizer() = nil, want the loaded tokenizer")
	}
	if model.Tokenizer() != model.Tok {
		t.Fatal("Tokenizer() did not return the model's Tok field")
	}
}

// ── MoETextDecodeFamily / MoETextRuntimeAvailable (reporter capability) ─────

func TestGptOss_MoETextDecodeFamily_Good(t *testing.T) {
	m := &GptOssModel{}
	if got := m.MoETextDecodeFamily(); got != "gpt_oss" {
		t.Fatalf("MoETextDecodeFamily() = %q, want gpt_oss", got)
	}
}

func TestGptOss_MoETextRuntimeAvailable_Bad(t *testing.T) {
	if (&GptOssModel{Layers: []*GptOssDecoderLayer{{Dense: &metal.DenseDecoderLayer{}}}}).MoETextRuntimeAvailable() {
		t.Fatal("GptOssModel.MoETextRuntimeAvailable(incomplete) = true, want false")
	}
}

func TestGptOss_MoETextRuntimeAvailable_NilReceiver_Ugly(t *testing.T) {
	var m *GptOssModel
	if m.MoETextRuntimeAvailable() {
		t.Fatal("nil-receiver MoETextRuntimeAvailable() = true, want false")
	}
}

// TestGptOss_MoETextRuntimeAvailable_NilLayer_Ugly drives the per-layer nil
// guard inside the reporter callback (gptoss.go lines 91-93).
func TestGptOss_MoETextRuntimeAvailable_NilLayer_Ugly(t *testing.T) {
	m := &GptOssModel{Layers: []*GptOssDecoderLayer{nil}}
	if m.MoETextRuntimeAvailable() {
		t.Fatal("model with a nil layer MoETextRuntimeAvailable() = true, want false")
	}
}

func TestGptOss_MoETextRuntimeAvailable_Good(t *testing.T) {
	requireMetalRuntime(t)
	model := loadMixedGptOss(t)
	if !model.MoETextRuntimeAvailable() {
		t.Fatal("loaded model MoETextRuntimeAvailable() = false, want true (native MoE decode linked)")
	}
}

// ── FillModelInfo (sizing + quantization reporter) ─────────────────────────

func TestGptOss_FillModelInfo_Good(t *testing.T) {
	model := loadMixedGptOss(t)
	info := &metal.ModelInfo{}
	model.FillModelInfo(info)
	if info.VocabSize != 5 {
		t.Fatalf("info.VocabSize = %d, want 5", info.VocabSize)
	}
	if info.HiddenSize != 8 {
		t.Fatalf("info.HiddenSize = %d, want 8", info.HiddenSize)
	}
	if info.ContextLength != 32 {
		t.Fatalf("info.ContextLength = %d, want 32 (max_position_embeddings)", info.ContextLength)
	}
}

// TestGptOss_FillModelInfo_Quantized confirms the quant fields are copied when
// the config carries a quantization block (gptoss.go lines 510-513).
func TestGptOss_FillModelInfo_Quantized(t *testing.T) {
	model := &GptOssModel{Cfg: &GptOssConfig{
		VocabSize:             100,
		HiddenSize:            64,
		MaxPositionEmbeddings: 4096,
		Quantization:          &metal.QuantizationConfig{Bits: 4, GroupSize: 64},
	}}
	info := &metal.ModelInfo{}
	model.FillModelInfo(info)
	if info.QuantBits != 4 || info.QuantGroup != 64 {
		t.Fatalf("info quant = bits %d group %d, want bits 4 group 64", info.QuantBits, info.QuantGroup)
	}
}

// ── registry dispatch (init closure, gptoss.go 516-520) ────────────────────

// TestGptOss_RegistryDispatch_Good loads the mixed fixture through
// metal.LoadAndInit, which probes model_type="gpt_oss" and dispatches via the
// loader registry — exercising the closure registered in init (gptoss.go
// 516-520) that bridges the registry to LoadGptOss. This is the synthetic path
// the kimi twin uses to cover its init; no import cycle is needed.
func TestGptOss_RegistryDispatch_Good(t *testing.T) {
	requireMetalRuntime(t)
	dir := t.TempDir()
	writeMixedGptOssModel(t, dir)

	model, err := metal.LoadAndInit(dir)
	if err != nil {
		t.Fatalf("LoadAndInit(gpt_oss) error = %v", err)
	}
	defer model.Close()
	if model.ModelType() != "gpt_oss" {
		t.Fatalf("dispatched ModelType() = %q, want gpt_oss", model.ModelType())
	}
}
