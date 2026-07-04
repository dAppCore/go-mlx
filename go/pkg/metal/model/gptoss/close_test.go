// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gptoss

import "testing"

// ── closeGptOss / CloseModel ───────────────────────────────────────────────

func TestGptOss_CloseModel_Good(t *testing.T) {
	requireMetalRuntime(t)
	dir := t.TempDir()
	writeMixedGptOssModel(t, dir)
	model, err := LoadGptOss(dir)
	if err != nil {
		t.Fatalf("LoadGptOss error = %v", err)
	}
	embedW := model.EmbedTokens.Weight
	outW := model.Output.Weight
	denseGate := model.Layers[0].Dense.MLP.GateProj.Weight
	qW := model.Layers[0].Dense.Attention.QProj.Weight

	model.CloseModel()

	if embedW.Valid() {
		t.Error("embed weight should be freed after CloseModel")
	}
	if outW != embedW && outW.Valid() {
		t.Error("output weight should be freed after CloseModel")
	}
	if denseGate.Valid() {
		t.Error("dense MLP gate weight should be freed after CloseModel")
	}
	if qW.Valid() {
		t.Error("q_proj weight should be freed after CloseModel")
	}
	if model.Layers != nil {
		t.Error("Layers should be nil after CloseModel")
	}
}

func TestGptOss_CloseModel_NilModel_Ugly(t *testing.T) {
	defer func() {
		if recovered := recover(); recovered != nil {
			t.Fatalf("CloseModel on nil model panicked: %v", recovered)
		}
	}()
	var m *GptOssModel
	m.CloseModel()
}

// TestGptOss_CloseModel_NilLayer_Ugly drives the nil/Dense-less layer continue
// in closeGptOss (gptoss.go lines 472-474): a model carrying a nil layer entry
// and a Dense-less layer must skip both without panicking. No Metal — the free
// helpers are nil-tolerant.
func TestGptOss_CloseModel_NilLayer_Ugly(t *testing.T) {
	defer func() {
		if recovered := recover(); recovered != nil {
			t.Fatalf("CloseModel with a nil layer panicked: %v", recovered)
		}
	}()
	m := &GptOssModel{Layers: []*GptOssDecoderLayer{nil, {Dense: nil}}}
	m.CloseModel()
	if m.Layers != nil {
		t.Fatal("Layers should be nil after CloseModel")
	}
}
