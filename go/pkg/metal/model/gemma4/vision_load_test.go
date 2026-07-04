// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/metal"
)

// These tests drive the vision-tower BUILD paths directly with synthetic weight
// maps — the error/branch lanes a full happy-path load (loadGemma4VisionTestModel
// in vision_forward_test.go) does not reach: deliberately-broken weight maps for
// the validate* guards, the projection-only checkpoint shape, the patch-conv
// reshape variants, and the unified-skip predicate.

// TestVisionLoad_ShouldBuildEncoderTower_Good distinguishes the plain gemma4
// family (builds the SigLIP encoder tower) from the unified families and the
// unified-vision sub-config, which carry no separate encoder tower and must skip
// the build. The nil-config case defaults to building.
func TestVisionLoad_ShouldBuildEncoderTower_Good(t *testing.T) {
	if !gemma4VisionShouldBuildEncoderTower(nil) {
		t.Fatal("nil config should default to building the tower")
	}
	if !gemma4VisionShouldBuildEncoderTower(&Gemma4TextConfig{TransformerConfig: metal.TransformerConfig{ModelType: "gemma4"}}) {
		t.Fatal("plain gemma4 should build the encoder tower")
	}
	for _, mt := range []string{"gemma4_unified", "gemma4_unified_text"} {
		cfg := &Gemma4TextConfig{TransformerConfig: metal.TransformerConfig{ModelType: mt}}
		if gemma4VisionShouldBuildEncoderTower(cfg) {
			t.Fatalf("%s should skip the encoder tower (no separate SigLIP stack)", mt)
		}
	}
	unifiedVision := &Gemma4TextConfig{
		TransformerConfig: metal.TransformerConfig{ModelType: "gemma4"},
		VisionConfig:      &Gemma4VisionConfig{TransformerConfig: metal.TransformerConfig{ModelType: "gemma4_unified_vision"}},
	}
	if gemma4VisionShouldBuildEncoderTower(unifiedVision) {
		t.Fatal("gemma4_unified_vision sub-config should skip the encoder tower")
	}
}

// TestVisionLoad_BuildComponents_NoWeights_Good pins the no-vision-weights branch:
// a weight map with neither tower nor projection weights builds nothing (nil
// tower, nil projector, no error) — a text-only checkpoint.
func TestVisionLoad_BuildComponents_NoWeights_Good(t *testing.T) {
	requireMetalRuntime(t)

	cfg := &Gemma4TextConfig{TransformerConfig: metal.TransformerConfig{ModelType: "gemma4", HiddenSize: 8}}
	weights := map[string]*metal.Array{"unrelated.weight": seqArray(0.1, 4)}
	vision, projector, err := buildGemma4VisionComponents(cfg, weights)
	if err != nil {
		t.Fatalf("buildGemma4VisionComponents: %v", err)
	}
	if vision != nil || projector != nil {
		t.Fatalf("text-only weights built vision=%v projector=%v, want both nil", vision, projector)
	}
}

// TestVisionLoad_BuildComponents_ProjectionOnly_Good pins the projection-only
// branch: a checkpoint with the multimodal projector but no patch-embedder tower
// weights (the unified shape, where the SigLIP tower lives elsewhere) builds a
// projector and no tower. The projector maps the vision embed dim to text hidden.
func TestVisionLoad_BuildComponents_ProjectionOnly_Good(t *testing.T) {
	requireMetalRuntime(t)

	cfg := &Gemma4TextConfig{
		TransformerConfig: metal.TransformerConfig{ModelType: "gemma4", HiddenSize: gemma4VisionTextHid},
		VisionConfig:      &Gemma4VisionConfig{TransformerConfig: metal.TransformerConfig{HiddenSize: gemma4VisionHidden}},
	}
	weights := map[string]*metal.Array{
		"multi_modal_projector.embedding_projection.weight": seqArray(0.5, gemma4VisionTextHid, gemma4VisionHidden),
	}
	vision, projector, err := buildGemma4VisionComponents(cfg, weights)
	if err != nil {
		t.Fatalf("buildGemma4VisionComponents: %v", err)
	}
	defer closeGemma4Vision(vision, projector)
	if vision != nil {
		t.Fatal("projection-only checkpoint built a tower, want nil")
	}
	if projector == nil || projector.Projection == nil {
		t.Fatal("projection-only checkpoint did not build a projector")
	}
}

// TestVisionLoad_BuildVisionModel_MissingPatchWeight_Bad pins the fail-loud guard:
// a weight map with encoder layers but no patch-embedding weight is rejected
// rather than building a tower with no front-end.
func TestVisionLoad_BuildVisionModel_MissingPatchWeight_Bad(t *testing.T) {
	requireMetalRuntime(t)

	cfg := normalizeGemma4VisionConfig(&Gemma4VisionConfig{
		TransformerConfig: metal.TransformerConfig{HiddenSize: gemma4VisionHidden, NumAttentionHeads: gemma4VisionHeads, NumHiddenLayers: 1},
		PatchSize:         gemma4VisionPatch,
	})
	weights := map[string]*metal.Array{"unrelated.weight": seqArray(0.1, 4)}
	defer freeWeightMap(weights)

	if _, err := buildGemma4VisionModel(cfg, weights); err == nil {
		t.Fatal("buildGemma4VisionModel with no patch weight returned nil error")
	}
}

// TestVisionLoad_BuildVisionModel_MissingLayerWeight_Bad pins the per-layer
// validation: a tower whose first encoder layer is missing its q-projection is
// rejected by validateGemma4VisionEncoderLayer with a layer-scoped error.
func TestVisionLoad_BuildVisionModel_MissingLayerWeight_Bad(t *testing.T) {
	requireMetalRuntime(t)

	cfg := normalizeGemma4VisionConfig(&Gemma4VisionConfig{
		TransformerConfig: metal.TransformerConfig{HiddenSize: gemma4VisionHidden, NumAttentionHeads: gemma4VisionHeads, NumHiddenLayers: 1},
		PatchSize:         gemma4VisionPatch,
	})
	// Patch weight present, but the single encoder layer has no attention/MLP
	// weights → validateGemma4VisionEncoderLayer must fail.
	weights := map[string]*metal.Array{
		"patch_embedder.input_proj.weight": seqArray(0.01, gemma4VisionHidden, gemma4VisionPatchDim),
	}
	defer freeWeightMap(weights)

	_, err := buildGemma4VisionModel(cfg, weights)
	if err == nil {
		t.Fatal("buildGemma4VisionModel with an incomplete encoder layer returned nil error")
	}
	if !core.Contains(err.Error(), "encoder layer 0") {
		t.Fatalf("error = %v, want a layer-0-scoped validation failure", err)
	}
}

// TestVisionLoad_ValidateEncoderLayer_Good walks validateGemma4VisionEncoderLayer
// across each missing-component case — nil attention, nil MLP, and a missing norm
// — confirming each returns a distinct, component-named error, and that a fully
// populated layer validates clean.
func TestVisionLoad_ValidateEncoderLayer_Good(t *testing.T) {
	requireMetalRuntime(t)

	ones := func() *metal.RMSNormModule {
		return &metal.RMSNormModule{Weight: gemma4Ones([]int32{gemma4VisionHidden})}
	}
	lin := func() *metal.Linear {
		return metal.NewLinear(seqArray(0.1, gemma4VisionHidden, gemma4VisionHidden), nil)
	}
	full := func() *Gemma4VisionEncoderLayer {
		return &Gemma4VisionEncoderLayer{
			InputNorm:    ones(),
			PostAttnNorm: ones(),
			PreFFNorm:    ones(),
			PostFFNorm:   ones(),
			Attention: &Gemma4VisionAttention{
				QProj: lin(), KProj: lin(), VProj: lin(), OProj: lin(),
				QNorm: ones(), KNorm: ones(),
			},
			MLP: &Gemma4VisionMLP{GateProj: lin(), UpProj: lin(), DownProj: lin()},
		}
	}

	if err := validateGemma4VisionEncoderLayer(full(), 0); err != nil {
		t.Fatalf("a fully populated layer failed validation: %v", err)
	}

	noAttn := full()
	noAttn.Attention = nil
	if err := validateGemma4VisionEncoderLayer(noAttn, 1); err == nil || !core.Contains(err.Error(), "attention") {
		t.Fatalf("nil-attention error = %v, want an attention-named failure", err)
	}

	noMLP := full()
	noMLP.MLP = nil
	if err := validateGemma4VisionEncoderLayer(noMLP, 2); err == nil || !core.Contains(err.Error(), "mlp") {
		t.Fatalf("nil-MLP error = %v, want an mlp-named failure", err)
	}

	noNorm := full()
	noNorm.InputNorm = nil
	if err := validateGemma4VisionEncoderLayer(noNorm, 3); err == nil || !core.Contains(err.Error(), "input norm") {
		t.Fatalf("missing-input-norm error = %v, want an input-norm-named failure", err)
	}

	noQNorm := full()
	noQNorm.Attention.QNorm = nil
	if err := validateGemma4VisionEncoderLayer(noQNorm, 4); err == nil || !core.Contains(err.Error(), "q norm") {
		t.Fatalf("missing-q-norm error = %v, want a q-norm-named failure", err)
	}

	// The remaining per-component missing cases, each driving a distinct
	// component-named decline so every norm/linear guard in the ladder fires.
	type drop struct {
		name   string
		mutate func(*Gemma4VisionEncoderLayer)
		want   string
	}
	drops := []drop{
		{"post-attn norm", func(l *Gemma4VisionEncoderLayer) { l.PostAttnNorm = nil }, "post-attention norm"},
		{"pre-ff norm", func(l *Gemma4VisionEncoderLayer) { l.PreFFNorm = nil }, "pre-feedforward norm"},
		{"post-ff norm", func(l *Gemma4VisionEncoderLayer) { l.PostFFNorm = nil }, "post-feedforward norm"},
		{"k projection", func(l *Gemma4VisionEncoderLayer) { l.Attention.KProj = nil }, "k projection"},
		{"v projection", func(l *Gemma4VisionEncoderLayer) { l.Attention.VProj = nil }, "v projection"},
		{"o projection", func(l *Gemma4VisionEncoderLayer) { l.Attention.OProj = nil }, "output projection"},
		{"k norm", func(l *Gemma4VisionEncoderLayer) { l.Attention.KNorm = nil }, "k norm"},
		{"gate projection", func(l *Gemma4VisionEncoderLayer) { l.MLP.GateProj = nil }, "gate projection"},
		{"up projection", func(l *Gemma4VisionEncoderLayer) { l.MLP.UpProj = nil }, "up projection"},
		{"down projection", func(l *Gemma4VisionEncoderLayer) { l.MLP.DownProj = nil }, "down projection"},
	}
	for i, d := range drops {
		layer := full()
		d.mutate(layer)
		err := validateGemma4VisionEncoderLayer(layer, int32(5+i))
		if err == nil || !core.Contains(err.Error(), d.want) {
			t.Fatalf("missing %s: error = %v, want a %q-named failure", d.name, err, d.want)
		}
	}
}

// TestVisionLoad_NormalizePatchProjection_Good walks normalizeGemma4PatchProjection
// across its weight-rank branches: the 2-D linear form (reshaped to a conv
// kernel), the 4-D channels-last conv weight (used directly), the 4-D
// channels-first weight (transposed to channels-last), and the nil guard.
func TestVisionLoad_NormalizePatchProjection_Good(t *testing.T) {
	requireMetalRuntime(t)

	cfg := normalizeGemma4VisionConfig(&Gemma4VisionConfig{
		TransformerConfig: metal.TransformerConfig{HiddenSize: gemma4VisionHidden},
		PatchSize:         gemma4VisionPatch,
		NumChannels:       3,
	})

	// nil weight → not ok.
	if _, _, ok := normalizeGemma4PatchProjection(nil, cfg); ok {
		t.Fatal("nil patch weight reported ok")
	}

	// 2-D [hidden, patchDim] → reshaped to conv [hidden, patch, patch, channels].
	lin2d := seqArray(0.01, gemma4VisionHidden, gemma4VisionPatchDim)
	defer metal.Free(lin2d)
	linW, convW, ok := normalizeGemma4PatchProjection(lin2d, cfg)
	if !ok || linW == nil || convW == nil {
		t.Fatalf("2-D projection = (%v,%v,%v), want both tensors", linW, convW, ok)
	}
	if err := metal.Eval(convW); err != nil {
		t.Fatalf("Eval 2-D conv: %v", err)
	}
	if got := convW.Shape(); len(got) != 4 || got[0] != gemma4VisionHidden || got[3] != 3 {
		t.Fatalf("2-D→conv shape = %v, want [%d %d %d 3]", got, gemma4VisionHidden, gemma4VisionPatch, gemma4VisionPatch)
	}
	metal.Free(convW)

	// 4-D channels-last [hidden, patch, patch, channels] → used directly.
	convLast := seqArray(0.02, gemma4VisionHidden, gemma4VisionPatch, gemma4VisionPatch, 3)
	defer metal.Free(convLast)
	linW2, convW2, ok := normalizeGemma4PatchProjection(convLast, cfg)
	if !ok || linW2 == nil || convW2 == nil {
		t.Fatalf("4-D channels-last = (%v,%v,%v), want both tensors", linW2, convW2, ok)
	}
	if err := metal.Eval(linW2); err != nil {
		t.Fatalf("Eval 4-D channels-last linear: %v", err)
	}
	if got := linW2.Shape(); len(got) != 2 || got[0] != gemma4VisionHidden || got[1] != gemma4VisionPatchDim {
		t.Fatalf("4-D channels-last linear = %v, want [%d %d]", got, gemma4VisionHidden, gemma4VisionPatchDim)
	}
	metal.Free(linW2)

	// 4-D channels-first [hidden, channels, patch, patch] → transposed then flat.
	convFirst := seqArray(0.03, gemma4VisionHidden, 3, gemma4VisionPatch, gemma4VisionPatch)
	defer metal.Free(convFirst)
	linW3, convW3, ok := normalizeGemma4PatchProjection(convFirst, cfg)
	if !ok || linW3 == nil || convW3 == nil {
		t.Fatalf("4-D channels-first = (%v,%v,%v), want both tensors", linW3, convW3, ok)
	}
	if err := metal.Eval(linW3); err != nil {
		t.Fatalf("Eval 4-D channels-first linear: %v", err)
	}
	if got := convW3.Shape(); len(got) != 4 || got[3] != 3 {
		t.Fatalf("4-D channels-first conv = %v, want channels-last [.. 3]", got)
	}
	metal.Free(linW3)
}

// TestVisionLoad_BuildMultiModalProjector_Linear1Linear2_Good covers the
// two-layer MLP projector branch (linear_1 + gelu + linear_2) used when a
// checkpoint ships an MLP projector instead of a single projection matrix.
func TestVisionLoad_BuildMultiModalProjector_Linear1Linear2_Good(t *testing.T) {
	requireMetalRuntime(t)

	textCfg := &Gemma4TextConfig{TransformerConfig: metal.TransformerConfig{HiddenSize: gemma4VisionTextHid}}
	visionCfg := normalizeGemma4VisionConfig(&Gemma4VisionConfig{TransformerConfig: metal.TransformerConfig{HiddenSize: gemma4VisionHidden}})
	weights := map[string]*metal.Array{
		"multi_modal_projector.linear_1.weight": seqArray(0.1, gemma4VisionInterm, gemma4VisionHidden),
		"multi_modal_projector.linear_2.weight": seqArray(0.2, gemma4VisionTextHid, gemma4VisionInterm),
	}
	defer freeWeightMap(weights)

	projector := buildGemma4MultiModalProjector(textCfg, visionCfg, weights)
	if projector == nil || projector.Linear1 == nil || projector.Linear2 == nil {
		t.Fatalf("MLP projector = %+v, want linear_1 + linear_2 wired", projector)
	}
	if projector.Projection != nil {
		t.Fatal("single-projection set on an MLP-only projector, want nil")
	}
}

// TestVisionLoad_BuildMultiModalProjector_DimMismatch_Bad pins the guard that a
// projector with no usable weights AND a vision/text hidden-size mismatch is
// rejected (nil) rather than returned half-built.
func TestVisionLoad_BuildMultiModalProjector_DimMismatch_Bad(t *testing.T) {
	textCfg := &Gemma4TextConfig{TransformerConfig: metal.TransformerConfig{HiddenSize: gemma4VisionTextHid}}
	visionCfg := normalizeGemma4VisionConfig(&Gemma4VisionConfig{TransformerConfig: metal.TransformerConfig{HiddenSize: gemma4VisionHidden}})
	// Empty weights → no projection, no MLP → not ready; hidden sizes differ.
	weights := map[string]*metal.Array{}

	if projector := buildGemma4MultiModalProjector(textCfg, visionCfg, weights); projector != nil {
		t.Fatalf("buildGemma4MultiModalProjector = %+v, want nil on mismatch with no weights", projector)
	}
}

// TestVisionLoad_VisionNorm_FallbackToOnes_Good covers gemma4VisionNorm's
// fallback: when the named weight is absent it returns a ones-initialised norm
// at the requested width rather than nil, so a tower with an absent optional norm
// still validates.
func TestVisionLoad_VisionNorm_FallbackToOnes_Good(t *testing.T) {
	requireMetalRuntime(t)

	present := map[string]*metal.Array{"some.norm.weight": seqArray(0.3, gemma4VisionHidden)}
	defer freeWeightMap(present)

	got := gemma4VisionNorm(present, gemma4VisionHidden, "some.norm.weight")
	if got == nil || got.Weight == nil {
		t.Fatal("gemma4VisionNorm returned nil for a present weight")
	}

	// Absent → ones fallback at the requested width.
	fallback := gemma4VisionNorm(map[string]*metal.Array{}, gemma4VisionHidden, "absent.norm.weight")
	if fallback == nil || fallback.Weight == nil {
		t.Fatal("gemma4VisionNorm fallback returned nil weight")
	}
	defer metal.Free(fallback.Weight)
	if err := metal.Eval(fallback.Weight); err != nil {
		t.Fatalf("Eval fallback norm: %v", err)
	}
	if got := fallback.Weight.Shape(); len(got) != 1 || got[0] != gemma4VisionHidden {
		t.Fatalf("fallback norm shape = %v, want [%d]", got, gemma4VisionHidden)
	}
}
