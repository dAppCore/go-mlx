// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	"testing"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
	"dappco.re/go/mlx/pkg/metal"
)

// gemma4VisionConfigJSON is a synthetic gemma4 multimodal config: the tiny dense
// text trunk (matching gemma4TinyWeights) plus a small SigLIP-derived vision
// tower and an image token id that fits the 10-row embed table. model_type is
// the plain "gemma4" so gemma4VisionShouldBuildEncoderTower builds the encoder
// (a *_unified type would skip the tower). The vision dims are deliberately tiny
// — hidden 16, 2 layers, 4 heads, patch 8, pooling kernel 2 — so a 32x32 image
// makes a 4x4 patch grid that pools cleanly through poolByGrid into 4 soft
// tokens, and the projector maps those 16-wide rows down to the text hidden 8.
const gemma4VisionConfigJSON = `{
	"model_type": "gemma4",
	"image_token_id": 5,
	"text_config": {
		"model_type": "gemma4_text",
		"hidden_size": 8,
		"num_hidden_layers": 2,
		"intermediate_size": 16,
		"num_attention_heads": 1,
		"num_key_value_heads": 1,
		"head_dim": 4,
		"global_head_dim": 8,
		"vocab_size": 10,
		"max_position_embeddings": 16,
		"rms_norm_eps": 1e-6,
		"sliding_window": 4,
		"sliding_window_pattern": 2,
		"num_kv_shared_layers": 0,
		"hidden_size_per_layer_input": 0,
		"use_double_wide_mlp": false,
		"layer_types": ["sliding_attention", "full_attention"]
	},
	"vision_config": {
		"model_type": "gemma4_vision",
		"hidden_size": 16,
		"intermediate_size": 32,
		"num_hidden_layers": 2,
		"num_attention_heads": 4,
		"num_key_value_heads": 4,
		"patch_size": 8,
		"pooling_kernel_size": 2,
		"position_embedding_size": 64,
		"rope_parameters": {"rope_type": "default", "rope_theta": 100}
	}
}`

const (
	gemma4VisionHidden    = 16
	gemma4VisionPatch     = 8
	gemma4VisionLayers    = 2
	gemma4VisionHeads     = 4
	gemma4VisionHeadDim   = gemma4VisionHidden / gemma4VisionHeads
	gemma4VisionInterm    = 32
	gemma4VisionPatchDim  = gemma4VisionPatch * gemma4VisionPatch * 3 // 192
	gemma4VisionTextHid   = 8
	gemma4VisionImageTok  = 5
	gemma4VisionPosEmbCnt = 64
)

// gemma4VisionTinyWeights extends gemma4TinyWeights with a synthetic SigLIP
// vision tower and the multimodal projector, using the raw checkpoint key names
// (vision_tower.* and multi_modal_projector.*) that sanitizeGemma4VisionWeights
// canonicalises at load. Shapes mirror buildGemma4VisionModel's reads: the patch
// projection is the 2-D [hidden, patchDim] form normalizeGemma4PatchProjection
// reshapes into a conv kernel, each encoder layer carries q/k/v/o projections +
// q/k norms + the four layernorms + the gated MLP, and the projector maps the
// 16-wide pooled rows down to the text hidden size 8.
func gemma4VisionTinyWeights() map[string]*metal.Array {
	w := gemma4TinyWeights()

	// Patch embedder: 2-D input projection [hidden, patchDim] + learned 2-D
	// position table [posEmbCount, hidden] (flat fallback branch).
	w["vision_tower.patch_embedder.input_proj.weight"] = seqArray(0.01, gemma4VisionHidden, gemma4VisionPatchDim)
	w["vision_tower.patch_embedder.position_embedding_table"] = seqArray(0.02, gemma4VisionPosEmbCnt, gemma4VisionHidden)
	w["vision_tower.post_layernorm.weight"] = seqArray(0.03, gemma4VisionHidden)

	for i := 0; i < gemma4VisionLayers; i++ {
		prefix := core.Sprintf("vision_tower.encoder.layers.%d", i)
		base := 0.10 + float32(i)
		w[prefix+".input_layernorm.weight"] = seqArray(base+0.01, gemma4VisionHidden)
		w[prefix+".post_attention_layernorm.weight"] = seqArray(base+0.02, gemma4VisionHidden)
		w[prefix+".pre_feedforward_layernorm.weight"] = seqArray(base+0.03, gemma4VisionHidden)
		w[prefix+".post_feedforward_layernorm.weight"] = seqArray(base+0.04, gemma4VisionHidden)

		w[prefix+".self_attn.q_proj.weight"] = seqArray(base+0.05, gemma4VisionHeads*gemma4VisionHeadDim, gemma4VisionHidden)
		w[prefix+".self_attn.k_proj.weight"] = seqArray(base+0.06, gemma4VisionHeads*gemma4VisionHeadDim, gemma4VisionHidden)
		w[prefix+".self_attn.v_proj.weight"] = seqArray(base+0.07, gemma4VisionHeads*gemma4VisionHeadDim, gemma4VisionHidden)
		w[prefix+".self_attn.o_proj.weight"] = seqArray(base+0.08, gemma4VisionHidden, gemma4VisionHeads*gemma4VisionHeadDim)
		w[prefix+".self_attn.q_norm.weight"] = seqArray(base+0.09, gemma4VisionHeadDim)
		w[prefix+".self_attn.k_norm.weight"] = seqArray(base+0.11, gemma4VisionHeadDim)

		w[prefix+".mlp.gate_proj.weight"] = seqArray(base+0.12, gemma4VisionInterm, gemma4VisionHidden)
		w[prefix+".mlp.up_proj.weight"] = seqArray(base+0.13, gemma4VisionInterm, gemma4VisionHidden)
		w[prefix+".mlp.down_proj.weight"] = seqArray(base+0.14, gemma4VisionHidden, gemma4VisionInterm)
	}

	// Multimodal projector: pooled vision rows [*, 16] → text hidden [*, 8].
	w["multi_modal_projector.embedding_projection.weight"] = seqArray(0.5, gemma4VisionTextHid, gemma4VisionHidden)
	return w
}

// loadGemma4VisionTestModel writes the synthetic multimodal checkpoint (config +
// tokenizer + gemma4VisionTinyWeights) and loads it, registering cleanup. The
// loaded model carries a real VisionTower + MultiModalProjector built through the
// production buildGemma4VisionComponents path on top of the tiny dense text
// trunk, so the whole vision cascade (patch embed → encoder → pooler → project →
// inject → masked text forward) runs on synthetic weights. This is the vision
// sibling of loadGemma4DiffusionTestModel — no real multi-GB checkpoint.
func loadGemma4VisionTestModel(t *testing.T) *Gemma4Model {
	t.Helper()
	requireMetalRuntime(t)

	dir := t.TempDir()
	if err := coreio.Local.Write(core.JoinPath(dir, "config.json"), gemma4VisionConfigJSON); err != nil {
		t.Fatalf("write config.json: %v", err)
	}
	writeMinimalTokenizer(t, dir)
	if err := metal.SaveSafetensors(core.JoinPath(dir, "model.safetensors"), gemma4VisionTinyWeights()); err != nil {
		t.Fatalf("SaveSafetensors: %v", err)
	}

	model, err := LoadGemma4(dir)
	if err != nil {
		t.Fatalf("LoadGemma4: %v", err)
	}
	t.Cleanup(func() { closeGemma4(model) })
	return model
}

// visionTestPixels returns a synthetic [1, 32, 32, 3] image — NHWC rank-4, the
// shape the engine hands the tower. 32/patch 8 = a 4x4 patch grid.
func visionTestPixels() *metal.Array {
	return metal.Zeros([]int32{1, 32, 32, 3}, metal.DTypeFloat32)
}

// TestVisionForward_TowerBuilt_Good asserts the synthetic checkpoint actually
// built a vision tower + projector through the production load path — the
// precondition every other forward test in this file depends on. If this fails,
// the dims in gemma4VisionTinyWeights disagree with the config, not the code.
func TestVisionForward_TowerBuilt_Good(t *testing.T) {
	model := loadGemma4VisionTestModel(t)

	if model.VisionTower == nil {
		t.Fatal("VisionTower = nil, want a built tower")
	}
	if model.MultiModalProjector == nil {
		t.Fatal("MultiModalProjector = nil, want a built projector")
	}
	if got := len(model.VisionTower.Encoder.Layers); got != gemma4VisionLayers {
		t.Fatalf("encoder layers = %d, want %d", got, gemma4VisionLayers)
	}
	if got := model.VisionTower.Cfg.HiddenSize; got != gemma4VisionHidden {
		t.Fatalf("vision hidden = %d, want %d (inferred from patch weight)", got, gemma4VisionHidden)
	}
	if !model.AcceptsImageInput() {
		t.Fatal("AcceptsImageInput = false on a vision checkpoint, want true")
	}
}

// TestVisionForward_PatchEmbedder_Conv_Good drives the rank-4 NHWC patch-embed
// path (prepare case 4 → prepareRawNHWC → Conv2d) and asserts the patches come
// back as [1, nPatches, hidden] with a 4x4 grid. This is the first debug lever:
// a nil here is a dims problem, not a code bug.
func TestVisionForward_PatchEmbedder_Conv_Good(t *testing.T) {
	model := loadGemma4VisionTestModel(t)

	pixels := visionTestPixels()
	patches, gridH, gridW := model.VisionTower.PatchEmbedder.Forward(pixels)
	if patches == nil || !patches.Valid() {
		t.Fatal("PatchEmbedder.Forward returned nil patches — check fixture dims")
	}
	if err := metal.Eval(patches); err != nil {
		t.Fatalf("Eval patches: %v", err)
	}
	defer metal.Free(pixels, patches)

	if gridH != 4 || gridW != 4 {
		t.Fatalf("grid = %dx%d, want 4x4", gridH, gridW)
	}
	shape := patches.Shape()
	if len(shape) != 3 || shape[0] != 1 || shape[1] != 16 || shape[2] != gemma4VisionHidden {
		t.Fatalf("patches shape = %v, want [1 16 %d]", shape, gemma4VisionHidden)
	}
}

// TestVisionForward_PatchEmbedder_PreProjected_Good drives the rank-3 pre-patched
// path (prepare case 3, shape[2]==patchDim → projected=false → InputProj fires +
// the -0.5/×2 rescale), covering InputProj.Forward which the Conv2d branch skips.
func TestVisionForward_PatchEmbedder_PreProjected_Good(t *testing.T) {
	model := loadGemma4VisionTestModel(t)

	// [1, 16 patches, patchDim] — already flattened to patch vectors.
	prePatched := metal.Zeros([]int32{1, 16, gemma4VisionPatchDim}, metal.DTypeFloat32)
	patches, gridH, gridW := model.VisionTower.PatchEmbedder.Forward(prePatched)
	if patches == nil || !patches.Valid() {
		t.Fatal("PatchEmbedder.Forward (pre-patched) returned nil")
	}
	if err := metal.Eval(patches); err != nil {
		t.Fatalf("Eval patches: %v", err)
	}
	defer metal.Free(prePatched, patches)

	if gridH != 4 || gridW != 4 {
		t.Fatalf("grid = %dx%d, want 4x4", gridH, gridW)
	}
	shape := patches.Shape()
	if len(shape) != 3 || shape[1] != 16 || shape[2] != gemma4VisionHidden {
		t.Fatalf("pre-patched patches shape = %v, want [1 16 %d]", shape, gemma4VisionHidden)
	}
}

// TestVisionForward_VisionTowerForward_Good runs the whole tower (patch embed →
// encoder stack → post-layernorm → pooler → projection-free std) and asserts the
// pooled output is [nSoftTokens, hidden]. Pooling kernel 2 over the 4x4 grid
// halves both axes → 2x2 = 4 soft tokens, each hidden-wide before projection.
func TestVisionForward_VisionTowerForward_Good(t *testing.T) {
	model := loadGemma4VisionTestModel(t)

	pixels := visionTestPixels()
	pooled := model.VisionTower.Forward(pixels)
	if pooled == nil || !pooled.Valid() {
		t.Fatal("VisionTower.Forward returned nil")
	}
	if err := metal.Eval(pooled); err != nil {
		t.Fatalf("Eval pooled: %v", err)
	}
	defer metal.Free(pixels, pooled)

	shape := pooled.Shape()
	if len(shape) != 2 || shape[0] != 4 || shape[1] != gemma4VisionHidden {
		t.Fatalf("pooled shape = %v, want [4 %d] (poolByGrid: 4x4 grid, kernel 2)", shape, gemma4VisionHidden)
	}
}

// TestVisionForward_ProjectImageFeatures_Good runs the engine-facing projection
// seam (#98): tower forward + multimodal projector. The projector maps the
// 16-wide pooled rows to the text hidden 8 — the shape that injects into the
// token stream. This is the happy path the _Bad guards (in vision_chat_test.go)
// only fence.
func TestVisionForward_ProjectImageFeatures_Good(t *testing.T) {
	model := loadGemma4VisionTestModel(t)

	pixels := visionTestPixels()
	features, err := model.ProjectImageFeatures(pixels)
	if err != nil {
		t.Fatalf("ProjectImageFeatures: %v", err)
	}
	if err := metal.Eval(features); err != nil {
		t.Fatalf("Eval features: %v", err)
	}
	defer metal.Free(pixels, features)

	shape := features.Shape()
	if len(shape) != 2 || shape[1] != gemma4VisionTextHid {
		t.Fatalf("projected features shape = %v, want [_ %d] (text hidden)", shape, gemma4VisionTextHid)
	}
}

// TestVisionForward_ForwardMultiModal_Good runs the image-bearing prefill end to
// end: a token sequence with 4 image-placeholder slots (matching the 4 pooled
// soft tokens) plus the raw image, through encode → inject → masked text forward.
// The image token id (5) is below the 10-row embed table, so the full-sequence
// embed gather that precedes injection stays in bounds. Asserts full logits
// [1, L, vocab] — the slot/feature counts match, so this is a true happy path,
// not a logged-mismatch fallback.
func TestVisionForward_ForwardMultiModal_Good(t *testing.T) {
	model := loadGemma4VisionTestModel(t)

	// [bos, img, img, img, img, world] — 4 image slots = 4 pooled features.
	tokenIDs := []int32{2, gemma4VisionImageTok, gemma4VisionImageTok, gemma4VisionImageTok, gemma4VisionImageTok, 4}
	tokens := metal.FromValues(tokenIDs, 1, int(len(tokenIDs)))
	pixels := visionTestPixels()
	caches := model.NewCache()

	logits := model.ForwardMultiModal(tokens, []*metal.Array{pixels}, caches)
	if err := metal.Eval(logits); err != nil {
		t.Fatalf("Eval logits: %v", err)
	}
	defer func() {
		metal.Free(tokens, pixels, logits)
		metal.FreeCaches(caches)
	}()

	shape := logits.Shape()
	if len(shape) != 3 || shape[0] != 1 || shape[1] != int32(len(tokenIDs)) || shape[2] != 10 {
		t.Fatalf("multimodal logits shape = %v, want [1 %d 10]", shape, len(tokenIDs))
	}
}

// TestVisionForward_ForwardMultiModal_NoImageTokens_Good pins the fast path: when
// the prompt carries no image-placeholder ids, ForwardUnifiedVideoMultiModal
// detects zero modal tokens and falls back to a plain text Forward even though an
// image was supplied. Logits stay well-shaped over the text-only sequence.
func TestVisionForward_ForwardMultiModal_NoImageTokens_Good(t *testing.T) {
	model := loadGemma4VisionTestModel(t)

	tokens := metal.FromValues([]int32{2, 3, 4}, 1, 3) // no image token id 5
	pixels := visionTestPixels()
	caches := model.NewCache()

	logits := model.ForwardMultiModal(tokens, []*metal.Array{pixels}, caches)
	if err := metal.Eval(logits); err != nil {
		t.Fatalf("Eval logits: %v", err)
	}
	defer func() {
		metal.Free(tokens, pixels, logits)
		metal.FreeCaches(caches)
	}()

	shape := logits.Shape()
	if len(shape) != 3 || shape[0] != 1 || shape[1] != 3 || shape[2] != 10 {
		t.Fatalf("no-image-token logits shape = %v, want [1 3 10]", shape)
	}
}

// TestVisionForward_ForwardImageFeatures_Good drives the pre-projected prefill
// (#99 feature-cache seam): features already projected to text hidden are
// injected at placeholder positions without re-running the tower. Mirrors the
// multimodal happy path but enters through the lifted-encode entrypoint.
func TestVisionForward_ForwardImageFeatures_Good(t *testing.T) {
	model := loadGemma4VisionTestModel(t)

	pixels := visionTestPixels()
	features, err := model.ProjectImageFeatures(pixels)
	if err != nil {
		t.Fatalf("ProjectImageFeatures: %v", err)
	}
	metal.Free(pixels)

	// One image worth of features: 4 rows = 4 placeholder slots.
	tokenIDs := []int32{2, gemma4VisionImageTok, gemma4VisionImageTok, gemma4VisionImageTok, gemma4VisionImageTok, 4}
	tokens := metal.FromValues(tokenIDs, 1, int(len(tokenIDs)))
	caches := model.NewCache()

	logits := model.ForwardImageFeatures(tokens, []*metal.Array{features}, caches)
	if err := metal.Eval(logits); err != nil {
		t.Fatalf("Eval logits: %v", err)
	}
	defer func() {
		metal.Free(tokens, features, logits)
		metal.FreeCaches(caches)
	}()

	shape := logits.Shape()
	if len(shape) != 3 || shape[0] != 1 || shape[1] != int32(len(tokenIDs)) || shape[2] != 10 {
		t.Fatalf("pre-projected logits shape = %v, want [1 %d 10]", shape, len(tokenIDs))
	}
}

// TestVisionForward_EncodeGemma4Images_Multi_Good drives the multi-image batch
// branch of encodeGemma4Images: two images encode + project, then concatenate
// into one feature block (len(features) > 1 → Concatenate). The combined rows
// are the two images' pooled-and-projected soft tokens stacked.
func TestVisionForward_EncodeGemma4Images_Multi_Good(t *testing.T) {
	model := loadGemma4VisionTestModel(t)

	a := visionTestPixels()
	b := visionTestPixels()
	combined := model.encodeGemma4Images([]*metal.Array{a, b})
	if combined == nil || !combined.Valid() {
		t.Fatal("encodeGemma4Images returned nil for two images")
	}
	if err := metal.Eval(combined); err != nil {
		t.Fatalf("Eval combined: %v", err)
	}
	defer metal.Free(a, b, combined)

	shape := combined.Shape()
	// Each image → 4 projected rows of text hidden 8; two images → 8 rows.
	if len(shape) != 2 || shape[0] != 8 || shape[1] != gemma4VisionTextHid {
		t.Fatalf("combined features shape = %v, want [8 %d]", shape, gemma4VisionTextHid)
	}
}

// TestVisionForward_EncodeGemma4Images_AllInvalid_Bad pins the empty-result
// guard: every input image invalid → no features encoded → nil return.
func TestVisionForward_EncodeGemma4Images_AllInvalid_Bad(t *testing.T) {
	model := loadGemma4VisionTestModel(t)

	if got := model.encodeGemma4Images([]*metal.Array{nil, nil}); got != nil {
		metal.Free(got)
		t.Fatal("encodeGemma4Images returned non-nil for all-invalid input, want nil")
	}
}

// TestVisionForward_ForwardMultiModal_SlotFeatureMismatch_Ugly drives the
// logged-mismatch branch of injectGemma4TokenFeatures: the prompt has 4
// image-placeholder slots but only one image's worth of features (4 pooled rows)
// is supplied for a sequence that also reuses the slot id elsewhere — here we
// over-declare slots (5) against the 4 pooled features so nFeatures != tokenSlots.
// The mismatch only logs (it does not fail the forward), and injection fills the
// features it has, so the call still produces well-shaped logits [1, L, vocab].
func TestVisionForward_ForwardMultiModal_SlotFeatureMismatch_Ugly(t *testing.T) {
	model := loadGemma4VisionTestModel(t)

	// 5 image slots, but one image pools to only 4 feature rows → count mismatch.
	tokenIDs := []int32{2, gemma4VisionImageTok, gemma4VisionImageTok, gemma4VisionImageTok, gemma4VisionImageTok, gemma4VisionImageTok}
	tokens := metal.FromValues(tokenIDs, 1, int(len(tokenIDs)))
	pixels := visionTestPixels()
	caches := model.NewCache()

	logits := model.ForwardMultiModal(tokens, []*metal.Array{pixels}, caches)
	if err := metal.Eval(logits); err != nil {
		t.Fatalf("Eval logits: %v", err)
	}
	defer func() {
		metal.Free(tokens, pixels, logits)
		metal.FreeCaches(caches)
	}()

	shape := logits.Shape()
	if len(shape) != 3 || shape[0] != 1 || shape[1] != int32(len(tokenIDs)) || shape[2] != 10 {
		t.Fatalf("slot/feature-mismatch logits shape = %v, want [1 %d 10] (mismatch logs, does not fail)", shape, len(tokenIDs))
	}
}
