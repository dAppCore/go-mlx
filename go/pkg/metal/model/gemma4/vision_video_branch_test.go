// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	"testing"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
	"dappco.re/go/mlx/pkg/metal"
)

// Video lane of ForwardUnifiedVideoMultiModal. The video branch (vision_forward.go
// lines 71-79) reuses the vision tower to encode raw frames, then injects them at
// VideoTokenID placeholders. It only fires when the prompt carries video tokens AND
// a tower is present, so it needs a fully-loaded model with a distinct
// video_token_id — a separate config from the shared gemma4VisionConfigJSON so
// pass-2's vision tests (and the bench set) stay byte-identical.

// gemma4VideoConfigJSON is the vision multimodal config with a video_token_id (6)
// distinct from the image_token_id (5); both stay below the 10-row embed table so
// the full-sequence gather before injection stays in bounds. Same tiny vision tower
// as gemma4VisionConfigJSON — frames encode through it exactly like images.
const gemma4VideoConfigJSON = `{
	"model_type": "gemma4",
	"image_token_id": 5,
	"video_token_id": 6,
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

const gemma4VisionVideoTok = 6

// loadGemma4VideoTestModel writes the video-capable multimodal checkpoint (the
// gemma4VideoConfigJSON config + tokenizer + the same gemma4VisionTinyWeights tower)
// and loads it. The loaded model carries VideoTokenID=6 plus a real vision tower, so
// ForwardUnifiedVideoMultiModal's video branch (encode frames through the tower →
// inject at video slots) runs on synthetic weights.
func loadGemma4VideoTestModel(t *testing.T) *Gemma4Model {
	t.Helper()
	requireMetalRuntime(t)

	dir := t.TempDir()
	if err := coreio.Local.Write(core.JoinPath(dir, "config.json"), gemma4VideoConfigJSON); err != nil {
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

// TestVisionVideoBranch_VideoTokenWired_Good asserts the checkpoint set VideoTokenID
// and built a tower — the precondition the forward test depends on.
func TestVisionVideoBranch_VideoTokenWired_Good(t *testing.T) {
	model := loadGemma4VideoTestModel(t)

	if model.Cfg.VideoTokenID != gemma4VisionVideoTok {
		t.Fatalf("VideoTokenID = %d, want %d", model.Cfg.VideoTokenID, gemma4VisionVideoTok)
	}
	if model.VisionTower == nil {
		t.Fatal("VisionTower = nil, want a built tower for the video lane")
	}
}

// TestVisionVideoBranch_ForwardVideo_Good drives the video branch end to end:
// a prompt with 4 video-placeholder slots plus one raw frame, through
// ForwardUnifiedVideoMultiModal. The frame encodes through the vision tower
// (4 pooled+projected soft tokens) and injects at the video slots, then the masked
// text forward produces full logits [1, L, vocab].
func TestVisionVideoBranch_ForwardVideo_Good(t *testing.T) {
	model := loadGemma4VideoTestModel(t)

	// [bos, vid, vid, vid, vid, world] — 4 video slots = 4 pooled frame features.
	tokenIDs := []int32{2, gemma4VisionVideoTok, gemma4VisionVideoTok, gemma4VisionVideoTok, gemma4VisionVideoTok, 4}
	tokens := metal.FromValues(tokenIDs, 1, int(len(tokenIDs)))
	frame := visionTestPixels()
	caches := model.NewCache()

	logits := model.ForwardUnifiedVideoMultiModal(tokens, nil, nil, []*metal.Array{frame}, caches)
	if err := metal.Eval(logits); err != nil {
		t.Fatalf("Eval logits: %v", err)
	}
	defer func() {
		metal.Free(tokens, frame, logits)
		metal.FreeCaches(caches)
	}()

	shape := logits.Shape()
	if len(shape) != 3 || shape[0] != 1 || shape[1] != int32(len(tokenIDs)) || shape[2] != 10 {
		t.Fatalf("video-lane logits shape = %v, want [1 %d 10]", shape, len(tokenIDs))
	}
}

// TestVisionVideoBranch_ForwardVideo_InvalidFrame_Bad drives the video encode-failure
// fallback: when every supplied frame is invalid, encodeGemma4Images returns nil, so
// the video branch frees the working hidden state and falls back to a plain text
// Forward. Logits stay well-shaped over the full sequence.
func TestVisionVideoBranch_ForwardVideo_InvalidFrame_Bad(t *testing.T) {
	model := loadGemma4VideoTestModel(t)

	tokenIDs := []int32{2, gemma4VisionVideoTok, gemma4VisionVideoTok, gemma4VisionVideoTok, gemma4VisionVideoTok, 4}
	tokens := metal.FromValues(tokenIDs, 1, int(len(tokenIDs)))
	caches := model.NewCache()

	// A single nil frame: video token count > 0 (so the branch is entered) but the
	// encode yields nil → the fallback Forward runs.
	logits := model.ForwardUnifiedVideoMultiModal(tokens, nil, nil, []*metal.Array{nil}, caches)
	if err := metal.Eval(logits); err != nil {
		t.Fatalf("Eval logits: %v", err)
	}
	defer func() {
		metal.Free(tokens, logits)
		metal.FreeCaches(caches)
	}()

	shape := logits.Shape()
	if len(shape) != 3 || shape[0] != 1 || shape[1] != int32(len(tokenIDs)) || shape[2] != 10 {
		t.Fatalf("invalid-frame fallback logits shape = %v, want [1 %d 10]", shape, len(tokenIDs))
	}
}
