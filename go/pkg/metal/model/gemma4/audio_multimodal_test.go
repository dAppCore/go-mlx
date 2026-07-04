// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	"testing"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
	"dappco.re/go/mlx/pkg/metal"
)

// End-to-end coverage for the audio branch of ForwardUnifiedVideoMultiModal — the
// integration path (audio tokens in the stream -> encodeGemma4Audio -> inject ->
// masked text forward) that the isolated encodeGemma4Audio tests in
// vision_audio_branch_test.go never drive because they carry no text trunk. This
// is the audio sibling of loadGemma4VisionTestModel: a synthetic checkpoint with
// the tiny dense text trunk plus a real Conformer audio tower + projector, loaded
// through the production LoadGemma4 path so the whole audio cascade runs on
// synthetic weights — no real multi-GB checkpoint.

// gemma4AudioMMConfigJSON is a synthetic gemma4 unified-audio config: the tiny
// dense text trunk (matching gemma4TinyWeights, hidden 8, vocab 10) plus the
// audio_config that audioTestWeights/audioExampleWeights build a tower for, and
// an audio_token_id (6) below the 10-row embed table so the full-sequence embed
// gather that precedes injection stays in bounds. model_type "gemma4" keeps the
// plain text trunk; the audio tower is built purely from the audio_tower.* and
// embed_audio.* weights being present in the checkpoint.
const gemma4AudioMMConfigJSON = `{
	"model_type": "gemma4",
	"audio_token_id": 6,
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
	"audio_config": {
		"model_type": "gemma4_unified_audio",
		"hidden_size": 16,
		"num_hidden_layers": 2,
		"num_attention_heads": 2,
		"attention_chunk_size": 4,
		"attention_context_left": 5,
		"attention_context_right": 0,
		"attention_logit_cap": 50,
		"conv_kernel_size": 3,
		"subsampling_conv_channels": [8, 4],
		"residual_weight": 0.5,
		"hidden_act": "silu",
		"output_proj_dims": 24
	}
}`

// gemma4AudioMMWeights extends the tiny dense text trunk with the synthetic
// Conformer tower (audio_tower.*) and the audio projector (embed_audio.*). The
// projector maps the tower's 24-wide output rows (output_proj_dims) down to the
// text hidden size 8, so projected audio features splice into the embed stream.
func gemma4AudioMMWeights() map[string]*metal.Array {
	w := gemma4TinyWeights()
	for name, arr := range audioExampleWeights() {
		w[name] = arr
	}
	// Audio projector: tower output rows [*, 24] -> text hidden [*, 8].
	w["embed_audio.embedding_projection.weight"] = seqArray(0.5, 8, audioTestProj)
	return w
}

// loadGemma4AudioMMTestModel writes the synthetic unified-audio checkpoint
// (config + tokenizer + trunk + audio tower + projector) and loads it through the
// production LoadGemma4 path, registering cleanup. The loaded model carries a real
// AudioEncoder + AudioProjector built by buildGemma4AudioEncoder /
// buildGemma4AudioProjector on top of the dense text trunk.
func loadGemma4AudioMMTestModel(t *testing.T) *Gemma4Model {
	t.Helper()
	requireMetalRuntime(t)

	dir := t.TempDir()
	if err := coreio.Local.Write(core.JoinPath(dir, "config.json"), gemma4AudioMMConfigJSON); err != nil {
		t.Fatalf("write config.json: %v", err)
	}
	writeMinimalTokenizer(t, dir)
	if err := metal.SaveSafetensors(core.JoinPath(dir, "model.safetensors"), gemma4AudioMMWeights()); err != nil {
		t.Fatalf("SaveSafetensors: %v", err)
	}

	model, err := LoadGemma4(dir)
	if err != nil {
		t.Fatalf("LoadGemma4: %v", err)
	}
	t.Cleanup(func() { closeGemma4(model) })
	return model
}

// TestAudioMultiModal_ForwardUnifiedMultiModal_Good runs the audio-bearing prefill
// end to end: a token sequence with 6 audio-placeholder slots (matching the 6 soft
// tokens 24 mel frames pool to) plus the raw mel clip, through the audio branch of
// ForwardUnifiedVideoMultiModal — encode (tower + projector) -> inject at the
// AudioTokenID positions -> masked text forward. This is the integration path the
// isolated encodeGemma4Audio tests cannot reach (no trunk). Asserts full logits
// [1, L, vocab]: the slot/feature counts match, so it is a true happy path, not a
// logged-mismatch fallback.
func TestAudioMultiModal_ForwardUnifiedMultiModal_Good(t *testing.T) {
	model := loadGemma4AudioMMTestModel(t)
	if model.AudioEncoder == nil || model.AudioProjector == nil {
		t.Fatalf("loaded model missing audio components: encoder=%v projector=%v", model.AudioEncoder != nil, model.AudioProjector != nil)
	}

	// [bos, aud x6, world] — 6 audio slots = 6 soft tokens from 24 mel frames.
	const aud = int32(6)
	tokenIDs := []int32{2, aud, aud, aud, aud, aud, aud, 4}
	tokens := metal.FromValues(tokenIDs, 1, int(len(tokenIDs)))
	mel := metal.Zeros([]int32{24, audioTestMelBins}, metal.DTypeFloat32)
	caches := model.NewCache()

	logits := model.ForwardUnifiedMultiModal(tokens, nil, []*metal.Array{mel}, caches)
	if err := metal.Eval(logits); err != nil {
		t.Fatalf("Eval logits: %v", err)
	}
	defer func() {
		metal.Free(tokens, mel, logits)
		metal.FreeCaches(caches)
	}()

	shape := logits.Shape()
	if len(shape) != 3 || shape[0] != 1 || shape[1] != int32(len(tokenIDs)) || shape[2] != 10 {
		t.Fatalf("audio multimodal logits shape = %v, want [1 %d 10]", shape, len(tokenIDs))
	}
}

// TestAudioMultiModal_ForwardMultiModal_NoAudioTokens_Good pins the fast path: when
// the prompt carries no audio-placeholder ids, ForwardUnifiedVideoMultiModal counts
// zero modal tokens and falls back to a plain text Forward even though audio
// features were supplied. Logits stay well-shaped over the text-only sequence.
func TestAudioMultiModal_ForwardMultiModal_NoAudioTokens_Good(t *testing.T) {
	model := loadGemma4AudioMMTestModel(t)

	tokens := metal.FromValues([]int32{2, 3, 4}, 1, 3) // no audio token id 6
	mel := metal.Zeros([]int32{24, audioTestMelBins}, metal.DTypeFloat32)
	caches := model.NewCache()

	logits := model.ForwardUnifiedMultiModal(tokens, nil, []*metal.Array{mel}, caches)
	if err := metal.Eval(logits); err != nil {
		t.Fatalf("Eval logits: %v", err)
	}
	defer func() {
		metal.Free(tokens, mel, logits)
		metal.FreeCaches(caches)
	}()

	shape := logits.Shape()
	if len(shape) != 3 || shape[0] != 1 || shape[1] != 3 || shape[2] != 10 {
		t.Fatalf("no-audio-token logits shape = %v, want [1 3 10]", shape)
	}
}

// TestAudioMultiModal_ForwardMultiModal_NoFeatures_Good pins the other fast path:
// audio tokens are present but no audio features are supplied, so hasAudio is false
// and the forward falls back to plain text. The model must not dereference a nil
// feature list.
func TestAudioMultiModal_ForwardMultiModal_NoFeatures_Good(t *testing.T) {
	model := loadGemma4AudioMMTestModel(t)

	const aud = int32(6)
	tokenIDs := []int32{2, aud, aud, 4}
	tokens := metal.FromValues(tokenIDs, 1, int(len(tokenIDs)))
	caches := model.NewCache()

	logits := model.ForwardUnifiedMultiModal(tokens, nil, nil, caches) // no features
	if err := metal.Eval(logits); err != nil {
		t.Fatalf("Eval logits: %v", err)
	}
	defer func() {
		metal.Free(tokens, logits)
		metal.FreeCaches(caches)
	}()

	shape := logits.Shape()
	if len(shape) != 3 || shape[0] != 1 || shape[1] != int32(len(tokenIDs)) || shape[2] != 10 {
		t.Fatalf("no-feature logits shape = %v, want [1 %d 10]", shape, len(tokenIDs))
	}
}
