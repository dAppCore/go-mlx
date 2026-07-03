// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/metal"
)

// Residual branch coverage for the audio projector + clippable-linear seams a
// real checkpoint reaches only under specific configs (use_clipped_linears) or
// never reaches at all (the projector-only/empty guards). The package sees every
// unexported field, so the structs are built directly in the exact shapes the
// loader produces for those configs — synthetic scalar clamp bounds, a one-entry
// weight map, a duplicate-canonical raw map.

// synthClipScalar returns a 1-element clamp bound (the scalar shape the
// checkpoint ships for input_min / output_max etc).
func synthClipScalar(v float32) *metal.Array {
	return metal.FromValue(v)
}

// audioExampleWeights builds the synthetic Conformer tower weight map in torch
// layouts (the loader owns the MLX transposes), using seqArray so it needs no
// *testing.T — the runnable Examples in audio_example_test.go (which have no t)
// build the same toy tower the audio_encoder_test.go fixture builds with t. The
// shapes mirror buildGemma4AudioEncoder's reads exactly; values are deterministic.
func audioExampleWeights() map[string]*metal.Array {
	w := map[string]*metal.Array{}
	put := func(name string, seed float32, dims ...int) {
		w["audio_tower."+name] = seqArray(seed, dims...)
	}
	put("subsample_conv_projection.layer0.conv.weight", 0.01, audioTestMelBins, 1, 3, 3)
	put("subsample_conv_projection.layer0.norm.weight", 0.02, audioTestMelBins)
	put("subsample_conv_projection.layer1.conv.weight", 0.03, 4, audioTestMelBins, 3, 3)
	put("subsample_conv_projection.layer1.norm.weight", 0.04, 4)
	put("subsample_conv_projection.input_proj_linear.weight", 0.05, audioTestHidden, (audioTestMelBins/4)*4)
	for i := 0; i < audioTestLayers; i++ {
		base := core.Sprintf("layers.%d.", i)
		seed := 0.10 + float32(i)*0.5
		for _, ff := range []string{"feed_forward1", "feed_forward2"} {
			put(base+ff+".ffw_layer_1.linear.weight", seed+0.01, audioTestFFW, audioTestHidden)
			put(base+ff+".ffw_layer_2.linear.weight", seed+0.02, audioTestHidden, audioTestFFW)
			put(base+ff+".pre_layer_norm.weight", seed+0.03, audioTestHidden)
			put(base+ff+".post_layer_norm.weight", seed+0.04, audioTestHidden)
			seed += 0.1
		}
		put(base+"self_attn.q_proj.linear.weight", seed+0.01, audioTestHidden, audioTestHidden)
		put(base+"self_attn.k_proj.linear.weight", seed+0.02, audioTestHidden, audioTestHidden)
		put(base+"self_attn.v_proj.linear.weight", seed+0.03, audioTestHidden, audioTestHidden)
		put(base+"self_attn.post.linear.weight", seed+0.04, audioTestHidden, audioTestHidden)
		put(base+"self_attn.relative_k_proj.weight", seed+0.05, audioTestHidden, audioTestHidden)
		put(base+"self_attn.per_dim_scale", seed+0.06, audioTestHidden/audioTestHeads)
		put(base+"lconv1d.linear_start.linear.weight", seed+0.07, 2*audioTestHidden, audioTestHidden)
		put(base+"lconv1d.linear_end.linear.weight", seed+0.08, audioTestHidden, audioTestHidden)
		put(base+"lconv1d.depthwise_conv1d.weight", seed+0.09, audioTestHidden, 1, audioTestKernel)
		put(base+"lconv1d.pre_layer_norm.weight", seed+0.11, audioTestHidden)
		put(base+"lconv1d.conv_norm.weight", seed+0.12, audioTestHidden)
		put(base+"norm_pre_attn.weight", seed+0.13, audioTestHidden)
		put(base+"norm_post_attn.weight", seed+0.14, audioTestHidden)
		put(base+"norm_out.weight", seed+0.15, audioTestHidden)
	}
	put("output_proj.weight", 0.9, audioTestProj, audioTestHidden)
	put("output_proj.bias", 0.91, audioTestProj)
	return w
}

// TestAudio_ClippableLinear_Forward_InputAndOutputClip_Good drives both clamp
// arms of Gemma4AudioClippableLinear.Forward: a linear wrapped with input and
// output bounds clamps its input before the matmul and its output after. The
// uncovered arm is exactly use_clipped_linears=true, which the synthetic tower
// (no recorded bounds) never sets.
//
// The output-clamp check alone cannot fail even if the input-clamp arm were
// silently dropped (out.Forward's OutputMin/Max clip runs unconditionally
// last, so every result lands in [-1,1] regardless), so this also builds two
// independent references through the SAME linear weight — one honouring the
// input clamp, one not — and asserts Forward's real output matches the
// clamped reference and genuinely differs from the unclamped one.
func TestAudio_ClippableLinear_Forward_InputAndOutputClip_Good(t *testing.T) {
	requireMetalRuntime(t)

	const dim = 4
	lin := synthVisionLinear("clip.linear", dim, dim)
	clip := &Gemma4AudioClippableLinear{
		Linear:    lin,
		InputMin:  synthClipScalar(-0.5),
		InputMax:  synthClipScalar(0.5),
		OutputMin: synthClipScalar(-1.0),
		OutputMax: synthClipScalar(1.0),
	}
	defer func() {
		metal.FreeLinear(lin)
		metal.Free(clip.InputMin, clip.InputMax, clip.OutputMin, clip.OutputMax)
	}()

	// Inputs far outside [-0.5, 0.5] so the input clamp is observable: every
	// value saturates to the bound before the matmul.
	x := seqArray(5.0, 1, dim)
	defer metal.Free(x)
	out := clip.Forward(x)
	defer metal.Free(out)
	if err := metal.Eval(out); err != nil {
		t.Fatalf("Eval out: %v", err)
	}

	// Output clamp pins every result into [-1, 1].
	for i, v := range out.Floats() {
		if v < -1.0001 || v > 1.0001 {
			t.Fatalf("clipped-linear output[%d] = %v, want within [-1, 1] (output clamp arm not applied)", i, v)
		}
	}

	// Reference A: input clamped then matmul then output clamped — what
	// Forward should compute.
	preclipped := metal.Clip(x, clip.InputMin, clip.InputMax)
	withClampOut := lin.Forward(preclipped)
	withClampClipped := metal.Clip(withClampOut, clip.OutputMin, clip.OutputMax)
	defer metal.Free(preclipped, withClampOut, withClampClipped)

	// Reference B: matmul on the RAW (unclamped) input, then output clamped —
	// what a regression that dropped the input-clamp arm would compute.
	noClampOut := lin.Forward(x)
	noClampClipped := metal.Clip(noClampOut, clip.OutputMin, clip.OutputMax)
	defer metal.Free(noClampOut, noClampClipped)

	if err := metal.Eval(withClampClipped, noClampClipped); err != nil {
		t.Fatalf("Eval reference clamps: %v", err)
	}
	floatSliceApprox(t, out.Floats(), withClampClipped.Floats())

	gotVals, noClampVals := out.Floats(), noClampClipped.Floats()
	differs := false
	for i := range gotVals {
		diff := gotVals[i] - noClampVals[i]
		if diff < 0 {
			diff = -diff
		}
		if diff >= floatApproxTol {
			differs = true
			break
		}
	}
	if !differs {
		t.Fatal("clippable-linear output matches the no-input-clamp reference — the input clamp arm made no observable difference")
	}
}

// TestAudio_ClippableLinear_Forward_NoBounds_Ugly pins the no-clamp arm: a linear
// with nil bounds is a plain matmul (the 12B-unified projector path, no recorded
// clamps), so the input and output passthrough untouched.
func TestAudio_ClippableLinear_Forward_NoBounds_Ugly(t *testing.T) {
	requireMetalRuntime(t)

	const dim = 4
	lin := synthVisionLinear("noclip.linear", dim, dim)
	clip := &Gemma4AudioClippableLinear{Linear: lin}
	defer metal.FreeLinear(lin)

	x := seqArray(0.1, 1, dim)
	defer metal.Free(x)
	out := clip.Forward(x)
	defer metal.Free(out)
	if err := metal.Eval(out); err != nil {
		t.Fatalf("Eval out: %v", err)
	}
	if out == x {
		t.Fatal("no-bounds clippable linear returned its input, want the matmul result")
	}
	if shape := out.Shape(); len(shape) != 2 || shape[1] != dim {
		t.Fatalf("no-bounds clippable linear shape = %v, want [1 %d]", shape, dim)
	}
}

// TestAudio_buildGemma4AudioProjector_EmptyWeights_Bad pins the empty-map guard:
// no embed_audio weights at all → nil projector (the model carries no audio
// modality), no Linear constructor plumbing attempted.
func TestAudio_buildGemma4AudioProjector_EmptyWeights_Bad(t *testing.T) {
	if proj := buildGemma4AudioProjector(nil, map[string]*metal.Array{}); proj != nil {
		closeGemma4AudioProjector(proj)
		t.Fatal("buildGemma4AudioProjector(empty) returned non-nil, want nil for the no-audio-weights guard")
	}
}

// TestAudio_buildGemma4AudioProjector_NoProjection_Bad pins the missing-projection
// guard: a non-empty audio weight map that lacks the embedding_projection tensor
// builds no usable projector (the gemma4Linear lookup misses), so the builder
// returns nil rather than a projector with a dangling Projection.
func TestAudio_buildGemma4AudioProjector_NoProjection_Bad(t *testing.T) {
	requireMetalRuntime(t)

	weights := map[string]*metal.Array{
		// Present but not the embedding_projection the projector reads.
		"embed_audio.some_other_tensor.weight": seqArray(0.1, 4, 4),
	}
	defer freeWeightMap(weights)

	if proj := buildGemma4AudioProjector(nil, weights); proj != nil {
		closeGemma4AudioProjector(proj)
		t.Fatal("buildGemma4AudioProjector(no projection weight) returned non-nil, want nil")
	}
}

// TestAudio_sanitizeGemma4AudioWeights_DuplicateCanonical_Ugly drives the
// duplicate-canonical-name arm: two raw keys that strip to the same canonical
// audio name (a wrapper-prefixed twin and the bare name) collide, so one array
// is freed and the other wins — the sanitizer must not leak the loser, and
// must not retain both or neither under the shared canonical key.
//
// Go map iteration order is randomised per run, and sanitizeGemma4AudioWeights
// dedups by iterating `raw`, so which of the two ORIGINAL arrays ends up the
// survivor is not deterministic across runs (that would make "the later one"
// an unenforceable, flaky claim). This asserts the invariant that IS
// guaranteed regardless of iteration order: exactly one of the two original
// handles survives valid under the canonical key, and the other was freed.
func TestAudio_sanitizeGemma4AudioWeights_DuplicateCanonical_Ugly(t *testing.T) {
	requireMetalRuntime(t)

	arrA := seqArray(0.1, 2, 2) // "audio_tower.output_proj.weight"
	arrB := seqArray(0.2, 2, 2) // "model.audio_tower.output_proj.weight" (same canonical name)
	raw := map[string]*metal.Array{
		"audio_tower.output_proj.weight":             arrA,
		"model.audio_tower.output_proj.weight":       arrB,
		"language_model.embed_audio.projection.bias": seqArray(0.3, 2),
		"unrelated.text_tower.weight":                seqArray(0.4, 2, 2), // dropped: not audio
	}

	audio := sanitizeGemma4AudioWeights(raw)
	defer freeWeightMap(audio)
	// The non-audio key is left in raw and must be freed by the caller.
	defer freeWeightMap(raw)

	// Both audio_tower.output_proj.weight spellings collapse to one canonical
	// entry; the embed_audio key canonicalises to its own.
	winner, ok := audio["audio_tower.output_proj.weight"]
	if !ok {
		t.Fatal("canonical audio_tower.output_proj.weight missing after sanitize")
	}
	if _, ok := audio["embed_audio.projection.bias"]; !ok {
		t.Fatal("canonical embed_audio.projection.bias missing after sanitize")
	}
	if _, ok := raw["unrelated.text_tower.weight"]; !ok {
		t.Fatal("non-audio weight should remain in the raw map, not be claimed by the audio sanitizer")
	}

	aWins := winner == arrA
	bWins := winner == arrB
	if aWins == bWins {
		t.Fatalf("winner is neither/both of the two original arrays (aWins=%v bWins=%v) — dedup lost or duplicated the handle", aWins, bWins)
	}
	if aWins && arrB.Valid() {
		t.Fatal("duplicate-canonical collision: A survived but B (the loser) was not freed")
	}
	if bWins && arrA.Valid() {
		t.Fatal("duplicate-canonical collision: B survived but A (the loser) was not freed")
	}
	if !winner.Valid() {
		t.Fatal("surviving audio_tower.output_proj.weight handle is not valid")
	}
}
