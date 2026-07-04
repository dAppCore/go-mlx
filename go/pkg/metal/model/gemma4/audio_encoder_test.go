// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	"math"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/metal"
)

// Synthetic Conformer geometry: small enough for instant tests, structured
// enough to exercise every block (2 layers, 2 heads, chunked attention with
// a real past horizon, mel bins == conv channel[0] per the reference's
// input_proj coupling).
const (
	audioTestHidden  = 16
	audioTestHeads   = 2
	audioTestFFW     = 64
	audioTestLayers  = 2
	audioTestChunk   = 4
	audioTestLeft    = 5
	audioTestKernel  = 3
	audioTestMelBins = 8
	audioTestProj    = 24
)

func audioTestConfig() *Gemma4AudioConfig {
	return normalizeGemma4AudioConfig(&Gemma4AudioConfig{
		HiddenSize:              audioTestHidden,
		NumHiddenLayers:         audioTestLayers,
		NumAttentionHeads:       audioTestHeads,
		AttentionChunkSize:      audioTestChunk,
		AttentionContextLeft:    audioTestLeft,
		AttentionContextRight:   0,
		AttentionLogitCap:       50,
		ConvKernelSize:          audioTestKernel,
		SubsamplingConvChannels: []int32{audioTestMelBins, 4},
		ResidualWeight:          0.5,
		HiddenAct:               "silu",
		OutputProjDims:          audioTestProj,
	})
}

// audioTestArray fills a tensor with small decorrelated values — sin-hashed
// so attention has structure (an all-equal fill degenerates every probe).
func audioTestArray(t *testing.T, seed float64, dims ...int) *metal.Array {
	t.Helper()
	n := 1
	for _, d := range dims {
		n *= d
	}
	vals := make([]float32, n)
	for i := range vals {
		vals[i] = float32(0.08 * math.Sin(seed+float64(i)*0.7113))
	}
	arr := metal.FromValues(vals, dims...)
	if err := metal.Eval(arr); err != nil {
		t.Fatalf("audioTestArray eval: %v", err)
	}
	return arr
}

// audioTestWeights builds the complete synthetic tower in torch layouts
// (the loader owns the MLX transposes).
func audioTestWeights(t *testing.T) map[string]*metal.Array {
	t.Helper()
	w := map[string]*metal.Array{}
	put := func(name string, seed float64, dims ...int) {
		w["audio_tower."+name] = audioTestArray(t, seed, dims...)
	}
	put("subsample_conv_projection.layer0.conv.weight", 1, audioTestMelBins, 1, 3, 3)
	put("subsample_conv_projection.layer0.norm.weight", 2, audioTestMelBins)
	put("subsample_conv_projection.layer1.conv.weight", 3, 4, audioTestMelBins, 3, 3)
	put("subsample_conv_projection.layer1.norm.weight", 4, 4)
	put("subsample_conv_projection.input_proj_linear.weight", 5, audioTestHidden, (audioTestMelBins/4)*4)
	for i := range audioTestLayers {
		base := core.Sprintf("layers.%d.", i)
		seed := float64(10 + i*100)
		for _, ff := range []string{"feed_forward1", "feed_forward2"} {
			put(base+ff+".ffw_layer_1.linear.weight", seed+1, audioTestFFW, audioTestHidden)
			put(base+ff+".ffw_layer_2.linear.weight", seed+2, audioTestHidden, audioTestFFW)
			put(base+ff+".pre_layer_norm.weight", seed+3, audioTestHidden)
			put(base+ff+".post_layer_norm.weight", seed+4, audioTestHidden)
			seed += 10
		}
		put(base+"self_attn.q_proj.linear.weight", seed+1, audioTestHidden, audioTestHidden)
		put(base+"self_attn.k_proj.linear.weight", seed+2, audioTestHidden, audioTestHidden)
		put(base+"self_attn.v_proj.linear.weight", seed+3, audioTestHidden, audioTestHidden)
		put(base+"self_attn.post.linear.weight", seed+4, audioTestHidden, audioTestHidden)
		put(base+"self_attn.relative_k_proj.weight", seed+5, audioTestHidden, audioTestHidden)
		put(base+"self_attn.per_dim_scale", seed+6, audioTestHidden/audioTestHeads)
		put(base+"lconv1d.linear_start.linear.weight", seed+7, 2*audioTestHidden, audioTestHidden)
		put(base+"lconv1d.linear_end.linear.weight", seed+8, audioTestHidden, audioTestHidden)
		put(base+"lconv1d.depthwise_conv1d.weight", seed+9, audioTestHidden, 1, audioTestKernel)
		put(base+"lconv1d.pre_layer_norm.weight", seed+10, audioTestHidden)
		put(base+"lconv1d.conv_norm.weight", seed+11, audioTestHidden)
		put(base+"norm_pre_attn.weight", seed+12, audioTestHidden)
		put(base+"norm_post_attn.weight", seed+13, audioTestHidden)
		put(base+"norm_out.weight", seed+14, audioTestHidden)
	}
	put("output_proj.weight", 90, audioTestProj, audioTestHidden)
	put("output_proj.bias", 91, audioTestProj)
	return w
}

func audioTestTextConfig() *Gemma4TextConfig {
	return &Gemma4TextConfig{AudioConfig: audioTestConfig()}
}

func buildAudioTestEncoder(t *testing.T) *Gemma4AudioEncoder {
	t.Helper()
	enc, err := buildGemma4AudioEncoder(audioTestTextConfig(), audioTestWeights(t))
	if err != nil {
		t.Fatalf("buildGemma4AudioEncoder: %v", err)
	}
	if enc == nil {
		t.Fatal("encoder = nil, want built Conformer")
	}
	return enc
}

func audioEncodeFloats(t *testing.T, enc *Gemma4AudioEncoder, features *metal.Array) []float32 {
	t.Helper()
	out := enc.Forward(features)
	defer metal.Free(out)
	if err := metal.Eval(out); err != nil {
		t.Fatalf("encoder forward eval: %v", err)
	}
	return out.Floats()
}

func TestGemma4_AudioEncoder_BuildAndShape_Good(t *testing.T) {
	requireMetalRuntime(t)
	enc := buildAudioTestEncoder(t)
	defer closeGemma4AudioEncoder(enc)

	if len(enc.Layers) != audioTestLayers || enc.Subsample == nil || enc.OutputProj == nil {
		t.Fatalf("encoder incomplete: layers=%d subsample=%v proj=%v", len(enc.Layers), enc.Subsample != nil, enc.OutputProj != nil)
	}

	// 19 mel frames: two stride-2 convs (pad 1) give ceil-chains 19→10→5.
	features := audioTestArray(t, 42, 1, 19, audioTestMelBins)
	defer metal.Free(features)
	out := enc.Forward(features)
	defer metal.Free(out)
	if err := metal.Eval(out); err != nil {
		t.Fatalf("forward eval: %v", err)
	}
	shape := out.Shape()
	if len(shape) != 3 || shape[0] != 1 || shape[1] != 5 || shape[2] != audioTestProj {
		t.Fatalf("encoder output shape = %v, want [1 5 %d]", shape, audioTestProj)
	}

	// The retained-weight walk must keep the load-derived constants alive.
	model := &Gemma4Model{AudioEncoder: enc}
	retained := gemma4RetainedWeights(model)
	if !arraySetContains(retained, enc.Subsample.Layer0.ConvWeight) ||
		!arraySetContains(retained, enc.PosEmbed) ||
		!arraySetContains(retained, enc.Layers[0].SelfAttn.QScalePerDim) {
		t.Fatal("derived audio encoder arrays missing from the retained-weight walk")
	}
}

func TestGemma4_AudioEncoder_NoTower_Good(t *testing.T) {
	enc, err := buildGemma4AudioEncoder(audioTestTextConfig(), map[string]*metal.Array{
		"embed_audio.embedding_projection.weight": nil,
	})
	if err != nil || enc != nil {
		t.Fatalf("projector-only weights built %v err=%v, want nil encoder no error", enc, err)
	}
}

func TestGemma4_AudioEncoder_MissingLayerWeight_Bad(t *testing.T) {
	requireMetalRuntime(t)
	weights := audioTestWeights(t)
	metal.Free(weights["audio_tower.layers.1.self_attn.per_dim_scale"])
	delete(weights, "audio_tower.layers.1.self_attn.per_dim_scale")
	enc, err := buildGemma4AudioEncoder(audioTestTextConfig(), weights)
	if err == nil || enc != nil {
		closeGemma4AudioEncoder(enc)
		t.Fatal("expected loud failure on incomplete tower weights")
	}
	for _, arr := range weights {
		metal.Free(arr)
	}
}

func TestGemma4_AudioEncoder_ConfigIncomplete_Bad(t *testing.T) {
	requireMetalRuntime(t)
	weights := audioTestWeights(t)
	cfg := &Gemma4TextConfig{AudioConfig: &Gemma4AudioConfig{HiddenSize: audioTestHidden}}
	enc, err := buildGemma4AudioEncoder(cfg, weights)
	if err == nil || enc != nil {
		closeGemma4AudioEncoder(enc)
		t.Fatal("expected loud failure on dimensionless audio_config")
	}
	for _, arr := range weights {
		metal.Free(arr)
	}
}

func TestGemma4_AudioEncoder_Deterministic_Good(t *testing.T) {
	requireMetalRuntime(t)
	enc := buildAudioTestEncoder(t)
	defer closeGemma4AudioEncoder(enc)

	features := audioTestArray(t, 7, 1, 24, audioTestMelBins)
	defer metal.Free(features)
	first := audioEncodeFloats(t, enc, features)
	second := audioEncodeFloats(t, enc, features)
	for i := range first {
		if first[i] != second[i] {
			t.Fatalf("encoder non-deterministic at %d: %v vs %v", i, first[i], second[i])
		}
	}
}

// The chunked attention runs with context_right=0, the depthwise conv is
// causal, and the subsampler's receptive cone for output frame j ends at
// input frame 4j+3. Changing only input frames ≥ 24 must therefore leave
// output frames 0..5 byte-identical — any drift means the blocked mask,
// the relative shift or the causal padding is misaligned (exactly the
// silent-garbage failure mode the reference port guards against).
func TestGemma4_AudioEncoder_NoFutureLeak_Good(t *testing.T) {
	requireMetalRuntime(t)
	enc := buildAudioTestEncoder(t)
	defer closeGemma4AudioEncoder(enc)

	const frames = 40
	const changeFrom = 24
	base := make([]float32, frames*audioTestMelBins)
	for i := range base {
		base[i] = float32(0.1 * math.Sin(float64(i)*0.3717))
	}
	altered := append([]float32(nil), base...)
	for i := changeFrom * audioTestMelBins; i < len(altered); i++ {
		altered[i] += 0.5
	}

	baseArr := metal.FromValues(base, 1, frames, audioTestMelBins)
	alteredArr := metal.FromValues(altered, 1, frames, audioTestMelBins)
	defer metal.Free(baseArr, alteredArr)

	baseOut := audioEncodeFloats(t, enc, baseArr)
	alteredOut := audioEncodeFloats(t, enc, alteredArr)
	if len(baseOut) != len(alteredOut) {
		t.Fatalf("output lengths diverge: %d vs %d", len(baseOut), len(alteredOut))
	}

	const safeFrames = 6 // 4j+3 < 24 ⇒ j ≤ 5
	for i := 0; i < safeFrames*audioTestProj; i++ {
		if baseOut[i] != alteredOut[i] {
			t.Fatalf("future leak: output frame %d dim %d changed (%v vs %v) when only inputs ≥ frame %d changed",
				i/audioTestProj, i%audioTestProj, baseOut[i], alteredOut[i], changeFrom)
		}
	}
	tailChanged := false
	for i := safeFrames * audioTestProj; i < len(baseOut); i++ {
		if baseOut[i] != alteredOut[i] {
			tailChanged = true
			break
		}
	}
	if !tailChanged {
		t.Fatal("altered tail produced identical outputs — the change never propagated, probe is dead")
	}
}
