// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package gemma4

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/metal"
)

// Coverage for buildGemma4CompiledLayerState's eligibility gate — the decline
// ladder and the linearWhy diagnostic switch. A real quantised layer (loaded
// via cov4clBuildQuantModel) is the eligible baseline; each case shallow-copies
// the decoder layer / attention and replaces exactly one field with an
// ineligible value, then asserts the builder declines with the expected reason.
// Nothing is freed — the loaded model owns every tensor and frees them via its
// own t.Cleanup; the copies only rebind pointers. Helpers prefixed cov4cls.

// cov4clsOwnerLayer loads a tiny quantised model and returns its first
// (KV-owner) decoder layer plus the text config. The layer is fully eligible:
// buildGemma4CompiledLayerState(layer, false, cfg) does not decline.
func cov4clsOwnerLayer(t *testing.T) (*Gemma4DecoderLayer, *Gemma4TextConfig) {
	t.Helper()
	m := cov4clBuildQuantModel(t, cov4clQuantConfig{
		hidden:       32,
		intermediate: 64,
		headDim:      32,
		layers:       1,
		sliding:      64,
		pattern:      1,
	})
	if len(m.Layers) == 0 {
		t.Fatal("loaded model has no layers")
	}
	return m.Layers[0], m.Cfg
}

// cov4clsCopyLayer returns a fresh decoder layer that reproduces exactly the
// fields buildGemma4CompiledLayerState reads, with a freshly-copied Attention,
// so a test can null a field on the copy without disturbing the loaded
// original. It is rebuilt field-by-field (not `*l`) because Gemma4DecoderLayer
// embeds an atomic.Pointer whose noCopy marker forbids struct assignment — and
// the builder never touches that field anyway.
func cov4clsCopyLayer(l *Gemma4DecoderLayer) *Gemma4DecoderLayer {
	cp := &Gemma4DecoderLayer{
		MLP:                         l.MLP,
		EnableMoE:                   l.EnableMoE,
		Router:                      l.Router,
		Experts:                     l.Experts,
		PreFFNorm2Scaled:            l.PreFFNorm2Scaled,
		PostFFNorm1Scaled:           l.PostFFNorm1Scaled,
		PostFFNorm2Scaled:           l.PostFFNorm2Scaled,
		PerLayerInputGate:           l.PerLayerInputGate,
		PerLayerProjection:          l.PerLayerProjection,
		PostPerLayerInputNormScaled: l.PostPerLayerInputNormScaled,
		LayerScalar:                 l.LayerScalar,
		InputNormScaled:             l.InputNormScaled,
		PostAttnNormScaled:          l.PostAttnNormScaled,
		PreFFNormScaled:             l.PreFFNormScaled,
		PostFFNormScaled:            l.PostFFNormScaled,
		LayerType:                   l.LayerType,
		IsSliding:                   l.IsSliding,
		DoubleWideMLP:               l.DoubleWideMLP,
		LayerIdx:                    l.LayerIdx,
	}
	if l.Attention != nil {
		att := *l.Attention
		cp.Attention = &att
	}
	return cp
}

// TestCompiledLayerState_Eligible confirms the loaded quant layer is accepted —
// the baseline the decline tests mutate away from.
func TestCompiledLayerState_Eligible(t *testing.T) {
	requireMetalRuntime(t)

	layer, cfg := cov4clsOwnerLayer(t)
	state := buildGemma4CompiledLayerState(layer, false, cfg)
	if state.declined {
		t.Fatalf("baseline owner layer declined: %q", state.reason)
	}
}

// TestCompiledLayerState_DeclineBranches nulls one structural prerequisite at a
// time and asserts the builder declines with the matching reason. Each case
// verifies the eligibility gate rejects that specific malformed layer.
func TestCompiledLayerState_DeclineBranches(t *testing.T) {
	requireMetalRuntime(t)

	layer, cfg := cov4clsOwnerLayer(t)

	cases := []struct {
		name   string
		mutate func(*Gemma4DecoderLayer)
		want   string
	}{
		{"nil attention", func(l *Gemma4DecoderLayer) { l.Attention = nil }, "attention shape invalid"},
		{"zero head dim", func(l *Gemma4DecoderLayer) { l.Attention.HeadDim = 0 }, "attention shape invalid"},
		{"zero kv heads", func(l *Gemma4DecoderLayer) { l.Attention.NKVHeads = 0 }, "attention shape invalid"},
		{"input norm scale", func(l *Gemma4DecoderLayer) { l.InputNormScaled = nil }, "input norm scale missing"},
		{"post-attn norm scale", func(l *Gemma4DecoderLayer) { l.PostAttnNormScaled = nil }, "post-attention norm scale missing"},
		{"pre-FF norm scale", func(l *Gemma4DecoderLayer) { l.PreFFNormScaled = nil }, "pre-FF norm scale missing"},
		{"post-FF norm scale", func(l *Gemma4DecoderLayer) { l.PostFFNormScaled = nil }, "post-FF norm scale missing"},
		{"q-norm scale", func(l *Gemma4DecoderLayer) { l.Attention.QNormScaled = nil }, "q-norm scale missing"},
		{"k-norm scale (owner)", func(l *Gemma4DecoderLayer) { l.Attention.KNormScaled = nil }, "k-norm scale missing"},
		{"nil MLP", func(l *Gemma4DecoderLayer) { l.MLP = nil }, "MLP module missing"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cp := cov4clsCopyLayer(layer)
			tc.mutate(cp)
			state := buildGemma4CompiledLayerState(cp, false, cfg)
			if !state.declined {
				t.Fatalf("%s: expected decline, got eligible", tc.name)
			}
			if !core.Contains(state.reason, tc.want) {
				t.Fatalf("%s: reason = %q, want substring %q", tc.name, state.reason, tc.want)
			}
		})
	}
}

// TestCompiledLayerState_LinearWhy drives the linearWhy diagnostic switch by
// replacing q_proj with a Linear that fails one eligibility condition at a time;
// the decline reason carries the per-weight cause.
func TestCompiledLayerState_LinearWhy(t *testing.T) {
	requireMetalRuntime(t)

	layer, cfg := cov4clsOwnerLayer(t)
	good := layer.Attention.QProj // eligible quantised linear, owned by the model

	// A small valid array to stand in for weight/scales/biases where a case
	// needs them present-but-not-the-failing-field.
	arr := func() *metal.Array { return seqArray(0.1, 8, 1) }

	cases := []struct {
		name string
		proj *metal.Linear
		want string
	}{
		{"nil", nil, "q_proj: nil"},
		{"lora", &metal.Linear{Weight: good.Weight, Scales: good.Scales, Biases: good.Biases, LoRA: &metal.LoRALinear{}}, "LoRA attached"},
		{"weight invalid", &metal.Linear{}, "weight invalid"},
		{"no scales", &metal.Linear{Weight: arr()}, "unquantized (no scales)"},
		{"no biases", &metal.Linear{Weight: arr(), Scales: arr()}, "quant biases missing"},
		{"additive bias", &metal.Linear{Weight: good.Weight, Scales: good.Scales, Biases: good.Biases, Bias: arr(), QuantizationMode: "affine"}, "additive bias present"},
		{"non-affine", &metal.Linear{Weight: good.Weight, Scales: good.Scales, Biases: good.Biases, QuantizationMode: "mxfp8"}, "non-affine quant mode"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cp := cov4clsCopyLayer(layer)
			cp.Attention.QProj = tc.proj
			state := buildGemma4CompiledLayerState(cp, false, cfg)
			if !state.declined {
				t.Fatalf("%s: expected decline, got eligible", tc.name)
			}
			if !core.Contains(state.reason, tc.want) {
				t.Fatalf("%s: reason = %q, want substring %q", tc.name, state.reason, tc.want)
			}
		})
	}

	// Free the standalone arrays created above for the present-but-not-failing
	// slots. The `good` tensors belong to the model and are NOT freed here.
	for _, tc := range cases {
		if tc.proj == nil || tc.proj == good {
			continue
		}
		if tc.proj.Weight != nil && tc.proj.Weight != good.Weight {
			metal.Free(tc.proj.Weight)
		}
		if tc.proj.Scales != nil && tc.proj.Scales != good.Scales {
			metal.Free(tc.proj.Scales)
		}
		if tc.proj.Biases != nil && tc.proj.Biases != good.Biases {
			metal.Free(tc.proj.Biases)
		}
		if tc.proj.Bias != nil {
			metal.Free(tc.proj.Bias)
		}
	}
}
