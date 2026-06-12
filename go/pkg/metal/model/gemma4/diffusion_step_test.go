// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	"math"
	"testing"

	"dappco.re/go/mlx/pkg/metal"
)

func TestDefaultDiffusionStepConfig_Good(t *testing.T) {
	cfg := DefaultDiffusionStepConfig(262144)
	if cfg.EntropyBound != 0.3 || cfg.MaxTemperature != 0.8 || cfg.MinTemperature != 0.4 || cfg.Exponent != 1.0 {
		t.Fatalf("config = %+v, want the measured decode profile", cfg)
	}
	if cfg.TextVocabSize != 262144 {
		t.Fatalf("TextVocabSize = %d, want pass-through", cfg.TextVocabSize)
	}
}

// The global canvas mask is all-zero by construction (every canvas token
// attends everywhere); the shape is the contract.
func TestDiffusionGlobalCanvasMask_Geometry_Good(t *testing.T) {
	mask := diffusionGlobalCanvasMask(1, 4, 10)
	defer metal.Free(mask)
	var shapeBuf [metal.MaxTensorRank]int32
	shape := mask.ShapeInto(shapeBuf[:0])
	if len(shape) != 4 || shape[0] != 1 || shape[1] != 1 || shape[2] != 4 || shape[3] != 10 {
		t.Fatalf("shape = %v, want [1 1 4 10]", shape)
	}
	for i, v := range mask.Floats() {
		if v != 0 {
			t.Fatalf("mask[%d] = %f, want 0 (attend everywhere)", i, v)
		}
	}
}

// The block-local mask: a shared context window [offset-window, offset)
// plus full canvas self-attention [offset, offset+L); everything else -inf.
// Wrong geometry = silent garbage, so walk every cell.
func TestDiffusionBlockLocalCanvasMask_Geometry_Good(t *testing.T) {
	const (
		L      = 3
		offset = 6
		window = 4
		keyLen = offset + L
	)
	mask := diffusionBlockLocalCanvasMask(1, L, keyLen, offset, window)
	defer metal.Free(mask)
	values := mask.Floats()
	negInf := float32(math.Inf(-1))
	for i := 0; i < L; i++ {
		for j := 0; j < keyLen; j++ {
			got := values[i*keyLen+j]
			inContext := j >= offset-window && j < offset
			inCanvas := j >= offset
			want := negInf
			if inContext || inCanvas {
				want = 0
			}
			if got != want {
				t.Fatalf("mask[%d][%d] = %f, want %f (context=[%d,%d) canvas=[%d,%d))",
					i, j, got, want, offset-window, offset, offset, offset+L)
			}
		}
	}
}

func TestDiffusionBlockLocalCanvasMask_ContextClampsAtZero_Ugly(t *testing.T) {
	// offset < window: the context window clamps to [0, offset) instead of
	// going negative.
	const (
		L      = 2
		offset = 2
		window = 8
		keyLen = offset + L
	)
	mask := diffusionBlockLocalCanvasMask(1, L, keyLen, offset, window)
	defer metal.Free(mask)
	for i, v := range mask.Floats() {
		if v != 0 {
			t.Fatalf("mask[%d] = %f, want all-attend when the clamped context covers the whole prefix", i, v)
		}
	}
}

// SampleDenoiseStep on synthetic peaked logits: every position carries one
// dominant token, so per-token entropy is ~0, every position is accepted
// within the budget, and the canvas converges to the argmax in one step.
// Exercises the full sampler chain (anneal, categorical, entropy sort,
// renoise, self-conditioning encode) on tiny arrays — no model load.
func TestSampleDenoiseStep_PeakedLogitsAcceptAll_Good(t *testing.T) {
	const (
		L = 4
		V = 8
		D = 4
	)
	peaks := []int32{3, 1, 7, 0}
	logits := make([]float32, L*V)
	for i, p := range peaks {
		logits[i*V+int(p)] = 32
	}
	logitsArr := metal.FromValues(logits, 1, L, V)
	defer metal.Free(logitsArr)

	embed := make([]float32, V*D)
	for i := range embed {
		embed[i] = float32(i%7) * 0.25
	}
	embedArr := metal.FromValues(embed, V, D)
	defer metal.Free(embedArr)

	m := &DiffusionGemmaModel{
		Gemma4Model: &Gemma4Model{
			Cfg:         &Gemma4TextConfig{TransformerConfig: metal.TransformerConfig{HiddenSize: D, VocabSize: V}},
			EmbedTokens: &metal.Embedding{Weight: embedArr},
		},
	}

	cfg := DefaultDiffusionStepConfig(V)
	cfg.Seed = 7
	prev := []int32{0, 0, 0, 0}
	res, err := m.SampleDenoiseStep(logitsArr, prev, 0, 1.0, cfg)
	if err != nil {
		t.Fatalf("SampleDenoiseStep() error = %v", err)
	}
	defer metal.Free(res.SCEmb)

	if len(res.Canvas) != L || len(res.Greedy) != L {
		t.Fatalf("canvas/greedy lengths = %d/%d, want %d", len(res.Canvas), len(res.Greedy), L)
	}
	for i, p := range peaks {
		if res.Greedy[i] != p {
			t.Fatalf("Greedy[%d] = %d, want peak %d", i, res.Greedy[i], p)
		}
		if res.Canvas[i] != p {
			t.Fatalf("Canvas[%d] = %d, want accepted peak %d (entropy ~0 accepts all)", i, res.Canvas[i], p)
		}
	}
	if res.Accepted != L {
		t.Fatalf("Accepted = %d, want all %d under near-zero entropy", res.Accepted, L)
	}
	if res.MeanEntropy > 0.01 {
		t.Fatalf("MeanEntropy = %f, want ~0 for peaked logits", res.MeanEntropy)
	}
	if res.SCEmb == nil || !res.SCEmb.Valid() {
		t.Fatal("SCEmb invalid, want the next step's self-conditioning signal")
	}
	var shapeBuf [metal.MaxTensorRank]int32
	shape := res.SCEmb.ShapeInto(shapeBuf[:0])
	if len(shape) != 3 || shape[0] != 1 || shape[1] != L || shape[2] != D {
		t.Fatalf("SCEmb shape = %v, want [1 %d %d]", shape, L, D)
	}
}

// Flat logits: entropy is at the vocab maximum, so the budget stops the
// acceptance sweep early — most positions renoise instead of locking.
func TestSampleDenoiseStep_FlatLogitsRespectBudget_Bad(t *testing.T) {
	const (
		L = 4
		V = 8
		D = 2
	)
	logitsArr := metal.FromValues(make([]float32, L*V), 1, L, V)
	embedArr := metal.FromValues(make([]float32, V*D), V, D)
	defer metal.Free(logitsArr, embedArr)

	m := &DiffusionGemmaModel{
		Gemma4Model: &Gemma4Model{
			Cfg:         &Gemma4TextConfig{TransformerConfig: metal.TransformerConfig{HiddenSize: D, VocabSize: V}},
			EmbedTokens: &metal.Embedding{Weight: embedArr},
		},
	}

	cfg := DefaultDiffusionStepConfig(V)
	cfg.Seed = 11
	res, err := m.SampleDenoiseStep(logitsArr, []int32{0, 0, 0, 0}, 0, 1.0, cfg)
	if err != nil {
		t.Fatalf("SampleDenoiseStep() error = %v", err)
	}
	defer metal.Free(res.SCEmb)

	// Uniform over 8 tokens = ln(8) ≈ 2.08 nats per position. The budget
	// check runs BEFORE each acceptance, so the first position always
	// locks (accumulated 0 ≤ 0.3) and the second never does (2.08 > 0.3)
	// — exactly one accepted, the rest renoise.
	if res.Accepted != 1 {
		t.Fatalf("Accepted = %d, want exactly 1 under the entropy budget on flat logits", res.Accepted)
	}
	if res.MeanEntropy < 1.5 {
		t.Fatalf("MeanEntropy = %f, want ~ln(8) for flat logits", res.MeanEntropy)
	}
}
