// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package native

import (
	"os"
	"testing"

	"dappco.re/go/mlx/pkg/model"
	g4 "dappco.re/go/mlx/pkg/model/gemma4"
)

// TestNativeTokenModel_ContractParity gates the token-loop CONTRACT against the
// proven native generation loop: model.Generate over a NativeTokenModel
// (whole-sequence decode through model.Backend + the embed/head bookends) must
// produce the EXACT greedy tokens GenerateGemma4BF16 produces (native's
// incremental persistent-cache loop) on the same bf16 gemma4. The two loops
// share no code — one is the contract loop in pkg/model, the other native's
// bespoke loop — so full-sequence equality proves the contract path yields real
// tokens identical to the path it generalises. The surface pkg/rocm drops into
// is proven, not asserted.
func TestNativeTokenModel_ContractParity(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 32
	arch, err := g4.Config{
		HiddenSize: dModel, NumHiddenLayers: 2, IntermediateSize: dFF,
		NumAttentionHeads: nHeads, NumKeyValueHeads: nKV, HeadDim: headDim,
		VocabSize: vocab, RMSNormEps: 1e-6,
	}.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	mk := func(n, salt int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = float32((i*salt+13)%97-48) * 0.02
		}
		return s
	}
	layers := make([]DecodeLayerWeights, len(arch.Layer))
	for li := range layers {
		layers[li] = forwardLayer(dModel, nHeads, nKV, headDim, dFF, (li+1)*100)
	}
	g := &Gemma4BF16{
		Layers:    layers,
		Embed:     toBF16Bytes(mk(vocab*dModel, 11)),
		FinalNorm: toBF16Bytes(mk(dModel, 7)),
	}
	g.LMHead, g.Tied = g.Embed, true // tied head

	prompt := []int32{1, 5, 3, 9}
	const maxNew, maxLen = 6, 16

	// reference: native's proven incremental (persistent-cache) generation loop.
	want, err := GenerateGemma4BF16(g, arch, prompt, maxNew, maxLen, -1)
	if err != nil {
		t.Fatalf("GenerateGemma4BF16: %v", err)
	}

	// the contract path: model.Generate over the NativeTokenModel (whole-seq).
	tm, err := NewBF16TokenModel(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewBF16TokenModel: %v", err)
	}
	got, err := model.Generate(tm, prompt, maxNew, -1)
	if err != nil {
		t.Fatalf("model.Generate: %v", err)
	}
	if len(got) != len(want) {
		t.Fatalf("contract generated %d tokens, want %d (%v vs %v)", len(got), len(want), got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("contract token %d = %d, native loop = %d (full: %v vs %v)", i, got[i], want[i], got, want)
		}
	}

	// the contract Vocab() reports the logit width Greedy reads.
	if tm.Vocab() != vocab {
		t.Fatalf("Vocab() = %d, want %d", tm.Vocab(), vocab)
	}

	// zero-temp sampled generation falls back to greedy → same sequence.
	sampled, err := model.GenerateSampled(tm, model.NewSampler(7), model.SampleParams{Temperature: 0}, prompt, maxNew, -1)
	if err != nil {
		t.Fatalf("GenerateSampled: %v", err)
	}
	for i := range want {
		if sampled[i] != want[i] {
			t.Fatalf("zero-temp sampled token %d = %d, want %d (%v)", i, sampled[i], want[i], sampled)
		}
	}

	t.Logf("token-loop contract ≡ native generation: model.Generate(NativeTokenModel) = GenerateGemma4BF16 = %v", got)
}
