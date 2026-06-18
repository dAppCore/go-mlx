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

	// whole-seq reference via the SAME model's contract pieces (tm.NativeBackend's
	// DecodeForward is the whole-sequence fallback): the incremental result must be
	// output-identical to the path it supersedes — the additive refinement changes
	// speed, not tokens.
	seq := make([][]byte, 0, len(prompt)+maxNew)
	for _, id := range prompt {
		e, eerr := tm.Embed(id)
		if eerr != nil {
			t.Fatalf("Embed: %v", eerr)
		}
		seq = append(seq, e)
	}
	var wholeSeq []int32
	for len(wholeSeq) < maxNew {
		hs, derr := tm.NativeBackend.DecodeForward(seq)
		if derr != nil {
			t.Fatalf("whole-seq DecodeForward: %v", derr)
		}
		logits, herr := tm.Head(hs[len(hs)-1])
		if herr != nil {
			t.Fatalf("Head: %v", herr)
		}
		nx, gerr := model.Greedy(logits, vocab)
		if gerr != nil {
			t.Fatalf("Greedy: %v", gerr)
		}
		wholeSeq = append(wholeSeq, nx)
		if len(wholeSeq) >= maxNew {
			break
		}
		e, eerr := tm.Embed(nx)
		if eerr != nil {
			t.Fatalf("Embed: %v", eerr)
		}
		seq = append(seq, e)
	}
	for i := range want {
		if wholeSeq[i] != want[i] {
			t.Fatalf("whole-seq token %d = %d, want %d (incremental %v vs whole-seq %v)", i, wholeSeq[i], want[i], got, wholeSeq)
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

	t.Logf("token-loop contract (incremental session) ≡ native generation ≡ whole-seq: model.Generate(NativeTokenModel) = GenerateGemma4BF16 = %v", got)
}

// TestNativeTokenModel_QuantContractParity is the 4-bit sibling: model.Generate
// over a quant NativeTokenModel (whole-sequence DecodeForwardArchQuant + the
// quant embed/head bookends) must produce the EXACT greedy tokens
// NewGemma4QuantSession produces (native's incremental quant loop) on the same
// synthetic 4-bit gemma4. The model is all-global, so the session's per-type
// RoPE coincides with the whole-seq one base — and the two independent loops
// agree token-for-token, proving the contract covers the serving quant too.
func TestNativeTokenModel_QuantContractParity(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const gs, bits = 32, 4
	const maxLen, n = 16, 6
	cfg := g4.Config{
		HiddenSize: 128, NumHiddenLayers: 2, IntermediateSize: 256,
		NumAttentionHeads: 2, NumKeyValueHeads: 1, HeadDim: 64, VocabSize: 32, RMSNormEps: 1e-6,
		Quantization: &g4.QuantConfig{GroupSize: gs, Bits: bits},
	}
	arch, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	ts := quantGemma4Tensors(t, arch, gs, bits)
	g, err := AssembleGemma4Quant(ts, arch, &g4.QuantConfig{GroupSize: gs, Bits: bits})
	if err != nil {
		t.Fatalf("AssembleGemma4Quant: %v", err)
	}
	prompt := []int32{1, 5, 3}

	// reference: native's proven incremental (persistent-cache) quant loop.
	sess, err := NewGemma4QuantSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewGemma4QuantSession: %v", err)
	}
	want, err := sess.Generate(prompt, n, -1)
	if err != nil {
		t.Fatalf("quant session Generate: %v", err)
	}

	// the contract path: model.Generate over the quant NativeTokenModel (whole-seq).
	tm, err := NewQuantTokenModel(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewQuantTokenModel: %v", err)
	}
	got, err := model.Generate(tm, prompt, n, -1)
	if err != nil {
		t.Fatalf("model.Generate (quant): %v", err)
	}
	if len(got) != len(want) {
		t.Fatalf("quant contract generated %d tokens, want %d (%v vs %v)", len(got), len(want), got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("quant contract token %d = %d, native session = %d (full: %v vs %v)", i, got[i], want[i], got, want)
		}
	}
	t.Logf("4-bit token-loop contract ≡ native quant session: model.Generate(NewQuantTokenModel) = %v", got)
}
