// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/model"
)

// TestMTPDecodeBatchedTokenIdentity is the headline MTP invariant: speculative decode emits EXACTLY
// the token stream plain greedy Generate would, while engaging the batched verify (one pass over the
// resident stack per draft block, not K stepGreedy rounds). It builds a synthetic dense bf16 session
// (no PLE, no ICB on the bf16 path) so verifyBatched takes the batched path, uses draft==target weights
// so every draft is accepted (exercising the full accept loop + the batched verify), and asserts the
// MTP token stream equals Generate's token-for-token.
func TestMTPDecodeBatchedTokenIdentity(t *testing.T) {
	requireNativeRuntime(t)
	const dModel, nHeads, nKV, headDim, dFF = 512, 8, 4, 64, 1024
	const vocab, nL, maxLen, K, maxNew = 64, 3, 96, 4, 16

	layers := make([]DecodeLayerWeights, nL)
	types := make([]string, nL)
	for li := range layers {
		layers[li] = forwardLayer(dModel, nHeads, nKV, headDim, dFF, (li+1)*100)
		types[li] = "full_attention"
	}
	specs := model.DeriveLayers(types, 0)
	embed := toBF16Bytes(syntheticFloat32(vocab*dModel, 21))
	g := &BF16Model{
		Layers:    layers,
		Embed:     embed,
		FinalNorm: toBF16Bytes(syntheticFloat32(dModel, 22)),
		LMHead:    embed, // tied
		Tied:      true,
	}
	arch := model.Arch{
		Hidden: dModel, Heads: nHeads, KVHeads: nKV, HeadDim: headDim, FF: dFF, Vocab: vocab,
		GlobalHeadDim: headDim, GlobalKVHeads: nKV,
		Eps: 1e-5, AttnScale: 0.125, RopeBase: 10000, RopeScale: 1, RopeLocalBase: 10000,
		RotaryDim: headDim, RotaryDimLocal: headDim,
		Layer: specs,
	}
	mk := func() *ArchSession {
		s, err := NewArchSession(g, arch, maxLen)
		if err != nil {
			t.Fatalf("NewArchSession: %v", err)
		}
		return s
	}

	prompt := []int32{1, 2, 3, 4, 5}

	// reference: plain greedy Generate on a fresh session.
	ref, err := mk().Generate(prompt, maxNew, -1)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}

	// MTP speculative decode with draft==target weights → every draft accepted, batched verify engaged.
	res, err := MTPDecode(mk(), mk(), prompt, maxNew, -1, K)
	if err != nil {
		t.Fatalf("MTPDecode: %v", err)
	}

	if len(res.Tokens) != len(ref) {
		t.Fatalf("MTP emitted %d tokens, Generate emitted %d", len(res.Tokens), len(ref))
	}
	for i := range ref {
		if res.Tokens[i] != ref[i] {
			t.Fatalf("token %d diverged: MTP=%d Generate=%d", i, res.Tokens[i], ref[i])
		}
	}
	if res.Accepted == 0 {
		t.Fatal("no drafts accepted — the speculative/batched path did not engage")
	}
	// draft == target weights, so every proposed token IS the target's greedy → all must be accepted.
	// A drop below full acceptance means the draft cache drifted out of alignment with the target.
	if res.Accepted != res.Drafted {
		t.Fatalf("draft==target should accept every draft, got %d/%d (draft cache misaligned)", res.Accepted, res.Drafted)
	}
	t.Log(core.Sprintf("MTP batched == Generate over %d tokens; accepted %d/%d drafted in %d rounds",
		len(ref), res.Accepted, res.Drafted, res.Rounds))
}
