// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/model"
)

const (
	mtpFixtureDModel = 512
	mtpFixtureNHeads = 8
	mtpFixtureNKV    = 4
	mtpFixtureHead   = 64
	mtpFixtureDFF    = 1024
	mtpFixtureVocab  = 64
	mtpFixtureLayers = 3
	mtpFixtureMaxLen = 96
)

func newMTPDecodeFixture(t testing.TB) func() *ArchSession {
	t.Helper()
	layers := make([]DecodeLayerWeights, mtpFixtureLayers)
	types := make([]string, mtpFixtureLayers)
	for li := range layers {
		layers[li] = forwardLayer(mtpFixtureDModel, mtpFixtureNHeads, mtpFixtureNKV, mtpFixtureHead, mtpFixtureDFF, (li+1)*100)
		types[li] = "full_attention"
	}
	specs := model.DeriveLayers(types, 0)
	embed := toBF16Bytes(syntheticFloat32(mtpFixtureVocab*mtpFixtureDModel, 21))
	g := &BF16Model{
		Layers:    layers,
		Embed:     embed,
		FinalNorm: toBF16Bytes(syntheticFloat32(mtpFixtureDModel, 22)),
		LMHead:    embed,
		Tied:      true,
	}
	arch := model.Arch{
		Hidden: mtpFixtureDModel, Heads: mtpFixtureNHeads, KVHeads: mtpFixtureNKV, HeadDim: mtpFixtureHead, FF: mtpFixtureDFF, Vocab: mtpFixtureVocab,
		GlobalHeadDim: mtpFixtureHead, GlobalKVHeads: mtpFixtureNKV,
		Eps: 1e-5, AttnScale: 0.125, RopeBase: 10000, RopeScale: 1, RopeLocalBase: 10000,
		RotaryDim: mtpFixtureHead, RotaryDimLocal: mtpFixtureHead,
		Layer: specs,
	}
	return func() *ArchSession {
		s, err := NewArchSession(g, arch, mtpFixtureMaxLen)
		if err != nil {
			t.Fatalf("NewArchSession: %v", err)
		}
		return s
	}
}

func TestMTPDecodeInputGuards(t *testing.T) {
	session := func(maxLen int) *ArchSession { return &ArchSession{maxLen: maxLen} }
	prompt := []int32{1}
	tests := []struct {
		name   string
		target *ArchSession
		draft  *ArchSession
		prompt []int32
		maxNew int
		k      int
	}{
		{name: "nil target", target: nil, draft: session(8), prompt: prompt, maxNew: 1, k: 1},
		{name: "nil draft", target: session(8), draft: nil, prompt: prompt, maxNew: 1, k: 1},
		{name: "empty prompt", target: session(8), draft: session(8), prompt: nil, maxNew: 1, k: 1},
		{name: "zero maxNew", target: session(8), draft: session(8), prompt: prompt, maxNew: 0, k: 1},
		{name: "zero k", target: session(8), draft: session(8), prompt: prompt, maxNew: 1, k: 0},
		{name: "target headroom", target: session(2), draft: session(8), prompt: prompt, maxNew: 1, k: 1},
		{name: "draft headroom", target: session(8), draft: session(2), prompt: prompt, maxNew: 1, k: 1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := MTPDecode(tt.target, tt.draft, tt.prompt, tt.maxNew, -1, tt.k); err == nil {
				t.Fatal("MTPDecode error = nil")
			}
		})
	}
}

// TestMTPDecodeBatchedTokenIdentity is the headline MTP invariant: speculative decode emits EXACTLY
// the token stream plain greedy Generate would, while engaging the batched verify (one pass over the
// resident stack per draft block, not K stepGreedy rounds). It builds a synthetic dense bf16 session
// (no PLE, no ICB on the bf16 path) so verifyBatched takes the batched path, uses draft==target weights
// so every draft is accepted (exercising the full accept loop + the batched verify), and asserts the
// MTP token stream equals Generate's token-for-token.
func TestMTPDecodeBatchedTokenIdentity(t *testing.T) {
	requireNativeRuntime(t)
	const K, maxNew = 4, 16
	mk := newMTPDecodeFixture(t)

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

func TestMTPDecodeDraftEqualsTargetAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)
	const K, maxNew = 4, 16
	prompt := []int32{1, 2, 3, 4, 5}
	mk := newMTPDecodeFixture(t)
	target := mk()
	draft := mk()

	var decodeErr error
	allocs := testing.AllocsPerRun(2, func() {
		target.pos = 0
		draft.pos = 0
		var res *MTPResult
		res, decodeErr = MTPDecode(target, draft, prompt, maxNew, -1, K)
		if decodeErr == nil && res.Accepted != res.Drafted {
			decodeErr = core.NewError("MTP draft==target did not accept every draft")
		}
	})
	if decodeErr != nil {
		t.Fatalf("MTPDecode: %v", decodeErr)
	}
	if allocs > 253400 {
		t.Fatalf("MTPDecode allocations = %.0f, want <= 253400", allocs)
	}
}

func TestMTPGreedyOfUsesDirectGreedyWhenAvailable(t *testing.T) {
	s := &ArchSession{
		arch: model.Arch{Vocab: 16},
		head: func([]byte, bool) ([]byte, error) {
			return nil, core.NewError("full logits head should not be called")
		},
		greedy: func([]byte, []int32) (int32, bool, error) {
			return 7, true, nil
		},
	}

	got, err := s.greedyOf([]byte{1, 2})
	if err != nil {
		t.Fatalf("greedyOf: %v", err)
	}
	if got != 7 {
		t.Fatalf("greedyOf = %d, want direct greedy token 7", got)
	}
}

func mtpSequentialFallbackSession(s *ArchSession) *ArchSession {
	s.perLayerInput = func(_ int32, _ []byte) ([]byte, error) {
		return nil, nil
	}
	return s
}

func mtpIDsEqual(a, b []int32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func TestMTPDecodeSequentialFallbackTokenIdentity(t *testing.T) {
	requireNativeRuntime(t)
	const K, maxNew = 4, 12
	prompt := []int32{1, 2, 3, 4, 5}
	mk := newMTPDecodeFixture(t)

	ref, err := mk().Generate(prompt, maxNew, -1)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}

	res, err := MTPDecode(mtpSequentialFallbackSession(mk()), mtpSequentialFallbackSession(mk()), prompt, maxNew, -1, K)
	if err != nil {
		t.Fatalf("MTPDecode: %v", err)
	}
	if !mtpIDsEqual(res.Tokens, ref) {
		t.Fatalf("sequential fallback MTP tokens %v != Generate %v", res.Tokens, ref)
	}
	if res.Accepted != res.Drafted {
		t.Fatalf("draft==target sequential fallback accepted %d/%d", res.Accepted, res.Drafted)
	}
}

func TestMTPVerifyBatchedWrapperAndFallback(t *testing.T) {
	requireNativeRuntime(t)
	mk := newMTPDecodeFixture(t)
	dense := mk()
	for _, id := range []int32{1, 2, 3} {
		if _, err := dense.stepID(id); err != nil {
			t.Fatalf("prefill dense stepID(%d): %v", id, err)
		}
	}
	greedys, ok, err := dense.verifyBatched([]int32{4, 5})
	if err != nil {
		t.Fatalf("verifyBatched dense: %v", err)
	}
	if !ok {
		t.Fatal("verifyBatched dense ok = false")
	}
	if len(greedys) != 2 {
		t.Fatalf("verifyBatched dense returned %d greedys, want 2", len(greedys))
	}
	for i, id := range greedys {
		if id < 0 || int(id) >= mtpFixtureVocab {
			t.Fatalf("greedy %d = %d outside vocab", i, id)
		}
	}

	fallback := mtpSequentialFallbackSession(mk())
	if _, ok, err = fallback.verifyBatched([]int32{4}); err != nil {
		t.Fatalf("verifyBatched fallback: %v", err)
	} else if ok {
		t.Fatal("verifyBatched fallback ok = true")
	}
	if _, ok, err = dense.verifyBatched(nil); err == nil {
		t.Fatal("verifyBatched empty error = nil")
	} else if ok {
		t.Fatal("verifyBatched empty ok = true")
	}
}

func TestMTPDecodeSinglePromptEOSMatchesGenerate(t *testing.T) {
	requireNativeRuntime(t)
	const K, maxNew = 4, 8
	prompt := []int32{7}
	mk := newMTPDecodeFixture(t)

	first, err := mk().Generate(prompt, 1, -1)
	if err != nil {
		t.Fatalf("Generate first token: %v", err)
	}
	res, err := MTPDecode(mk(), mk(), prompt, maxNew, int(first[0]), K)
	if err != nil {
		t.Fatalf("MTPDecode: %v", err)
	}
	if !mtpIDsEqual(res.Tokens, first) {
		t.Fatalf("MTP EOS tokens %v != first greedy %v", res.Tokens, first)
	}
	if len(res.Tokens) != 1 {
		t.Fatalf("MTP EOS emitted %d tokens, want 1", len(res.Tokens))
	}
}

func TestMTPDecodeDensePromptPrefillAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)
	const K, maxNew = 4, 1
	prompt := []int32{1, 2, 3, 4, 5}
	mk := newMTPDecodeFixture(t)
	target := mk()
	draft := mk()

	var decodeErr error
	allocs := testing.AllocsPerRun(3, func() {
		target.pos = 0
		draft.pos = 0
		var res *MTPResult
		res, decodeErr = MTPDecode(target, draft, prompt, maxNew, -1, K)
		if decodeErr == nil && len(res.Tokens) != maxNew {
			decodeErr = core.NewError("MTPDecode prompt-prefill fixture emitted wrong token count")
		}
	})
	if decodeErr != nil {
		t.Fatalf("MTPDecode: %v", decodeErr)
	}
	if allocs > 64108 {
		t.Fatalf("MTPDecode dense prompt-prefill allocations = %.0f, want <= 64108", allocs)
	}
}
