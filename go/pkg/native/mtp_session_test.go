// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"testing"
	"unsafe"

	core "dappco.re/go"
	"dappco.re/go/mlx/kv"
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

	mtpWordedPromptText  = "speculative decoding works with a few words"
	mtpWordedPromptWords = 7
)

var mtpWordedPromptTokens = [...]int32{2, 18, 7, 41, 13, 5, 29}

func mtpWordedPromptIDs() []int32 {
	return mtpWordedPromptTokens[:]
}

func TestMTPWordedPromptFixtureUsesAFewWords(t *testing.T) {
	prompt := mtpWordedPromptIDs()
	if mtpWordedPromptWords < 5 {
		t.Fatalf("MTP worded prompt %q has %d words, want a few words", mtpWordedPromptText, mtpWordedPromptWords)
	}
	if len(prompt) != mtpWordedPromptWords {
		t.Fatalf("MTP worded prompt token count = %d, want one stable token id per word", len(prompt))
	}
	for i, id := range prompt {
		if id <= 0 || int(id) >= mtpFixtureVocab {
			t.Fatalf("MTP worded prompt token %d = %d outside fixture vocab %d", i, id, mtpFixtureVocab)
		}
	}
}

func newMTPDecodeFixture(t testing.TB) func() *ArchSession {
	t.Helper()
	return newMTPDecodeFixtureWithArch(t, nil)
}

func newMTPDecodeFixtureWithArch(t testing.TB, configure func(*model.Arch)) func() *ArchSession {
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
	if configure != nil {
		configure(&arch)
	}
	return func() *ArchSession {
		s, err := NewArchSession(g, arch, mtpFixtureMaxLen)
		if err != nil {
			t.Fatalf("NewArchSession: %v", err)
		}
		head := &headEncoder{
			finalNorm: copyView(g.FinalNorm),
			weight:    copyView(g.LMHead),
			dModel:    arch.Hidden,
			vocab:     arch.Vocab,
			eps:       arch.Eps,
			softCap:   arch.SoftCap,
		}
		s.headEnc = head
		s.head = func(hidden []byte, skipSoftcap bool) ([]byte, error) {
			return head.encode(hidden, skipSoftcap)
		}
		s.greedy = func(hidden []byte, suppress []int32) (int32, bool, error) {
			return head.greedyInPool(hidden, suppress)
		}
		s.markDefaultHeadFunc()
		s.markDefaultGreedyFunc()
		return s
	}
}

func TestMTPDecodeInputGuards(t *testing.T) {
	session := func(maxLen int) *ArchSession { return &ArchSession{maxLen: maxLen} }
	prompt := mtpWordedPromptIDs()
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
		{name: "target headroom", target: session(1), draft: session(8), prompt: prompt, maxNew: 1, k: 1},
		{name: "draft headroom", target: session(8), draft: session(1), prompt: prompt, maxNew: 1, k: 1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := MTPDecode(tt.target, tt.draft, tt.prompt, tt.maxNew, -1, tt.k); err == nil {
				t.Fatal("MTPDecode error = nil")
			}
		})
	}
}

func TestMTPDecodeSampledInputGuards(t *testing.T) {
	session := func(maxLen int) *ArchSession { return &ArchSession{maxLen: maxLen} }
	prompt := mtpWordedPromptIDs()
	targetSampler := model.NewSampler(1)
	draftSampler := model.NewSampler(2)
	sharedSampler := model.NewSampler(3)
	tests := []struct {
		name          string
		target        *ArchSession
		draft         *ArchSession
		prompt        []int32
		maxNew        int
		k             int
		targetSampler *model.Sampler
		draftSampler  *model.Sampler
	}{
		{name: "nil target", target: nil, draft: session(8), prompt: prompt, maxNew: 1, k: 1, targetSampler: targetSampler, draftSampler: draftSampler},
		{name: "nil draft", target: session(8), draft: nil, prompt: prompt, maxNew: 1, k: 1, targetSampler: targetSampler, draftSampler: draftSampler},
		{name: "nil target sampler", target: session(8), draft: session(8), prompt: prompt, maxNew: 1, k: 1, draftSampler: draftSampler},
		{name: "nil draft sampler", target: session(8), draft: session(8), prompt: prompt, maxNew: 1, k: 1, targetSampler: targetSampler},
		{name: "shared sampler", target: session(8), draft: session(8), prompt: prompt, maxNew: 1, k: 1, targetSampler: sharedSampler, draftSampler: sharedSampler},
		{name: "empty prompt", target: session(8), draft: session(8), prompt: nil, maxNew: 1, k: 1, targetSampler: targetSampler, draftSampler: draftSampler},
		{name: "zero maxNew", target: session(8), draft: session(8), prompt: prompt, maxNew: 0, k: 1, targetSampler: targetSampler, draftSampler: draftSampler},
		{name: "zero k", target: session(8), draft: session(8), prompt: prompt, maxNew: 1, k: 0, targetSampler: targetSampler, draftSampler: draftSampler},
		{name: "target headroom", target: session(1), draft: session(8), prompt: prompt, maxNew: 1, k: 1, targetSampler: targetSampler, draftSampler: draftSampler},
		{name: "draft headroom", target: session(8), draft: session(1), prompt: prompt, maxNew: 1, k: 1, targetSampler: targetSampler, draftSampler: draftSampler},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := MTPDecodeSampled(tt.target, tt.draft, tt.prompt, tt.maxNew, nil, tt.targetSampler, tt.draftSampler, model.SampleParams{}, tt.k); err == nil {
				t.Fatal("MTPDecodeSampled error = nil")
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

	prompt := mtpWordedPromptIDs()

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

func TestMTPDecodeEachYieldsCommittedTokens(t *testing.T) {
	requireNativeRuntime(t)
	const K, maxNew = 4, 10
	prompt := mtpWordedPromptIDs()
	mk := newMTPDecodeFixture(t)
	ref, err := mk().Generate(prompt, maxNew, -1)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	var yielded []int32
	res, err := MTPDecodeEach(mk(), mk(), prompt, maxNew, -1, K, func(id int32) bool {
		yielded = append(yielded, id)
		return true
	})
	if err != nil {
		t.Fatalf("MTPDecodeEach: %v", err)
	}
	if !mtpIDsEqual(res.Tokens, ref) {
		t.Fatalf("MTPDecodeEach tokens %v != Generate %v", res.Tokens, ref)
	}
	if !mtpIDsEqual(yielded, res.Tokens) {
		t.Fatalf("MTPDecodeEach yielded %v != result tokens %v", yielded, res.Tokens)
	}
}

func TestMTPDecodeUsesExactContextTailHeadroom(t *testing.T) {
	requireNativeRuntime(t)
	const K, maxNew = 4, 2
	prompt := mtpWordedPromptIDs()
	maxLen := len(prompt) + maxNew
	mk := newMTPDecodeFixture(t)
	limit := func(s *ArchSession) *ArchSession {
		s.maxLen = maxLen
		return s
	}

	ref, err := limit(mk()).Generate(prompt, maxNew, -1)
	if err != nil {
		t.Fatalf("Generate exact tail reference: %v", err)
	}
	res, err := MTPDecode(limit(mk()), limit(mk()), prompt, maxNew, -1, K)
	if err != nil {
		t.Fatalf("MTPDecode exact tail: %v", err)
	}
	if !mtpIDsEqual(res.Tokens, ref) {
		t.Fatalf("MTP exact tail tokens %v != Generate %v", res.Tokens, ref)
	}
}

func TestMTPDensePromptPrefillWordedHiddenMatchesSequential(t *testing.T) {
	requireNativeRuntime(t)
	prompt := mtpWordedPromptIDs()
	mk := newMTPDecodeFixture(t)
	ref := mk()
	sess := mk()

	if err := ref.PrefillTokens(prompt); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	got, ok, err := sess.prefillMTPPrompt(prompt, true)
	if err != nil {
		t.Fatalf("prefillMTPPrompt: %v", err)
	}
	if !ok {
		t.Fatal("prefillMTPPrompt ok = false")
	}
	if !bytes.Equal(got, ref.retainedHidden) {
		t.Fatal("prefillMTPPrompt hidden differs from sequential prompt prefill")
	}
	for _, id := range []int32{13, 37, 41} {
		wantHidden, err := ref.stepID(id)
		if err != nil {
			t.Fatalf("reference stepID(%d): %v", id, err)
		}
		gotHidden, err := sess.stepID(id)
		if err != nil {
			t.Fatalf("dense-prefill stepID(%d): %v", id, err)
		}
		if !bytes.Equal(gotHidden, wantHidden) {
			t.Fatalf("hidden after stepping %d differs after dense prompt prefill", id)
		}
	}
}

func TestMTPDecodeSampledMatchesGenerateSampled(t *testing.T) {
	requireNativeRuntime(t)
	const K, maxNew = 4, 12
	const seed uint64 = 53
	prompt := mtpWordedPromptIDs()
	params := model.SampleParams{
		Temperature:   0.8,
		TopK:          7,
		TopP:          0.75,
		MinP:          0.01,
		RepeatPenalty: 1.2,
		SuppressTokens: []int32{
			2,
			7,
		},
	}
	mk := newMTPDecodeFixture(t)

	ref, err := mk().GenerateSampledEach(prompt, maxNew, nil, model.NewSampler(seed), params, nil, nil)
	if err != nil {
		t.Fatalf("GenerateSampledEach: %v", err)
	}
	res, err := MTPDecodeSampled(mk(), mk(), prompt, maxNew, nil, model.NewSampler(seed), model.NewSampler(seed+1), params, K)
	if err != nil {
		t.Fatalf("MTPDecodeSampled: %v", err)
	}
	if !mtpIDsEqual(res.Tokens, ref) {
		t.Fatalf("sampled MTP tokens %v != GenerateSampledEach %v (accepted=%d drafted=%d rounds=%d)", res.Tokens, ref, res.Accepted, res.Drafted, res.Rounds)
	}
	if res.Drafted == 0 {
		t.Fatal("sampled MTP proposed no draft tokens")
	}
}

func TestMTPDecodeSampledEachYieldsCommittedTokens(t *testing.T) {
	requireNativeRuntime(t)
	const K, maxNew = 4, 10
	const seed uint64 = 53
	prompt := mtpWordedPromptIDs()
	params := model.SampleParams{
		Temperature:   0.8,
		TopK:          7,
		TopP:          0.75,
		MinP:          0.01,
		RepeatPenalty: 1.2,
		SuppressTokens: []int32{
			2,
			7,
		},
	}
	mk := newMTPDecodeFixture(t)
	ref, err := mk().GenerateSampledEach(prompt, maxNew, nil, model.NewSampler(seed), params, nil, nil)
	if err != nil {
		t.Fatalf("GenerateSampledEach: %v", err)
	}
	var yielded []int32
	res, err := MTPDecodeSampledEach(mk(), mk(), prompt, maxNew, nil, model.NewSampler(seed), model.NewSampler(seed+1), params, K, func(id int32) bool {
		yielded = append(yielded, id)
		return true
	})
	if err != nil {
		t.Fatalf("MTPDecodeSampledEach: %v", err)
	}
	if !mtpIDsEqual(res.Tokens, ref) {
		t.Fatalf("MTPDecodeSampledEach tokens %v != GenerateSampledEach %v (accepted=%d drafted=%d rounds=%d)", res.Tokens, ref, res.Accepted, res.Drafted, res.Rounds)
	}
	if !mtpIDsEqual(yielded, res.Tokens) {
		t.Fatalf("MTPDecodeSampledEach yielded %v != result tokens %v", yielded, res.Tokens)
	}
}

func TestMTPSampledPickerMatchesGenerateSampledOnWordedPrompt(t *testing.T) {
	requireNativeRuntime(t)
	const maxNew = 12
	const seed uint64 = 53
	prompt := mtpWordedPromptIDs()
	params := model.SampleParams{
		Temperature:   0.8,
		TopK:          7,
		TopP:          0.75,
		MinP:          0.01,
		RepeatPenalty: 1.2,
		SuppressTokens: []int32{
			2,
			7,
		},
	}
	mk := newMTPDecodeFixture(t)

	ref, err := mk().GenerateSampledEach(prompt, maxNew, nil, model.NewSampler(seed), params, nil, nil)
	if err != nil {
		t.Fatalf("GenerateSampledEach: %v", err)
	}
	sess := mk()
	hidden, ok, err := sess.prefillMTPPrompt(prompt, true)
	if err != nil {
		t.Fatalf("prefillMTPPrompt: %v", err)
	}
	if !ok {
		t.Fatal("prefillMTPPrompt ok = false")
	}
	history := sess.sampleHistoryScratchFor(params, maxNew)
	var got []int32
	sampler := model.NewSampler(seed)
	for len(got) < maxNew {
		pickParams := sess.mtpSamplePickParams(params, nil, len(got))
		next, err := sess.sampleMTPTokenFromHidden(hidden, sampler, pickParams, history)
		if err != nil {
			t.Fatalf("sampleMTPTokenFromHidden token %d: %v", len(got), err)
		}
		got = append(got, next)
		if params.RepeatPenalty > 1 {
			history = append(history, next)
		}
		hidden, err = sess.stepID(next)
		if err != nil {
			t.Fatalf("stepID(%d): %v", next, err)
		}
	}
	if !mtpIDsEqual(got, ref) {
		t.Fatalf("MTP sampled picker tokens %v != GenerateSampledEach %v", got, ref)
	}
}

func TestMTPDecodeSlidingRingWrapMatchesGenerate(t *testing.T) {
	requireNativeRuntime(t)
	const K, maxNew = 3, 10
	mk := newMTPDecodeFixtureWithArch(t, func(arch *model.Arch) {
		arch.SlidingWindow = 4
		arch.Layer[0].Attention = model.SlidingAttention
	})
	prompt := mtpWordedPromptIDs()

	ref, err := mk().Generate(prompt, maxNew, -1)
	if err != nil {
		t.Fatalf("Generate sliding reference: %v", err)
	}
	res, err := MTPDecode(mk(), mk(), prompt, maxNew, -1, K)
	if err != nil {
		t.Fatalf("MTPDecode sliding: %v", err)
	}
	if !mtpIDsEqual(res.Tokens, ref) {
		t.Fatalf("sliding MTP tokens %v != Generate %v", res.Tokens, ref)
	}
	if res.Accepted == 0 {
		t.Fatal("sliding MTP accepted no drafts; batched verify did not engage")
	}
}

func TestMTPDecodeDraftEqualsTargetAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)
	const K, maxNew = 4, 16
	prompt := mtpWordedPromptIDs()
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

// mtpSequentialFallbackSession forces s onto the byte-identical sequential verify
// lane by flipping the test-only verifyBatchedDisabledForTest guard, so
// verifyBatchedHiddens / verifyBatchedInto decline (ok=false) and MTPDecode /
// the assistant pair step token-by-token. This is the honest hook: it does not
// rely on any arch property (every resident arch — dense and PLE — now batches),
// so the sequential lane is exercised on the same fixture the batched lane uses.
func mtpSequentialFallbackSession(s *ArchSession) *ArchSession {
	s.verifyBatchedDisabledForTest = true
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
	prompt := mtpWordedPromptIDs()
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

func TestMTPVerifyBatchedSlidingRingWrapMatchesSequential(t *testing.T) {
	requireNativeRuntime(t)
	mk := newMTPDecodeFixtureWithArch(t, func(arch *model.Arch) {
		arch.SlidingWindow = 4
		arch.Layer[0].Attention = model.SlidingAttention
	})
	ref := mk()
	sess := mk()
	prompt := mtpWordedPromptIDs()
	if err := ref.PrefillTokens(prompt); err != nil {
		t.Fatalf("reference PrefillTokens: %v", err)
	}
	if err := sess.PrefillTokens(prompt); err != nil {
		t.Fatalf("candidate PrefillTokens: %v", err)
	}
	if sess.Pos() != len(prompt) {
		t.Fatalf("prefill pos = %d, want %d", sess.Pos(), len(prompt))
	}
	ids := []int32{4, 5}
	want := make([]int32, len(ids))
	for i, id := range ids {
		hidden, err := ref.stepID(id)
		if err != nil {
			t.Fatalf("reference stepID(%d): %v", id, err)
		}
		want[i], err = ref.greedyOf(hidden)
		if err != nil {
			t.Fatalf("reference greedyOf(%d): %v", id, err)
		}
	}
	greedys, ok, err := sess.verifyBatchedInto(ids, make([]int32, len(ids)))
	if err != nil {
		t.Fatalf("verifyBatchedInto sliding wrap: %v", err)
	}
	if !ok {
		t.Fatal("verifyBatchedInto sliding wrap ok = false")
	}
	if !mtpIDsEqual(greedys, want) {
		t.Fatalf("verifyBatchedInto sliding wrap greedys = %v, want sequential %v", greedys, want)
	}
	if sess.Pos() != len(prompt) {
		t.Fatalf("verifyBatchedInto sliding wrap changed pos = %d, want %d", sess.Pos(), len(prompt))
	}
}

func TestMTPVerifyBatchedHiddensMatchesSequential(t *testing.T) {
	requireNativeRuntime(t)
	mk := newMTPDecodeFixture(t)
	ref := mk()
	sess := mk()
	prompt := mtpWordedPromptIDs()
	if err := ref.PrefillTokens(prompt); err != nil {
		t.Fatalf("reference PrefillTokens: %v", err)
	}
	if err := sess.PrefillTokens(prompt); err != nil {
		t.Fatalf("candidate PrefillTokens: %v", err)
	}
	ids := []int32{4, 5, 6}
	want := make([][]byte, len(ids))
	for i, id := range ids {
		hidden, err := ref.stepID(id)
		if err != nil {
			t.Fatalf("reference stepID(%d): %v", id, err)
		}
		want[i] = append([]byte(nil), hidden...)
	}
	got, ok, err := sess.verifyBatchedHiddens(ids)
	if err != nil {
		t.Fatalf("verifyBatchedHiddens: %v", err)
	}
	if !ok {
		t.Fatal("verifyBatchedHiddens ok = false")
	}
	if len(got) != len(want) {
		t.Fatalf("verifyBatchedHiddens returned %d rows, want %d", len(got), len(want))
	}
	for i := range got {
		eqBytes(t, core.Sprintf("batched hidden row %d", i), got[i], want[i])
	}
	if sess.Pos() != len(prompt) {
		t.Fatalf("verifyBatchedHiddens changed pos = %d, want %d", sess.Pos(), len(prompt))
	}
}

func TestMTPSampledDenseBatchRowPickerMatchesHiddenOnWordedPrompt(t *testing.T) {
	requireNativeRuntime(t)
	mk := newMTPDecodeFixture(t)
	sess := mk()
	prompt := mtpWordedPromptIDs()
	if err := sess.PrefillTokens(prompt); err != nil {
		t.Fatalf("PrefillTokens(%q): %v", mtpWordedPromptText, err)
	}
	ids := []int32{4, 5, 6}
	hiddens, ok, err := sess.verifyBatchedHiddens(ids)
	if err != nil {
		t.Fatalf("verifyBatchedHiddens: %v", err)
	}
	if !ok {
		t.Fatal("verifyBatchedHiddens ok = false")
	}
	params := model.SampleParams{
		Temperature:    0.8,
		TopK:           7,
		TopP:           0.75,
		MinP:           0.01,
		RepeatPenalty:  1.2,
		SuppressTokens: []int32{2, 7},
	}
	history := []int32{3, 5, 8}
	const row = 1
	want, err := sess.sampleMTPTokenFromHidden(hiddens[row], model.NewSampler(83), params, history)
	if err != nil {
		t.Fatalf("sampleMTPTokenFromHidden: %v", err)
	}
	got, direct, err := sess.sampleMTPTokenFromDenseBatchRow(row, model.NewSampler(83), params, history)
	if err != nil {
		t.Fatalf("sampleMTPTokenFromDenseBatchRow: %v", err)
	}
	if !direct {
		t.Fatal("sampleMTPTokenFromDenseBatchRow declined the worded prompt batch row")
	}
	if got != want {
		t.Fatalf("sampleMTPTokenFromDenseBatchRow = %d, want hidden sample %d", got, want)
	}
}

func TestMTPVerifyBatchedUsesEmbedInto(t *testing.T) {
	requireNativeRuntime(t)
	mk := newMTPDecodeFixture(t)
	control := mk()
	candidate := mk()
	for _, sess := range []*ArchSession{control, candidate} {
		for _, id := range mtpWordedPromptIDs() {
			if _, err := sess.stepID(id); err != nil {
				t.Fatalf("prefill stepID(%d): %v", id, err)
			}
		}
	}
	ids := []int32{4, 5, 6, 7}
	want := make([]int32, len(ids))
	if _, ok, err := control.verifyBatchedInto(ids, want); err != nil {
		t.Fatalf("control verifyBatchedInto: %v", err)
	} else if !ok {
		t.Fatal("control verifyBatchedInto ok = false")
	}

	candidate.embed = func(int32) ([]byte, error) {
		return nil, core.NewError("allocating embed path called")
	}
	candidate.embedFuncPtr = 0
	got := make([]int32, len(ids))
	if _, ok, err := candidate.verifyBatchedInto(ids, got); err != nil {
		t.Fatalf("candidate verifyBatchedInto: %v", err)
	} else if !ok {
		t.Fatal("candidate verifyBatchedInto ok = false")
	}
	if !mtpIDsEqual(got, want) {
		t.Fatalf("verifyBatchedInto embedInto greedys %v != allocating reference %v", got, want)
	}
}

func TestMTPPrefillPromptUsesEmbedInto(t *testing.T) {
	requireNativeRuntime(t)
	mk := newMTPDecodeFixture(t)
	control := mk()
	candidate := mk()
	ids := mtpWordedPromptIDs()
	want, ok, err := control.prefillMTPPrompt(ids, true)
	if err != nil {
		t.Fatalf("control prefillMTPPrompt: %v", err)
	}
	if !ok {
		t.Fatal("control prefillMTPPrompt ok = false")
	}

	candidate.embed = func(int32) ([]byte, error) {
		return nil, core.NewError("allocating embed path called")
	}
	candidate.embedFuncPtr = 0
	got, ok, err := candidate.prefillMTPPrompt(ids, true)
	if err != nil {
		t.Fatalf("candidate prefillMTPPrompt: %v", err)
	}
	if !ok {
		t.Fatal("candidate prefillMTPPrompt ok = false")
	}
	if !bytes.Equal(got, want) {
		t.Fatal("prefillMTPPrompt embedInto hidden differs from allocating reference")
	}
}

func TestMTPPrefillPromptRetainsLastHiddenNoCopy(t *testing.T) {
	requireNativeRuntime(t)
	mk := newMTPDecodeFixture(t)
	sess := mk()
	ids := mtpWordedPromptIDs()

	hidden, ok, err := sess.prefillMTPPrompt(ids, true)
	if err != nil {
		t.Fatalf("prefillMTPPrompt: %v", err)
	}
	if !ok {
		t.Fatal("prefillMTPPrompt ok = false")
	}
	if sess.retainedHiddenPinned == nil || sess.retainedHiddenPinned.buf == nil {
		t.Fatal("prefillMTPPrompt did not retain a pinned last hidden")
	}
	if len(hidden) != len(sess.retainedHiddenPinned.bytes) {
		t.Fatalf("prefillMTPPrompt hidden len = %d, want retained pinned len %d", len(hidden), len(sess.retainedHiddenPinned.bytes))
	}
	if unsafe.Pointer(&hidden[0]) != unsafe.Pointer(&sess.retainedHiddenPinned.bytes[0]) {
		t.Fatal("prefillMTPPrompt hidden does not alias retained pinned backing")
	}
	if sess.retainedHiddenBufferFor(hidden) == nil {
		t.Fatal("prefillMTPPrompt retained hidden is not exposed as a no-copy buffer")
	}
}

func TestMTPStepIDRetainsHiddenNoCopy(t *testing.T) {
	requireNativeRuntime(t)
	mk := newMTPDecodeFixture(t)
	sess := mk()

	hidden, err := sess.stepID(3)
	if err != nil {
		t.Fatalf("stepID: %v", err)
	}
	if sess.retainedHiddenPinned == nil || sess.retainedHiddenPinned.buf == nil {
		t.Fatal("stepID did not retain a pinned hidden")
	}
	if len(hidden) != len(sess.retainedHiddenPinned.bytes) {
		t.Fatalf("stepID hidden len = %d, want retained pinned len %d", len(hidden), len(sess.retainedHiddenPinned.bytes))
	}
	if unsafe.Pointer(&hidden[0]) != unsafe.Pointer(&sess.retainedHiddenPinned.bytes[0]) {
		t.Fatal("stepID hidden does not alias retained pinned backing")
	}
	if sess.retainedHiddenBufferFor(hidden) == nil {
		t.Fatal("stepID retained hidden is not exposed as a no-copy buffer")
	}
}

func TestGreedyFallbackUsesHeadLogitsScratch(t *testing.T) {
	requireNativeRuntime(t)
	mk := newMTPDecodeFixture(t)
	sess := mk()

	hidden, err := sess.stepID(3)
	if err != nil {
		t.Fatalf("stepID: %v", err)
	}
	if sess.retainedHiddenBufferFor(hidden) == nil {
		t.Fatal("test setup did not retain hidden as a no-copy buffer")
	}
	sess.greedy = nil
	sess.sampleHeadLogits = nil

	got, err := sess.greedyFromHiddenInPool(hidden, nil)
	if err != nil {
		t.Fatalf("greedyFromHiddenInPool: %v", err)
	}
	if got < 0 || int(got) >= sess.arch.Vocab {
		t.Fatalf("greedyFromHiddenInPool token = %d outside vocab %d", got, sess.arch.Vocab)
	}
	if len(sess.sampleHeadLogits) != sess.arch.Vocab*bf16Size {
		t.Fatalf("fallback logits scratch len = %d, want %d", len(sess.sampleHeadLogits), sess.arch.Vocab*bf16Size)
	}
}

func TestMTPVerifyBatchedDirectHeadAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)
	mk := newMTPDecodeFixture(t)
	dense := mk()
	for _, id := range mtpWordedPromptIDs() {
		if _, err := dense.stepID(id); err != nil {
			t.Fatalf("prefill dense stepID(%d): %v", id, err)
		}
	}
	ids := []int32{4, 5, 6, 7}
	greedys := make([]int32, len(ids))
	if _, ok, err := dense.verifyBatchedInto(ids, greedys); err != nil {
		t.Fatalf("verifyBatched warmup: %v", err)
	} else if !ok {
		t.Fatal("verifyBatched warmup ok = false")
	}

	var verifyErr error
	var verifyOK bool
	allocs := testing.AllocsPerRun(5, func() {
		_, verifyOK, verifyErr = dense.verifyBatchedInto(ids, greedys)
	})
	if verifyErr != nil {
		t.Fatalf("verifyBatched: %v", verifyErr)
	}
	if !verifyOK {
		t.Fatal("verifyBatched ok = false")
	}
	if allocs > 680 {
		t.Fatalf("verifyBatched allocations = %.0f, want <= 680", allocs)
	}
}

func TestMTPVerifyBatchedFallbackReusesPinnedHiddenRows(t *testing.T) {
	requireNativeRuntime(t)
	mk := newMTPDecodeFixture(t)
	dense := mk()
	for _, id := range mtpWordedPromptIDs() {
		if _, err := dense.stepID(id); err != nil {
			t.Fatalf("prefill dense stepID(%d): %v", id, err)
		}
	}
	dense.greedy = func(hidden []byte, suppress []int32) (int32, bool, error) {
		return dense.headEnc.greedyInPool(hidden, suppress)
	}
	if dense.canUseDirectHeadGreedy() {
		t.Fatal("test setup still has direct head greedy enabled")
	}

	ids := []int32{4, 5, 6, 7}
	if _, ok, err := dense.verifyBatchedInto(ids, make([]int32, len(ids))); err != nil {
		t.Fatalf("verifyBatched fallback: %v", err)
	} else if !ok {
		t.Fatal("verifyBatched fallback ok = false")
	}
	if dense.mtpVerifyHiddenPinned == nil || dense.mtpVerifyHiddenPinned.buf == nil {
		t.Fatal("verifyBatched fallback did not retain pinned packed hidden rows")
	}
	if len(dense.mtpVerifyHiddenRows) != len(ids) {
		t.Fatalf("verifyBatched fallback retained %d rows, want %d", len(dense.mtpVerifyHiddenRows), len(ids))
	}
	base := unsafe.Pointer(&dense.mtpVerifyHiddenPinned.bytes[0])
	rowBytes := dense.arch.Hidden * bf16Size
	for i, row := range dense.mtpVerifyHiddenRows {
		if len(row) != rowBytes {
			t.Fatalf("hidden row %d length = %d, want %d", i, len(row), rowBytes)
		}
		if unsafe.Pointer(&row[0]) != unsafe.Pointer(&dense.mtpVerifyHiddenPinned.bytes[i*rowBytes]) {
			t.Fatalf("hidden row %d does not alias the pinned packed backing at %p", i, base)
		}
	}
}

func TestMTPDecodeWordedPromptEOSMatchesGenerate(t *testing.T) {
	requireNativeRuntime(t)
	const K, maxNew = 4, 8
	prompt := mtpWordedPromptIDs()
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

func TestMTPDecodeEOSRollsBackTargetPosition(t *testing.T) {
	requireNativeRuntime(t)
	const K, maxNew = 4, 8
	prompt := mtpWordedPromptIDs()
	mk := newMTPDecodeFixture(t)

	first, err := mk().Generate(prompt, 1, -1)
	if err != nil {
		t.Fatalf("Generate first token: %v", err)
	}
	target := mk()
	draft := mk()
	res, err := MTPDecode(target, draft, prompt, maxNew, int(first[0]), K)
	if err != nil {
		t.Fatalf("MTPDecode: %v", err)
	}
	if len(res.Tokens) != 1 {
		t.Fatalf("MTP EOS emitted %d tokens, want 1", len(res.Tokens))
	}
	wantPos := len(prompt) + len(res.Tokens)
	if target.Pos() != wantPos {
		t.Fatalf("target pos after EOS = %d, want prompt+emitted %d", target.Pos(), wantPos)
	}
}

func TestMTPDecodeEOSRetainsDraftBoundaryForContinuation(t *testing.T) {
	requireNativeRuntime(t)
	const K, maxNew = 4, 8
	prompt := mtpWordedPromptIDs()
	mk := newMTPDecodeFixture(t)

	want, err := mk().Generate(prompt, 2, -1)
	if err != nil {
		t.Fatalf("Generate reference: %v", err)
	}
	target := mk()
	draft := mk()
	res, err := MTPDecode(target, draft, prompt, maxNew, int(want[0]), K)
	if err != nil {
		t.Fatalf("MTPDecode: %v", err)
	}
	if !mtpIDsEqual(res.Tokens, want[:1]) {
		t.Fatalf("MTP EOS tokens = %v, want first greedy %v", res.Tokens, want[:1])
	}
	wantPos := len(prompt) + len(res.Tokens)
	if draft.Pos() != wantPos {
		t.Fatalf("draft pos after EOS = %d, want prompt+emitted %d", draft.Pos(), wantPos)
	}
	got, err := draft.GenerateFromCache(1, -1)
	if err != nil {
		t.Fatalf("draft GenerateFromCache after MTPDecode: %v", err)
	}
	if !mtpIDsEqual(got, want[1:]) {
		t.Fatalf("draft GenerateFromCache after MTPDecode = %v, want next token %v", got, want[1:])
	}
}

func TestMTPDecodePopulatesTargetKVSnapshotTokens(t *testing.T) {
	requireNativeRuntime(t)
	const K, maxNew = 3, 4
	prompt := mtpWordedPromptIDs()
	mk := newMTPDecodeFixture(t)
	target := mk()
	draft := mk()

	res, err := MTPDecode(target, draft, prompt, maxNew, -1, K)
	if err != nil {
		t.Fatalf("MTPDecode: %v", err)
	}
	snapshot, err := target.CaptureKVWithOptions(kv.CaptureOptions{RawKVOnly: true})
	if err != nil {
		t.Fatalf("CaptureKVWithOptions after MTPDecode: %v", err)
	}
	want := append(append([]int32(nil), prompt...), res.Tokens...)
	if !mtpIDsEqual(snapshot.Tokens, want) {
		t.Fatalf("snapshot tokens after MTPDecode = %v, want %v", snapshot.Tokens, want)
	}
	if snapshot.TokenOffset != len(want) {
		t.Fatalf("snapshot token offset = %d, want %d", snapshot.TokenOffset, len(want))
	}
}

func TestMTPDecodeMaxNewRetainsBoundaryForContinuation(t *testing.T) {
	requireNativeRuntime(t)
	const K, maxNew = 4, 2
	prompt := mtpWordedPromptIDs()
	mk := newMTPDecodeFixture(t)

	want, err := mk().Generate(prompt, maxNew+1, -1)
	if err != nil {
		t.Fatalf("Generate reference: %v", err)
	}
	target := mk()
	draft := mk()
	res, err := MTPDecode(target, draft, prompt, maxNew, -1, K)
	if err != nil {
		t.Fatalf("MTPDecode: %v", err)
	}
	if !mtpIDsEqual(res.Tokens, want[:maxNew]) {
		t.Fatalf("MTP tokens = %v, want prefix %v", res.Tokens, want[:maxNew])
	}
	got, err := target.GenerateFromCache(1, -1)
	if err != nil {
		t.Fatalf("GenerateFromCache after MTPDecode: %v", err)
	}
	if !mtpIDsEqual(got, want[maxNew:]) {
		t.Fatalf("GenerateFromCache after MTPDecode = %v, want next token %v", got, want[maxNew:])
	}
}

func TestMTPDecodeDensePromptPrefillAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)
	const K, maxNew = 4, 1
	prompt := mtpWordedPromptIDs()
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
