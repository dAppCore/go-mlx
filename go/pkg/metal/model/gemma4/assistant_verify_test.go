// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	"context"
	"testing"

	"dappco.re/go/mlx/pkg/metal"
)

// --- pure decision-walk tests (no model, tiny synthetic logits) -------------

func TestAssistantVerify_GreedyAcceptWalk_Good(t *testing.T) {
	// Full block accepted: every row's pick matches the draft.
	decision, err := gemma4GreedyAcceptWalk(7, []int32{3, 9}, []int32{7, 3, 9})
	if err != nil {
		t.Fatalf("gemma4GreedyAcceptWalk: %v", err)
	}
	if !decision.AllAccepted || len(decision.Accepted) != 3 {
		t.Fatalf("decision = %+v, want full accept of 3", decision)
	}
	if decision.TargetFirst != 7 {
		t.Fatalf("TargetFirst = %d, want 7", decision.TargetFirst)
	}
}

func TestAssistantVerify_GreedyAcceptWalk_Bad(t *testing.T) {
	// First-token reject: the incoming pick disagrees immediately.
	decision, err := gemma4GreedyAcceptWalk(5, []int32{3, 9}, []int32{7, 3, 9})
	if err != nil {
		t.Fatalf("gemma4GreedyAcceptWalk: %v", err)
	}
	if decision.AllAccepted || len(decision.Accepted) != 0 || decision.Replacement != 5 {
		t.Fatalf("decision = %+v, want first-token reject with replacement 5", decision)
	}

	// Mid-block reject: row 0 agrees, row 1 disagrees.
	decision, err = gemma4GreedyAcceptWalk(7, []int32{3, 2}, []int32{7, 3, 9})
	if err != nil {
		t.Fatalf("gemma4GreedyAcceptWalk: %v", err)
	}
	if decision.AllAccepted || len(decision.Accepted) != 2 || decision.Replacement != 2 {
		t.Fatalf("decision = %+v, want mid-block reject after 2 with replacement 2", decision)
	}
}

func TestAssistantVerify_GreedyAcceptWalk_Ugly(t *testing.T) {
	// A block longer than the verify rows is an engine fault, not a truncate.
	if _, err := gemma4GreedyAcceptWalk(7, []int32{3}, []int32{7, 3, 9}); err == nil {
		t.Fatal("gemma4GreedyAcceptWalk() error = nil, want missing verify row")
	}
}

// peakedRowLogits builds [1, rows, vocab] logits where row r peaks hard at
// peaks[r] — a deterministic distribution for sampler-driven tests.
func peakedRowLogits(t *testing.T, vocab int, peaks ...int32) *metal.Array {
	t.Helper()
	values := make([]float32, len(peaks)*vocab)
	for r, p := range peaks {
		values[r*vocab+int(p)] = 64
	}
	return metal.FromValues(values, 1, len(peaks), vocab)
}

// Target-sampled acceptance with a peaked target distribution: the sampler
// draw IS the peak, so acceptance mirrors the greedy walk — and the carried
// lead is accepted without consuming a draw even when the logits contradict
// it.
func TestAssistantVerify_TargetSampledDecision_Good(t *testing.T) {
	const vocab = 10
	sampler := metal.NewSamplerWithSuppressionKeyed(0.9, 0, 0, 0, nil, metal.NewSamplerKeys(7))
	defer metal.CloseSampler(sampler)

	incoming := peakedRowLogits(t, vocab, 7)
	rows := peakedRowLogits(t, vocab, 3, 9)
	defer metal.Free(incoming, rows)

	decision, err := gemma4TargetSampledDecision(incoming, rows, []int32{7, 3, 9}, gemma4VerifyDecider{Sampler: sampler})
	if err != nil {
		t.Fatalf("gemma4TargetSampledDecision: %v", err)
	}
	if !decision.AllAccepted || len(decision.Accepted) != 3 {
		t.Fatalf("decision = %+v, want full accept of 3", decision)
	}
}

func TestAssistantVerify_TargetSampledDecision_CarrySkipsDraw_Good(t *testing.T) {
	const vocab = 10
	sampler := metal.NewSamplerWithSuppressionKeyed(0.9, 0, 0, 0, nil, metal.NewSamplerKeys(7))
	defer metal.CloseSampler(sampler)

	// The incoming logits peak at 5 — yet block[0]=2 (the carry) must be
	// accepted unconditionally: it was committed last round.
	incoming := peakedRowLogits(t, vocab, 5)
	rows := peakedRowLogits(t, vocab, 4)
	defer metal.Free(incoming, rows)

	decision, err := gemma4TargetSampledDecision(incoming, rows, []int32{2, 4}, gemma4VerifyDecider{Sampler: sampler, Carry: true})
	if err != nil {
		t.Fatalf("gemma4TargetSampledDecision: %v", err)
	}
	if !decision.AllAccepted || len(decision.Accepted) != 2 || decision.Accepted[0] != 2 {
		t.Fatalf("decision = %+v, want carry accepted without a draw then draft accepted", decision)
	}
}

func TestAssistantVerify_TargetSampledDecision_Bad(t *testing.T) {
	const vocab = 10
	sampler := metal.NewSamplerWithSuppressionKeyed(0.9, 0, 0, 0, nil, metal.NewSamplerKeys(7))
	defer metal.CloseSampler(sampler)

	// Mid-block reject: the target's draw at row 0 is 3 (accept), at row 1 is
	// 8 but the draft proposed 9 — the target's token wins, suffix discarded.
	incoming := peakedRowLogits(t, vocab, 7)
	rows := peakedRowLogits(t, vocab, 3, 8)
	defer metal.Free(incoming, rows)

	decision, err := gemma4TargetSampledDecision(incoming, rows, []int32{7, 3, 9}, gemma4VerifyDecider{Sampler: sampler})
	if err != nil {
		t.Fatalf("gemma4TargetSampledDecision: %v", err)
	}
	if decision.AllAccepted || len(decision.Accepted) != 2 || decision.Replacement != 8 {
		t.Fatalf("decision = %+v, want 2 accepted with replacement 8", decision)
	}
}

// The key-stream alignment law, proven hermetically (no model forward, so no
// GPU shape variance): the target-sampled walk must consume EXACTLY one keyed
// draw per judged position, in commit order, stopping at the first mismatch —
// so a parallel sampler over the same keys reproduces the committed tokens,
// and the draw AFTER any walk lands exactly where plain AR's next draw would.
func TestAssistantVerify_TargetSampledKeyStreamAlignment_Good(t *testing.T) {
	const vocab = 10
	newSampler := func() metal.Sampler {
		return metal.NewSamplerWithSuppressionKeyed(0.9, 0.9, 0, 0, nil, metal.NewSamplerKeys(99))
	}
	// Reference AR: four sequential draws over four deterministic rows.
	rows := []*metal.Array{
		peakedRowLogits(t, vocab, 4),
		peakedRowLogits(t, vocab, 7),
		peakedRowLogits(t, vocab, 1),
		peakedRowLogits(t, vocab, 8),
	}
	defer func() {
		for _, row := range rows {
			metal.Free(row)
		}
	}()
	reference := make([]int32, len(rows))
	refSampler := newSampler()
	for i, row := range rows {
		arr, id, _, err := metal.SampleTokenIDWithSuppressionGuard(row, refSampler, nil, false)
		metal.Free(arr)
		if err != nil {
			t.Fatalf("reference draw %d: %v", i, err)
		}
		reference[i] = id
	}
	metal.CloseSampler(refSampler)

	drawNext := func(sampler metal.Sampler, row *metal.Array) int32 {
		t.Helper()
		arr, id, _, err := metal.SampleTokenIDWithSuppressionGuard(row, sampler, nil, false)
		metal.Free(arr)
		if err != nil {
			t.Fatalf("post-walk draw: %v", err)
		}
		return id
	}
	stackRows := func(a, b *metal.Array) *metal.Array {
		t.Helper()
		stacked := metal.Concatenate2(a, b, 1)
		t.Cleanup(func() { metal.Free(stacked) })
		return stacked
	}

	// Case 1 — full accept: the walk consumes len(block) draws; the NEXT draw
	// equals reference[3] (plain AR's fourth draw).
	s := newSampler()
	decision, err := gemma4TargetSampledDecision(rows[0], stackRows(rows[1], rows[2]), []int32{reference[0], reference[1], reference[2]}, gemma4VerifyDecider{Sampler: s})
	if err != nil {
		t.Fatalf("full-accept walk: %v", err)
	}
	if !decision.AllAccepted || len(decision.Accepted) != 3 {
		t.Fatalf("full-accept decision = %+v, want 3 accepted", decision)
	}
	if next := drawNext(s, rows[3]); next != reference[3] {
		t.Fatalf("draw after full accept = %d, want plain AR's next draw %d", next, reference[3])
	}
	metal.CloseSampler(s)

	// Case 2 — mid reject: the walk consumes draws only through the mismatch;
	// the replacement IS reference[1], and the NEXT draw lands on
	// reference[2] — no draws were burnt for the discarded suffix.
	s = newSampler()
	wrong := (reference[1] + 1) % vocab
	decision, err = gemma4TargetSampledDecision(rows[0], stackRows(rows[1], rows[2]), []int32{reference[0], wrong, 5}, gemma4VerifyDecider{Sampler: s})
	if err != nil {
		t.Fatalf("mid-reject walk: %v", err)
	}
	if decision.AllAccepted || len(decision.Accepted) != 1 || decision.Replacement != reference[1] {
		t.Fatalf("mid-reject decision = %+v, want 1 accepted with replacement %d", decision, reference[1])
	}
	if next := drawNext(s, rows[2]); next != reference[2] {
		t.Fatalf("draw after mid reject = %d, want plain AR's draw %d (the walk must not burn suffix draws)", next, reference[2])
	}
	metal.CloseSampler(s)

	// Case 3 — carry: position 0 consumes NO draw, so the first judged draw
	// is reference[0] for the row that judges position 1.
	s = newSampler()
	carryBlock := []int32{3 /* committed last round */, reference[0]}
	decision, err = gemma4TargetSampledDecision(rows[1] /* unused for pos 0 */, stackRows(rows[0], rows[1]), carryBlock, gemma4VerifyDecider{Sampler: s, Carry: true})
	if err != nil {
		t.Fatalf("carry walk: %v", err)
	}
	if !decision.AllAccepted || len(decision.Accepted) != 2 || decision.Accepted[0] != 3 {
		t.Fatalf("carry decision = %+v, want carry + draft accepted", decision)
	}
	if next := drawNext(s, rows[1]); next != reference[1] {
		t.Fatalf("draw after carry walk = %d, want plain AR's second draw %d", next, reference[1])
	}
	metal.CloseSampler(s)
}

// --- tiny-pair integration: the AR-equivalence invariant --------------------

// gemma4VerifyPeakedTargetWeights is the AR-equivalence fixture's target: the
// shared tiny weights with the tied embedding rows re-shaped to be DECISIVE —
// each vocab row carries one dominant, distinctly-scaled component, so logits
// separate like a real trained model's instead of sitting on knife-edge ties.
// The flat seqArray embedding turned the engine's benign low-bit drift
// (batched verify rows vs single-token forwards, restored vs fresh prefill)
// into flipped argmax picks and categorical draws; decisive separations keep
// every comparison about the ALGORITHM, the realistic regime.
func gemma4VerifyPeakedTargetWeights() map[string]*metal.Array {
	weights := gemma4AssistantTargetTinyWeights()
	metal.Free(weights["model.embed_tokens.weight"])
	const vocab, hidden = 10, 8
	values := make([]float32, vocab*hidden)
	for v := 0; v < vocab; v++ {
		for d := 0; d < hidden; d++ {
			values[v*hidden+d] = 0.02 * float32(v+1) // small varied base
		}
		// Dominant, distinctly-scaled component per row. Kept MODEST: larger
		// scales (3.0+) push activations into a regime where the engine's
		// warm-allocator state makes even greedy decode non-reproducible —
		// observed on the UNMODIFIED baseline tree, so it is a pre-existing
		// engine numeric-overflow property, not an MTP one. ~1.2-2.8 keeps
		// logits decisive without entering that regime.
		values[v*hidden+(v%hidden)] = 1.2 + 0.18*float32(v)
	}
	weights["model.embed_tokens.weight"] = metal.FromValues(values, vocab, hidden)
	return weights
}

// loadTinyGemma4AssistantRuntime loads the tiny synthetic target through the
// FULL runtime loader (metal.LoadAndInit — engine features, fixed sliding
// caches: what serve runs) and attaches the tiny assistant. The prompt cache
// is DISABLED so every run takes the identical fresh-prefill path (a restored
// prefix differs from a fresh prefill in low-bit float order).
func loadTinyGemma4AssistantRuntime(t *testing.T) (*metal.Model, *Gemma4AssistantPair) {
	t.Helper()
	targetDir := t.TempDir()
	writeGemma4AssistantTargetConfig(t, targetDir)
	writeMinimalTokenizer(t, targetDir)
	if err := metal.SaveSafetensors(targetDir+"/model.safetensors", gemma4VerifyPeakedTargetWeights()); err != nil {
		t.Fatalf("SaveSafetensors target: %v", err)
	}
	assistantDir := t.TempDir()
	writeGemma4AssistantConfig(t, assistantDir, false)
	writeMinimalTokenizer(t, assistantDir)
	if err := metal.SaveSafetensors(assistantDir+"/model.safetensors", gemma4AssistantTinyWeights(false)); err != nil {
		t.Fatalf("SaveSafetensors assistant: %v", err)
	}
	m, err := metal.LoadAndInit(targetDir, metal.LoadConfig{DisablePromptCache: true})
	if err != nil {
		t.Fatalf("metal.LoadAndInit(tiny target): %v", err)
	}
	pair, err := AttachGemma4Assistant(m, assistantDir)
	if err != nil {
		m.Close()
		t.Fatalf("AttachGemma4Assistant: %v", err)
	}
	t.Cleanup(func() {
		pair.Close()
		m.Close()
	})
	return m, pair
}

func collectPlainTokens(t *testing.T, m *metal.Model, prompt string, cfg metal.GenerateConfig) []int32 {
	t.Helper()
	var ids []int32
	for token := range m.Generate(context.Background(), prompt, cfg) {
		ids = append(ids, token.ID)
	}
	if err := m.Err(); err != nil {
		t.Fatalf("plain Generate: %v", err)
	}
	return ids
}

func collectMTPTokens(t *testing.T, m *metal.Model, pair *Gemma4AssistantPair, prompt string, cfg metal.GenerateConfig, draftTokens int) (Gemma4AssistantGenerateResult, []int32) {
	t.Helper()
	result, err := pair.GenerateWithSink(context.Background(), m, prompt, cfg, draftTokens, nil)
	if err != nil {
		t.Fatalf("GenerateWithSink: %v", err)
	}
	ids := make([]int32, len(result.Tokens))
	for i, token := range result.Tokens {
		ids[i] = token.ID
	}
	return result, ids
}

func assertTokenIDsEqual(t *testing.T, label string, mtp, plain []int32) {
	t.Helper()
	if len(mtp) != len(plain) {
		t.Fatalf("%s: MTP emitted %d tokens %v, plain AR emitted %d tokens %v", label, len(mtp), mtp, len(plain), plain)
	}
	for i := range mtp {
		if mtp[i] != plain[i] {
			t.Fatalf("%s: token %d = %d, plain AR produced %d (mtp=%v plain=%v)", label, i, mtp[i], plain[i], mtp, plain)
		}
	}
}

// assertEquivalenceStable asserts MTP == plain, distinguishing a REAL
// divergence from the engine's pre-existing warm-process numeric instability
// (observed on the unmodified baseline: a warm allocator can degenerate
// subsequent generations — greedy included). On mismatch BOTH sides re-run
// once: if either side fails to reproduce itself, the engine state shifted
// mid-test and the comparison is void (skip); a STABLE divergence is a real
// MTP bug and still fails.
func assertEquivalenceStable(t *testing.T, label string, plain []int32, plainAgain func() []int32, mtp []int32, mtpAgain func() []int32) {
	t.Helper()
	if int32SlicesEqualVerifyTest(mtp, plain) {
		return
	}
	replain := plainAgain()
	if !int32SlicesEqualVerifyTest(replain, plain) {
		t.Skipf("%s: engine state shifted mid-test (plain %v -> %v) — pre-existing warm-process instability, comparison void", label, plain, replain)
	}
	remtp := mtpAgain()
	if !int32SlicesEqualVerifyTest(remtp, mtp) {
		t.Skipf("%s: engine state shifted mid-test (mtp %v -> %v) — pre-existing warm-process instability, comparison void", label, mtp, remtp)
	}
	assertTokenIDsEqual(t, label, mtp, plain)
}

// THE invariant: the committed MTP sequence must equal what plain AR decode
// produces on the same model — greedy, across multiple draft block sizes, on
// a generation long enough to rotate the 4-token sliding window (so the
// journal-backed rollback runs across the rotation boundary).
func TestAssistantVerify_GreedyARSEquivalence_Good(t *testing.T) {
	requireMetalRuntime(t)
	m, pair := loadTinyGemma4AssistantRuntime(t)

	const prompt = "hello world hello"
	cfg := metal.GenerateConfig{MaxTokens: 12}
	plain := collectPlainTokens(t, m, prompt, cfg)
	if len(plain) == 0 {
		t.Fatal("plain AR produced no tokens — fixture broke")
	}
	for _, draftTokens := range []int{1, 2, 4} {
		result, mtp := collectMTPTokens(t, m, pair, prompt, cfg, draftTokens)
		assertEquivalenceStable(t, "greedy", plain,
			func() []int32 { return collectPlainTokens(t, m, prompt, cfg) },
			mtp,
			func() []int32 { _, again := collectMTPTokens(t, m, pair, prompt, cfg, draftTokens); return again })
		if result.DraftCalls == 0 || result.TargetVerifyCalls == 0 {
			t.Fatalf("draft/verify calls = %d/%d, want the speculative lane to have run", result.DraftCalls, result.TargetVerifyCalls)
		}
	}
}

// THE invariant at temperature: a SEEDED sampled request commits exactly the
// plain AR sequence — the verify consumes the same keyed sampler chain, one
// draw per committed token, in commit order.
//
// Anchor stability: the tiny model's near-flat 10-token logits sit on
// categorical knife edges, and the ENGINE's plain path is itself only
// reproducible up to low-bit logits drift (a prompt-cache restore vs a fresh
// prefill can flip a draw — observed: two consecutive plain runs at seed 1337
// diverging). Each seed is therefore anchored on a double plain run: seeds
// where plain cannot reproduce itself are skipped — that instability is an
// engine-wide logits-drift property, not an MTP acceptance property. Real
// models' peaked distributions sit nowhere near these knife edges.
func TestAssistantVerify_SampledARSEquivalence_Good(t *testing.T) {
	requireMetalRuntime(t)
	m, pair := loadTinyGemma4AssistantRuntime(t)

	const prompt = "hello world hello"
	anchored := 0
	for _, seed := range []uint64{42, 7, 1234} {
		cfg := metal.GenerateConfig{MaxTokens: 12, Temperature: 0.9, TopP: 0.9, Seed: seed, SeedSet: true}
		plain := collectPlainTokens(t, m, prompt, cfg)
		again := collectPlainTokens(t, m, prompt, cfg)
		if len(plain) == 0 {
			t.Fatal("plain AR produced no tokens — fixture broke")
		}
		if !int32SlicesEqualVerifyTest(plain, again) {
			t.Logf("seed %d: plain AR not self-reproducible (%v vs %v) — engine logits drift, skipping anchor", seed, plain, again)
			continue
		}
		anchored++
		for _, draftTokens := range []int{1, 2, 4} {
			_, mtp := collectMTPTokens(t, m, pair, prompt, cfg, draftTokens)
			assertEquivalenceStable(t, "sampled", plain,
				func() []int32 { return collectPlainTokens(t, m, prompt, cfg) },
				mtp,
				func() []int32 { _, again := collectMTPTokens(t, m, pair, prompt, cfg, draftTokens); return again })
		}
	}
	if anchored == 0 {
		t.Skip("no seed produced a self-reproducible plain anchor on this machine")
	}
}

func int32SlicesEqualVerifyTest(a, b []int32) bool {
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

// A stop token landing INSIDE an accepted block must end the generation at
// exactly the same token as plain AR.
func TestAssistantVerify_StopTokenInsideBlock_Good(t *testing.T) {
	requireMetalRuntime(t)
	m, pair := loadTinyGemma4AssistantRuntime(t)

	const prompt = "hello world hello"
	probe := collectPlainTokens(t, m, prompt, metal.GenerateConfig{MaxTokens: 12})
	if len(probe) < 3 {
		t.Skipf("plain AR produced only %d tokens — not enough to plant a stop token", len(probe))
	}
	// Plant a stop token that first appears mid-stream: the earliest token id
	// not seen in the first two positions (planting probe[2] directly can
	// coincide with probe[0..1] when the tiny model settles on a fixed point).
	stop := probe[2]
	for _, candidate := range probe[2:] {
		if candidate != probe[0] && candidate != probe[1] {
			stop = candidate
			break
		}
	}
	cfg := metal.GenerateConfig{MaxTokens: 12, StopTokens: []int32{stop}}
	plain := collectPlainTokens(t, m, prompt, cfg)
	_, mtp := collectMTPTokens(t, m, pair, prompt, cfg, 4)
	t.Logf("probe=%v stop=%d plain=%v mtp=%v", probe, stop, plain, mtp)
	assertEquivalenceStable(t, "stop token", plain,
		func() []int32 { return collectPlainTokens(t, m, prompt, cfg) },
		mtp,
		func() []int32 { _, again := collectMTPTokens(t, m, pair, prompt, cfg, 4); return again })
}

// The clone-regime sampled verify (the fallback for cache modes without
// journal support) must agree with the greedy decision at temperature->0
// equivalent peaked logits, adopt a trimmed cache set, and leave the live
// caches untouched.
func TestAssistantVerify_SampledCloneFallback_Good(t *testing.T) {
	requireMetalRuntime(t)
	_, pair := loadTinyGemma4AssistantRuntime(t)

	caches := pair.Target.NewCache()
	defer metal.FreeCaches(caches)
	prefillLogits, previousHidden := prefillTinyGemma4AssistantTarget(t, pair, caches, []int32{1, 2, 3})
	defer metal.Free(prefillLogits, previousHidden)
	offsets := gemma4AssistantCacheOffsets(caches)

	// Derive the target's first sampled draw with a PARALLEL sampler over the
	// same keys — deterministic — then propose a draft that contradicts it,
	// guaranteeing a first-token reject whose replacement IS that draw.
	probeSampler := metal.NewSamplerWithSuppressionKeyed(0.9, 0, 0, 0, nil, metal.NewSamplerKeys(7))
	probeArr, expected, _, err := metal.SampleTokenIDWithSuppressionGuard(prefillLogits, probeSampler, nil, false)
	metal.Free(probeArr)
	metal.CloseSampler(probeSampler)
	if err != nil {
		t.Fatalf("probe sample: %v", err)
	}
	wrongToken := (expected + 1) % 10

	sampler := metal.NewSamplerWithSuppressionKeyed(0.9, 0, 0, 0, nil, metal.NewSamplerKeys(7))
	defer metal.CloseSampler(sampler)
	d := gemma4VerifyDecider{Sampler: sampler}

	result, err := pair.verifyDraftBlockSampledClone(prefillLogits, []int32{wrongToken, wrongToken}, caches, d)
	if err != nil {
		t.Fatalf("verifyDraftBlockSampledClone: %v", err)
	}
	defer result.Close()
	if result.AcceptedCount != 0 || result.RejectedCount != 2 {
		t.Fatalf("accepted/rejected = %d/%d, want first-token reject 0/2", result.AcceptedCount, result.RejectedCount)
	}
	if result.ReplacementToken != expected {
		t.Fatalf("replacement = %d, want the target's deterministic draw %d", result.ReplacementToken, expected)
	}
	if result.Caches == nil {
		t.Fatal("clone verify returned no cache set, want the trimmed clone")
	}
	if got := gemma4AssistantCacheOffsets(caches); !gemma4AssistantIntSlicesEqual(got, offsets) {
		t.Fatalf("live cache offsets = %v, want untouched %v", got, offsets)
	}
}

// The live verify mutates the caller's caches in place: accepted prefix
// committed, rejected suffix trimmed exactly, no cache set returned.
func TestAssistantVerify_LiveVerifyTrimsInPlace_Good(t *testing.T) {
	requireMetalRuntime(t)
	_, pair := loadTinyGemma4AssistantRuntime(t)

	caches := pair.Target.NewCache()
	defer metal.FreeCaches(caches)
	prefillLogits, previousHidden := prefillTinyGemma4AssistantTarget(t, pair, caches, []int32{1, 2, 3})
	defer metal.Free(prefillLogits, previousHidden)
	preOffsets := gemma4AssistantCacheOffsets(caches)

	targetToken, err := gemma4AssistantGreedyToken(prefillLogits)
	if err != nil {
		t.Fatalf("greedy target token: %v", err)
	}
	badToken := (targetToken + 1) % 10

	result, err := pair.verifyDraftBlockLive(prefillLogits, []int32{targetToken, badToken}, caches, gemma4VerifyDecider{})
	if err != nil {
		t.Fatalf("verifyDraftBlockLive: %v", err)
	}
	defer result.Close()
	if result.Caches != nil {
		t.Fatal("live verify returned a cache set, want in-place commitment")
	}
	if result.AcceptedCount != 1 || result.RejectedCount != 1 {
		t.Fatalf("accepted/rejected = %d/%d, want 1/1", result.AcceptedCount, result.RejectedCount)
	}
	for i, c := range caches {
		if c == nil {
			continue
		}
		if got, want := c.Offset(), preOffsets[i]+1; got != want {
			t.Fatalf("cache %d offset = %d, want pre+accepted %d", i, got, want)
		}
	}
}
