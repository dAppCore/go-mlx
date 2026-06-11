// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"math"
	"runtime"
	"testing"
	"unsafe"
)

func TestSample_Greedy_Good(t *testing.T) {
	// Logits heavily favour index 2
	logits := FromValues([]float32{-10, -10, 100, -10}, 1, 4)
	s := newSampler(0, 0, 0, 0) // temp=0 → Greedy
	token := s.Sample(logits)
	Materialize(token)

	if token.Int() != 2 {
		t.Errorf("Greedy sample = %d, want 2", token.Int())
	}
}

func TestSample_Temperature_HighTemp_Good(t *testing.T) {
	// High temperature should still produce a valid index
	logits := FromValues([]float32{1, 2, 3, 4}, 1, 4)
	s := newSampler(100.0, 0, 0, 0) // very high temp → near uniform
	token := s.Sample(logits)
	Materialize(token)

	idx := token.Int()
	if idx < 0 || idx >= 4 {
		t.Errorf("sample index = %d, out of range [0, 4)", idx)
	}
}

func TestSample_Temperature_LowTemp_Good(t *testing.T) {
	// Very low temperature should behave like Greedy
	logits := FromValues([]float32{-10, -10, 100, -10}, 1, 4)
	s := newSampler(0.001, 0, 0, 0) // near-zero temp → near-Greedy
	token := s.Sample(logits)
	Materialize(token)

	if token.Int() != 2 {
		t.Errorf("low-temp sample = %d, want 2 (near Greedy)", token.Int())
	}
}

func TestSample_TopKSampler_Good(t *testing.T) {
	// TopK=1 with clear winner should always pick that token
	logits := FromValues([]float32{-100, 100, -100, -100}, 1, 4)
	s := newSampler(1.0, 0, 0, 1) // topK=1
	token := s.Sample(logits)
	Materialize(token)

	if token.Int() != 1 {
		t.Errorf("topk=1 sample = %d, want 1", token.Int())
	}
}

func TestSample_TopKSampler_MultipleTokens_Good(t *testing.T) {
	// TopK=2, both high logits — should pick one of them
	logits := FromValues([]float32{-100, 50, 50, -100}, 1, 4)
	s := newSampler(1.0, 0, 0, 2) // topK=2

	seen := map[int]bool{}
	for range 20 {
		token := s.Sample(logits)
		Materialize(token)
		seen[token.Int()] = true
	}

	// Should only ever pick index 1 or 2
	for idx := range seen {
		if idx != 1 && idx != 2 {
			t.Errorf("topk=2 sampled index %d, expected only 1 or 2", idx)
		}
	}
}

func TestSample_TopKSampler_OverLargeK_NoOp_Good(t *testing.T) {
	logits := FromValues([]float32{1, 2, 3, 4}, 1, 4)
	filtered := TopKSampler(99).Sample(logits)
	Materialize(filtered)

	got := filtered.Floats()
	want := []float32{1, 2, 3, 4}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("filtered[%d] = %f, want %f", i, got[i], want[i])
		}
	}
}

func TestSample_TopKSampler_NonPositiveK_NoOp_Good(t *testing.T) {
	logits := FromValues([]float32{1, 2, 3, 4}, 1, 4)
	filtered := TopKSampler(0).Sample(logits)
	Materialize(filtered)

	got := filtered.Floats()
	want := []float32{1, 2, 3, 4}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("filtered[%d] = %f, want %f", i, got[i], want[i])
		}
	}
}

func TestSample_SuppressTokenLogits_Good(t *testing.T) {
	logits := FromValues([]float32{100, 1, 2, 3}, 1, 4)
	filtered := suppressTokenLogits(logits, []int32{0})
	defer Free(logits, filtered)
	if err := Eval(filtered); err != nil {
		t.Fatalf("Eval(suppressTokenLogits) error = %v", err)
	}
	got := filtered.Floats()
	if got[0] >= got[3] {
		t.Fatalf("suppressed logits = %v, want token 0 below token 3", got)
	}
}

func TestSample_SuppressTokenLogits3D_Good(t *testing.T) {
	logits := FromValues([]float32{0.1, 9, 3, 2}, 1, 1, 4)
	filtered := suppressTokenLogits(logits, []int32{1})
	defer Free(logits, filtered)
	if err := Eval(filtered); err != nil {
		t.Fatalf("Eval(suppressTokenLogits) error = %v", err)
	}
	got := filtered.Floats()
	if got[1] >= got[2] {
		t.Fatalf("suppressed 3D logits = %v, want token 1 below token 2", got)
	}
}

func TestSample_SuppressTokenLogitsThenTopK_Good(t *testing.T) {
	logits := FromValues([]float32{100, 1, 2, 3}, 1, 4)
	filtered := suppressTokenLogits(logits, []int32{0})
	defer Free(logits, filtered)
	s := newSampler(1.0, 0, 0, 1)
	token := s.Sample(filtered)
	defer Free(token)
	if err := Eval(token); err != nil {
		t.Fatalf("Eval(sample) error = %v", err)
	}
	if token.Int() == 0 {
		t.Fatal("sampled suppressed token 0")
	}
}

func TestSample_SuppressTokenLogitsThenTopPTopK_Good(t *testing.T) {
	logits := FromValues([]float32{100, 1, 2, 3}, 1, 4)
	filtered := suppressTokenLogits(logits, []int32{0})
	defer Free(logits, filtered)
	s := newSampler(1.0, 0.95, 0, 3)
	for range 10 {
		token := s.Sample(filtered)
		if err := Eval(token); err != nil {
			Free(token)
			t.Fatalf("Eval(sample) error = %v", err)
		}
		got := token.Int()
		Free(token)
		if got == 0 {
			t.Fatal("sampled suppressed token 0")
		}
	}
}

func TestSample_NewSamplerWithSuppression_Good(t *testing.T) {
	logits := FromValues([]float32{100, 1, 2, 3}, 1, 4)
	defer Free(logits)
	s := NewSamplerWithSuppression(1.0, 0.95, 0, 3, []int32{0})
	defer CloseSampler(s)
	for range 10 {
		token := s.Sample(logits)
		if err := Eval(token); err != nil {
			Free(token)
			t.Fatalf("Eval(sample) error = %v", err)
		}
		got := token.Int()
		Free(token)
		if got == 0 {
			t.Fatal("sampled suppressed token 0")
		}
	}
}

func TestSample_TopKTopPChainMapsGlobalToken_Good(t *testing.T) {
	logits := FromValues([]float32{0, 1, 100, 80}, 1, 4)
	defer Free(logits)
	s := newSampler(1.0, 0.5, 0, 2)
	token := s.Sample(logits)
	defer Free(token)
	if err := Eval(token); err != nil {
		t.Fatalf("Eval(sample) error = %v", err)
	}
	if got := token.Int(); got != 2 {
		t.Fatalf("sample = %d, want global token 2", got)
	}
}

type fixedTokenSampler struct {
	id int32
}

func (s fixedTokenSampler) Sample(logits *Array) *Array {
	return FromValues([]int32{s.id}, 1)
}

func TestSample_SuppressionGuardFallsBackBeforeAppend_Good(t *testing.T) {
	logits := FromValues([]float32{100, 1, 2, 3}, 1, 4)
	defer Free(logits)

	token, err := SampleTokenWithSuppressionGuard(logits, fixedTokenSampler{id: 0}, []int32{0})
	if err != nil {
		t.Fatalf("suppression guard: %v", err)
	}
	defer Free(token)
	if got := int32(token.Int()); got == 0 {
		t.Fatalf("suppression guard token = %d, want non-suppressed fallback", got)
	}
}

func TestSample_SuppressionGuardGemmaSizedIDs_Good(t *testing.T) {
	values := make([]float32, 258885)
	values[0] = 100
	values[123] = 10
	logits := FromValues(values, 1, len(values))
	defer Free(logits)
	suppressTokens := []int32{0, 2, 3, 4, 46, 47, 48, 49, 50, 51, 52, 98, 100, 101, 105, 255999, 256000, 258880, 258881, 258882, 258883, 258884}

	token, err := SampleTokenWithSuppressionGuard(logits, fixedTokenSampler{id: 0}, suppressTokens)
	if err != nil {
		t.Fatalf("suppression guard: %v", err)
	}
	defer Free(token)
	if got := int32(token.Int()); got == 0 || TokenIDSuppressed(got, suppressTokens) {
		t.Fatalf("suppression guard token = %d, want non-suppressed Gemma-sized fallback", got)
	}
}

func TestSample_SuppressionGuardGemmaSizedBFloat16IDs_Good(t *testing.T) {
	values := make([]float32, 258885)
	values[0] = 100
	values[123] = 10
	base := FromValues(values, 1, len(values))
	logits := AsType(base, DTypeBFloat16)
	defer Free(base, logits)
	suppressTokens := []int32{0, 2, 3, 4, 46, 47, 48, 49, 50, 51, 52, 98, 100, 101, 105, 255999, 256000, 258880, 258881, 258882, 258883, 258884}

	token, err := SampleTokenWithSuppressionGuard(logits, fixedTokenSampler{id: 0}, suppressTokens)
	if err != nil {
		t.Fatalf("suppression guard: %v", err)
	}
	defer Free(token)
	if got := int32(token.Int()); got != 123 {
		t.Fatalf("suppression guard token = %d, want 123", got)
	}
}

func TestSample_SuppressionGuardLastTokenView_Good(t *testing.T) {
	values := make([]float32, 2*258885)
	values[258885] = 100
	values[258885+123] = 10
	base := FromValues(values, 1, 2, 258885)
	logits := AsType(base, DTypeBFloat16)
	last, err := lastTokenLogits(logits)
	if err != nil {
		t.Fatalf("lastTokenLogits: %v", err)
	}
	defer Free(base, logits, last)
	suppressTokens := []int32{0, 2, 3, 4, 46, 47, 48, 49, 50, 51, 52, 98, 100, 101, 105, 255999, 256000, 258880, 258881, 258882, 258883, 258884}

	token, err := SampleTokenWithSuppressionGuard(last, fixedTokenSampler{id: 0}, suppressTokens)
	if err != nil {
		t.Fatalf("suppression guard: %v", err)
	}
	defer Free(token)
	if got := int32(token.Int()); got != 123 {
		t.Fatalf("suppression guard token = %d, want 123", got)
	}
}

func TestSample_HostUnsuppressedGreedyTokenSkipsSuppressedAndNaN_Good(t *testing.T) {
	logits := FromValues([]float32{100, float32(math.NaN()), 9, 11}, 1, 4)
	defer Free(logits)

	token, err := hostUnsuppressedGreedyToken(logits, []int32{0})
	if err != nil {
		t.Fatalf("hostUnsuppressedGreedyToken: %v", err)
	}
	defer Free(token)
	if got := int32(token.Int()); got != 3 {
		t.Fatalf("hostUnsuppressedGreedyToken = %d, want 3", got)
	}
}

func TestSample_HostUnsuppressedGreedyTokenMaterializesLazyFloat32_Good(t *testing.T) {
	base := FromValues([]float32{100, 1, 9, 11}, 1, 4)
	zero := Zeros([]int32{1, 4}, DTypeFloat32)
	logits := Add(base, zero)
	defer Free(base, zero, logits)

	token, err := hostUnsuppressedGreedyToken(logits, []int32{0})
	if err != nil {
		t.Fatalf("hostUnsuppressedGreedyToken: %v", err)
	}
	defer Free(token)
	if got := int32(token.Int()); got != 3 {
		t.Fatalf("hostUnsuppressedGreedyToken = %d, want 3", got)
	}
}

func TestSample_NewSamplerWithSuppressionBeforeTopPTopK_Good(t *testing.T) {
	s := NewSamplerWithSuppression(1.0, 0.95, 0, 3, []int32{0})
	defer CloseSampler(s)
	c, ok := s.(*topKTopPChain)
	if !ok {
		t.Fatalf("NewSamplerWithSuppression returned %T, want topKTopPChain", s)
	}
	if c.topK != 3 {
		t.Fatalf("topK = %d, want 3", c.topK)
	}
	if c.topP != 0.95 {
		t.Fatalf("topP = %f, want 0.95", c.topP)
	}
	if len(c.prefix.steps) != 0 {
		t.Fatalf("len(prefix) = %d, want fused suppression without prefix", len(c.prefix.steps))
	}
	if c.suppress == nil {
		t.Fatal("suppress = nil, want fused suppress-token sampler")
	}
	if len(c.suppress.tokens) != 1 || c.suppress.tokens[0] != 0 {
		t.Fatalf("suppress tokens = %v, want [0]", c.suppress.tokens)
	}
}

func TestSample_NewSamplerSkipsUnitTemperature_Good(t *testing.T) {
	s := newSampler(1.0, 0.95, 0, 64)
	c, ok := s.(*topKTopPChain)
	if !ok {
		t.Fatalf("newSampler returned %T, want topKTopPChain", s)
	}
	if len(c.prefix.steps) != 0 {
		t.Fatalf("len(prefix) = %d, want no no-op Temperature sampler", len(c.prefix.steps))
	}
}

func TestSample_PrefetchTokenEvalParity_Good(t *testing.T) {
	requireMetalRuntime(t)

	const seed = 240524
	suppress := []int32{0, 7}
	directID := sampleParityTokenID(t, seed, suppress, false)
	prefetchedID := sampleParityTokenID(t, seed, suppress, true)
	if prefetchedID != directID {
		t.Fatalf("prefetched token id = %d, want direct token id %d", prefetchedID, directID)
	}
}

func sampleParityTokenID(t *testing.T, seed uint64, suppress []int32, prefetch bool) int32 {
	t.Helper()
	base := FromValues([]float32{9.0, 3.4, 3.2, 3.0, 2.8, 2.6, 2.4, 9.0}, 1, 8)
	zero := Zeros([]int32{1, 8}, DTypeFloat32)
	logits := Add(base, zero)
	defer Free(base, zero, logits)

	// Seeded keys make the two parity runs draw identically — sampler draw
	// reproducibility comes from the explicit key sequence, not the global
	// mlx_random_seed state.
	s := NewSamplerWithSuppressionKeyed(1.0, 0.95, 0, 4, suppress, NewSamplerKeys(seed))
	defer CloseSampler(s)
	if !prefetch {
		token, id, _, err := SampleTokenIDWithSuppressionGuard(logits, s, suppress, false)
		if err != nil {
			t.Fatalf("SampleTokenIDWithSuppressionGuard: %v", err)
		}
		Free(token)
		return id
	}

	token := s.Sample(logits)
	if err := EvalAsync(logits, token); err != nil {
		Free(token)
		t.Fatalf("EvalAsync(logits, token): %v", err)
	}
	id := int32(token.Int())
	Free(token)
	if TokenIDSuppressed(id, suppress) {
		t.Fatalf("prefetched token id = %d, want unsuppressed token", id)
	}
	return id
}

func TestSample_Chain_Good(t *testing.T) {
	// Full chain: topK + temperature
	logits := FromValues([]float32{1, 2, 3, 4, 5}, 1, 5)
	s := newSampler(0.5, 0, 0, 3) // temp=0.5, topK=3

	token := s.Sample(logits)
	Materialize(token)

	idx := token.Int()
	if idx < 0 || idx >= 5 {
		t.Errorf("chain sample index = %d, out of range", idx)
	}
}

func TestSample_ChainOrder_Good(t *testing.T) {
	s := newSampler(0.7, 0.9, 0.05, 20)
	c, ok := s.(chain)
	if !ok {
		t.Fatalf("newSampler returned %T, want chain", s)
	}
	if len(c.steps) != 4 {
		t.Fatalf("len(chain) = %d, want 4", len(c.steps))
	}
	if _, ok := c.steps[0].(Temperature); !ok {
		t.Fatalf("chain[0] = %T, want Temperature", c.steps[0])
	}
	if _, ok := c.steps[1].(TopP); !ok {
		t.Fatalf("chain[1] = %T, want TopP", c.steps[1])
	}
	if _, ok := c.steps[2].(TopKSampler); !ok {
		t.Fatalf("chain[2] = %T, want TopKSampler", c.steps[2])
	}
	if _, ok := c.steps[3].(MinPSampler); !ok {
		t.Fatalf("chain[3] = %T, want MinPSampler", c.steps[3])
	}
}

func TestSample_TopPSamplesWithoutTemperature_Good(t *testing.T) {
	s := newSampler(0, 0.9, 0, 0)
	c, ok := s.(chain)
	if !ok {
		t.Fatalf("newSampler returned %T, want chain", s)
	}
	if len(c.steps) != 1 {
		t.Fatalf("len(chain) = %d, want 1", len(c.steps))
	}
	if _, ok := c.steps[0].(TopP); !ok {
		t.Fatalf("chain[0] = %T, want TopP", c.steps[0])
	}
}

func TestSample_TopKSamplesWithoutTemperature_Good(t *testing.T) {
	s := newSampler(0, 0, 0, 20)
	c, ok := s.(chain)
	if !ok {
		t.Fatalf("newSampler returned %T, want chain", s)
	}
	if len(c.steps) != 1 {
		t.Fatalf("len(chain) = %d, want 1", len(c.steps))
	}
	if _, ok := c.steps[0].(TopKSampler); !ok {
		t.Fatalf("chain[0] = %T, want TopKSampler", c.steps[0])
	}
}

func TestSample_MinPSamplesWithoutTemperature_Good(t *testing.T) {
	s := newSampler(0, 0, 0.05, 0)
	c, ok := s.(chain)
	if !ok {
		t.Fatalf("newSampler returned %T, want chain", s)
	}
	if len(c.steps) != 1 {
		t.Fatalf("len(chain) = %d, want 1", len(c.steps))
	}
	if _, ok := c.steps[0].(MinPSampler); !ok {
		t.Fatalf("chain[0] = %T, want MinPSampler", c.steps[0])
	}
}

func TestSample_TopP_DominantLogit_Good(t *testing.T) {
	// With one dominant logit, TopP should always pick it
	logits := FromValues([]float32{-10, -10, 100, -10}, 1, 4)
	s := newSampler(0.5, 0.9, 0, 0) // topP=0.9, temp=0.5
	token := s.Sample(logits)
	Materialize(token)

	if token.Int() != 2 {
		t.Errorf("topP dominant sample = %d, want 2", token.Int())
	}
}

func TestSample_TopP_RestrictsOptions_Good(t *testing.T) {
	// Two equal high logits, two low. TopP=0.5 should mostly restrict to top tokens.
	logits := FromValues([]float32{10, 10, -100, -100}, 1, 4)
	s := newSampler(1.0, 0.5, 0, 0) // topP=0.5, temp=1.0

	seen := map[int]bool{}
	for range 30 {
		token := s.Sample(logits)
		Materialize(token)
		seen[token.Int()] = true
	}

	// Should only pick indices 0 or 1 (the two high-probability tokens)
	for idx := range seen {
		if idx != 0 && idx != 1 {
			t.Errorf("topP=0.5 sampled index %d, expected only 0 or 1", idx)
		}
	}
}

func TestSample_MinP_DominantLogit_Good(t *testing.T) {
	// With one dominant logit, MinP should always pick it
	logits := FromValues([]float32{-10, -10, 100, -10}, 1, 4)
	s := newSampler(0.5, 0, 0.1, 0) // minP=0.1, temp=0.5
	token := s.Sample(logits)
	Materialize(token)

	if token.Int() != 2 {
		t.Errorf("minP dominant sample = %d, want 2", token.Int())
	}
}

func TestSample_MinP_RestrictsOptions_Good(t *testing.T) {
	// One very high logit, rest are low. MinP=0.1 should mask the low tokens.
	logits := FromValues([]float32{-100, 50, -100, -100}, 1, 4)
	s := newSampler(1.0, 0, 0.1, 0) // minP=0.1, temp=1.0

	for range 20 {
		token := s.Sample(logits)
		Materialize(token)
		if token.Int() != 1 {
			t.Errorf("minP with dominant logit sampled %d, want 1", token.Int())
		}
	}
}

func TestSample_ApplyRepeatPenalty_Good(t *testing.T) {
	// Logits: [1, 4] with values [5.0, -3.0, 1.0, 0.0]
	// History: tokens 0 and 1 have been seen.
	// Penalty 2.0:
	//   token 0 (logit 5.0 > 0): 5.0 / 2.0 = 2.5
	//   token 1 (logit -3.0 < 0): -3.0 * 2.0 = -6.0
	//   token 2 (not in history): unchanged = 1.0
	//   token 3 (not in history): unchanged = 0.0
	logits := FromValues([]float32{5.0, -3.0, 1.0, 0.0}, 1, 4)
	Materialize(logits)

	result := applyRepeatPenalty(logits, []int32{0, 1, 0}, 2.0) // duplicate 0 should be deduped
	Materialize(result)

	got := result.Floats()
	want := []float32{2.5, -6.0, 1.0, 0.0}
	for i := range got {
		diff := got[i] - want[i]
		if diff > 0.01 || diff < -0.01 {
			t.Errorf("repeatPenalty[%d] = %f, want %f", i, got[i], want[i])
		}
	}
}

func TestSample_ApplyRepeatPenalty_NoHistory_Good(t *testing.T) {
	// With empty history, logits should be unchanged.
	logits := FromValues([]float32{5.0, -3.0, 1.0}, 1, 3)
	Materialize(logits)

	// applyRepeatPenalty is not called when history is empty (checked in generate loop),
	// but verify the function handles it gracefully if called directly.
	result := applyRepeatPenalty(logits, []int32{1}, 1.0) // penalty=1.0 → no change
	Materialize(result)

	got := result.Floats()
	want := []float32{5.0, -3.0, 1.0}
	for i := range got {
		diff := got[i] - want[i]
		if diff > 0.01 || diff < -0.01 {
			t.Errorf("penalty=1.0[%d] = %f, want %f", i, got[i], want[i])
		}
	}
}

// TestMaterialiseFloat32ViewFast_FastPath_Good asserts that the fast-path
// (already DTypeFloat32 + row-contiguous) yields a view bit-exact to the
// underlying tensor data — no Materialize crossing, no dtype conversion.
func TestMaterialiseFloat32ViewFast_FastPath_Good(t *testing.T) {
	values := []float32{0.1, -0.2, 3.14, -42, 1e-6, 1e6, math.MaxFloat32, -math.MaxFloat32}
	arr := FromValues(values, 1, len(values))
	Materialize(arr) // pre-materialise so backing store exists
	defer Free(arr)

	if !arr.IsRowContiguous() {
		t.Fatalf("test pre-condition: arr must be row-contiguous, got !IsRowContiguous")
	}
	if arr.Dtype() != DTypeFloat32 {
		t.Fatalf("test pre-condition: arr must be DTypeFloat32, got %v", arr.Dtype())
	}

	view, cleanup, err := materialiseFloat32ViewFast(arr)
	if err != nil {
		t.Fatalf("materialiseFloat32ViewFast: %v", err)
	}
	defer cleanup()

	if len(view) != len(values) {
		t.Fatalf("view len = %d, want %d", len(view), len(values))
	}
	for i, want := range values {
		if view[i] != want {
			t.Errorf("view[%d] = %v, want %v (bit-exact required)", i, view[i], want)
		}
	}
}

// TestMaterialiseFloat32ViewFast_SlowPathDtype_Good asserts parity with the
// legacy helper when arr is non-float32 — fall-through path must produce a
// bit-exact view via AsType + Materialize.
func TestMaterialiseFloat32ViewFast_SlowPathDtype_Good(t *testing.T) {
	// Build a float32 array, then AsType to float16 to force the slow path.
	values := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	src := FromValues(values, 1, len(values))
	Materialize(src)
	defer Free(src)
	f16 := AsType(src, DTypeFloat16)
	Materialize(f16)
	defer Free(f16)

	view, cleanup, err := materialiseFloat32ViewFast(f16)
	if err != nil {
		t.Fatalf("materialiseFloat32ViewFast (slow): %v", err)
	}
	defer cleanup()

	if len(view) != len(values) {
		t.Fatalf("view len = %d, want %d", len(view), len(values))
	}
	// float16 -> float32 round-trip is exact for these small integers
	for i, want := range values {
		if view[i] != want {
			t.Errorf("view[%d] = %v, want %v (float16 round-trip exact for ints)", i, view[i], want)
		}
	}
}

// TestMaterialiseFloat32ViewFast_LegacyParity_Good asserts the fast-path
// helper produces bit-exact output vs the legacy materialiseFloat32View on
// the same input.  Identical contract = safe migration.
func TestMaterialiseFloat32ViewFast_LegacyParity_Good(t *testing.T) {
	values := make([]float32, 1024)
	for i := range values {
		values[i] = float32(i)*0.001 - 0.5
	}
	arr := FromValues(values, 1, len(values))
	Materialize(arr)
	defer Free(arr)

	fastView, fastCleanup, err := materialiseFloat32ViewFast(arr)
	if err != nil {
		t.Fatalf("fast: %v", err)
	}
	defer fastCleanup()

	slowSrc, slowConverted, err := materialiseFloat32View(arr)
	if err != nil {
		t.Fatalf("slow: %v", err)
	}
	defer Free(slowConverted)
	slowN := slowSrc.Size()
	slowPtr := (*float32)(rawArrayDataPointer(slowSrc))
	slowView := unsafe.Slice(slowPtr, slowN)
	defer runtime.KeepAlive(slowSrc)

	if len(fastView) != len(slowView) {
		t.Fatalf("len mismatch: fast=%d slow=%d", len(fastView), len(slowView))
	}
	for i := range fastView {
		if fastView[i] != slowView[i] {
			t.Errorf("parity[%d]: fast=%v slow=%v", i, fastView[i], slowView[i])
		}
	}
}

// TestMaterialiseFloat32ViewFast_NonContiguous_Ugly asserts that a sliced
// (and so potentially non-contiguous) view falls through to the slow path and
// still produces correct float32 data — the dtype + contiguity gate must
// route non-contiguous tensors to materialiseFloat32View without panic.
func TestMaterialiseFloat32ViewFast_NonContiguous_Ugly(t *testing.T) {
	// 2x4 then slice a non-row-aligned axis to force a non-contiguous view.
	values := []float32{
		0, 1, 2, 3,
		4, 5, 6, 7,
	}
	arr := FromValues(values, 2, 4)
	Materialize(arr)
	defer Free(arr)
	sliced := SliceAxis(arr, -1, 1, 3) // shape [2, 2] — strided view
	Materialize(sliced)
	defer Free(sliced)

	view, cleanup, err := materialiseFloat32ViewFast(sliced)
	if err != nil {
		t.Fatalf("non-contig: %v", err)
	}
	defer cleanup()

	want := []float32{1, 2, 5, 6}
	if len(view) != len(want) {
		t.Fatalf("view len = %d, want %d", len(view), len(want))
	}
	for i, w := range want {
		if view[i] != w {
			t.Errorf("view[%d] = %v, want %v", i, view[i], w)
		}
	}
}
