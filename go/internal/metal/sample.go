// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"math"
	"runtime"
	"slices"
	"sync"
	"unsafe"

	core "dappco.re/go"
)

// suppressIDsScratch is a pooled []int32 buffer reused for dedup +
// validity-filter inside suppressTokenLogits and hostUnsuppressedGreedyToken.
// These fire per-token when the suppression guard activates, so eliminating
// the map[int32]bool + slice growth pair pays back across the generation.
var suppressIDsScratch = sync.Pool{
	New: func() any {
		buf := make([]int32, 0, 64)
		return &buf
	},
}

// Sampler transforms logits into a sampled token index.
//
//	s := newSampler(0.7, 0.9, 0, 40) // temp=0.7, topP=0.9, minP=0, topK=40
//	tokenID := s.Sample(logits)
type Sampler interface {
	Sample(logits *Array) *Array
}

// newSampler creates a composable sampler chain from the given parameters.
// Order: Temperature -> TopP -> TopK -> MinP -> categorical sample.
//
//	s := newSampler(0, 0, 0, 0)        // greedy (temp=0)
//	s := newSampler(0.7, 0.9, 0, 40)   // top-p + top-k + temperature
//	s := newSampler(1.0, 0, 0.05, 0)   // min-p sampling
func newSampler(temp, topP, minP float32, topK int) Sampler {
	return newSamplerWithSuppression(temp, topP, minP, topK, nil)
}

func newSamplerWithSuppression(temp, topP, minP float32, topK int, suppressTokens []int32) Sampler {
	if temp <= 0 && topP <= 0 && minP <= 0 && topK <= 0 && len(suppressTokens) > 0 {
		return suppressedGreedy{tokens: append([]int32(nil), suppressTokens...)}
	}
	samplers := make([]Sampler, 0, 4)
	if temp > 0 {
		samplers = append(samplers, Temperature(temp))
	}
	if len(suppressTokens) > 0 {
		samplers = append(samplers, SuppressTokensSampler{tokens: append([]int32(nil), suppressTokens...)})
	}
	if topP > 0 && topP < 1 {
		samplers = append(samplers, TopP(topP))
	}
	if topK > 0 {
		samplers = append(samplers, TopKSampler(topK))
	}
	if minP > 0 {
		samplers = append(samplers, MinPSampler(minP))
	}
	if len(samplers) == 0 {
		return greedy{}
	}
	return chain(samplers)
}

func suppressTokenLogits(logits *Array, ids []int32) *Array {
	if logits == nil || len(ids) == 0 {
		if logits == nil {
			return nil
		}
		return logits.Clone()
	}
	lastDim := logits.Dim(logits.NumDims() - 1)

	// Build the valid + deduped id set via pooled scratch — replaces
	// per-call map[int32]bool + slice growth.  Filter pass appends only
	// in-range non-negative ids, then sort+compact removes duplicates.
	scratchPtr := suppressIDsScratch.Get().(*[]int32)
	scratch := (*scratchPtr)[:0]
	if cap(scratch) < len(ids) {
		scratch = make([]int32, 0, len(ids))
	}
	for _, id := range ids {
		if id < 0 || int(id) >= lastDim {
			continue
		}
		scratch = append(scratch, id)
	}
	if len(scratch) == 0 {
		*scratchPtr = scratch
		suppressIDsScratch.Put(scratchPtr)
		return logits.Clone()
	}
	slices.Sort(scratch)
	valid := slices.Compact(scratch)

	idx := FromValues(valid, 1, len(valid))
	inf := FromValue(float32(math.Inf(-1)))
	if dtype := logits.Dtype(); dtype != DTypeFloat32 {
		cast := AsType(inf, dtype)
		Free(inf)
		inf = cast
	}
	res := PutAlongAxis(logits, idx, inf, -1)
	Free(idx, inf)

	// FromValues has copied valid into MLX memory, scratch is safe to recycle.
	*scratchPtr = scratch
	suppressIDsScratch.Put(scratchPtr)
	return res
}

// chain applies a sequence of samplers in order, then draws a categorical sample.
//
//	chain{TopP(0.9), TopKSampler(40), Temperature(0.7)}.Sample(logits)
type chain []Sampler

func (c chain) Sample(logits *Array) *Array {
	curr := logits
	for _, s := range c {
		next := s.Sample(curr)
		if curr != logits {
			Free(curr)
		}
		curr = next
	}
	// Final categorical sample from log-probabilities
	res := RandomCategorical(curr)
	if curr != logits {
		Free(curr)
	}
	return res
}

// greedy returns the argmax token (deterministic, no sampling).
//
//	greedy{}.Sample(logits) // picks the single most likely token
type greedy struct{}

func (greedy) Sample(logits *Array) *Array {
	return Argmax(logits, -1, false)
}

type suppressedGreedy struct {
	tokens []int32
}

func (s suppressedGreedy) Sample(logits *Array) *Array {
	filtered := suppressTokenLogits(logits, s.tokens)
	token := Argmax(filtered, -1, false)
	Free(filtered)
	return token
}

type SuppressTokensSampler struct {
	tokens []int32
}

func (s SuppressTokensSampler) Sample(logits *Array) *Array {
	return suppressTokenLogits(logits, s.tokens)
}

func sampleTokenWithSuppressionGuard(logits *Array, sampler Sampler, suppressTokens []int32) (*Array, error) {
	next := sampler.Sample(logits)
	if err := Eval(next); err != nil {
		Free(next)
		return nil, err
	}
	if !tokenIDSuppressed(int32(next.Int()), suppressTokens) {
		return next, nil
	}
	Free(next)
	filtered := suppressTokenLogits(logits, suppressTokens)
	if err := Eval(filtered); err != nil {
		Free(filtered)
		return nil, err
	}
	next = greedy{}.Sample(filtered)
	Free(filtered)
	if err := Eval(next); err != nil {
		Free(next)
		return nil, err
	}
	if tokenIDSuppressed(int32(next.Int()), suppressTokens) {
		Free(next)
		next, err := hostUnsuppressedGreedyToken(logits, suppressTokens)
		if err != nil {
			return nil, err
		}
		if err := Eval(next); err != nil {
			Free(next)
			return nil, err
		}
		if !tokenIDSuppressed(int32(next.Int()), suppressTokens) {
			return next, nil
		}
		id := int32(next.Int())
		Free(next)
		return nil, core.NewError(core.Sprintf("mlx: sampler returned suppressed token %d after suppression guard", id))
	}
	return next, nil
}

func hostUnsuppressedGreedyToken(logits *Array, suppressTokens []int32) (*Array, error) {
	if logits == nil || !logits.Valid() {
		return nil, core.NewError("mlx: logits are empty")
	}

	// Dedup + sort suppressTokens via pooled scratch so the inner loop can
	// use binary search instead of a per-call map[int32]bool allocation
	// (the original cost ~16B/entry + 8 allocs on a Gemma-sized suppress
	// list).  Per-token hot path — fires whenever the sampler tries a
	// suppressed id and falls through the guard.
	scratchPtr := suppressIDsScratch.Get().(*[]int32)
	scratch := (*scratchPtr)[:0]
	if cap(scratch) < len(suppressTokens) {
		scratch = make([]int32, 0, len(suppressTokens))
	}
	for _, id := range suppressTokens {
		if id >= 0 {
			scratch = append(scratch, id)
		}
	}
	slices.Sort(scratch)
	suppressed := slices.Compact(scratch)

	// Scan logits via a borrowed MLX-memory view rather than copying to a
	// freshly-allocated Go []float32 (logits.Floats() does make([]float32, n)
	// + per-element copy — ~1MB on a 258k Gemma vocab).  Argmax is read-only,
	// no copy needed.  Dtype-convert via AsType if non-float32 so the view
	// remains float32-typed.
	src, converted, err := materialiseFloat32View(logits)
	if err != nil {
		*scratchPtr = scratch
		suppressIDsScratch.Put(scratchPtr)
		return nil, err
	}
	n := src.Size()
	if n == 0 {
		Free(converted)
		*scratchPtr = scratch
		suppressIDsScratch.Put(scratchPtr)
		return nil, core.NewError("mlx: logits are empty")
	}
	ptr := (*float32)(rawArrayDataPointer(src))
	if ptr == nil {
		Free(converted)
		*scratchPtr = scratch
		suppressIDsScratch.Put(scratchPtr)
		return nil, core.NewError("mlx: logits are empty")
	}
	view := unsafe.Slice(ptr, n)

	bestID := int32(-1)
	bestValue := float32(math.Inf(-1))
	for id, value := range view {
		tokenID := int32(id)
		if math.IsNaN(float64(value)) {
			continue
		}
		if _, ok := slices.BinarySearch(suppressed, tokenID); ok {
			continue
		}
		if bestID < 0 || value > bestValue {
			bestID = tokenID
			bestValue = value
		}
	}
	runtime.KeepAlive(src)
	Free(converted)

	*scratchPtr = scratch
	suppressIDsScratch.Put(scratchPtr)

	if bestID < 0 {
		return nil, core.NewError("mlx: no finite unsuppressed logits available")
	}
	return fromSingleInt32(bestID), nil
}

// materialiseFloat32View returns a borrowed view-source for hostside scans of
// a logits tensor.  Result.converted is non-nil iff a dtype conversion was
// needed (caller must Free it after the scan finishes).
func materialiseFloat32View(t *Array) (src, converted *Array, err error) {
	src = t
	if t.Dtype() != DTypeFloat32 {
		converted = AsType(t, DTypeFloat32)
		Materialize(converted)
		src = converted
	}
	if !src.IsRowContiguous() {
		c := Contiguous(src)
		Materialize(c)
		if converted != nil {
			Free(converted)
		}
		converted = c
		src = c
	}
	Materialize(src)
	return src, converted, nil
}

// materialiseFloat32ViewFast returns a borrowed []float32 view of arr plus a
// cleanup func that the caller MUST defer.  The view is tied to arr via
// runtime.KeepAlive inside cleanup, so callers do not need their own KeepAlive.
//
// Fast-path: when arr is already DTypeFloat32 + row-contiguous, the helper
// skips every internal Materialize cgo crossing — the legacy
// materialiseFloat32View calls Materialize on src unconditionally at the end,
// even when dtype + layout already match.  At ~30-60 ns per cgo crossing,
// dropping that one Materialize shifts the zero-copy threshold from ~1KB down
// to a few hundred bytes (the dtype + contiguity check is ~5-10 ns).
//
// Slow-path: when arr needs dtype conversion or contiguity copy, the helper
// falls through to materialiseFloat32View — same ceremony, same overhead.
//
//	view, cleanup, err := materialiseFloat32ViewFast(logits)
//	if err != nil { return err }
//	defer cleanup()
//	bestID := argmax(view)
func materialiseFloat32ViewFast(arr *Array) ([]float32, func(), error) {
	if arr.Dtype() == DTypeFloat32 && arr.IsRowContiguous() {
		// Fast-path: dtype + layout already match.  Skip Materialize entirely
		// — the only invariant the caller needs is a valid float32 backing
		// store, which the dtype+contiguity check already proves.
		n := arr.Size()
		if n == 0 {
			return nil, func() {}, nil
		}
		ptr := (*float32)(rawArrayDataPointer(arr))
		if ptr == nil {
			return nil, func() {}, core.NewError("mlx: array data pointer is nil")
		}
		view := unsafe.Slice(ptr, n)
		cleanup := func() { runtime.KeepAlive(arr) }
		return view, cleanup, nil
	}
	// Slow-path: fall through to the legacy helper.  AsType / Contiguous
	// crossings are unavoidable when dtype or layout doesn't match.
	src, converted, err := materialiseFloat32View(arr)
	if err != nil {
		return nil, func() {}, err
	}
	n := src.Size()
	if n == 0 {
		Free(converted)
		return nil, func() {}, nil
	}
	ptr := (*float32)(rawArrayDataPointer(src))
	if ptr == nil {
		Free(converted)
		return nil, func() {}, core.NewError("mlx: array data pointer is nil")
	}
	view := unsafe.Slice(ptr, n)
	cleanup := func() {
		runtime.KeepAlive(src)
		Free(converted)
	}
	return view, cleanup, nil
}

func tokenIDSuppressed(id int32, suppressTokens []int32) bool {
	for _, suppressed := range suppressTokens {
		if id == suppressed {
			return true
		}
	}
	return false
}

// Temperature scales logits by 1/temp before categorical sampling.
// Higher values produce more random output; lower values approach greedy.
//
//	Temperature(0.7).Sample(logits) // moderate creativity
//	Temperature(0.1).Sample(logits) // near-greedy, focused output
type Temperature float32

func (t Temperature) Sample(logits *Array) *Array {
	return MulScalar(logits, 1.0/float32(t))
}

// TopKSampler masks all but the top-k logits, setting the rest to -inf.
//
//	TopKSampler(40).Sample(logits) // keep only top 40 candidates
//	TopKSampler(10).Sample(logits) // very focused — top 10 only
type TopKSampler int

func (k TopKSampler) Sample(logits *Array) *Array {
	lastDim := logits.Dim(logits.NumDims() - 1)
	if lastDim <= 0 || int(k) <= 0 || int(k) >= lastDim {
		return logits.Clone()
	}
	neg := Negative(logits)
	maskIdx := Argpartition(neg, int(k)-1, -1)
	Free(neg)
	// Slice the indices beyond top-k
	mask := SliceAxis(maskIdx, -1, int32(k), int32(lastDim))
	Free(maskIdx)
	// W11-R: inline the -inf scalar into PutAlongAxis via a scalar-shape
	// FromValue; PutAlongAxis broadcasts.  Cannot collapse further without
	// an MLX put_along_axis_scalar bridge — the FromValue cost is a single
	// rank-0 alloc which is at floor for this op.
	inf := FromValue(float32(math.Inf(-1)))
	res := PutAlongAxis(logits, mask, inf, -1)
	Free(mask, inf)
	return res
}

// TopP implements nucleus (top-p) sampling.
// Keeps the smallest set of tokens whose cumulative probability exceeds p.
//
//	TopP(0.9).Sample(logits) // include tokens covering 90% of probability mass
//	TopP(0.5).Sample(logits) // conservative — only highest-probability half
type TopP float32

func (p TopP) Sample(logits *Array) *Array {
	// Convert logits to probabilities
	probs := Softmax(logits)

	// Sort descending via argsort of negated probs
	neg := Negative(probs)
	sortIdx := Argsort(neg, -1)
	Free(neg)
	sortedProbs := TakeAlongAxis(probs, sortIdx, -1)

	// Cumulative sum of sorted probabilities
	cumProbs := CumSum(sortedProbs, -1, false, true)

	// Mask in sorted space: keep tokens where cumprob (excluding current) <= threshold
	shiftedCum := Subtract(cumProbs, sortedProbs)

	// W11-R: inline the scalar compare + scalar/scalar where into single cgo
	// crossings.  Was 3× FromValue + Greater + Where + 3× Free; now
	// greaterScalar + whereScalarScalar (2 cgo crossings, 0 Go-side scalar
	// *Array wrappers).
	gt := greaterScalar(shiftedCum, float32(p))
	sortedMask := whereScalarScalar(gt, float32(math.Inf(-1)), 0)
	Free(gt, shiftedCum, cumProbs, sortedProbs)

	// Scatter mask back to original positions
	emptyMask := Zeros(logits.Shape(), DTypeFloat32)
	mask := PutAlongAxis(emptyMask, sortIdx, sortedMask, -1)
	Free(emptyMask, sortIdx, sortedMask)

	// W11-R: replace zeroArr + Greater(zeroArr, mask) + inf2 + Where(gt0, inf2, logits)
	// with scalarGreater + whereScalarArray (2 cgo crossings, 0 Go-side scalar
	// *Array wrappers).
	gt0 := scalarGreater(0, mask)
	res := whereScalarArray(gt0, float32(math.Inf(-1)), logits)
	Free(gt0, mask, probs)

	return res
}

// MinPSampler masks tokens whose probability falls below min_p * max_prob.
// Adapts the threshold relative to the best token, so the cut-off scales with confidence.
//
//	MinPSampler(0.05).Sample(logits) // drop tokens less than 5% of top-token probability
//	MinPSampler(0.1).Sample(logits)  // stricter — drop tokens below 10% of max
type MinPSampler float32

func (p MinPSampler) Sample(logits *Array) *Array {
	// Convert logits to probabilities
	probs := Softmax(logits)

	// Find the maximum probability
	maxProb := MaxAxis(probs, -1, true)

	// Threshold = min_p * max_prob
	threshold := MulScalar(maxProb, float32(p))
	Free(maxProb)

	// W11-R: inline the scalar -inf into the where call — replaces FromValue
	// + Where + Free(scalar) triple with a single cgo crossing.
	gt := Greater(threshold, probs)
	mask := whereScalarArray(gt, float32(math.Inf(-1)), logits)
	Free(probs, threshold, gt)
	return mask
}
