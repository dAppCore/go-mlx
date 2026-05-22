// SPDX-Licence-Identifier: EUPL-1.2

package kv

import "math"

const (
	kvCoherenceThreshold = 0.7
	kvCollapseThreshold  = 0.5
)

// Analysis contains K/V cache coherence metrics for one prefill snapshot.
type Analysis struct {
	MeanKeyCoherence       float64
	MeanValueCoherence     float64
	MeanCrossAlignment     float64
	MeanHeadEntropy        float64
	PhaseLockScore         float64
	MeanKVCoupling         float64
	JointCollapseCount     int
	LayerKeyCoherence      []float64
	LayerValueCoherence    []float64
	LayerCrossAlignment    []float64
	LayerKVCoupling        []float64
	SharedCacheLayerGroups map[int][]int
	GQA                    bool
}

// Composite returns a 0-10000 integer score from K/V posture metrics.
func (r *Analysis) Composite() int {
	if r == nil {
		return 0
	}
	jointStability := math.Max(0, 1.0-float64(r.JointCollapseCount)*0.2)
	var score float64
	if r.GQA {
		score = (0.30*r.MeanKeyCoherence +
			0.20*r.MeanValueCoherence +
			0.20*r.MeanCrossAlignment +
			0.15*r.MeanKVCoupling +
			0.10*r.MeanHeadEntropy +
			0.05*jointStability) * 10000.0
	} else {
		score = (0.22*r.MeanKeyCoherence +
			0.18*r.MeanValueCoherence +
			0.20*r.MeanCrossAlignment +
			0.15*r.PhaseLockScore +
			0.15*r.MeanKVCoupling +
			0.05*r.MeanHeadEntropy +
			0.05*jointStability) * 10000.0
	}
	return min(10000, max(0, int(score)))
}

// Analyze computes coherence metrics from a CPU-readable KV cache snapshot.
func Analyze(snapshot *Snapshot) *Analysis {
	if snapshot == nil || len(snapshot.Layers) == 0 {
		return &Analysis{}
	}
	if kvAnalysisNumHeads(snapshot) <= 4 {
		return analyzeKVGQA(snapshot)
	}
	return analyzeKVMultiHead(snapshot)
}

func analyzeKVMultiHead(snapshot *Snapshot) *Analysis {
	numLayers := kvAnalysisNumLayers(snapshot)
	result := &Analysis{
		LayerKeyCoherence:      make([]float64, numLayers),
		LayerValueCoherence:    make([]float64, numLayers),
		LayerCrossAlignment:    make([]float64, max(0, numLayers-1)),
		LayerKVCoupling:        make([]float64, numLayers),
		SharedCacheLayerGroups: kvSharedCacheLayerGroups(snapshot),
	}

	layerStates := make([][]float32, numLayers)
	var keyTotal, valueTotal, entropyTotal, couplingTotal float64
	var layerCount, entropyCount, couplingCount int
	var lockedPairs, totalPairs int

	// One magnitudes scratch reused across every kvAnalysisHeadEntropy
	// call (every layer × head × side). Was per-call alloc before.
	var entropyScratch []float64
	if snapshot.SeqLen > 0 {
		entropyScratch = make([]float64, snapshot.SeqLen)
	}

	for layer := range numLayers {
		layerSnapshot, ok := snapshot.layer(layer)
		if !ok || len(layerSnapshot.Heads) == 0 {
			continue
		}
		keyHeads := kvAnalysisHeadVectors(layerSnapshot.Heads, true)
		valueHeads := kvAnalysisHeadVectors(layerSnapshot.Heads, false)
		keyCoherence, keyLocked, keyPairs := kvAnalysisPairCoherence(keyHeads)
		valueCoherence, valueLocked, valuePairs := kvAnalysisPairCoherence(valueHeads)
		coupling, couplingN := kvAnalysisLayerCoupling(layerSnapshot.Heads)

		result.LayerKeyCoherence[layer] = keyCoherence
		result.LayerValueCoherence[layer] = valueCoherence
		result.LayerKVCoupling[layer] = coupling
		layerStates[layer] = kvAnalysisLayerState(layerSnapshot.Heads)

		keyTotal += keyCoherence
		valueTotal += valueCoherence
		layerCount++
		lockedPairs += keyLocked + valueLocked
		totalPairs += keyPairs + valuePairs
		if couplingN > 0 {
			couplingTotal += coupling
			couplingCount++
		}
		for _, head := range layerSnapshot.Heads {
			if len(head.Key) > 0 {
				entropyTotal += kvAnalysisHeadEntropy(head.Key, snapshot.SeqLen, snapshot.HeadDim, entropyScratch)
				entropyCount++
			}
			if len(head.Value) > 0 {
				entropyTotal += kvAnalysisHeadEntropy(head.Value, snapshot.SeqLen, snapshot.HeadDim, entropyScratch)
				entropyCount++
			}
		}
	}

	var crossTotal float64
	var crossCount int
	for layer := 0; layer < numLayers-1; layer++ {
		if len(layerStates[layer]) == 0 || len(layerStates[layer+1]) == 0 {
			continue
		}
		alignment := kvAnalysisCosine32(layerStates[layer], layerStates[layer+1])
		result.LayerCrossAlignment[layer] = alignment
		crossTotal += alignment
		crossCount++
		if alignment < kvCollapseThreshold {
			result.JointCollapseCount++
		}
	}

	if layerCount > 0 {
		result.MeanKeyCoherence = keyTotal / float64(layerCount)
		result.MeanValueCoherence = valueTotal / float64(layerCount)
	}
	if crossCount > 0 {
		result.MeanCrossAlignment = crossTotal / float64(crossCount)
	}
	if entropyCount > 0 {
		result.MeanHeadEntropy = entropyTotal / float64(entropyCount)
	}
	if couplingCount > 0 {
		result.MeanKVCoupling = couplingTotal / float64(couplingCount)
	}
	if totalPairs > 0 {
		result.PhaseLockScore = float64(lockedPairs) / float64(totalPairs)
	}
	return result
}

func analyzeKVGQA(snapshot *Snapshot) *Analysis {
	numLayers := kvAnalysisNumLayers(snapshot)
	result := &Analysis{
		GQA:                    true,
		LayerKeyCoherence:      make([]float64, numLayers),
		LayerValueCoherence:    make([]float64, numLayers),
		LayerCrossAlignment:    make([]float64, max(0, numLayers-1)),
		LayerKVCoupling:        make([]float64, numLayers),
		SharedCacheLayerGroups: kvSharedCacheLayerGroups(snapshot),
	}

	var keyTotal, valueTotal, entropyTotal, couplingTotal float64
	var layerCount, entropyCount, couplingCount int
	var lockedPairs, totalPairs int

	// One scaled-vector scratch per Analyze — reused across all layer
	// keys+values calls to avoid per-layer/per-side allocations.
	// Sized to seqLen × headDim (the pair-loop pre-scaled rows); the
	// entropy helper reuses the same buffer (it only needs seqLen
	// float64s for magnitudes — fits trivially).
	var scratch []float64
	if snapshot.SeqLen > 0 && snapshot.HeadDim > 0 {
		scratch = make([]float64, snapshot.SeqLen*snapshot.HeadDim)
	} else if snapshot.SeqLen > 0 {
		scratch = make([]float64, snapshot.SeqLen)
	}

	for layer := range numLayers {
		layerSnapshot, ok := snapshot.layer(layer)
		if !ok || len(layerSnapshot.Heads) == 0 {
			continue
		}
		keyDiff, keyLocked, keyPairs := kvAnalysisPositionDifferentiation(layerSnapshot.Heads, snapshot.SeqLen, snapshot.HeadDim, true, scratch)
		valueDiff, valueLocked, valuePairs := kvAnalysisPositionDifferentiation(layerSnapshot.Heads, snapshot.SeqLen, snapshot.HeadDim, false, scratch)
		coupling, couplingN := kvAnalysisLayerCoupling(layerSnapshot.Heads)

		result.LayerKeyCoherence[layer] = keyDiff
		result.LayerValueCoherence[layer] = valueDiff
		result.LayerKVCoupling[layer] = coupling
		keyTotal += keyDiff
		valueTotal += valueDiff
		layerCount++
		lockedPairs += keyLocked + valueLocked
		totalPairs += keyPairs + valuePairs
		if couplingN > 0 {
			couplingTotal += coupling
			couplingCount++
		}
		for _, head := range layerSnapshot.Heads {
			if len(head.Key) > 0 {
				// scratch double-duty: reuse as the entropy magnitudes
				// scratch since the position-differentiation pair loop
				// has finished consuming it for this layer. cap(scratch)
				// ≥ seqLen·headDim ≥ seqLen, so head-entropy's
				// seqLen-sized request always fits.
				entropyTotal += kvAnalysisHeadEntropy(head.Key, snapshot.SeqLen, snapshot.HeadDim, scratch)
				entropyCount++
			}
			if len(head.Value) > 0 {
				entropyTotal += kvAnalysisHeadEntropy(head.Value, snapshot.SeqLen, snapshot.HeadDim, scratch)
				entropyCount++
			}
		}
	}

	var crossTotal float64
	var crossCount int
	for layer := 0; layer < numLayers-1; layer++ {
		keyDelta := math.Abs(result.LayerKeyCoherence[layer+1] - result.LayerKeyCoherence[layer])
		valueDelta := math.Abs(result.LayerValueCoherence[layer+1] - result.LayerValueCoherence[layer])
		smoothness := 1.0 - (keyDelta+valueDelta)/2
		result.LayerCrossAlignment[layer] = smoothness
		crossTotal += smoothness
		crossCount++
		if smoothness < kvCollapseThreshold {
			result.JointCollapseCount++
		}
	}

	if layerCount > 0 {
		result.MeanKeyCoherence = keyTotal / float64(layerCount)
		result.MeanValueCoherence = valueTotal / float64(layerCount)
	}
	if crossCount > 0 {
		result.MeanCrossAlignment = crossTotal / float64(crossCount)
	}
	if entropyCount > 0 {
		result.MeanHeadEntropy = entropyTotal / float64(entropyCount)
	}
	if couplingCount > 0 {
		result.MeanKVCoupling = couplingTotal / float64(couplingCount)
	}
	if totalPairs > 0 {
		result.PhaseLockScore = float64(lockedPairs) / float64(totalPairs)
	}
	return result
}

// Features returns the 7D model-state feature vector from K/V metrics.
func Features(result *Analysis) []float64 {
	if result == nil {
		return make([]float64, 7)
	}
	return []float64{
		result.MeanKeyCoherence,
		result.MeanValueCoherence,
		result.MeanCrossAlignment,
		result.MeanHeadEntropy,
		result.PhaseLockScore,
		result.MeanKVCoupling,
		math.Max(0, 1.0-float64(result.JointCollapseCount)*0.2),
	}
}

// FeatureLabels returns labels matching Features order.
func FeatureLabels() []string {
	return []string{
		"key_coherence",
		"value_coherence",
		"cross_alignment",
		"head_entropy",
		"phase_lock",
		"kv_coupling",
		"joint_stability",
	}
}

func kvAnalysisNumLayers(snapshot *Snapshot) int {
	if snapshot == nil {
		return 0
	}
	if snapshot.NumLayers > 0 {
		return snapshot.NumLayers
	}
	return len(snapshot.Layers)
}

func kvAnalysisNumHeads(snapshot *Snapshot) int {
	if snapshot == nil {
		return 0
	}
	if snapshot.NumHeads > 0 {
		return snapshot.NumHeads
	}
	for _, layer := range snapshot.Layers {
		if len(layer.Heads) > 0 {
			return len(layer.Heads)
		}
	}
	return 0
}

func kvSharedCacheLayerGroups(snapshot *Snapshot) map[int][]int {
	if snapshot == nil {
		return map[int][]int{}
	}
	// Pre-size the hint map against layer count — Analyze callers
	// always have len(Layers) layers to bucket, so the runtime can
	// skip its rehash cycle on the bucket map.
	groups := make(map[int][]int, len(snapshot.Layers))
	for _, layer := range snapshot.Layers {
		groups[layer.CacheIndex] = append(groups[layer.CacheIndex], layer.Layer)
	}
	for cacheIndex, layers := range groups {
		if len(layers) < 2 {
			delete(groups, cacheIndex)
		}
	}
	return groups
}

func kvAnalysisHeadVectors(heads []HeadSnapshot, keys bool) [][]float32 {
	// Pre-extend instead of pre-allocate-empty + N appends — len is
	// known up-front (one slot per head). Hoists the keys/values branch
	// out of the inner loop too.
	vectors := make([][]float32, len(heads))
	if keys {
		for i := range heads {
			vectors[i] = heads[i].Key
		}
	} else {
		for i := range heads {
			vectors[i] = heads[i].Value
		}
	}
	return vectors
}

func kvAnalysisPairCoherence(vectors [][]float32) (float64, int, int) {
	// Precompute per-vector 1/|v| once so the O(N²) pair loop only
	// pays a dot product + 2 muls — same self-norm-recompute waste
	// kvAnalysisPositionDifferentiation had.
	n := len(vectors)
	invNorms := make([]float64, n)
	for i, vec := range vectors {
		var sum float64
		for _, value := range vec {
			v := float64(value)
			sum += v * v
		}
		if sum > 0 {
			invNorms[i] = 1.0 / math.Sqrt(sum)
		}
	}
	var total float64
	var locked, pairs int
	for i := 0; i < n; i++ {
		invA := invNorms[i]
		rowA := vectors[i]
		for j := i + 1; j < n; j++ {
			rowB := vectors[j]
			// Match the original kvAnalysisCosine32 semantics: count
			// the pair, with similarity = 0 when lengths mismatch or
			// either norm is zero.
			pairs++
			if len(rowA) != len(rowB) || len(rowA) == 0 || invA == 0 || invNorms[j] == 0 {
				continue
			}
			invB := invNorms[j]
			var dot float64
			for k, av := range rowA {
				dot += float64(av) * float64(rowB[k])
			}
			similarity := dot * invA * invB
			total += similarity
			if similarity >= kvCoherenceThreshold {
				locked++
			}
		}
	}
	if pairs == 0 {
		return 0, locked, pairs
	}
	return total / float64(pairs), locked, pairs
}

func kvAnalysisLayerCoupling(heads []HeadSnapshot) (float64, int) {
	var total float64
	var count int
	for _, head := range heads {
		if len(head.Key) == 0 || len(head.Value) == 0 {
			continue
		}
		total += kvAnalysisCosine32(head.Key, head.Value)
		count++
	}
	if count == 0 {
		return 0, 0
	}
	return total / float64(count), count
}

func kvAnalysisLayerState(heads []HeadSnapshot) []float32 {
	if len(heads) == 0 {
		return nil
	}
	// Find the first contributor head — its (Key+Value) length is the
	// shared mean-vector size; heads that don't match that exact shape
	// are skipped (matches the original kvAnalysisMeanVector behaviour).
	var size int
	for _, head := range heads {
		if l := len(head.Key) + len(head.Value); l > 0 {
			size = l
			break
		}
	}
	if size == 0 {
		return nil
	}
	// Sum-into-place + multiply-by-inverse: skip the per-head combined
	// alloc + the intermediate [][]float32 by aggregating directly into
	// the mean buffer. The original allocated len(heads) backing slices
	// + len(heads) combined buffers for every layer Analyze touched.
	mean := make([]float32, size)
	var count int
	for _, head := range heads {
		keyLen := len(head.Key)
		valLen := len(head.Value)
		if keyLen+valLen != size {
			continue
		}
		for i, v := range head.Key {
			mean[i] += v
		}
		for j, v := range head.Value {
			mean[keyLen+j] += v
		}
		count++
	}
	if count == 0 {
		return nil
	}
	invScale := float32(1) / float32(count)
	for i := range mean {
		mean[i] *= invScale
	}
	return mean
}

func kvAnalysisMeanVector(vectors [][]float32) []float32 {
	if len(vectors) == 0 || len(vectors[0]) == 0 {
		return nil
	}
	size := len(vectors[0])
	mean := make([]float32, size)
	var count int
	for _, vector := range vectors {
		if len(vector) != size {
			continue
		}
		for i, value := range vector {
			mean[i] += value
		}
		count++
	}
	if count == 0 {
		return nil
	}
	// Multiply-by-inverse avoids the per-element float divide; for the
	// multi-head non-GQA analysis path this loop runs through every
	// flat-state element of every layer.
	invScale := float32(1) / float32(count)
	for i := range mean {
		mean[i] *= invScale
	}
	return mean
}

func kvAnalysisPositionDifferentiation(heads []HeadSnapshot, seqLen, headDim int, keys bool, scratch []float64) (float64, int, int) {
	if seqLen < 2 || headDim <= 0 {
		return 0, 0, 0
	}
	// Pre-scale each position into float64 with `scaled[i][k] = v[i][k]/|v[i]|`
	// stored in a flat seqLen·headDim slice. The pair loop then computes
	// the cosine via a pure float64 dot product — no per-pair invA·invB
	// muls, no per-pair float32→float64 conversions (which previously
	// cost O(seqLen²·headDim) conversions vs O(seqLen·headDim) now), and
	// no per-pair invNorms[i]/invNorms[j] loads. Zero-norm positions are
	// left as all-zero rows in scratch — their dot product is 0 which is
	// below threshold=0.3, contributing locked++ + 0 similarity (matches
	// the original kvAnalysisCosine32 semantics). caller-owned `scratch`
	// is reused across all keys+values+layers; sized seqLen×headDim
	// float64s.
	scaledSize := seqLen * headDim
	if cap(scratch) < scaledSize {
		scratch = make([]float64, scaledSize)
	} else {
		scratch = scratch[:scaledSize]
	}
	threshold := 1.0 - kvCoherenceThreshold
	var totalSimilarity float64
	var locked, pairs int
	for _, head := range heads {
		flat := head.Value
		if keys {
			flat = head.Key
		}
		if len(flat) < scaledSize {
			continue
		}
		// Pass 1: convert + scale each position into float64 land. We
		// fold the 1/|v| scaling directly into the stored vector so the
		// pair loop is a plain dot product. Zero-norm positions get an
		// all-zero scratch row (dot product will be 0 → < threshold →
		// locked++), matching the original cosine-of-zero-vector
		// semantics. Accumulate totalSum here so the headDim=1 path
		// doesn't have to walk scratch[] a second time below.
		var totalSum float64
		for pos := 0; pos < seqLen; pos++ {
			start := pos * headDim
			row := flat[start : start+headDim]
			out := scratch[start : start+headDim]
			var sum float64
			for k, value := range row {
				v := float64(value)
				out[k] = v
				sum += v * v
			}
			if sum == 0 {
				// Zero the row — covers both the genuine zero-norm
				// case and any prior layer/head leftover.
				for k := range out {
					out[k] = 0
				}
				continue
			}
			inv := 1.0 / math.Sqrt(sum)
			for k := range out {
				out[k] *= inv
				totalSum += out[k]
			}
		}
		// Pass 2: pure float64 dot product. The cosine is the dot of
		// the pre-scaled rows directly — no per-pair multiplies needed.
		// Specialise headDim=1 — the inner k loop overhead is the
		// dominant cost when the loop only runs once.
		if headDim == 1 {
			// Split the per-pair similarity check by sign of ai so the
			// inner-loop locked compare is a direct compare-against-
			// constant (no per-iter mul + cmp serial dep). For ai>0
			// the condition (ai·aj < threshold) is equivalent to
			// aj < threshold/ai; for ai<0 it flips because we divided
			// by a negative. ai==0 short-circuits the whole row to
			// locked = (seqLen-i-1) since dot ≡ 0 < threshold.
			//
			// subSum = sum_{j>i} scratch[j] reduces to O(1) per i via
			// a running totalSum that subtracts scratch[i] as i
			// advances. Pulls the O(N²) FADDD chain out of the inner
			// loop, leaving the inner loop as load + compare + cinc
			// only (the M3 FCMPD/CINC dual-issue can ~saturate at
			// pair / cycle).
			//
			// Loops unrolled 4× to expose ILP — the OoO window covers
			// the L1 latency of scratch[j] loads. The locked compare
			// stays as a branch + counter (M3's FCMPD + CSEL fast path
			// beats the FMOV→shift trick whose float→int register move
			// has ~5-cycle latency on Apple Silicon).
			// totalSum was accumulated in Pass 1; the GQA path with
			// headDim>1 ignores it (we'd need per-position totals for
			// the general dot product, not a flat sum).
			subSum := totalSum
			for i := 0; i < seqLen; i++ {
				ai := scratch[i]
				remaining := seqLen - i - 1
				// subSum tracks sum_{j>i} scratch[j]. Subtract ai
				// before using since we need sum over j > i (exclusive).
				subSum -= ai
				if ai == 0 {
					// dot ≡ 0 for the rest of this row.
					locked += remaining
					continue
				}
				totalSimilarity += ai * subSum
				invT := threshold / ai
				// Re-slice scratch to the j-tail so bounds-check
				// elimination can prove each unrolled load is in range
				// from a single per-iteration length check.
				tail := scratch[i+1:]
				m := len(tail)
				k := 0
				if ai > 0 {
					for ; k+3 < m; k += 4 {
						// Re-slice to a fixed 4-element window so the
						// 4 loads share a single length check (BCE
						// sees window[3] cap=4 → no further checks).
						window := tail[k : k+4 : k+4]
						a0 := window[0]
						a1 := window[1]
						a2 := window[2]
						a3 := window[3]
						if a0 < invT {
							locked++
						}
						if a1 < invT {
							locked++
						}
						if a2 < invT {
							locked++
						}
						if a3 < invT {
							locked++
						}
					}
					for ; k < m; k++ {
						if tail[k] < invT {
							locked++
						}
					}
				} else {
					// ai < 0: condition is aj > invT (sign flipped).
					for ; k+3 < m; k += 4 {
						window := tail[k : k+4 : k+4]
						a0 := window[0]
						a1 := window[1]
						a2 := window[2]
						a3 := window[3]
						if a0 > invT {
							locked++
						}
						if a1 > invT {
							locked++
						}
						if a2 > invT {
							locked++
						}
						if a3 > invT {
							locked++
						}
					}
					for ; k < m; k++ {
						if tail[k] > invT {
							locked++
						}
					}
				}
			}
			pairs += seqLen * (seqLen - 1) / 2
			continue
		}
		for i := 0; i < seqLen; i++ {
			baseA := i * headDim
			rowA := scratch[baseA : baseA+headDim]
			for j := i + 1; j < seqLen; j++ {
				baseB := j * headDim
				rowB := scratch[baseB : baseB+headDim]
				// Pure float64 dot product — no float32 conversions,
				// no per-pair inverse-norm multiplications. Split the
				// accumulation across 4 parallel chains to break the
				// loop-carried FADDD dependency (3-cycle latency on M3);
				// the 4 chains issue on independent FADDD units, giving
				// ~4× throughput on the arithmetic side. Cache-bound for
				// large headDim·seqLen, but the per-pair tail still
				// benefits. Inlined here because Go won't inline a
				// helper call inside this O(seqLen²) loop and the call
				// overhead measured larger than the unroll win.
				var d0, d1, d2, d3 float64
				k := 0
				for ; k+3 < headDim; k += 4 {
					d0 += rowA[k] * rowB[k]
					d1 += rowA[k+1] * rowB[k+1]
					d2 += rowA[k+2] * rowB[k+2]
					d3 += rowA[k+3] * rowB[k+3]
				}
				dot := (d0 + d1) + (d2 + d3)
				for ; k < headDim; k++ {
					dot += rowA[k] * rowB[k]
				}
				totalSimilarity += dot
				if dot < threshold {
					locked++
				}
			}
		}
		pairs += seqLen * (seqLen - 1) / 2
	}
	if pairs == 0 {
		return 0, locked, pairs
	}
	return 1.0 - totalSimilarity/float64(pairs), locked, pairs
}

func kvAnalysisPositionVector(flat []float32, position, headDim int) []float32 {
	start := position * headDim
	end := start + headDim
	if start < 0 || end > len(flat) {
		return nil
	}
	return flat[start:end]
}

func kvAnalysisCosine32(a, b []float32) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot, normA, normB float64
	for i := range a {
		ai, bi := float64(a[i]), float64(b[i])
		dot += ai * bi
		normA += ai * ai
		normB += bi * bi
	}
	denom := math.Sqrt(normA) * math.Sqrt(normB)
	if denom == 0 {
		return 0
	}
	return dot / denom
}

func kvAnalysisHeadEntropy(head []float32, seqLen, headDim int, scratch []float64) float64 {
	if seqLen <= 1 || headDim <= 0 {
		return 0
	}
	// Single-pass via caller-owned scratch slice. The prior
	// implementation paid 2× sqrt + 2× inner FMA loop to avoid the
	// per-head allocation, but with analyzeKVGQA passing in a shared
	// buffer (reused across all heads + layers + sides) the alloc
	// cost falls to zero. scratch is cap-checked so over-eager callers
	// don't have to size it perfectly.
	if cap(scratch) < seqLen {
		scratch = make([]float64, seqLen)
	} else {
		scratch = scratch[:seqLen]
	}
	var total float64
	n := 0
	for pos := 0; pos < seqLen; pos++ {
		start := pos * headDim
		if start >= len(head) {
			break
		}
		end := start + headDim
		if end > len(head) {
			end = len(head)
		}
		var sum float64
		for _, value := range head[start:end] {
			v := float64(value)
			sum += v * v
		}
		mag := math.Sqrt(sum)
		scratch[n] = mag
		total += mag
		n++
	}
	if total == 0 {
		return 0
	}
	maxEntropy := math.Log2(float64(seqLen))
	if maxEntropy == 0 {
		return 0
	}
	invTotal := 1 / total
	var entropy float64
	for _, magnitude := range scratch[:n] {
		p := magnitude * invTotal
		if p > 0 {
			entropy -= p * math.Log2(p)
		}
	}
	return entropy / maxEntropy
}
