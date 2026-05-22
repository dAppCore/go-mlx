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
				entropyTotal += kvAnalysisHeadEntropy(head.Key, snapshot.SeqLen, snapshot.HeadDim)
				entropyCount++
			}
			if len(head.Value) > 0 {
				entropyTotal += kvAnalysisHeadEntropy(head.Value, snapshot.SeqLen, snapshot.HeadDim)
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

	// One invNorms scratch per Analyze — reused across all layer
	// keys+values calls to avoid per-layer/per-side allocations
	// (snapshot.SeqLen × 8 bytes × layers × 2 sides).
	var invNorms []float64
	if snapshot.SeqLen > 0 {
		invNorms = make([]float64, snapshot.SeqLen)
	}

	for layer := range numLayers {
		layerSnapshot, ok := snapshot.layer(layer)
		if !ok || len(layerSnapshot.Heads) == 0 {
			continue
		}
		keyDiff, keyLocked, keyPairs := kvAnalysisPositionDifferentiation(layerSnapshot.Heads, snapshot.SeqLen, snapshot.HeadDim, true, invNorms)
		valueDiff, valueLocked, valuePairs := kvAnalysisPositionDifferentiation(layerSnapshot.Heads, snapshot.SeqLen, snapshot.HeadDim, false, invNorms)
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
				entropyTotal += kvAnalysisHeadEntropy(head.Key, snapshot.SeqLen, snapshot.HeadDim)
				entropyCount++
			}
			if len(head.Value) > 0 {
				entropyTotal += kvAnalysisHeadEntropy(head.Value, snapshot.SeqLen, snapshot.HeadDim)
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
	var total float64
	var locked, pairs int
	for i := 0; i < len(vectors); i++ {
		for j := i + 1; j < len(vectors); j++ {
			similarity := kvAnalysisCosine32(vectors[i], vectors[j])
			total += similarity
			pairs++
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
	// At most one state slot per head — pre-size to skip the
	// geometric-grow append cycle.
	states := make([][]float32, 0, len(heads))
	for _, head := range heads {
		if len(head.Key) == 0 && len(head.Value) == 0 {
			continue
		}
		combined := make([]float32, 0, len(head.Key)+len(head.Value))
		combined = append(combined, head.Key...)
		combined = append(combined, head.Value...)
		states = append(states, combined)
	}
	return kvAnalysisMeanVector(states)
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

func kvAnalysisPositionDifferentiation(heads []HeadSnapshot, seqLen, headDim int, keys bool, invNorms []float64) (float64, int, int) {
	if seqLen < 2 || headDim <= 0 {
		return 0, 0, 0
	}
	// Precompute per-position inverse norms once (O(seqLen)) so the
	// O(seqLen²) pair loop only pays a dot product + 2 muls. Previously
	// each pair recomputed normA + normB inside kvAnalysisCosine32,
	// giving O(seqLen²·headDim) self-norm work — pure waste because
	// normA only depends on position i. Drops Analyze_2048Tokens from
	// ~28ms to a fraction. invNorms is caller-owned scratch reused
	// across keys+values+layers.
	if cap(invNorms) < seqLen {
		invNorms = make([]float64, seqLen)
	} else {
		invNorms = invNorms[:seqLen]
	}
	threshold := 1.0 - kvCoherenceThreshold
	var totalSimilarity float64
	var locked, pairs int
	for _, head := range heads {
		flat := head.Value
		if keys {
			flat = head.Key
		}
		if len(flat) < seqLen*headDim {
			continue
		}
		// Pass 1: per-position |v| as 1/|v| (or 0 for zero positions).
		for pos := 0; pos < seqLen; pos++ {
			start := pos * headDim
			row := flat[start : start+headDim]
			var sum float64
			for _, value := range row {
				v := float64(value)
				sum += v * v
			}
			if sum == 0 {
				invNorms[pos] = 0
			} else {
				invNorms[pos] = 1.0 / math.Sqrt(sum)
			}
		}
		// Pass 2: pairwise dot products only — divide once with
		// precomputed inverse norms, no per-pair sqrt.
		for i := 0; i < seqLen; i++ {
			invA := invNorms[i]
			if invA == 0 {
				// Pairs with the zero-norm row still increment counters
				// (matches the original kvAnalysisCosine32 behaviour of
				// returning 0 similarity on degenerate inputs).
				for j := i + 1; j < seqLen; j++ {
					if invNorms[j] == 0 {
						continue
					}
					pairs++
					if threshold > 0 {
						locked++
					}
				}
				continue
			}
			rowA := flat[i*headDim : (i+1)*headDim]
			for j := i + 1; j < seqLen; j++ {
				invB := invNorms[j]
				if invB == 0 {
					pairs++
					if threshold > 0 {
						locked++
					}
					continue
				}
				rowB := flat[j*headDim : (j+1)*headDim]
				var dot float64
				for k := range rowA {
					dot += float64(rowA[k]) * float64(rowB[k])
				}
				similarity := dot * invA * invB
				totalSimilarity += similarity
				pairs++
				if similarity < threshold {
					locked++
				}
			}
		}
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

func kvAnalysisHeadEntropy(head []float32, seqLen, headDim int) float64 {
	if seqLen <= 1 || headDim <= 0 {
		return 0
	}
	// Two-pass without retaining magnitudes — first pass accumulates
	// sqrt(sum-of-squares) per position into the running total; second
	// pass recomputes the same magnitudes for entropy. This drops the
	// per-head []float64{seqLen} allocation (16KB at seqLen=2048) which
	// dominated Analyze's per-call alloc footprint.
	var total float64
	for pos := 0; pos < seqLen; pos++ {
		start := pos * headDim
		if start >= len(head) {
			break
		}
		var sum float64
		end := start + headDim
		if end > len(head) {
			end = len(head)
		}
		for _, value := range head[start:end] {
			v := float64(value)
			sum += v * v
		}
		total += math.Sqrt(sum)
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
	for pos := 0; pos < seqLen; pos++ {
		start := pos * headDim
		if start >= len(head) {
			break
		}
		var sum float64
		end := start + headDim
		if end > len(head) {
			end = len(head)
		}
		for _, value := range head[start:end] {
			v := float64(value)
			sum += v * v
		}
		magnitude := math.Sqrt(sum)
		p := magnitude * invTotal
		if p > 0 {
			entropy -= p * math.Log2(p)
		}
	}
	return entropy / maxEntropy
}
