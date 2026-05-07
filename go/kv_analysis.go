// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import "math"

const (
	kvCoherenceThreshold = 0.7
	kvCollapseThreshold  = 0.5
)

// KVAnalysis contains K/V cache coherence metrics for one prefill snapshot.
type KVAnalysis struct {
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
func (r *KVAnalysis) Composite() int {
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

// AnalyzeKV computes coherence metrics from a CPU-readable KV cache snapshot.
func AnalyzeKV(snapshot *KVSnapshot) *KVAnalysis {
	if snapshot == nil || len(snapshot.Layers) == 0 {
		return &KVAnalysis{}
	}
	if kvAnalysisNumHeads(snapshot) <= 4 {
		return analyzeKVGQA(snapshot)
	}
	return analyzeKVMultiHead(snapshot)
}

func analyzeKVMultiHead(snapshot *KVSnapshot) *KVAnalysis {
	numLayers := kvAnalysisNumLayers(snapshot)
	result := &KVAnalysis{
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

func analyzeKVGQA(snapshot *KVSnapshot) *KVAnalysis {
	numLayers := kvAnalysisNumLayers(snapshot)
	result := &KVAnalysis{
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

	for layer := range numLayers {
		layerSnapshot, ok := snapshot.layer(layer)
		if !ok || len(layerSnapshot.Heads) == 0 {
			continue
		}
		keyDiff, keyLocked, keyPairs := kvAnalysisPositionDifferentiation(layerSnapshot.Heads, snapshot.SeqLen, snapshot.HeadDim, true)
		valueDiff, valueLocked, valuePairs := kvAnalysisPositionDifferentiation(layerSnapshot.Heads, snapshot.SeqLen, snapshot.HeadDim, false)
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

// KVFeatures returns the 7D model-state feature vector from K/V metrics.
func KVFeatures(result *KVAnalysis) []float64 {
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

// KVFeatureLabels returns labels matching KVFeatures order.
func KVFeatureLabels() []string {
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

func kvAnalysisNumLayers(snapshot *KVSnapshot) int {
	if snapshot == nil {
		return 0
	}
	if snapshot.NumLayers > 0 {
		return snapshot.NumLayers
	}
	return len(snapshot.Layers)
}

func kvAnalysisNumHeads(snapshot *KVSnapshot) int {
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

func kvSharedCacheLayerGroups(snapshot *KVSnapshot) map[int][]int {
	groups := make(map[int][]int)
	if snapshot == nil {
		return groups
	}
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

func kvAnalysisHeadVectors(heads []KVHeadSnapshot, keys bool) [][]float32 {
	vectors := make([][]float32, 0, len(heads))
	for _, head := range heads {
		if keys {
			vectors = append(vectors, head.Key)
			continue
		}
		vectors = append(vectors, head.Value)
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

func kvAnalysisLayerCoupling(heads []KVHeadSnapshot) (float64, int) {
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

func kvAnalysisLayerState(heads []KVHeadSnapshot) []float32 {
	if len(heads) == 0 {
		return nil
	}
	var states [][]float32
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
	scale := float32(count)
	for i := range mean {
		mean[i] /= scale
	}
	return mean
}

func kvAnalysisPositionDifferentiation(heads []KVHeadSnapshot, seqLen, headDim int, keys bool) (float64, int, int) {
	if seqLen < 2 || headDim <= 0 {
		return 0, 0, 0
	}
	var totalSimilarity float64
	var locked, pairs int
	for _, head := range heads {
		flat := head.Value
		if keys {
			flat = head.Key
		}
		for i := 0; i < seqLen; i++ {
			first := kvAnalysisPositionVector(flat, i, headDim)
			if first == nil {
				continue
			}
			for j := i + 1; j < seqLen; j++ {
				second := kvAnalysisPositionVector(flat, j, headDim)
				if second == nil {
					continue
				}
				similarity := kvAnalysisCosine32(first, second)
				totalSimilarity += similarity
				pairs++
				if similarity < 1.0-kvCoherenceThreshold {
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
	magnitudes := make([]float64, seqLen)
	var total float64
	for pos := 0; pos < seqLen; pos++ {
		start := pos * headDim
		if start >= len(head) {
			break
		}
		var sum float64
		for dim := 0; dim < headDim && start+dim < len(head); dim++ {
			value := float64(head[start+dim])
			sum += value * value
		}
		magnitudes[pos] = math.Sqrt(sum)
		total += magnitudes[pos]
	}
	if total == 0 {
		return 0
	}
	var entropy float64
	for _, magnitude := range magnitudes {
		p := magnitude / total
		if p > 0 {
			entropy -= p * math.Log2(p)
		}
	}
	maxEntropy := math.Log2(float64(seqLen))
	if maxEntropy == 0 {
		return 0
	}
	return entropy / maxEntropy
}
