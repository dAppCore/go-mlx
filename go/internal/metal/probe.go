// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"math"

	core "dappco.re/go"
)

const defaultProbeTopK = 8

// ProbeEventKind names the typed payload carried by a probe event.
type ProbeEventKind string

const (
	ProbeEventToken          ProbeEventKind = "token"
	ProbeEventLogits         ProbeEventKind = "logits"
	ProbeEventEntropy        ProbeEventKind = "entropy"
	ProbeEventSelectedHeads  ProbeEventKind = "selected_heads"
	ProbeEventLayerCoherence ProbeEventKind = "layer_coherence"
	ProbeEventRouterDecision ProbeEventKind = "router_decision"
	ProbeEventResidual       ProbeEventKind = "residual_summary"
	ProbeEventCachePressure  ProbeEventKind = "cache_pressure"
	ProbeEventMemoryPressure ProbeEventKind = "memory_pressure"
	ProbeEventTraining       ProbeEventKind = "training"
)

// ProbePhase identifies where the event was emitted in the runtime.
type ProbePhase string

const (
	ProbePhasePrefill  ProbePhase = "prefill"
	ProbePhaseDecode   ProbePhase = "decode"
	ProbePhaseTraining ProbePhase = "training"
)

// ProbeEvent is the event envelope used by native inference and training.
type ProbeEvent struct {
	Kind           ProbeEventKind
	Phase          ProbePhase
	Step           int
	Token          *ProbeToken
	Logits         *ProbeLogits
	Entropy        *ProbeEntropy
	SelectedHeads  *ProbeHeadSelection
	LayerCoherence *ProbeLayerCoherence
	RouterDecision *ProbeRouterDecision
	Residual       *ProbeResidualSummary
	Cache          *ProbeCachePressure
	Memory         *ProbeMemoryPressure
	Training       *ProbeTraining
	Meta           map[string]string
}

// ProbeToken records a selected token and local decode position.
type ProbeToken struct {
	ID              int32
	Text            string
	PromptTokens    int
	GeneratedTokens int
}

// ProbeLogit records one high-scoring token from a logit vector.
type ProbeLogit struct {
	TokenID     int32
	Logit       float32
	Probability float64
}

// ProbeLogits records a compact summary of a logit vector.
type ProbeLogits struct {
	Shape      []int32
	VocabSize  int
	MaxTokenID int32
	MaxLogit   float32
	MinTokenID int32
	MinLogit   float32
	MeanLogit  float64
	Top        []ProbeLogit
	Values     []float32
	Meta       map[string]string
}

// ProbeEntropy records the Shannon entropy of a probability distribution.
type ProbeEntropy struct {
	Value float64
	Unit  string
}

// ProbeHeadSelection records attention heads selected for a probe or analysis pass.
type ProbeHeadSelection struct {
	Layer  int
	Heads  []int
	Scores []float64
}

// ProbeLayerCoherence records per-layer K/V and residual posture metrics.
type ProbeLayerCoherence struct {
	Layer          int
	KeyCoherence   float64
	ValueCoherence float64
	CrossAlignment float64
	KVCoupling     float64
	HeadEntropy    float64
	PhaseLock      float64
}

// ProbeRouterDecision records MoE or routing decisions when the architecture exposes them.
type ProbeRouterDecision struct {
	Layer       int
	TokenID     int32
	ExpertIDs   []int
	Weights     []float32
	Temperature float32
}

// ProbeResidualSummary records compact residual-stream statistics.
type ProbeResidualSummary struct {
	Layer    int
	Mean     float64
	Variance float64
	RMS      float64
	L2Norm   float64
	MaxAbs   float64
}

// ProbeCachePressure records KV cache posture for local memory-aware runs.
type ProbeCachePressure struct {
	PromptTokens    int
	GeneratedTokens int
	LayerCount      int
	CacheTokens     int
	ProcessedTokens int
	MaxCacheTokens  int
	Utilization     float64
	Rotating        bool
}

// ProbeMemoryPressure records MLX allocator pressure.
type ProbeMemoryPressure struct {
	ActiveBytes uint64
	PeakBytes   uint64
	CacheBytes  uint64
}

// ProbeTraining records training-loop scalars.
type ProbeTraining struct {
	Step         int
	Epoch        int
	Loss         float64
	LearningRate float64
	GradNorm     float64
}

// ProbeSink consumes typed probe events.
type ProbeSink interface {
	EmitProbe(ProbeEvent)
}

// ProbeSinkFunc adapts a function into a ProbeSink.
type ProbeSinkFunc func(ProbeEvent)

// EmitProbe emits an event to the wrapped function.
func (f ProbeSinkFunc) EmitProbe(event ProbeEvent) {
	if f != nil {
		f(event)
	}
}

func emitProbe(sink ProbeSink, event ProbeEvent) {
	if sink != nil {
		sink.EmitProbe(event)
	}
}

func emitProbeLogits(sink ProbeSink, phase ProbePhase, step int, logits *Array) error {
	if sink == nil {
		return nil
	}
	summary, entropy, ok, err := summarizeProbeLogits(logits, defaultProbeTopK)
	if err != nil || !ok {
		return err
	}
	emitProbe(sink, ProbeEvent{
		Kind:   ProbeEventLogits,
		Phase:  phase,
		Step:   step,
		Logits: &summary,
	})
	emitProbe(sink, ProbeEvent{
		Kind:    ProbeEventEntropy,
		Phase:   phase,
		Step:    step,
		Entropy: &entropy,
	})
	return nil
}

func emitProbeToken(sink ProbeSink, phase ProbePhase, step int, id int32, text string, promptTokens, generatedTokens int) {
	if sink == nil {
		return
	}
	emitProbe(sink, ProbeEvent{
		Kind:  ProbeEventToken,
		Phase: phase,
		Step:  step,
		Token: &ProbeToken{
			ID:              id,
			Text:            text,
			PromptTokens:    promptTokens,
			GeneratedTokens: generatedTokens,
		},
	})
}

func emitProbeCachePressure(sink ProbeSink, phase ProbePhase, promptTokens, generatedTokens, step int, caches []Cache) {
	if sink == nil {
		return
	}
	emitProbe(sink, probeCachePressure(phase, promptTokens, generatedTokens, step, caches))
}

func probeCachePressure(phase ProbePhase, promptTokens, generatedTokens, step int, caches []Cache) ProbeEvent {
	cache := &ProbeCachePressure{
		PromptTokens:    promptTokens,
		GeneratedTokens: generatedTokens,
		LayerCount:      len(caches),
	}
	for _, layerCache := range caches {
		if layerCache == nil {
			continue
		}
		cache.CacheTokens = max(cache.CacheTokens, layerCache.Len())
		cache.ProcessedTokens = max(cache.ProcessedTokens, layerCache.Offset())
		if rotating, ok := layerCache.(*RotatingKVCache); ok {
			cache.Rotating = true
			cache.MaxCacheTokens = max(cache.MaxCacheTokens, rotating.maxSize)
		}
	}
	if cache.ProcessedTokens == 0 {
		cache.ProcessedTokens = promptTokens + generatedTokens
	}
	if cache.MaxCacheTokens > 0 {
		cache.Utilization = float64(cache.CacheTokens) / float64(cache.MaxCacheTokens)
	}
	return ProbeEvent{
		Kind:  ProbeEventCachePressure,
		Phase: phase,
		Step:  step,
		Cache: cache,
	}
}

func emitProbeMemoryPressure(sink ProbeSink, phase ProbePhase, step int) {
	if sink == nil {
		return
	}
	emitProbe(sink, ProbeEvent{
		Kind:  ProbeEventMemoryPressure,
		Phase: phase,
		Step:  step,
		Memory: &ProbeMemoryPressure{
			ActiveBytes: GetActiveMemory(),
			PeakBytes:   GetPeakMemory(),
			CacheBytes:  GetCacheMemory(),
		},
	})
}

func summarizeProbeLogits(logits *Array, topK int) (ProbeLogits, ProbeEntropy, bool, error) {
	if logits == nil || !logits.Valid() {
		return ProbeLogits{}, ProbeEntropy{}, false, nil
	}
	if err := Eval(logits); err != nil {
		return ProbeLogits{}, ProbeEntropy{}, false, core.E("probe.logits", "eval", err)
	}
	shape := logits.Shape()
	if len(shape) == 0 {
		return ProbeLogits{}, ProbeEntropy{}, false, nil
	}
	vocabSize := int(shape[len(shape)-1])
	if vocabSize <= 0 {
		return ProbeLogits{}, ProbeEntropy{}, false, nil
	}
	flat := logits.Floats()
	if len(flat) < vocabSize {
		return ProbeLogits{}, ProbeEntropy{}, false, nil
	}
	row := flat[len(flat)-vocabSize:]
	if topK <= 0 || topK > len(row) {
		topK = len(row)
	}

	summary := ProbeLogits{
		Shape:     append([]int32(nil), shape...),
		VocabSize: vocabSize,
		Top:       make([]ProbeLogit, 0, topK),
	}
	var (
		maxLogit    = math.Inf(-1)
		minLogit    = math.Inf(1)
		finiteSum   float64
		finiteCount int
		validCount  int
		posInfCount int
	)
	for idx, value32 := range row {
		value := float64(value32)
		if math.IsNaN(value) {
			continue
		}
		validCount++
		if value > maxLogit {
			maxLogit = value
			summary.MaxTokenID = int32(idx)
			summary.MaxLogit = value32
		}
		if value < minLogit {
			minLogit = value
			summary.MinTokenID = int32(idx)
			summary.MinLogit = value32
		}
		if !math.IsInf(value, 0) {
			finiteSum += value
			finiteCount++
		}
		if math.IsInf(value, 1) {
			posInfCount++
		}
		summary.Top = insertProbeTop(summary.Top, ProbeLogit{
			TokenID: int32(idx),
			Logit:   value32,
		}, topK)
	}
	if validCount == 0 {
		return ProbeLogits{}, ProbeEntropy{}, false, nil
	}
	if finiteCount > 0 {
		summary.MeanLogit = finiteSum / float64(finiteCount)
	}

	entropyValue, probabilities := probeEntropyAndTopProbabilities(row, summary.Top, maxLogit, posInfCount)
	for i := range summary.Top {
		summary.Top[i].Probability = probabilities[i]
	}
	return summary, ProbeEntropy{Value: entropyValue, Unit: "nats"}, true, nil
}

func insertProbeTop(top []ProbeLogit, candidate ProbeLogit, limit int) []ProbeLogit {
	if limit <= 0 {
		return top
	}
	pos := len(top)
	for i, existing := range top {
		if candidate.Logit > existing.Logit {
			pos = i
			break
		}
	}
	if pos >= limit {
		return top
	}
	top = append(top, ProbeLogit{})
	copy(top[pos+1:], top[pos:])
	top[pos] = candidate
	if len(top) > limit {
		top = top[:limit]
	}
	return top
}

func probeEntropyAndTopProbabilities(row []float32, top []ProbeLogit, maxLogit float64, posInfCount int) (float64, []float64) {
	probabilities := make([]float64, len(top))
	if len(row) == 0 {
		return 0, probabilities
	}
	if posInfCount > 0 {
		probability := 1.0 / float64(posInfCount)
		for i, candidate := range top {
			if math.IsInf(float64(candidate.Logit), 1) {
				probabilities[i] = probability
			}
		}
		return math.Log(float64(posInfCount)), probabilities
	}

	var sumExp float64
	for _, value32 := range row {
		value := float64(value32)
		if math.IsNaN(value) {
			continue
		}
		sumExp += math.Exp(value - maxLogit)
	}
	if sumExp <= 0 {
		return 0, probabilities
	}

	var entropy float64
	for _, value32 := range row {
		value := float64(value32)
		if math.IsNaN(value) {
			continue
		}
		probability := math.Exp(value-maxLogit) / sumExp
		if probability > 0 {
			entropy -= probability * math.Log(probability)
		}
	}
	for i, candidate := range top {
		value := float64(candidate.Logit)
		if math.IsNaN(value) {
			continue
		}
		probabilities[i] = math.Exp(value-maxLogit) / sumExp
	}
	return entropy, probabilities
}
