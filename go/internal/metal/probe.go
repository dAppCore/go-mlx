// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"math"
	"sort"

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
	shape := logits.Shape()
	if len(shape) == 0 {
		return ProbeLogits{}, ProbeEntropy{}, false, nil
	}
	vocabSize := int(shape[len(shape)-1])
	if vocabSize <= 0 {
		return ProbeLogits{}, ProbeEntropy{}, false, nil
	}
	topK = compactProbeTopK(topK, vocabSize)
	row, cleanup, ok := lastProbeLogitRow(logits, shape, vocabSize)
	defer Free(cleanup...)
	if !ok {
		return ProbeLogits{}, ProbeEntropy{}, false, nil
	}

	summary, entropy, err := summarizeProbeLogitsCompact(row, shape, vocabSize, topK)
	if err != nil {
		return ProbeLogits{}, ProbeEntropy{}, false, err
	}
	return summary, entropy, true, nil
}

func compactProbeTopK(topK, vocabSize int) int {
	if topK <= 0 {
		topK = defaultProbeTopK
	}
	if topK > vocabSize {
		topK = vocabSize
	}
	return topK
}

func lastProbeLogitRow(logits *Array, shape []int32, vocabSize int) (*Array, []*Array, bool) {
	rows := 1
	for _, dim := range shape[:len(shape)-1] {
		if dim <= 0 {
			return nil, nil, false
		}
		rows *= int(dim)
	}
	if rows <= 0 {
		return nil, nil, false
	}
	reshaped := Reshape(logits, int32(rows), int32(vocabSize))
	row := SliceAxis(reshaped, 0, int32(rows-1), int32(rows))
	return row, []*Array{reshaped, row}, true
}

func summarizeProbeLogitsCompact(row *Array, shape []int32, vocabSize, topK int) (ProbeLogits, ProbeEntropy, error) {
	neg := Negative(row)
	topIndicesAll := Argpartition(neg, topK-1, -1)
	topIndices := SliceAxis(topIndicesAll, -1, 0, int32(topK))
	topValues := TakeAlongAxis(row, topIndices, -1)
	maxTokenID := Argmax(row, -1, false)
	maxLogit := MaxAxis(row, -1, false)
	minTokenID := Argmax(neg, -1, false)
	negMinLogit := MaxAxis(neg, -1, false)
	meanLogit := Mean(row, -1, false)
	logSumExp := LogSumExp(row, -1, false)
	probabilities := Softmax(row)
	weightedLogits := Mul(probabilities, row)
	expectedLogit := Sum(weightedLogits, -1, false)
	entropy := Subtract(logSumExp, expectedLogit)
	defer Free(
		neg,
		topIndicesAll,
		topIndices,
		topValues,
		maxTokenID,
		maxLogit,
		minTokenID,
		negMinLogit,
		meanLogit,
		logSumExp,
		probabilities,
		weightedLogits,
		expectedLogit,
		entropy,
	)
	if err := Eval(topIndices, topValues, maxTokenID, maxLogit, minTokenID, negMinLogit, meanLogit, logSumExp, entropy); err != nil {
		return ProbeLogits{}, ProbeEntropy{}, core.E("probe.logits", "compact", err)
	}

	topIDs := topIndices.Ints()
	topLogits := topValues.Floats()

	summary := ProbeLogits{
		Shape:      append([]int32(nil), shape...),
		VocabSize:  vocabSize,
		MaxTokenID: int32(maxTokenID.Int()),
		MaxLogit:   float32(maxLogit.Float()),
		MinTokenID: int32(minTokenID.Int()),
		MinLogit:   float32(-negMinLogit.Float()),
		MeanLogit:  meanLogit.Float(),
		Top:        make([]ProbeLogit, 0, len(topIDs)),
		Meta:       map[string]string{"cpu_transfer": "compact_topk"},
	}
	logZ := logSumExp.Float()
	for i, id := range topIDs {
		if i >= len(topLogits) {
			continue
		}
		value := topLogits[i]
		summary.Top = append(summary.Top, ProbeLogit{
			TokenID:     int32(id),
			Logit:       value,
			Probability: math.Exp(float64(value) - logZ),
		})
	}
	sort.Slice(summary.Top, func(i, j int) bool {
		if summary.Top[i].Logit == summary.Top[j].Logit {
			return summary.Top[i].TokenID < summary.Top[j].TokenID
		}
		return summary.Top[i].Logit > summary.Top[j].Logit
	})
	return summary, ProbeEntropy{Value: entropy.Float(), Unit: "nats"}, nil
}
