// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import "sync"

// ProbeEventKind names the typed payload carried by a probe event.
type ProbeEventKind string

const (
	ProbeEventToken           ProbeEventKind = "token"
	ProbeEventLogits          ProbeEventKind = "logits"
	ProbeEventEntropy         ProbeEventKind = "entropy"
	ProbeEventSelectedHeads   ProbeEventKind = "selected_heads"
	ProbeEventLayerCoherence  ProbeEventKind = "layer_coherence"
	ProbeEventRouterDecision  ProbeEventKind = "router_decision"
	ProbeEventExpertResidency ProbeEventKind = "expert_residency"
	ProbeEventResidual        ProbeEventKind = "residual_summary"
	ProbeEventCachePressure   ProbeEventKind = "cache_pressure"
	ProbeEventMemoryPressure  ProbeEventKind = "memory_pressure"
	ProbeEventTraining        ProbeEventKind = "training"
)

// ProbePhase identifies where the event was emitted in the runtime.
type ProbePhase string

const (
	ProbePhasePrefill  ProbePhase = "prefill"
	ProbePhaseDecode   ProbePhase = "decode"
	ProbePhaseTraining ProbePhase = "training"
)

// ProbeEvent is the first-class event envelope for inference and training probes.
type ProbeEvent struct {
	Kind            ProbeEventKind        `json:"kind"`
	Phase           ProbePhase            `json:"phase,omitempty"`
	Step            int                   `json:"step"`
	Token           *ProbeToken           `json:"token,omitempty"`
	Logits          *ProbeLogits          `json:"logits,omitempty"`
	Entropy         *ProbeEntropy         `json:"entropy,omitempty"`
	SelectedHeads   *ProbeHeadSelection   `json:"selected_heads,omitempty"`
	LayerCoherence  *ProbeLayerCoherence  `json:"layer_coherence,omitempty"`
	RouterDecision  *ProbeRouterDecision  `json:"router_decision,omitempty"`
	ExpertResidency *ProbeExpertResidency `json:"expert_residency,omitempty"`
	Residual        *ProbeResidualSummary `json:"residual,omitempty"`
	Cache           *ProbeCachePressure   `json:"cache,omitempty"`
	Memory          *ProbeMemoryPressure  `json:"memory,omitempty"`
	Training        *ProbeTraining        `json:"training,omitempty"`
	Meta            map[string]string     `json:"meta,omitempty"`
}

// ProbeToken records a selected token and local decode position.
type ProbeToken struct {
	ID              int32  `json:"id"`
	Text            string `json:"text,omitempty"`
	PromptTokens    int    `json:"prompt_tokens,omitempty"`
	GeneratedTokens int    `json:"generated_tokens,omitempty"`
}

// ProbeLogit records one high-scoring token from a logit vector.
type ProbeLogit struct {
	TokenID     int32   `json:"token_id"`
	Logit       float32 `json:"logit"`
	Probability float64 `json:"probability,omitempty"`
}

// ProbeLogits records a compact summary of a logit vector.
type ProbeLogits struct {
	Shape      []int32           `json:"shape,omitempty"`
	VocabSize  int               `json:"vocab_size,omitempty"`
	MaxTokenID int32             `json:"max_token_id"`
	MaxLogit   float32           `json:"max_logit"`
	MinTokenID int32             `json:"min_token_id"`
	MinLogit   float32           `json:"min_logit"`
	MeanLogit  float64           `json:"mean_logit"`
	Top        []ProbeLogit      `json:"top,omitempty"`
	Values     []float32         `json:"values,omitempty"`
	Meta       map[string]string `json:"meta,omitempty"`
}

// ProbeEntropy records the Shannon entropy of a probability distribution.
type ProbeEntropy struct {
	Value float64 `json:"value"`
	Unit  string  `json:"unit,omitempty"`
}

// ProbeHeadSelection records attention heads selected for a probe or analysis pass.
type ProbeHeadSelection struct {
	Layer  int       `json:"layer,omitempty"`
	Heads  []int     `json:"heads,omitempty"`
	Scores []float64 `json:"scores,omitempty"`
}

// ProbeLayerCoherence records per-layer K/V and residual posture metrics.
type ProbeLayerCoherence struct {
	Layer          int     `json:"layer,omitempty"`
	KeyCoherence   float64 `json:"key_coherence,omitempty"`
	ValueCoherence float64 `json:"value_coherence,omitempty"`
	CrossAlignment float64 `json:"cross_alignment,omitempty"`
	KVCoupling     float64 `json:"kv_coupling,omitempty"`
	HeadEntropy    float64 `json:"head_entropy,omitempty"`
	PhaseLock      float64 `json:"phase_lock,omitempty"`
}

// ProbeRouterDecision records MoE or routing decisions when the architecture exposes them.
type ProbeRouterDecision struct {
	Layer       int       `json:"layer,omitempty"`
	TokenID     int32     `json:"token_id,omitempty"`
	ExpertIDs   []int     `json:"expert_ids,omitempty"`
	Weights     []float32 `json:"weights,omitempty"`
	Temperature float32   `json:"temperature,omitempty"`
}

// ProbeExpertResidency records MoE expert paging and residency transitions.
type ProbeExpertResidency struct {
	Action             ExpertResidencyAction `json:"action"`
	Layer              int                   `json:"layer,omitempty"`
	ExpertIDs          []int                 `json:"expert_ids,omitempty"`
	ResidentExperts    int                   `json:"resident_experts,omitempty"`
	MaxResidentExperts int                   `json:"max_resident_experts,omitempty"`
	LoadedBytes        uint64                `json:"loaded_bytes,omitempty"`
	EvictedBytes       uint64                `json:"evicted_bytes,omitempty"`
	Duration           int64                 `json:"duration,omitempty"`
}

// ProbeResidualSummary records compact residual-stream statistics.
type ProbeResidualSummary struct {
	Layer    int     `json:"layer,omitempty"`
	Mean     float64 `json:"mean,omitempty"`
	Variance float64 `json:"variance,omitempty"`
	RMS      float64 `json:"rms,omitempty"`
	L2Norm   float64 `json:"l2_norm,omitempty"`
	MaxAbs   float64 `json:"max_abs,omitempty"`
}

// ProbeCachePressure records KV cache posture for local memory-aware runs.
type ProbeCachePressure struct {
	PromptTokens    int     `json:"prompt_tokens,omitempty"`
	GeneratedTokens int     `json:"generated_tokens,omitempty"`
	LayerCount      int     `json:"layer_count,omitempty"`
	CacheTokens     int     `json:"cache_tokens,omitempty"`
	ProcessedTokens int     `json:"processed_tokens,omitempty"`
	MaxCacheTokens  int     `json:"max_cache_tokens,omitempty"`
	Utilization     float64 `json:"utilization,omitempty"`
	Rotating        bool    `json:"rotating,omitempty"`
}

// ProbeMemoryPressure records MLX allocator pressure.
type ProbeMemoryPressure struct {
	ActiveBytes uint64 `json:"active_bytes,omitempty"`
	PeakBytes   uint64 `json:"peak_bytes,omitempty"`
	CacheBytes  uint64 `json:"cache_bytes,omitempty"`
}

// ProbeTraining records training-loop scalars.
type ProbeTraining struct {
	Step         int     `json:"step,omitempty"`
	Epoch        int     `json:"epoch,omitempty"`
	Loss         float64 `json:"loss,omitempty"`
	LearningRate float64 `json:"learning_rate,omitempty"`
	GradNorm     float64 `json:"grad_norm,omitempty"`
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

// ProbeBus fans probe events out to one or more sinks.
type ProbeBus struct {
	mu    sync.RWMutex
	sinks []ProbeSink
}

// NewProbeBus creates a fanout sink.
func NewProbeBus(sinks ...ProbeSink) *ProbeBus {
	bus := &ProbeBus{}
	for _, sink := range sinks {
		bus.Add(sink)
	}
	return bus
}

// Add appends a sink to the bus.
func (b *ProbeBus) Add(sink ProbeSink) {
	if b == nil || sink == nil {
		return
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	b.sinks = append(b.sinks, sink)
}

// EmitProbe emits an event to every sink.
func (b *ProbeBus) EmitProbe(event ProbeEvent) {
	if b == nil {
		return
	}
	b.mu.RLock()
	sinks := append([]ProbeSink(nil), b.sinks...)
	b.mu.RUnlock()
	for _, sink := range sinks {
		if sink != nil {
			sink.EmitProbe(cloneProbeEvent(event))
		}
	}
}

// ProbeRecorder stores probe events in memory for tests, reproducible probes, or artifacts.
type ProbeRecorder struct {
	mu     sync.Mutex
	events []ProbeEvent
}

// NewProbeRecorder returns a recorder sink.
func NewProbeRecorder() *ProbeRecorder {
	return &ProbeRecorder{}
}

// EmitProbe records an event.
func (r *ProbeRecorder) EmitProbe(event ProbeEvent) {
	if r == nil {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.events = append(r.events, cloneProbeEvent(event))
}

// Events returns recorded events without aliasing recorder storage.
func (r *ProbeRecorder) Events() []ProbeEvent {
	if r == nil {
		return nil
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	out := make([]ProbeEvent, len(r.events))
	for i, event := range r.events {
		out[i] = cloneProbeEvent(event)
	}
	return out
}

// WithProbeSink streams typed probe events during generation.
func WithProbeSink(sink ProbeSink) GenerateOption {
	return func(c *GenerateConfig) {
		c.ProbeSink = sink
	}
}

// WithProbeCallback streams typed probe events to a callback during generation.
func WithProbeCallback(callback func(ProbeEvent)) GenerateOption {
	if callback == nil {
		return func(*GenerateConfig) {}
	}
	return WithProbeSink(ProbeSinkFunc(callback))
}

func cloneProbeEvent(event ProbeEvent) ProbeEvent {
	out := event
	if event.Token != nil {
		token := *event.Token
		out.Token = &token
	}
	if event.Logits != nil {
		logits := *event.Logits
		logits.Shape = append([]int32(nil), event.Logits.Shape...)
		logits.Top = append([]ProbeLogit(nil), event.Logits.Top...)
		logits.Values = append([]float32(nil), event.Logits.Values...)
		logits.Meta = cloneProbeMeta(event.Logits.Meta)
		out.Logits = &logits
	}
	if event.Entropy != nil {
		entropy := *event.Entropy
		out.Entropy = &entropy
	}
	if event.SelectedHeads != nil {
		heads := *event.SelectedHeads
		heads.Heads = append([]int(nil), event.SelectedHeads.Heads...)
		heads.Scores = append([]float64(nil), event.SelectedHeads.Scores...)
		out.SelectedHeads = &heads
	}
	if event.LayerCoherence != nil {
		coherence := *event.LayerCoherence
		out.LayerCoherence = &coherence
	}
	if event.RouterDecision != nil {
		router := *event.RouterDecision
		router.ExpertIDs = append([]int(nil), event.RouterDecision.ExpertIDs...)
		router.Weights = append([]float32(nil), event.RouterDecision.Weights...)
		out.RouterDecision = &router
	}
	if event.ExpertResidency != nil {
		residency := *event.ExpertResidency
		residency.ExpertIDs = append([]int(nil), event.ExpertResidency.ExpertIDs...)
		out.ExpertResidency = &residency
	}
	if event.Residual != nil {
		residual := *event.Residual
		out.Residual = &residual
	}
	if event.Cache != nil {
		cache := *event.Cache
		out.Cache = &cache
	}
	if event.Memory != nil {
		memory := *event.Memory
		out.Memory = &memory
	}
	if event.Training != nil {
		training := *event.Training
		out.Training = &training
	}
	out.Meta = cloneProbeMeta(event.Meta)
	return out
}

func cloneProbeMeta(meta map[string]string) map[string]string {
	if len(meta) == 0 {
		return nil
	}
	out := make(map[string]string, len(meta))
	for key, value := range meta {
		out[key] = value
	}
	return out
}
