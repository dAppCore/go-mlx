// SPDX-Licence-Identifier: EUPL-1.2

// Package probe is the go-mlx event-vocabulary for first-class
// observability of inference and training. Backends emit typed Events
// through a Sink; Bus fans events out to multiple sinks, Recorder stores
// them in memory for tests and reproducible probes.
//
//	recorder := probe.NewRecorder()
//	bus := probe.NewBus(recorder, callerSink)
//	bus.EmitProbe(probe.Event{Kind: probe.KindToken, Token: &probe.Token{ID: 7}})
//	events := recorder.Events()
package probe

import (
	"sync"

	core "dappco.re/go"
)

// Kind names the typed payload carried by a probe event.
type Kind string

// Phase identifies where the event was emitted in the runtime.
type Phase string

const (
	KindToken           Kind = "token"
	KindLogits          Kind = "logits"
	KindEntropy         Kind = "entropy"
	KindSelectedHeads   Kind = "selected_heads"
	KindLayerCoherence  Kind = "layer_coherence"
	KindRouterDecision  Kind = "router_decision"
	KindExpertResidency Kind = "expert_residency"
	KindResidual        Kind = "residual_summary"
	KindCachePressure   Kind = "cache_pressure"
	KindMemoryPressure  Kind = "memory_pressure"
	KindTraining        Kind = "training"

	PhasePrefill  Phase = "prefill"
	PhaseDecode   Phase = "decode"
	PhaseTraining Phase = "training"
)

// Event is the first-class event envelope for inference and training probes.
type Event struct {
	Kind            Kind              `json:"kind"`
	Phase           Phase             `json:"phase,omitempty"`
	Step            int               `json:"step"`
	Token           *Token            `json:"token,omitempty"`
	Logits          *Logits           `json:"logits,omitempty"`
	Entropy         *Entropy          `json:"entropy,omitempty"`
	SelectedHeads   *HeadSelection    `json:"selected_heads,omitempty"`
	LayerCoherence  *LayerCoherence   `json:"layer_coherence,omitempty"`
	RouterDecision  *RouterDecision   `json:"router_decision,omitempty"`
	ExpertResidency *ExpertResidency  `json:"expert_residency,omitempty"`
	Residual        *ResidualSummary  `json:"residual,omitempty"`
	Cache           *CachePressure    `json:"cache,omitempty"`
	Memory          *MemoryPressure   `json:"memory,omitempty"`
	Training        *Training         `json:"training,omitempty"`
	Meta            map[string]string `json:"meta,omitempty"`
}

// Token records a selected token and local decode position.
type Token struct {
	ID              int32  `json:"id"`
	Text            string `json:"text,omitempty"`
	PromptTokens    int    `json:"prompt_tokens,omitempty"`
	GeneratedTokens int    `json:"generated_tokens,omitempty"`
}

// Logit records one high-scoring token from a logit vector.
type Logit struct {
	TokenID     int32   `json:"token_id"`
	Logit       float32 `json:"logit"`
	Probability float64 `json:"probability,omitempty"`
}

// Logits records a compact summary of a logit vector.
type Logits struct {
	Shape      []int32           `json:"shape,omitempty"`
	VocabSize  int               `json:"vocab_size,omitempty"`
	MaxTokenID int32             `json:"max_token_id"`
	MaxLogit   float32           `json:"max_logit"`
	MinTokenID int32             `json:"min_token_id"`
	MinLogit   float32           `json:"min_logit"`
	MeanLogit  float64           `json:"mean_logit"`
	Top        []Logit           `json:"top,omitempty"`
	Values     []float32         `json:"values,omitempty"`
	Meta       map[string]string `json:"meta,omitempty"`
}

// Entropy records the Shannon entropy of a probability distribution.
type Entropy struct {
	Value float64 `json:"value"`
	Unit  string  `json:"unit,omitempty"`
}

// HeadSelection records attention heads selected for a probe or analysis pass.
type HeadSelection struct {
	Layer  int       `json:"layer,omitempty"`
	Heads  []int     `json:"heads,omitempty"`
	Scores []float64 `json:"scores,omitempty"`
}

// LayerCoherence records per-layer K/V and residual posture metrics.
type LayerCoherence struct {
	Layer          int     `json:"layer,omitempty"`
	KeyCoherence   float64 `json:"key_coherence,omitempty"`
	ValueCoherence float64 `json:"value_coherence,omitempty"`
	CrossAlignment float64 `json:"cross_alignment,omitempty"`
	KVCoupling     float64 `json:"kv_coupling,omitempty"`
	HeadEntropy    float64 `json:"head_entropy,omitempty"`
	PhaseLock      float64 `json:"phase_lock,omitempty"`
}

// RouterDecision records MoE or routing decisions when the architecture exposes them.
type RouterDecision struct {
	Layer       int       `json:"layer,omitempty"`
	TokenID     int32     `json:"token_id,omitempty"`
	ExpertIDs   []int     `json:"expert_ids,omitempty"`
	Weights     []float32 `json:"weights,omitempty"`
	Temperature float32   `json:"temperature,omitempty"`
}

// ExpertResidencyAction names probe-visible expert residency transitions.
type ExpertResidencyAction string

const (
	ExpertResidencyActionStartup ExpertResidencyAction = "startup"
	ExpertResidencyActionPageIn  ExpertResidencyAction = "page_in"
	ExpertResidencyActionEvict   ExpertResidencyAction = "evict"
	ExpertResidencyActionHit     ExpertResidencyAction = "hit"
)

// ExpertResidency records MoE expert paging and residency transitions.
type ExpertResidency struct {
	Action             ExpertResidencyAction `json:"action"`
	Layer              int                   `json:"layer,omitempty"`
	ExpertIDs          []int                 `json:"expert_ids,omitempty"`
	ResidentExperts    int                   `json:"resident_experts,omitempty"`
	MaxResidentExperts int                   `json:"max_resident_experts,omitempty"`
	LoadedBytes        uint64                `json:"loaded_bytes,omitempty"`
	EvictedBytes       uint64                `json:"evicted_bytes,omitempty"`
	Duration           int64                 `json:"duration,omitempty"`
}

// ResidualSummary records compact residual-stream statistics.
type ResidualSummary struct {
	Layer    int     `json:"layer,omitempty"`
	Mean     float64 `json:"mean,omitempty"`
	Variance float64 `json:"variance,omitempty"`
	RMS      float64 `json:"rms,omitempty"`
	L2Norm   float64 `json:"l2_norm,omitempty"`
	MaxAbs   float64 `json:"max_abs,omitempty"`
}

// CachePressure records KV cache posture for local memory-aware runs.
type CachePressure struct {
	PromptTokens    int     `json:"prompt_tokens,omitempty"`
	GeneratedTokens int     `json:"generated_tokens,omitempty"`
	LayerCount      int     `json:"layer_count,omitempty"`
	CacheTokens     int     `json:"cache_tokens,omitempty"`
	ProcessedTokens int     `json:"processed_tokens,omitempty"`
	MaxCacheTokens  int     `json:"max_cache_tokens,omitempty"`
	Utilization     float64 `json:"utilization,omitempty"`
	Rotating        bool    `json:"rotating,omitempty"`
}

// MemoryPressure records MLX allocator pressure.
type MemoryPressure struct {
	ActiveBytes uint64 `json:"active_bytes,omitempty"`
	PeakBytes   uint64 `json:"peak_bytes,omitempty"`
	CacheBytes  uint64 `json:"cache_bytes,omitempty"`
}

// Training records training-loop scalars.
type Training struct {
	Step         int     `json:"step,omitempty"`
	Epoch        int     `json:"epoch,omitempty"`
	Loss         float64 `json:"loss,omitempty"`
	LearningRate float64 `json:"learning_rate,omitempty"`
	GradNorm     float64 `json:"grad_norm,omitempty"`
}

// Sink consumes typed probe events.
type Sink interface {
	EmitProbe(Event)
}

// SinkFunc adapts a function into a Sink.
type SinkFunc func(Event)

// EmitProbe emits an event to the wrapped function.
//
//	probe.SinkFunc(func(e probe.Event) { … }).EmitProbe(event)
func (f SinkFunc) EmitProbe(event Event) {
	if f != nil {
		f(event)
	}
}

// Bus fans probe events out to one or more sinks.
type Bus struct {
	mu    sync.RWMutex
	sinks []Sink
}

// NewBus creates a fanout sink.
//
//	bus := probe.NewBus(sink1, sink2)
func NewBus(sinks ...Sink) *Bus {
	bus := &Bus{}
	for _, sink := range sinks {
		bus.Add(sink)
	}
	return bus
}

// Add appends a sink to the bus. Nil receivers and nil sinks are ignored.
//
//	bus.Add(sink)
func (b *Bus) Add(sink Sink) {
	if b == nil || sink == nil {
		return
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	b.sinks = append(b.sinks, sink)
}

// EmitProbe emits an event to every sink.
//
//	bus.EmitProbe(event)
func (b *Bus) EmitProbe(event Event) {
	if b == nil {
		return
	}
	b.mu.RLock()
	sinks := core.SliceClone(b.sinks)
	b.mu.RUnlock()
	for _, sink := range sinks {
		if sink != nil {
			sink.EmitProbe(CloneEvent(event))
		}
	}
}

// Recorder stores probe events in memory for tests, reproducible probes,
// or artifacts.
type Recorder struct {
	mu     sync.Mutex
	events []Event
}

// NewRecorder returns a recorder sink.
//
//	r := probe.NewRecorder()
func NewRecorder() *Recorder {
	return &Recorder{}
}

// EmitProbe records an event.
//
//	r.EmitProbe(event)
func (r *Recorder) EmitProbe(event Event) {
	if r == nil {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.events = append(r.events, CloneEvent(event))
}

// Events returns recorded events without aliasing recorder storage.
//
//	events := r.Events()
func (r *Recorder) Events() []Event {
	if r == nil {
		return nil
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	out := make([]Event, len(r.events))
	for i := range r.events {
		out[i] = CloneEvent(r.events[i])
	}
	return out
}

// CloneEvent returns a deep copy of an Event so emitters can safely
// share immutable references downstream.
//
//	out := probe.CloneEvent(event)
func CloneEvent(event Event) Event {
	out := event
	if event.Token != nil {
		token := *event.Token
		out.Token = &token
	}
	if event.Logits != nil {
		logits := *event.Logits
		logits.Shape = core.SliceClone(event.Logits.Shape)
		logits.Top = core.SliceClone(event.Logits.Top)
		logits.Values = core.SliceClone(event.Logits.Values)
		logits.Meta = cloneMeta(event.Logits.Meta)
		out.Logits = &logits
	}
	if event.Entropy != nil {
		entropy := *event.Entropy
		out.Entropy = &entropy
	}
	if event.SelectedHeads != nil {
		heads := *event.SelectedHeads
		heads.Heads = core.SliceClone(event.SelectedHeads.Heads)
		heads.Scores = core.SliceClone(event.SelectedHeads.Scores)
		out.SelectedHeads = &heads
	}
	if event.LayerCoherence != nil {
		coherence := *event.LayerCoherence
		out.LayerCoherence = &coherence
	}
	if event.RouterDecision != nil {
		router := *event.RouterDecision
		router.ExpertIDs = core.SliceClone(event.RouterDecision.ExpertIDs)
		router.Weights = core.SliceClone(event.RouterDecision.Weights)
		out.RouterDecision = &router
	}
	if event.ExpertResidency != nil {
		residency := *event.ExpertResidency
		residency.ExpertIDs = core.SliceClone(event.ExpertResidency.ExpertIDs)
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
	out.Meta = cloneMeta(event.Meta)
	return out
}

func cloneMeta(meta map[string]string) map[string]string {
	if len(meta) == 0 {
		return nil
	}
	return core.MapClone(meta)
}
