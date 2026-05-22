// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"strconv"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/bench"
)

// SplitPlacementRole describes where a component is expected to execute.
type SplitPlacementRole string

const (
	SplitPlacementRoleLocalMetal     SplitPlacementRole = "local_metal"
	SplitPlacementRoleExternalNeeded SplitPlacementRole = "external_needed"
)

// SplitComponentPlacement records one component's runtime placement.
type SplitComponentPlacement struct {
	Component inference.ModelComponent `json:"component"`
	Role      SplitPlacementRole       `json:"role"`
	Ready     bool                     `json:"ready"`
	Required  bool                     `json:"required,omitempty"`
	Bytes     int64                    `json:"bytes,omitempty"`
	Note      string                   `json:"note,omitempty"`
}

// SplitExecutorPlacement is the executable view of a materialised slice.
type SplitExecutorPlacement struct {
	SlicePath              string                     `json:"slice_path"`
	SourcePath             string                     `json:"source_path,omitempty"`
	Preset                 inference.ModelSlicePreset `json:"preset,omitempty"`
	Ready                  bool                       `json:"ready"`
	Standalone             bool                       `json:"standalone"`
	RequiresSplitPlacement bool                       `json:"requires_split_placement"`
	LocalTensorBytes       int64                      `json:"local_tensor_bytes,omitempty"`
	OffloadTensorBytes     int64                      `json:"offload_tensor_bytes,omitempty"`
	RetainedTensorRatio    float64                    `json:"retained_tensor_ratio,omitempty"`
	LocalComponents        []inference.ModelComponent `json:"local_components,omitempty"`
	RequiredPlacements     []SplitComponentPlacement  `json:"required_placements,omitempty"`
	AllPlacements          []SplitComponentPlacement  `json:"all_placements,omitempty"`
}

// Requires reports whether placement still needs component supplied externally.
func (plan SplitExecutorPlacement) Requires(component inference.ModelComponent) bool {
	for _, placement := range plan.RequiredPlacements {
		if placement.Component == component {
			return true
		}
	}
	return false
}

// SplitFFNExecutor is the FFN/expert execution seam for split inference.
type SplitFFNExecutor interface {
	ForwardFFN(context.Context, SplitFFNRequest) (SplitFFNResult, error)
}

type splitFFNMemoryReporter interface {
	MemoryReport() CPUSplitFFNMemoryReport
}

type splitFFNMemoryEstimator interface {
	EstimateMemoryReport(context.Context) (CPUSplitFFNMemoryReport, error)
}

// SplitPowerSample is one host power reading captured during split execution.
type SplitPowerSample struct {
	Phase  string  `json:"phase,omitempty"`
	Watts  float64 `json:"watts,omitempty"`
	Source string  `json:"source,omitempty"`
}

// SplitPowerMeter supplies optional host-specific power readings.
type SplitPowerMeter interface {
	SampleSplitPower(context.Context, string) (SplitPowerSample, error)
}

// SplitPowerReport records the power samples captured for one split run.
type SplitPowerReport struct {
	Available    bool               `json:"available"`
	Source       string             `json:"source,omitempty"`
	SampleCount  int                `json:"sample_count,omitempty"`
	AverageWatts float64            `json:"average_watts,omitempty"`
	PeakWatts    float64            `json:"peak_watts,omitempty"`
	Samples      []SplitPowerSample `json:"samples,omitempty"`
	Error        string             `json:"error,omitempty"`
}

// SplitExecutorMetrics reports the most recent split generation timing,
// throughput, memory, and optional power readings.
type SplitExecutorMetrics struct {
	PromptTokens        int                      `json:"prompt_tokens,omitempty"`
	GeneratedTokens     int                      `json:"generated_tokens,omitempty"`
	FirstTokenDuration  time.Duration            `json:"first_token_duration,omitempty"`
	PrefillDuration     time.Duration            `json:"prefill_duration,omitempty"`
	DecodeDuration      time.Duration            `json:"decode_duration,omitempty"`
	TotalDuration       time.Duration            `json:"total_duration,omitempty"`
	PrefillTokensPerSec float64                  `json:"prefill_tokens_per_sec,omitempty"`
	DecodeTokensPerSec  float64                  `json:"decode_tokens_per_sec,omitempty"`
	PeakMemoryBytes     uint64                   `json:"peak_memory_bytes,omitempty"`
	ActiveMemoryBytes   uint64                   `json:"active_memory_bytes,omitempty"`
	CPUFFNMemory        *CPUSplitFFNMemoryReport `json:"cpu_ffn_memory,omitempty"`
	Power               SplitPowerReport         `json:"power,omitempty"`
}

// SplitFFNRequest is the minimal FFN boundary shape. Hidden states are flat for
// now; later versions can add layer ranges and quantised buffer views.
type SplitFFNRequest struct {
	Layer  int       `json:"layer"`
	Hidden []float32 `json:"hidden,omitempty"`
}

// SplitFFNResult is the hidden-state result from an FFN placement.
type SplitFFNResult struct {
	Hidden []float32 `json:"hidden,omitempty"`
}

// SplitLocalRuntime is the local attention/logits side of split inference.
// Implementations own the Metal-resident slice state; SplitExecutor owns the
// cross-placement orchestration.
type SplitLocalRuntime interface {
	Prefill(context.Context, SplitPrefillRequest) (SplitPrefillResult, error)
	ForwardAttention(context.Context, SplitAttentionRequest) (SplitAttentionResult, error)
	Sample(context.Context, SplitSampleRequest) (SplitSampleResult, error)
	DecodeToken(context.Context, int32) (string, error)
}

// SplitPrefillRequest starts a split decode session from a prompt.
type SplitPrefillRequest struct {
	Prompt    string                 `json:"prompt"`
	Config    GenerateConfig         `json:"config,omitempty"`
	Placement SplitExecutorPlacement `json:"placement"`
}

// SplitPrefillResult is the local runtime state needed by the orchestrator.
type SplitPrefillResult struct {
	Tokens []int32   `json:"tokens,omitempty"`
	Hidden []float32 `json:"hidden,omitempty"`
	Layers int       `json:"layers,omitempty"`
}

// SplitAttentionRequest asks the local runtime to run one attention layer.
type SplitAttentionRequest struct {
	Step   int            `json:"step"`
	Layer  int            `json:"layer"`
	Tokens []int32        `json:"tokens,omitempty"`
	Hidden []float32      `json:"hidden,omitempty"`
	Config GenerateConfig `json:"config,omitempty"`
}

// SplitAttentionResult returns the hidden state after local attention.
type SplitAttentionResult struct {
	Hidden []float32 `json:"hidden,omitempty"`
}

// SplitSampleRequest asks the local runtime to project logits and sample.
type SplitSampleRequest struct {
	Step   int            `json:"step"`
	Tokens []int32        `json:"tokens,omitempty"`
	Hidden []float32      `json:"hidden,omitempty"`
	Config GenerateConfig `json:"config,omitempty"`
}

// SplitSampleResult is one sampled token from the local logits path.
type SplitSampleResult struct {
	TokenID int32     `json:"token_id"`
	Hidden  []float32 `json:"hidden,omitempty"`
}

// SplitExecutorOption configures a split executor.
type SplitExecutorOption func(*splitExecutorConfig)

type splitExecutorConfig struct {
	ffn               SplitFFNExecutor
	cpuFFN            bool
	cpuFFNConfig      CPUSplitFFNConfig
	local             SplitLocalRuntime
	nativeLocal       bool
	nativeLocalConfig LoadConfig
	powerMeter        SplitPowerMeter
}

// WithSplitFFNExecutor supplies the FFN/expert placement used by client slices.
func WithSplitFFNExecutor(executor SplitFFNExecutor) SplitExecutorOption {
	return func(cfg *splitExecutorConfig) {
		cfg.ffn = executor
	}
}

// WithCPUSplitFFNExecutor loads omitted dense FFN weights on CPU from the
// source pack recorded in the slice manifest.
func WithCPUSplitFFNExecutor(opts ...CPUSplitFFNOption) SplitExecutorOption {
	return func(cfg *splitExecutorConfig) {
		cfg.cpuFFN = true
		cfg.cpuFFNConfig = applyCPUSplitFFNOptions(opts)
	}
}

// WithSplitLocalRuntime supplies the local attention/logits runtime.
func WithSplitLocalRuntime(runtime SplitLocalRuntime) SplitExecutorOption {
	return func(cfg *splitExecutorConfig) {
		cfg.local = runtime
	}
}

// WithNativeSplitLocalRuntime asks LoadSplitExecutor to load the local
// attention/logits runtime from the materialised slice.
func WithNativeSplitLocalRuntime(opts ...LoadOption) SplitExecutorOption {
	return func(cfg *splitExecutorConfig) {
		cfg.nativeLocal = true
		cfg.nativeLocalConfig = applyLoadOptions(opts)
	}
}

// WithSplitPowerMeter records host power samples during split generation.
func WithSplitPowerMeter(meter SplitPowerMeter) SplitExecutorOption {
	return func(cfg *splitExecutorConfig) {
		cfg.powerMeter = meter
	}
}

var loadNativeSplitLocalRuntime = func(ctx context.Context, slicePath string, cfg LoadConfig) (SplitLocalRuntime, error) {
	return LoadNativeSplitLocalRuntime(ctx, slicePath, cfg)
}

// SplitExecutor is a manifest-backed split runtime skeleton. It validates
// placement and owns the future local-attention/remote-FFN boundary.
type SplitExecutor struct {
	inspection ModelSliceInspection
	placement  SplitExecutorPlacement
	ffn        SplitFFNExecutor
	local      SplitLocalRuntime
	powerMeter SplitPowerMeter
	metrics    SplitExecutorMetrics
}

// LoadSplitExecutor prepares a split executor from a materialised slice.
func LoadSplitExecutor(ctx context.Context, slicePath string, opts ...SplitExecutorOption) (*SplitExecutor, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if core.Trim(slicePath) == "" {
		return nil, core.NewError("mlx: split executor requires a slice path")
	}
	cfg := splitExecutorConfig{}
	for _, opt := range opts {
		opt(&cfg)
	}
	inspection, err := InspectModelSlice(slicePath)
	if err != nil {
		return nil, err
	}
	if cfg.nativeLocal && cfg.local == nil {
		local, err := loadNativeSplitLocalRuntime(ctx, slicePath, cfg.nativeLocalConfig)
		if err != nil {
			return nil, err
		}
		cfg.local = local
	}
	if cfg.cpuFFN && cfg.ffn == nil {
		ffn, err := loadCPUSplitFFNExecutor(ctx, inspection.SourcePath, cfg.cpuFFNConfig)
		if err != nil {
			return nil, err
		}
		cfg.ffn = ffn
	}
	placement := buildSplitExecutorPlacement(inspection, cfg.ffn)
	return &SplitExecutor{
		inspection: inspection,
		placement:  placement,
		ffn:        cfg.ffn,
		local:      cfg.local,
		powerMeter: cfg.powerMeter,
	}, nil
}

// Placement returns the current split placement plan.
func (executor *SplitExecutor) Placement() SplitExecutorPlacement {
	if executor == nil {
		return SplitExecutorPlacement{}
	}
	return executor.placement
}

// Metrics returns the most recent split generation metrics.
func (executor *SplitExecutor) Metrics() SplitExecutorMetrics {
	if executor == nil {
		return SplitExecutorMetrics{}
	}
	return cloneSplitExecutorMetrics(executor.metrics)
}

// CPUSplitFFNMemoryReport returns CPU FFN memory counters when the split
// executor is backed by the built-in CPU FFN implementation.
func (executor *SplitExecutor) CPUSplitFFNMemoryReport() *CPUSplitFFNMemoryReport {
	if executor == nil {
		return nil
	}
	reporter, ok := executor.ffn.(splitFFNMemoryReporter)
	if !ok {
		return nil
	}
	report := reporter.MemoryReport()
	return &report
}

// CPUSplitFFNMemoryEstimate predicts CPU FFN residency without loading layers.
func (executor *SplitExecutor) CPUSplitFFNMemoryEstimate(ctx context.Context) (*CPUSplitFFNMemoryReport, error) {
	if executor == nil {
		return nil, nil
	}
	estimator, ok := executor.ffn.(splitFFNMemoryEstimator)
	if !ok {
		return nil, nil
	}
	report, err := estimator.EstimateMemoryReport(ctx)
	if err != nil {
		return nil, err
	}
	return &report, nil
}

// Generate is the future split decode entrypoint. It deliberately refuses to
// run until all required placements are supplied.
func (executor *SplitExecutor) Generate(ctx context.Context, prompt string, cfg GenerateConfig) (string, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return "", err
	}
	if executor == nil {
		return "", core.NewError("mlx: split executor is nil")
	}
	if executor.placement.Requires(inference.ModelComponentFFN) && executor.ffn == nil {
		return "", core.NewError("mlx: split executor requires an FFN executor for omitted feed-forward weights")
	}
	if executor.local == nil {
		return "", core.NewError("mlx: split executor local attention execution is not wired yet")
	}
	if cfg.MaxTokens <= 0 {
		cfg.MaxTokens = DefaultGenerateConfig().MaxTokens
	}
	executor.metrics = SplitExecutorMetrics{}
	totalStart := time.Now()
	ResetPeakMemory()
	power := newSplitPowerRecorder(ctx, executor.powerMeter)
	prefillStart := time.Now()
	state, err := executor.local.Prefill(ctx, SplitPrefillRequest{
		Prompt:    prompt,
		Config:    cfg,
		Placement: executor.placement,
	})
	if err != nil {
		return "", core.E("mlx.SplitExecutor.Generate", "prefill", err)
	}
	prefillDuration := bench.NonZeroDuration(time.Since(prefillStart))
	power.sample(ctx, "prefill")
	if state.Layers <= 0 {
		return "", core.NewError("mlx: split executor prefill returned no layers")
	}
	if len(state.Hidden) == 0 {
		return "", core.NewError("mlx: split executor prefill returned empty hidden state")
	}

	tokens := make([]int32, len(state.Tokens), len(state.Tokens)+cfg.MaxTokens)
	copy(tokens, state.Tokens)
	hidden := cloneSplitHidden(state.Hidden)
	builder := core.NewBuilder()
	decodeStart := time.Now()
	generatedTokens := 0
	var firstTokenDuration time.Duration
	requiresFFN := executor.placement.Requires(inference.ModelComponentFFN)
	for step := 0; step < cfg.MaxTokens; step++ {
		if err := ctx.Err(); err != nil {
			return "", err
		}
		for layer := 0; layer < state.Layers; layer++ {
			attention, err := executor.local.ForwardAttention(ctx, SplitAttentionRequest{
				Step:   step,
				Layer:  layer,
				Tokens: cloneSplitTokenIDs(tokens),
				Hidden: cloneSplitHidden(hidden),
				Config: cfg,
			})
			if err != nil {
				return "", core.E("mlx.SplitExecutor.Generate", splitExecutorLayerStepLabel("attention layer ", layer, step), err)
			}
			if len(attention.Hidden) == 0 {
				return "", core.Errorf("mlx: split executor attention layer %d step %d returned empty hidden state", layer, step)
			}
			hidden = cloneSplitHidden(attention.Hidden)
			if requiresFFN {
				ffn, err := executor.ffn.ForwardFFN(ctx, SplitFFNRequest{
					Layer:  layer,
					Hidden: cloneSplitHidden(hidden),
				})
				if err != nil {
					return "", core.E("mlx.SplitExecutor.Generate", splitExecutorLayerStepLabel("ffn layer ", layer, step), err)
				}
				if len(ffn.Hidden) == 0 {
					return "", core.Errorf("mlx: split executor ffn layer %d step %d returned empty hidden state", layer, step)
				}
				hidden = cloneSplitHidden(ffn.Hidden)
			}
		}

		sample, err := executor.local.Sample(ctx, SplitSampleRequest{
			Step:   step,
			Tokens: cloneSplitTokenIDs(tokens),
			Hidden: cloneSplitHidden(hidden),
			Config: cfg,
		})
		if err != nil {
			return "", core.E("mlx.SplitExecutor.Generate", splitExecutorStepLabel("sample step ", step), err)
		}
		tokens = append(tokens, sample.TokenID)
		if len(sample.Hidden) > 0 {
			hidden = cloneSplitHidden(sample.Hidden)
		}
		if splitExecutorStopToken(cfg.StopTokens, sample.TokenID) {
			break
		}
		text, err := executor.local.DecodeToken(ctx, sample.TokenID)
		if err != nil {
			return "", core.E("mlx.SplitExecutor.Generate", splitExecutorStepLabel("decode token step ", step), err)
		}
		generatedTokens++
		if firstTokenDuration == 0 {
			firstTokenDuration = bench.NonZeroDuration(time.Since(totalStart))
			power.sample(ctx, "first_token")
		}
		builder.WriteString(text)
	}
	decodeDuration := bench.NonZeroDuration(time.Since(decodeStart))
	totalDuration := bench.NonZeroDuration(time.Since(totalStart))
	metrics := SplitExecutorMetrics{
		PromptTokens:       len(state.Tokens),
		GeneratedTokens:    generatedTokens,
		FirstTokenDuration: firstTokenDuration,
		PrefillDuration:    prefillDuration,
		DecodeDuration:     decodeDuration,
		TotalDuration:      totalDuration,
		PeakMemoryBytes:    GetPeakMemory(),
		ActiveMemoryBytes:  GetActiveMemory(),
	}
	if metrics.PrefillDuration > 0 {
		metrics.PrefillTokensPerSec = float64(metrics.PromptTokens) / metrics.PrefillDuration.Seconds()
	}
	if metrics.DecodeDuration > 0 {
		metrics.DecodeTokensPerSec = float64(metrics.GeneratedTokens) / metrics.DecodeDuration.Seconds()
	}
	metrics.CPUFFNMemory = executor.CPUSplitFFNMemoryReport()
	power.sample(ctx, "complete")
	metrics.Power = power.report()
	executor.metrics = metrics
	return builder.String(), nil
}

func buildSplitExecutorPlacement(inspection ModelSliceInspection, ffn SplitFFNExecutor) SplitExecutorPlacement {
	componentCount := len(inspection.Plan.Components)
	missingCount := len(inspection.MissingRuntimeComponents)
	localComponents := make([]inference.ModelComponent, len(inspection.Plan.Components))
	copy(localComponents, inspection.Plan.Components)
	plan := SplitExecutorPlacement{
		SlicePath:              inspection.Path,
		SourcePath:             inspection.SourcePath,
		Preset:                 inspection.Plan.Preset,
		Standalone:             inspection.Standalone,
		RequiresSplitPlacement: inspection.RequiresSplitPlacement,
		LocalTensorBytes:       inspection.LocalTensorBytes,
		OffloadTensorBytes:     inspection.OffloadTensorBytes,
		RetainedTensorRatio:    inspection.RetainedTensorRatio,
		LocalComponents:        localComponents,
		AllPlacements:          make([]SplitComponentPlacement, 0, componentCount+missingCount),
		RequiredPlacements:     make([]SplitComponentPlacement, 0, missingCount),
	}
	for _, component := range inspection.Plan.Components {
		plan.AllPlacements = append(plan.AllPlacements, SplitComponentPlacement{
			Component: component,
			Role:      SplitPlacementRoleLocalMetal,
			Ready:     true,
		})
	}
	for _, component := range inspection.MissingRuntimeComponents {
		ready := component == inference.ModelComponentFFN && ffn != nil
		placement := SplitComponentPlacement{
			Component: component,
			Role:      SplitPlacementRoleExternalNeeded,
			Ready:     ready,
			Required:  true,
			Note:      "component was omitted from the local slice",
		}
		if component == inference.ModelComponentFFN {
			placement.Bytes = inspection.OffloadTensorBytes
		}
		plan.RequiredPlacements = append(plan.RequiredPlacements, placement)
		plan.AllPlacements = append(plan.AllPlacements, placement)
	}
	plan.Ready = splitExecutorPlacementsReady(plan.RequiredPlacements)
	if inspection.Standalone {
		plan.Ready = true
	}
	return plan
}

func splitExecutorPlacementsReady(placements []SplitComponentPlacement) bool {
	for _, placement := range placements {
		if placement.Required && !placement.Ready {
			return false
		}
	}
	return true
}

func cloneSplitTokenIDs(in []int32) []int32 {
	if len(in) == 0 {
		return nil
	}
	out := make([]int32, len(in))
	copy(out, in)
	return out
}

func cloneSplitHidden(in []float32) []float32 {
	if len(in) == 0 {
		return nil
	}
	out := make([]float32, len(in))
	copy(out, in)
	return out
}

type splitPowerRecorder struct {
	meter       SplitPowerMeter
	powerReport SplitPowerReport
	total       float64
}

// splitPowerExpectedSamples covers the standard recorder phases:
// start, prefill, first_token, complete.
const splitPowerExpectedSamples = 4

func newSplitPowerRecorder(ctx context.Context, meter SplitPowerMeter) *splitPowerRecorder {
	recorder := &splitPowerRecorder{meter: meter}
	if meter == nil {
		recorder.powerReport.Source = "not_configured"
		return recorder
	}
	recorder.powerReport.Samples = make([]SplitPowerSample, 0, splitPowerExpectedSamples)
	recorder.sample(ctx, "start")
	return recorder
}

func (recorder *splitPowerRecorder) sample(ctx context.Context, phase string) {
	if recorder == nil || recorder.meter == nil {
		return
	}
	sample, err := recorder.meter.SampleSplitPower(ctx, phase)
	if err != nil {
		recorder.powerReport.Error = err.Error()
		return
	}
	sample.Phase = firstNonEmpty(sample.Phase, phase)
	if sample.Source != "" && recorder.powerReport.Source == "" {
		recorder.powerReport.Source = sample.Source
	}
	recorder.powerReport.Samples = append(recorder.powerReport.Samples, sample)
	recorder.powerReport.SampleCount = len(recorder.powerReport.Samples)
	recorder.total += sample.Watts
	if sample.Watts > recorder.powerReport.PeakWatts {
		recorder.powerReport.PeakWatts = sample.Watts
	}
}

func (recorder *splitPowerRecorder) report() SplitPowerReport {
	if recorder == nil {
		return SplitPowerReport{Source: "not_configured"}
	}
	if recorder.powerReport.SampleCount == 0 {
		if recorder.powerReport.Source == "" {
			recorder.powerReport.Source = "not_configured"
		}
		return recorder.powerReport
	}
	recorder.powerReport.Available = true
	recorder.powerReport.AverageWatts = recorder.total / float64(recorder.powerReport.SampleCount)
	return recorder.powerReport
}

func cloneSplitExecutorMetrics(metrics SplitExecutorMetrics) SplitExecutorMetrics {
	if metrics.CPUFFNMemory != nil {
		report := *metrics.CPUFFNMemory
		metrics.CPUFFNMemory = &report
	}
	if n := len(metrics.Power.Samples); n > 0 {
		samples := make([]SplitPowerSample, n)
		copy(samples, metrics.Power.Samples)
		metrics.Power.Samples = samples
	}
	return metrics
}

func splitExecutorStopToken(stopTokens []int32, id int32) bool {
	for _, stop := range stopTokens {
		if stop == id {
			return true
		}
	}
	return false
}

func splitExecutorLayerStepLabel(prefix string, layer, step int) string {
	buf := make([]byte, 0, len(prefix)+24)
	buf = append(buf, prefix...)
	buf = strconv.AppendInt(buf, int64(layer), 10)
	buf = append(buf, " step "...)
	buf = strconv.AppendInt(buf, int64(step), 10)
	return string(buf)
}

func splitExecutorStepLabel(prefix string, step int) string {
	buf := make([]byte, 0, len(prefix)+12)
	buf = append(buf, prefix...)
	buf = strconv.AppendInt(buf, int64(step), 10)
	return string(buf)
}
