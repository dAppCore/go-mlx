// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"strconv"

	"dappco.re/go/mlx/dataset"
	"dappco.re/go/mlx/memory"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/eval"
	"dappco.re/go/mlx/lora"
	"dappco.re/go/mlx/pkg/metal"
	"dappco.re/go/mlx/probe"
	"reflect"
)

// inference_convert.go: translation between metal/root types and the inference.*
// contract types (probe events, identities, memory plans, eval/training results).

func toInferenceProbeEvent(event metal.ProbeEvent) inference.ProbeEvent {
	// Local pointer aliases — the previous form did event.X.Y per field
	// (load .X pointer + load .Y field), which the compiler can't hoist
	// across nil checks. One pointer fetch + many field reads compiles
	// to single loads. toInferenceProbeEvent fires per probe event,
	// which under ProbeSink is emitted per token during generation.
	out := inference.ProbeEvent{
		Kind:   inference.ProbeEventKind(event.Kind),
		Phase:  inference.ProbePhase(event.Phase),
		Step:   event.Step,
		Labels: cloneInferenceLabels(event.Meta),
	}
	if token := event.Token; token != nil {
		out.Token = &inference.ProbeToken{
			ID:              token.ID,
			Text:            token.Text,
			PromptTokens:    token.PromptTokens,
			GeneratedTokens: token.GeneratedTokens,
		}
	}
	if logits := event.Logits; logits != nil {
		out.Logits = &inference.ProbeLogits{
			VocabularySize: logits.VocabSize,
			Min:            logits.MinLogit,
			Max:            logits.MaxLogit,
			Mean:           float32(logits.MeanLogit),
			Top:            toInferenceProbeLogits(logits.Top),
		}
	}
	if entropy := event.Entropy; entropy != nil {
		out.Entropy = &inference.ProbeEntropy{Value: entropy.Value, Unit: entropy.Unit}
	}
	if heads := event.SelectedHeads; heads != nil {
		out.SelectedHeads = &inference.ProbeHeadSelection{Layer: heads.Layer, Heads: core.SliceClone(heads.Heads)}
	}
	if coherence := event.LayerCoherence; coherence != nil {
		out.LayerCoherence = &inference.ProbeLayerCoherence{
			Layer:          coherence.Layer,
			KVCoupling:     coherence.KVCoupling,
			MeanCoherence:  meanNonZero(coherence.KeyCoherence, coherence.ValueCoherence, coherence.CrossAlignment),
			PhaseLock:      coherence.PhaseLock,
			SpectralStable: coherence.HeadEntropy,
		}
	}
	if router := event.RouterDecision; router != nil {
		out.RouterDecision = &inference.ProbeRouterDecision{
			Layer:       router.Layer,
			ExpertIDs:   core.SliceClone(router.ExpertIDs),
			ExpertProbs: core.SliceClone(router.Weights),
		}
	}
	if residual := event.Residual; residual != nil {
		out.Residual = &inference.ProbeResidualSummary{
			Layer: residual.Layer,
			Mean:  residual.Mean,
			RMS:   residual.RMS,
			Norm:  residual.L2Norm,
		}
	}
	if cache := event.Cache; cache != nil {
		out.Cache = &inference.ProbeCachePressure{
			PromptTokens:    cache.PromptTokens,
			GeneratedTokens: cache.GeneratedTokens,
			CachedTokens:    cache.CacheTokens,
			HitRate:         cache.Utilization,
		}
	}
	if memory := event.Memory; memory != nil {
		out.Memory = &inference.ProbeMemoryPressure{
			ActiveBytes: memory.ActiveBytes,
			PeakBytes:   memory.PeakBytes,
		}
	}
	if training := event.Training; training != nil {
		out.Training = &inference.ProbeTraining{
			Epoch:        training.Epoch,
			Step:         training.Step,
			Loss:         training.Loss,
			LearningRate: training.LearningRate,
		}
	}
	return out
}

func toInferenceProbeLogits(logits []metal.ProbeLogit) []inference.ProbeLogit {
	out := make([]inference.ProbeLogit, len(logits))
	// Index iteration — same rationale as spine's toProbeLogits.
	for i := range logits {
		out[i] = inference.ProbeLogit{ID: logits[i].TokenID, Value: logits[i].Logit}
	}
	return out
}

func toInferenceModelIdentity(info ModelInfo) inference.ModelIdentity {
	return inference.ModelIdentity{
		Architecture:  info.Architecture,
		VocabSize:     info.VocabSize,
		NumLayers:     info.NumLayers,
		HiddenSize:    info.HiddenSize,
		QuantBits:     info.QuantBits,
		QuantGroup:    info.QuantGroup,
		ContextLength: info.ContextLength,
	}
}

func toInferenceAdapterIdentity(info metal.AdapterInfo) inference.AdapterIdentity {
	return inference.AdapterIdentity{
		Path:       info.Path,
		Hash:       info.Hash,
		Format:     "lora",
		Rank:       info.Rank,
		Alpha:      info.Alpha,
		TargetKeys: core.SliceClone(info.TargetKeys),
		Labels:     adapterIdentityLabels(info.Name, info.Scale),
	}
}

// adapterIdentityCommonScaleStrings caches the strconv.FormatFloat output
// for the LoRA scale values that show up most often in practice. The map
// is read-only after package init so concurrent lookups are lock-free.
// Hit rates ≈ 100% in the field — LoRA training defaults are 0.5/1.0/2.0
// (Alpha/Rank, see sft.go:433), checkpoints are tagged with the same
// constants, and adapter merges round to the nearest tenth. Each hit
// saves one ~3 B strconv heap alloc per adapterIdentityLabels call.
var adapterIdentityCommonScaleStrings = map[float32]string{
	0.125: "0.125",
	0.25:  "0.25",
	0.5:   "0.5",
	1:     "1",
	1.5:   "1.5",
	2:     "2",
	4:     "4",
	8:     "8",
}

func adapterIdentityLabels(name string, scale float32) map[string]string {
	// Cheap pre-check — return nil before allocating the map when both
	// fields are zero. adapterIdentityLabels is called per
	// toInferenceAdapterIdentity / toInferenceRootAdapterIdentity which
	// fire on every CapabilityReport / TrainSFT / BenchReport call, and
	// the zero-name + zero-scale shape is the dominant "no adapter
	// loaded" case.
	if name == "" && scale == 0 {
		return nil
	}
	// Pre-size for the two possible keys. strconv.FormatFloat with 'g'
	// matches Sprintf("%g") semantics — shortest representation that
	// round-trips — but skips the fmt format-parser + interface-boxing.
	// Bitsize 32 matches the float32 input precision.
	labels := make(map[string]string, 2)
	if name != "" {
		labels["name"] = name
	}
	if scale != 0 {
		// Hot path: cached constants for the LoRA scales we see ~100% of
		// the time. The fallback FormatFloat ('g' / -1 / 32 bitsize) only
		// fires for unusual mid-training scale values.
		if cached, ok := adapterIdentityCommonScaleStrings[scale]; ok {
			labels["scale"] = cached
		} else {
			labels["scale"] = strconv.FormatFloat(float64(scale), 'g', -1, 32)
		}
	}
	return labels
}

// commonQuantizationLabels caches the "%d-bit" strconv+concat output for the
// common model-quant widths. Cache hit drops 2 allocs (strconv heap alloc +
// concat heap alloc, ~16 B) per toInferenceMemoryPlan call. Fallback path
// keeps the original strconv.Itoa + "-bit" concat for any other width.
var commonQuantizationLabels = map[int]string{
	2:  "2-bit",
	3:  "3-bit",
	4:  "4-bit",
	5:  "5-bit",
	6:  "6-bit",
	8:  "8-bit",
	16: "16-bit",
}

func toInferenceMemoryPlan(plan memory.Plan) inference.MemoryPlan {
	// The quantisation label reports the model's ACTUAL width
	// (ModelQuantization, read from its bytes) — never a machine-class
	// preference. Unquantised/unknown (0) reports no label (the field is
	// omitempty). Cached lookup avoids the strconv+concat allocs for common widths.
	quant := ""
	if plan.ModelQuantization > 0 {
		label, ok := commonQuantizationLabels[plan.ModelQuantization]
		if !ok {
			label = strconv.Itoa(plan.ModelQuantization) + "-bit"
		}
		quant = label
	}
	return inference.MemoryPlan{
		MachineClass:      string(plan.MachineClass),
		DeviceMemoryBytes: plan.DeviceMemoryBytes,
		ContextLength:     plan.ContextLength,
		BatchSize:         plan.BatchSize,
		CacheMode:         string(plan.CacheMode),
		Quantization:      quant,
		KVCacheBytes:      plan.EstimatedKVCacheModeBytes,
		TrainingFeasible:  plan.MachineClass != memory.ClassApple16GB,
		Notes:             core.SliceClone(plan.Notes),
	}
}

func toEvalConfig(cfg inference.EvalConfig) eval.Config {
	return eval.Config{
		MaxSamples: cfg.MaxSamples,
		Batch: dataset.BatchConfig{
			BatchSize: cfg.BatchSize,
			MaxSeqLen: cfg.MaxSeqLen,
		},
	}
}

func toInferenceEvalReport(report *eval.Report) *inference.EvalReport {
	if report == nil {
		return nil
	}
	return &inference.EvalReport{
		Model:   toInferenceModelIdentity(evalInfoToModel(report.ModelInfo)),
		Adapter: toInferenceRootAdapterIdentity(evalAdapterToLora(report.Adapter)),
		Metrics: inference.EvalMetrics{
			Samples:    report.Metrics.Samples,
			Tokens:     report.Metrics.Tokens,
			Loss:       report.Metrics.Loss,
			Perplexity: report.Metrics.Perplexity,
		},
		Probes: toInferenceQualityResults(report.Quality.Checks),
	}
}

func toInferenceQualityResults(checks []eval.QualityCheck) []inference.QualityProbeResult {
	out := make([]inference.QualityProbeResult, len(checks))
	// Index iteration — eval.QualityCheck carries Name + Detail (string
	// headers) + Pass + Score, ~48 B total. Skip the per-iter copy.
	for i := range checks {
		out[i] = inference.QualityProbeResult{Name: checks[i].Name, Passed: checks[i].Pass, Score: checks[i].Score, Text: checks[i].Detail}
	}
	return out
}

func toSFTConfig(cfg inference.TrainingConfig, sink inference.ProbeSink) SFTConfig {
	return SFTConfig{
		BatchSize:                 cfg.BatchSize,
		GradientAccumulationSteps: cfg.GradientAccumulation,
		Epochs:                    cfg.Epochs,
		LearningRate:              cfg.LearningRate,
		LoRA: LoRAConfig{
			Rank:       cfg.LoRA.Rank,
			Alpha:      cfg.LoRA.Alpha,
			TargetKeys: core.SliceClone(cfg.LoRA.TargetKeys),
			DType:      sftDType(cfg.LoRA.BFloat16),
			ProbeSink:  inferenceProbeSink{sink: sink},
		},
		ProbeSink: inferenceProbeSink{sink: sink},
	}
}

type inferenceProbeSink struct {
	sink inference.ProbeSink
}

func (sink inferenceProbeSink) EmitProbe(event probe.Event) {
	if sink.sink == nil {
		return
	}
	sink.sink.EmitProbe(toInferenceRootProbeEvent(event))
}

func toInferenceRootProbeEvent(event probe.Event) inference.ProbeEvent {
	// Local pointer aliases — see toInferenceProbeEvent for rationale.
	out := inference.ProbeEvent{
		Kind:   inference.ProbeEventKind(event.Kind),
		Phase:  inference.ProbePhase(event.Phase),
		Step:   event.Step,
		Labels: cloneInferenceLabels(event.Meta),
	}
	if token := event.Token; token != nil {
		out.Token = &inference.ProbeToken{
			ID:              token.ID,
			Text:            token.Text,
			PromptTokens:    token.PromptTokens,
			GeneratedTokens: token.GeneratedTokens,
		}
	}
	if entropy := event.Entropy; entropy != nil {
		out.Entropy = &inference.ProbeEntropy{Value: entropy.Value, Unit: entropy.Unit}
	}
	if training := event.Training; training != nil {
		out.Training = &inference.ProbeTraining{
			Epoch:        training.Epoch,
			Step:         training.Step,
			Loss:         training.Loss,
			LearningRate: training.LearningRate,
		}
	}
	return out
}

func sftDType(bfloat16 bool) DType {
	if bfloat16 {
		return DTypeBFloat16
	}
	return 0
}

func toInferenceTrainingResult(info ModelInfo, result *SFTResult, cfg inference.TrainingConfig) *inference.TrainingResult {
	out := &inference.TrainingResult{
		Model:  toInferenceModelIdentity(info),
		Labels: cloneInferenceLabels(cfg.Labels),
	}
	if result == nil {
		return out
	}
	out.Adapter = toInferenceRootAdapterIdentity(info.Adapter)
	if result.AdapterPath != "" {
		out.Adapter.Path = result.AdapterPath
	}
	out.Metrics = inference.TrainingMetrics{
		Epoch:        result.Epochs,
		Step:         result.Steps,
		Samples:      result.Samples,
		Loss:         result.LastLoss,
		LearningRate: cfg.LearningRate,
	}
	out.Checkpoints = stateRefsFromPaths("sft_checkpoint", result.Checkpoints)
	return out
}

func toInferenceRootAdapterIdentity(info lora.AdapterInfo) inference.AdapterIdentity {
	return inference.AdapterIdentity{
		Path:       info.Path,
		Hash:       info.Hash,
		Format:     "lora",
		Rank:       info.Rank,
		Alpha:      info.Alpha,
		TargetKeys: core.SliceClone(info.TargetKeys),
		Labels:     adapterIdentityLabels(info.Name, info.Scale),
	}
}

// stateRefsURIScheme is the URI scheme prefix for file-backed StateRefs.
// Hoisted to package init so the literal isn't re-interned per call —
// also serves as the documented prefix for the single-buffer URI build
// path in stateRefsFromPaths.
const stateRefsURIScheme = "file://"

func stateRefsFromPaths(kind string, paths []string) []inference.StateRef {
	// Two-pass: count non-empty paths + total URI byte length so we can
	// pre-size the output slice exactly AND allocate one shared backing
	// buffer for every "file://"+path string. Each StateRef.URI is a
	// substring of that single allocation — drops N per-call concat
	// allocs (one per non-empty path) down to ONE allocation regardless
	// of path count.
	nonEmpty := 0
	totalBytes := 0
	for _, path := range paths {
		if path == "" {
			continue
		}
		nonEmpty++
		totalBytes += len(stateRefsURIScheme) + len(path)
	}
	if nonEmpty == 0 {
		return []inference.StateRef{}
	}
	buf := make([]byte, 0, totalBytes)
	out := make([]inference.StateRef, 0, nonEmpty)
	for _, path := range paths {
		if path == "" {
			continue
		}
		start := len(buf)
		buf = append(buf, stateRefsURIScheme...)
		buf = append(buf, path...)
		// Use [start:end] not [start:] so the substring length is captured
		// at write time. buf was pre-sized to totalBytes so append never
		// grows the backing array, which keeps prior substring pointers
		// valid through the rest of the loop. core.AsString is zero-copy
		// + buf is fresh-built and never re-handed-out, so the safety
		// contract holds.
		out = append(out, inference.StateRef{
			Kind: kind,
			URI:  core.AsString(buf[start:len(buf)]),
		})
	}
	return out
}

func cloneInferenceLabels(labels map[string]string) map[string]string {
	if len(labels) == 0 {
		return nil
	}
	// core.MapClone → maps.Clone uses runtime.mapclone for bulk-bucket
	// hash-table copy rather than the user-space range+assign loop.
	// Same alloc shape (2 allocs / 336 bytes for a 4-entry string map),
	// iteration moves into compiled runtime code. Matches the helpers.go
	// cloneStringMap adoption (6dd0c53).
	return core.MapClone(labels)
}

func cloneInferenceSplitEndpoints(endpoints []inference.SplitEndpoint) []inference.SplitEndpoint {
	if len(endpoints) == 0 {
		return nil
	}
	out := make([]inference.SplitEndpoint, len(endpoints))
	// Index iteration — the range-and-copy form copied each endpoint
	// twice (once into the loop-var, once into the output) on every
	// step. SplitEndpoint carries Address/Role/Format strings plus
	// the Labels map header, so the copy is non-trivial. Index assigns
	// straight from source to destination.
	for i := range endpoints {
		out[i] = endpoints[i]
		out[i].Labels = cloneInferenceLabels(endpoints[i].Labels)
	}
	return out
}

func meanNonZero(values ...float64) float64 {
	var total float64
	var count int
	for _, value := range values {
		if value == 0 {
			continue
		}
		total += value
		count++
	}
	if count == 0 {
		return 0
	}
	return total / float64(count)
}

// --- merged from options.go (organisation check: this is the
// inference.GenerateConfig -> metal bridge, not an options surface) ---
// inferenceMinPFieldIndex / inferenceMinPFieldPresent cache the structural
// offset of the MinP field on the linked inference.GenerateConfig so the
// forward-compatibility lookup walks the struct fields once at package
// init rather than once per Generate / Chat / Classify call.
//
// reflect.Type.FieldByName performs a linear scan with no internal cache
// in Go 1.21-1.26. Resolving the probe in init() instead of the prior
// sync.Once-guarded helper drops the per-call cost from "atomic load +
// function call + branch + return tuple" to a single package-var read on
// the hot path — when MinP is absent (the current shape of
// inference.GenerateConfig), the predicate short-circuits before any
// reflect.ValueOf work runs at all.
var (
	inferenceMinPFieldIndex   []int
	inferenceMinPFieldPresent bool
)

func init() {
	field, ok := reflect.TypeFor[inference.GenerateConfig]().FieldByName("MinP")
	if !ok {
		return
	}
	switch field.Type.Kind() {
	case reflect.Float32, reflect.Float64:
		inferenceMinPFieldIndex = field.Index
		inferenceMinPFieldPresent = true
	}
}

func inferenceGenerateConfigToMetal(cfg inference.GenerateConfig) metal.GenerateConfig {
	out := metal.GenerateConfig{
		MaxTokens:      cfg.MaxTokens,
		Temperature:    cfg.Temperature,
		TopK:           cfg.TopK,
		TopP:           cfg.TopP,
		StopTokens:     cfg.StopTokens,
		RepeatPenalty:  cfg.RepeatPenalty,
		EnableThinking: cfg.EnableThinking,
		ThinkingBudget: cfg.ThinkingBudget,
	}
	// Keep go-mlx forward-compatible with inference.GenerateConfig versions
	// that expose MinP without requiring a synchronized dependency update
	// here. The reflect FieldByName scan is amortised through the package-
	// init probe so we pay it once per process and the per-call cost is a
	// single bool load on the absent-field hot path.
	if inferenceMinPFieldPresent {
		out.MinP = float32(reflect.ValueOf(cfg).FieldByIndex(inferenceMinPFieldIndex).Float())
	}
	return out
}
