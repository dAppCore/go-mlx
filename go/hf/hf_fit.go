// SPDX-Licence-Identifier: EUPL-1.2

package hf

import (
	"context"
	"slices"

	core "dappco.re/go"
	sharedhf "dappco.re/go/inference/hf"
	"dappco.re/go/inference/memory"
	mp "dappco.re/go/inference/modelpack"
	"dappco.re/go/inference/profile"
)

// PlanFits discovers HF/local metadata and estimates local Apple fit.
func PlanFits(ctx context.Context, cfg FitConfig) (*FitReport, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if cfg.MaxResults <= 0 {
		cfg.MaxResults = 10
	}
	if cfg.LoRARank <= 0 {
		cfg.LoRARank = 16
	}
	if cfg.KVBytes <= 0 {
		cfg.KVBytes = 2
	}

	models, err := planFitEntries(ctx, cfg)
	if err != nil {
		return nil, err
	}
	if len(models) == 0 {
		return nil, core.NewError("mlx: no model metadata available for fit planning")
	}

	basePlan := memory.NewPlan(memory.Input{Device: cfg.Device})
	report := &FitReport{
		Query:       cfg.Query,
		Device:      cfg.Device,
		DeviceClass: basePlan.MachineClass,
		MemoryPlan:  basePlan,
		Models:      models,
	}
	slices.SortFunc(report.Models, func(a, b FitPlan) int {
		if a.InferenceFits != b.InferenceFits {
			if a.InferenceFits {
				return -1
			}
			return 1
		}
		if a.ExpectedTotalBytes < b.ExpectedTotalBytes {
			return -1
		}
		if a.ExpectedTotalBytes > b.ExpectedTotalBytes {
			return 1
		}
		return 0
	})
	return report, nil
}

type fitEntry struct {
	meta      ModelMetadata
	source    string
	localPath string
}

// planFitEntries discovers local + remote model metadata and plans each
// fit straight into the result slice. It folds the old collect-then-plan
// two-step: the previous form materialised an intermediate []fitEntry only
// to range over it once in PlanFits, costing a whole slice allocation on
// every call. Here each discovered fitEntry is a stack value handed
// directly to planFit, so the only slice allocated is the FitPlan result
// PlanFits already owns. Discovery order (local → search → IDs) and every
// error path are preserved exactly so the observable output is unchanged.
func planFitEntries(ctx context.Context, cfg FitConfig) ([]FitPlan, error) {
	// Hoist Source nil-check before the search/id loops — both used to
	// re-check inside the loop body.
	if (cfg.Query != "" || len(cfg.ModelIDs) > 0) && cfg.Source == nil {
		if cfg.Query != "" {
			return nil, core.NewError("mlx: HF metadata source is required for query search")
		}
		return nil, core.NewError("mlx: HF metadata source is required for model id lookup")
	}
	// Pre-size to the deterministic count (local paths + explicit IDs).
	// Sizing to this exact floor — rather than the local+IDs+MaxResults
	// upper bound — keeps the no-query path (the common "what's in my cache"
	// call) at a single exact-fit allocation and avoids reserving MaxResults
	// slots of the 752-byte FitPlan that a search rarely fills. The query
	// path extends the slice once, to the exact result count, via
	// slices.Grow below — so search growth is a single allocation, never the
	// geometric doubling a bare append loop would pay.
	models := make([]FitPlan, 0, len(cfg.LocalPaths)+len(cfg.ModelIDs))
	for _, path := range cfg.LocalPaths {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		meta, root, err := sharedhf.InspectLocalMetadata(path)
		if err != nil {
			return nil, err
		}
		models = append(models, planFit(fitEntry{meta: meta, source: SourceLocal, localPath: root}, cfg))
	}
	if cfg.Query != "" {
		found, err := cfg.Source.SearchModels(ctx, cfg.Query, cfg.MaxResults)
		if err != nil {
			return nil, err
		}
		// Grow once to fit every search result plus the model IDs still to
		// come, so neither the search-append loop below nor the trailing ID
		// loop triggers a runtime growslice.
		models = slices.Grow(models, len(found)+len(cfg.ModelIDs))
		for _, meta := range found {
			models = append(models, planFit(fitEntry{meta: meta, source: SourceRemote}, cfg))
		}
	}
	for _, id := range cfg.ModelIDs {
		meta, err := cfg.Source.ModelMetadata(ctx, id)
		if err != nil {
			return nil, err
		}
		if meta.ID == "" && meta.ModelID == "" {
			meta.ID = id
		}
		models = append(models, planFit(fitEntry{meta: meta, source: SourceRemote}, cfg))
	}
	return models, nil
}

func planFit(entry fitEntry, cfg FitConfig) FitPlan {
	meta := entry.meta
	config := normalizeModelConfig(meta.Config)
	modelID := firstNonEmpty(meta.ID, meta.ModelID)
	// Inline the architecture / contextLength / quantization /
	// quantizationType accessors here — each one normalizes config again
	// (a value copy of the ~96-byte ModelConfig struct) before reading a
	// single field. We've already normalised once at the top of the
	// function; read directly from the normalised local instead.
	arch := configArchitecture(&config)
	contextLimit := firstPositive(config.ContextLength, config.MaxPositionEmbeddings)
	quant := config.QuantizationConfig
	if quant == nil {
		quant = config.Quantization
	}
	var quantBits, quantGroup int
	var quantType string
	if quant != nil {
		quantBits = quant.Bits
		quantGroup = quant.GroupSize
		quantType = quant.Type
	}
	quantFamily := ""
	format, weightBytes := sharedhf.WeightFormatAndBytes(meta.Files)
	info := meta.JANG
	if info == nil {
		info = InferJANG(meta)
	}
	if info != nil {
		quantBits = firstPositive(info.BitsDefault, quantBits)
		quantGroup = firstPositive(info.GroupSize, quantGroup)
		if info.Packed != nil {
			quantType = info.Packed.Type
		}
		quantFamily = "jang"
	}
	// quantBits stays 0 (honest unknown) when neither the config
	// quantization block nor JANG declared a width — the filename is never
	// consulted. Quant is read from what the model actually ships, not what
	// the file is called; post-download the packed-tensor geometry
	// (model.ResolveQuant) settles it for sure.

	// Hoist the architecture profile lookup: previously planFit hit
	// profile.LookupArchitectureProfile up to 5 times per call
	// (archSupported x2, resolveArchitectureProfile, archNativeRuntime,
	// usesGenerationKVCache). Use the Ref form — read-only pointer into
	// the immutable registry, no 5-slice clone. pack.ArchitectureProfile
	// borrows the same pointer (the ModelPack is consumed inside this
	// function; nothing downstream mutates the profile's slice fields).
	archProfileRef, archProfileOK := profile.LookupArchitectureProfileRef(arch)
	supportedArch := archProfileOK
	nativeRuntime := archProfileOK && archProfileRef.NativeRuntime
	nonStandaloneNative := archProfileOK && archProfileRef.NativeRuntime && !archProfileRef.Generation && !archProfileRef.Embeddings && !archProfileRef.Rerank

	pack := mp.ModelPack{
		Architecture:          arch,
		SupportedArchitecture: supportedArch,
		QuantBits:             quantBits,
		QuantGroup:            quantGroup,
		QuantType:             quantType,
		QuantFamily:           quantFamily,
		ContextLength:         contextLimit,
		WeightBytes:           weightBytes,
	}
	if archProfileOK {
		pack.ArchitectureProfile = archProfileRef
	}
	memoryPlan := memory.NewPlan(memory.Input{Device: cfg.Device, Pack: &pack})
	if cfg.ContextHint > 0 && cfg.ContextHint < memoryPlan.ContextLength {
		memoryPlan.ContextLength = cfg.ContextHint
	}
	kvBytes := uint64(0)
	if packUsesKVCache(&pack, archProfileOK, archProfileRef) {
		kvBytes = estimateModelKVBytes(config, memoryPlan.ContextLength, memoryPlan.BatchSize, cfg.KVBytes)
	}
	runtimeBytes := estimateRuntimeOverheadBytes(weightBytes)
	totalBytes := weightBytes + kvBytes + runtimeBytes
	limit := memoryPlan.MemoryLimitBytes
	if limit == 0 {
		limit = cfg.Device.MaxRecommendedWorkingSetSize
	}
	if limit == 0 {
		limit = cfg.Device.MemorySize
	}

	plan := FitPlan{
		ModelID:               modelID,
		LocalPath:             entry.localPath,
		Source:                entry.source,
		Architecture:          arch,
		SupportedArchitecture: supportedArch,
		WeightFormat:          format,
		QuantBits:             quantBits,
		QuantGroup:            quantGroup,
		QuantType:             quantType,
		QuantFamily:           quantFamily,
		WeightBytes:           weightBytes,
		ExpectedKVBytes:       kvBytes,
		ExpectedRuntimeBytes:  runtimeBytes,
		ExpectedTotalBytes:    totalBytes,
		ContextLimit:          contextLimit,
		ContextRecommendation: memoryPlan.ContextLength,
		MemoryPlan:            memoryPlan,
		Embeddings:            archProfileOK && archProfileRef.Embeddings,
		Rerank:                archProfileOK && archProfileRef.Rerank,
	}
	plan.NativeLoadable = supportedArch && nativeRuntime && format != ""
	if nonStandaloneNative {
		plan.NativeLoadable = false
	}
	plan.MemoryFits = weightBytes > 0 && (limit == 0 || totalBytes <= limit)
	plan.InferenceFits = plan.NativeLoadable && plan.MemoryFits
	plan.Training = estimateTrainingFit(config, plan, limit, cfg.LoRARank)
	plan.Notes = fitNotes(plan, limit, nativeRuntime, nonStandaloneNative)
	return plan
}

// packUsesKVCache is the planFit-local variant of usesGenerationKVCache.
// Skips the per-call profile.LookupArchitectureProfile inside the public
// helper (the planFit caller already has the lookup result) and the
// pack.ArchitectureProfile probe (we set it from the same lookup).
// archProfile is a read-only pointer into the static registry; do not
// mutate.
func packUsesKVCache(pack *mp.ModelPack, archProfileOK bool, archProfile *profile.ModelArchitectureProfile) bool {
	if pack != nil {
		if pack.Embedding != nil || pack.Rerank != nil {
			return false
		}
	}
	if archProfileOK && archProfile != nil && (!archProfile.Generation || archProfile.Embeddings || archProfile.Rerank) {
		return false
	}
	return true
}

func estimateModelKVBytes(config ModelConfig, contextLength, batchSize, bytesPerElement int) uint64 {
	config = normalizeModelConfig(config)
	layers := config.NumHiddenLayers
	hidden := config.HiddenSize
	heads := config.NumAttentionHeads
	kvHeads := config.NumKeyValueHeads
	if kvHeads <= 0 {
		kvHeads = heads
	}
	headDim := config.HeadDim
	if headDim <= 0 && heads > 0 && hidden > 0 {
		headDim = hidden / heads
	}
	if batchSize <= 0 {
		batchSize = 1
	}
	if bytesPerElement <= 0 {
		bytesPerElement = 2
	}
	if layers <= 0 || contextLength <= 0 {
		return 0
	}
	var perToken int
	if kvHeads > 0 && headDim > 0 {
		perToken = 2 * layers * kvHeads * headDim * bytesPerElement
	} else if hidden > 0 {
		perToken = 2 * layers * hidden * bytesPerElement
	}
	if perToken <= 0 {
		return 0
	}
	return uint64(perToken) * uint64(contextLength) * uint64(batchSize)
}

func estimateRuntimeOverheadBytes(weightBytes uint64) uint64 {
	if weightBytes == 0 {
		return 0
	}
	overhead := weightBytes / 10
	if overhead < memory.GiB {
		return memory.GiB
	}
	return overhead
}

func estimateTrainingFit(config ModelConfig, plan FitPlan, memoryLimit uint64, rank int) TrainingFit {
	config = normalizeModelConfig(config)
	if rank <= 0 {
		rank = 16
	}
	hidden := config.HiddenSize
	layers := config.NumHiddenLayers
	targets := 4
	if hidden <= 0 || layers <= 0 {
		targets = 0
	}
	loraParams := uint64(positiveInt(hidden)) *
		uint64(positiveInt(layers)) *
		uint64(positiveInt(targets)) *
		uint64(rank) *
		2
	loraWeights := loraParams * 2
	optimizerBytes := loraParams * 8
	loraTotal := loraWeights + optimizerBytes
	totalWithLoRA := plan.ExpectedTotalBytes + loraTotal
	fit := TrainingFit{
		RecommendedLoRARank:     rank,
		EstimatedLoRABytes:      loraWeights,
		EstimatedOptimizerBytes: optimizerBytes,
	}
	fit.LoRAFeasible = plan.InferenceFits && (memoryLimit == 0 || totalWithLoRA <= memoryLimit)
	fullTuneBytes := plan.WeightBytes*6 + plan.ExpectedKVBytes + plan.ExpectedRuntimeBytes
	fit.FullFineTuneFeasible = plan.NativeLoadable && plan.QuantBits >= 16 && (memoryLimit == 0 || fullTuneBytes <= memoryLimit)
	// Pre-count the notes so the result slice is allocated exactly once
	// at the right capacity. The previous append-from-nil pattern paid a
	// cap-1 alloc plus a cap-1→2 growslice when both notes fired. nil for
	// the zero-note path keeps TrainingFit.Notes ungrown for the common
	// case (CPU/MPS-clean models).
	loraBudgetOver := !fit.LoRAFeasible
	quantBelowDense := plan.QuantBits > 0 && plan.QuantBits < 16
	count := 0
	if loraBudgetOver {
		count++
	}
	if quantBelowDense {
		count++
	}
	if count > 0 {
		notes := make([]string, 0, count)
		if loraBudgetOver {
			notes = append(notes, "LoRA training estimate exceeds local working-set budget")
		}
		if quantBelowDense {
			notes = append(notes, "full fine-tune requires dense trainable weights; quantized pack is LoRA-only")
		}
		fit.Notes = notes
	}
	return fit
}

func fitNotes(plan FitPlan, memoryLimit uint64, nativeRuntime bool, nonStandaloneNative bool) []string {
	// Caller already has the archNativeRuntime result from the hoisted
	// LookupArchitectureProfile in planFit — pass it through so fitNotes
	// doesn't repeat the full lookup-and-clone.
	//
	// Pre-count the notes so the result slice is allocated exactly once
	// at the right capacity. The previous append-from-nil pattern paid
	// 2-3 growslice allocs when 2+ notes fired (cap 1 → 2 → 4). For the
	// zero-note case we return nil so the FitPlan.Notes field stays nil.
	unsupported := !plan.SupportedArchitecture
	notNative := plan.SupportedArchitecture && !nativeRuntime
	unknownBytes := plan.WeightBytes == 0
	overBudget := memoryLimit > 0 && plan.ExpectedTotalBytes > memoryLimit
	contextCapped := plan.ContextLimit > 0 && plan.ContextRecommendation < plan.ContextLimit
	count := 0
	if unsupported {
		count++
	}
	if notNative {
		count++
	}
	if nonStandaloneNative {
		count++
	}
	if unknownBytes {
		count++
	}
	if overBudget {
		count++
	}
	if contextCapped {
		count++
	}
	if count == 0 {
		return nil
	}
	notes := make([]string, 0, count)
	if unsupported {
		notes = append(notes, "architecture is not currently supported by native go-mlx loaders")
	}
	if notNative {
		notes = append(notes, "architecture is recognized, but native runtime kernels are not implemented yet")
	}
	if nonStandaloneNative {
		switch plan.Architecture {
		case "gemma4_assistant":
			notes = append(notes, "Gemma 4 assistant is an attached MTP drafter; load with LoadSpeculativePair beside a Gemma 4 target")
		case "minimax_m2":
			notes = append(notes, "MiniMax M2 has a staged native JANGTQ/MXTQ tensor-plan loader; standalone sparse generation is still pending")
		default:
			notes = append(notes, "architecture has native runtime assets but is not a standalone generation target")
		}
	}
	if unknownBytes {
		notes = append(notes, "weight byte size is unknown")
	}
	if overBudget {
		notes = append(notes, "estimated model+KV memory exceeds local working-set budget")
	}
	if contextCapped {
		notes = append(notes, "context recommendation is capped by local machine class")
	}
	return notes
}

func isGemma4UnifiedConfig(config ModelConfig) bool {
	if profile.NormalizeArchitecture(config.ModelType) == "gemma4_unified" {
		return true
	}
	for _, arch := range config.Architectures {
		if profile.ArchitectureFromTransformersName(arch) == "gemma4_unified" {
			return true
		}
	}
	return false
}

func isGemma4AssistantConfig(config ModelConfig) bool {
	if profile.NormalizeArchitecture(config.ModelType) == "gemma4_assistant" {
		return true
	}
	for _, arch := range config.Architectures {
		if profile.ArchitectureFromTransformersName(arch) == "gemma4_assistant" {
			return true
		}
	}
	return false
}

// configArchitecture is the already-normalised, pointer-receiver variant
// for callers that have already done the normalize. Avoids the second
// normalize value-copy of ~96-byte ModelConfig.
func configArchitecture(config *ModelConfig) string {
	for _, arch := range config.Architectures {
		if modelType := profile.ArchitectureFromTransformersName(arch); modelType == "bert_rerank" {
			return modelType
		}
	}
	if config.ModelType != "" {
		return profile.NormalizeArchitecture(config.ModelType)
	}
	for _, arch := range config.Architectures {
		if modelType := profile.ArchitectureFromTransformersName(arch); modelType != "" {
			return modelType
		}
	}
	return ""
}

func positiveInt(value int) int {
	if value < 0 {
		return 0
	}
	return value
}

func fitResultError(result core.Result) error {
	if result.OK {
		return nil
	}
	if err, ok := result.Value.(error); ok {
		return err
	}
	return core.NewError("core result failed")
}
