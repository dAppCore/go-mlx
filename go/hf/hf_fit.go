// SPDX-Licence-Identifier: EUPL-1.2

package hf

import (
	"context"
	"slices"

	core "dappco.re/go"
	"dappco.re/go/inference/quant/jang"
	"dappco.re/go/mlx/memory"
	mp "dappco.re/go/mlx/pack"
	"dappco.re/go/mlx/profile"
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

	entries, err := collectFitEntries(ctx, cfg)
	if err != nil {
		return nil, err
	}
	if len(entries) == 0 {
		return nil, core.NewError("mlx: no model metadata available for fit planning")
	}

	basePlan := memory.NewPlan(memory.Input{Device: cfg.Device})
	report := &FitReport{
		Query:       cfg.Query,
		Device:      cfg.Device,
		DeviceClass: basePlan.MachineClass,
		MemoryPlan:  basePlan,
		Models:      make([]FitPlan, 0, len(entries)),
	}
	for _, entry := range entries {
		report.Models = append(report.Models, planFit(entry, cfg))
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

func collectFitEntries(ctx context.Context, cfg FitConfig) ([]fitEntry, error) {
	// Hoist Source nil-check before the search/id loops — both used to
	// re-check inside the loop body. Also pre-size entries to the known
	// minimum: local paths + IDs are deterministic, search adds at most
	// MaxResults. Saves the growslice walk inside the hot path.
	if (cfg.Query != "" || len(cfg.ModelIDs) > 0) && cfg.Source == nil {
		if cfg.Query != "" {
			return nil, core.NewError("mlx: HF metadata source is required for query search")
		}
		return nil, core.NewError("mlx: HF metadata source is required for model id lookup")
	}
	capacity := len(cfg.LocalPaths) + len(cfg.ModelIDs)
	if cfg.Query != "" && cfg.MaxResults > 0 {
		capacity += cfg.MaxResults
	}
	entries := make([]fitEntry, 0, capacity)
	for _, path := range cfg.LocalPaths {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		meta, root, err := inspectLocalMetadata(path)
		if err != nil {
			return nil, err
		}
		entries = append(entries, fitEntry{meta: meta, source: SourceLocal, localPath: root})
	}
	if cfg.Query != "" {
		found, err := cfg.Source.SearchModels(ctx, cfg.Query, cfg.MaxResults)
		if err != nil {
			return nil, err
		}
		for _, meta := range found {
			entries = append(entries, fitEntry{meta: meta, source: SourceRemote})
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
		entries = append(entries, fitEntry{meta: meta, source: SourceRemote})
	}
	return entries, nil
}

func inspectLocalMetadata(path string) (ModelMetadata, string, error) {
	root := resolveLocalMetadataRoot(path)
	read := core.ReadFile(core.PathJoin(root, "config.json"))
	if !read.OK {
		return ModelMetadata{}, root, core.E("PlanFits", "read local config.json", fitResultError(read))
	}
	var config ModelConfig
	if result := core.JSONUnmarshal(read.Value.([]byte), &config); !result.OK {
		return ModelMetadata{}, root, core.E("PlanFits", "parse local config.json", fitResultError(result))
	}
	files := localModelFiles(root)
	jang, _ := jang.ReadConfig(root)
	return ModelMetadata{
		ID:     localModelID(path, root),
		Config: config,
		Files:  files,
		JANG:   jang,
	}, root, nil
}

func resolveLocalMetadataRoot(path string) string {
	// Replace filepath.Glob(path/snapshots/*/config.json) with a single
	// ReadDir of path/snapshots. Glob runs a readdir then per-match stat
	// *and* allocates the full match path strings plus an outer []string.
	// ReadDir hands back DirEntry values; we pick the lexically-first
	// directory name and let the caller's subsequent ReadFile of
	// config.json surface a missing-file error if the snapshot is
	// incomplete (same observable shape as the previous Glob miss path).
	// For the dominant single-snapshot case this collapses the per-
	// candidate Stat into a single PathJoin.
	snapshotsDir := core.PathJoin(path, "snapshots")
	read := core.ReadDir(core.DirFS(snapshotsDir), ".")
	if read.OK {
		entries, ok := read.Value.([]core.FsDirEntry)
		if ok && len(entries) > 0 {
			// Find the lexically-first directory entry. ReadDir on
			// Darwin/Linux returns dirents in arbitrary order, so
			// scan all entries and track the smallest valid name.
			var winner string
			for _, entry := range entries {
				if !entry.IsDir() {
					continue
				}
				name := entry.Name()
				if winner == "" || name < winner {
					winner = name
				}
			}
			if winner != "" {
				return core.PathJoin(snapshotsDir, winner)
			}
		}
	}
	// hasSuffixFold avoids allocating a lowered copy of the full path
	// (paths can be long: ~/.cache/huggingface/hub/...) just to test a
	// 12-byte suffix.
	if hasSuffixFold(path, "config.json") {
		return core.PathDir(path)
	}
	return path
}

// localModelIDSearchPaths is the small array we walk in localModelID —
// hoisted so the slice literal isn't allocated per call.
var localModelIDSearchOrder = [2]int{0, 1}

func localModelID(inputPath, root string) string {
	paths := [2]string{root, inputPath}
	for _, idx := range localModelIDSearchOrder {
		path := paths[idx]
		for current := path; current != "" && current != "."; {
			base := core.PathBase(current)
			if core.HasPrefix(base, "models--") {
				return core.Replace(core.TrimPrefix(base, "models--"), "--", "/")
			}
			parent := core.PathDir(current)
			if parent == current {
				break
			}
			current = parent
		}
	}
	return core.PathBase(root)
}

func localModelFiles(root string) []ModelFile {
	// Pre-size: a typical pack has 1-4 safetensors shards + tokenizer.json
	// + tokenizer_config.json. 8 is a comfortable initial capacity that
	// avoids growslice for almost every real model.
	files := make([]ModelFile, 0, 8)
	// One ReadDir against the snapshot directory beats five filepath.Glob
	// passes (one per pattern). filepath.Glob does its own readdir per
	// pattern + per-entry filepath.Match alloc; a single ReadDir + inline
	// suffix/name match on the entries collapses the 5x readdir + 5x
	// match slice into a single syscall and a tight per-entry branch.
	read := core.ReadDir(core.DirFS(root), ".")
	if !read.OK {
		return files
	}
	entries, ok := read.Value.([]core.FsDirEntry)
	if !ok {
		return files
	}
	// core.ReadDir (via os.DirFS → os.ReadDir) already returns entries
	// sorted by name. Filtering preserves order, so the resulting files
	// slice is sorted by Name without a post-pass slices.SortFunc — the
	// previous explicit sort was a stale carry-over from the multi-Glob
	// shape where the per-pattern matches were appended in pattern order
	// rather than alphabetical.
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		name := entry.Name()
		if !isLocalModelFileName(name) {
			continue
		}
		var size uint64
		if info, err := entry.Info(); err == nil {
			size = uint64(info.Size())
		}
		files = append(files, ModelFile{Name: name, Size: size})
	}
	return files
}

// isLocalModelFileName reports whether name is one of the weight or
// tokenizer file shapes localModelFiles surfaces. The previous form ran
// five filepath.Glob passes; this inlined predicate replaces them with a
// single suffix/equality check per ReadDir entry.
func isLocalModelFileName(name string) bool {
	switch name {
	case "tokenizer.json", "tokenizer_config.json":
		return true
	}
	// Suffix tests on the weight extensions. The most common shape is
	// "*.safetensors" so put that first.
	return hasSuffixFold(name, ".safetensors") ||
		hasSuffixFold(name, ".gguf") ||
		hasSuffixFold(name, ".bin")
}

func planFit(entry fitEntry, cfg FitConfig) FitPlan {
	meta := entry.meta
	config := meta.Config.normalized()
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
	format, weightBytes := weightFormatAndBytes(meta.Files)
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

func weightFormatAndBytes(files []ModelFile) (string, uint64) {
	if len(files) == 0 {
		return "", 0
	}
	// Cache the format strings — pulling string(mp.ModelPackFormat...) out
	// of the loop avoids the implicit conversion per iteration and lets
	// the per-format pointer compare instead of a fresh string each time.
	const (
		fmtBin = "bin"
	)
	safetensors := string(mp.ModelPackFormatSafetensors)
	gguf := string(mp.ModelPackFormatGGUF)
	mixed := string(mp.ModelPackFormatMixed)

	var format string
	var total uint64
	for _, file := range files {
		// hasSuffixFold avoids the per-file Lower alloc — model weight
		// filenames are ASCII so case-folding the suffix is sufficient.
		name := file.filename()
		switch {
		case hasSuffixFold(name, ".safetensors"):
			if format == "" {
				format = safetensors
			} else if format != safetensors {
				format = mixed
			}
			total += file.byteSize()
		case hasSuffixFold(name, ".gguf"):
			if format == "" {
				format = gguf
			} else if format != gguf {
				format = mixed
			}
			total += file.byteSize()
		case hasSuffixFold(name, ".bin"):
			if format == "" {
				format = fmtBin
			}
			total += file.byteSize()
		}
	}
	return format, total
}

// hasSuffixFold reports whether s ends with suffix using ASCII case-folding.
// Suffix is required to be lowercase. Pure scan, no allocations.
func hasSuffixFold(s, suffix string) bool {
	if len(s) < len(suffix) {
		return false
	}
	off := len(s) - len(suffix)
	for i := 0; i < len(suffix); i++ {
		c := s[off+i]
		if c >= 'A' && c <= 'Z' {
			c += 'a' - 'A'
		}
		if c != suffix[i] {
			return false
		}
	}
	return true
}

func estimateModelKVBytes(config ModelConfig, contextLength, batchSize, bytesPerElement int) uint64 {
	config = config.normalized()
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
	config = config.normalized()
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
