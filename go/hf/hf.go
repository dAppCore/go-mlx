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

const (
	SourceRemote = "huggingface"
	SourceLocal  = "local"

	defaultBaseURL = "https://huggingface.co"
)

// ModelSource provides optional Hugging Face metadata lookup/search.
type ModelSource interface {
	SearchModels(context.Context, string, int) ([]ModelMetadata, error)
	ModelMetadata(context.Context, string) (ModelMetadata, error)
}

// RemoteConfig configures the optional HF Hub metadata source.
type RemoteConfig struct {
	BaseURL   string
	Token     string
	UserAgent string
	Client    *core.HTTPClient
}

// RemoteSource reads model metadata from the Hugging Face Hub API.
type RemoteSource struct {
	baseURL   string
	token     string
	userAgent string
	client    *core.HTTPClient
}

// NewRemoteSource creates a network-backed HF metadata source.
func NewRemoteSource(cfg RemoteConfig) *RemoteSource {
	baseURL := core.TrimSuffix(cfg.BaseURL, "/")
	if baseURL == "" {
		baseURL = defaultBaseURL
	}
	client := cfg.Client
	if client == nil {
		client = &core.HTTPClient{}
	}
	return &RemoteSource{
		baseURL:   baseURL,
		token:     cfg.Token,
		userAgent: firstNonEmpty(cfg.UserAgent, "go-mlx"),
		client:    client,
	}
}

// SearchModels queries HF model metadata. Network use is explicit via this source.
func (s *RemoteSource) SearchModels(ctx context.Context, query string, limit int) ([]ModelMetadata, error) {
	if s == nil {
		return nil, core.NewError("mlx: nil RemoteSource")
	}
	if limit <= 0 {
		limit = 10
	}
	values := core.URLValues{
		"search": []string{query},
		"limit":  []string{core.Itoa(limit)},
		"full":   []string{"true"},
	}
	var models []ModelMetadata
	target := core.Concat(s.baseURL, "/api/models?", values.Encode())
	if err := s.getJSON(ctx, target, &models); err != nil {
		return nil, err
	}
	return models, nil
}

// ModelMetadata returns detailed HF metadata for one model id.
func (s *RemoteSource) ModelMetadata(ctx context.Context, modelID string) (ModelMetadata, error) {
	if s == nil {
		return ModelMetadata{}, core.NewError("mlx: nil RemoteSource")
	}
	target := core.Concat(s.baseURL, "/api/models/", core.URLPathEscape(modelID))
	var meta ModelMetadata
	if err := s.getJSON(ctx, target, &meta); err != nil {
		return ModelMetadata{}, err
	}
	if meta.ID == "" && meta.ModelID == "" {
		meta.ID = modelID
	}
	return meta, nil
}

func (s *RemoteSource) getJSON(ctx context.Context, target string, out any) error {
	reqResult := core.NewHTTPRequestContext(ctx, "GET", target, nil)
	if !reqResult.OK {
		return core.E("RemoteSource", "build request", fitResultError(reqResult))
	}
	req := reqResult.Value.(*core.Request)
	req.Header.Set("Accept", "application/json")
	if s.userAgent != "" {
		req.Header.Set("User-Agent", s.userAgent)
	}
	if s.token != "" {
		req.Header.Set("Authorization", core.Concat("Bearer ", s.token))
	}
	resp, err := s.client.Do(req)
	if err != nil {
		return core.E("RemoteSource", "GET metadata", err)
	}
	read := core.ReadAll(resp.Body)
	if !read.OK {
		return core.E("RemoteSource", "read response", fitResultError(read))
	}
	body, ok := read.Value.(string)
	if !ok {
		return core.E("RemoteSource", "read response", core.NewError("unexpected response body shape"))
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return core.NewError(core.Sprintf("mlx: HF metadata request failed: %d %s", resp.StatusCode, core.Trim(body)))
	}
	if result := core.JSONUnmarshal([]byte(body), out); !result.OK {
		return core.E("RemoteSource", "parse response", fitResultError(result))
	}
	return nil
}

// FitConfig controls model discovery and local fit planning.
type FitConfig struct {
	Query       string
	ModelIDs    []string
	LocalPaths  []string
	MaxResults  int
	Device      memory.DeviceInfo
	Source      ModelSource
	LoRARank    int
	KVBytes     int
	ContextHint int
}

// ModelMetadata is the subset of Hugging Face/local metadata needed for fit planning.
type ModelMetadata struct {
	ID          string      `json:"id,omitempty"`
	ModelID     string      `json:"modelId,omitempty"`
	Tags        []string    `json:"tags,omitempty"`
	PipelineTag string      `json:"pipeline_tag,omitempty"`
	Config      ModelConfig `json:"config,omitempty"`
	Files       []ModelFile `json:"siblings,omitempty"`
	JANG        *jang.Info  `json:"jang,omitempty"`
}

// ModelFile describes one model repository file.
type ModelFile struct {
	Name      string `json:"name,omitempty"`
	RFilename string `json:"rfilename,omitempty"`
	Size      uint64 `json:"size,omitempty"`
	SizeBytes uint64 `json:"sizeBytes,omitempty"`
}

// ModelConfig mirrors common transformer config fields exposed by HF.
type ModelConfig struct {
	ModelType             string              `json:"model_type,omitempty"`
	Architectures         []string            `json:"architectures,omitempty"`
	VocabSize             int                 `json:"vocab_size,omitempty"`
	HiddenSize            int                 `json:"hidden_size,omitempty"`
	IntermediateSize      int                 `json:"intermediate_size,omitempty"`
	NumHiddenLayers       int                 `json:"num_hidden_layers,omitempty"`
	NumAttentionHeads     int                 `json:"num_attention_heads,omitempty"`
	NumKeyValueHeads      int                 `json:"num_key_value_heads,omitempty"`
	HeadDim               int                 `json:"head_dim,omitempty"`
	MaxPositionEmbeddings int                 `json:"max_position_embeddings,omitempty"`
	ContextLength         int                 `json:"context_length,omitempty"`
	Quantization          *QuantizationConfig `json:"quantization,omitempty"`
	QuantizationConfig    *QuantizationConfig `json:"quantization_config,omitempty"`
	TextConfig            *ModelConfig        `json:"text_config,omitempty"`
}

// QuantizationConfig captures quantization metadata when present.
type QuantizationConfig struct {
	Bits      int    `json:"bits,omitempty"`
	GroupSize int    `json:"group_size,omitempty"`
	Type      string `json:"type,omitempty"`
}

// FitReport is the top-level library output for HF/local model fit planning.
type FitReport struct {
	Query       string            `json:"query,omitempty"`
	Device      memory.DeviceInfo `json:"device"`
	DeviceClass memory.Class      `json:"device_class"`
	MemoryPlan  memory.Plan       `json:"memory_plan"`
	Models      []FitPlan         `json:"models"`
}

// FitPlan is one model's local Apple fit estimate.
type FitPlan struct {
	ModelID               string      `json:"model_id,omitempty"`
	LocalPath             string      `json:"local_path,omitempty"`
	Source                string      `json:"source"`
	Architecture          string      `json:"architecture,omitempty"`
	SupportedArchitecture bool        `json:"supported_architecture"`
	NativeLoadable        bool        `json:"native_loadable"`
	WeightFormat          string      `json:"weight_format,omitempty"`
	QuantBits             int         `json:"quant_bits,omitempty"`
	QuantGroup            int         `json:"quant_group,omitempty"`
	QuantType             string      `json:"quant_type,omitempty"`
	QuantFamily           string      `json:"quant_family,omitempty"`
	WeightBytes           uint64      `json:"weight_bytes,omitempty"`
	ExpectedKVBytes       uint64      `json:"expected_kv_bytes,omitempty"`
	ExpectedRuntimeBytes  uint64      `json:"expected_runtime_bytes,omitempty"`
	ExpectedTotalBytes    uint64      `json:"expected_total_bytes,omitempty"`
	ContextLimit          int         `json:"context_limit,omitempty"`
	ContextRecommendation int         `json:"context_recommendation,omitempty"`
	MemoryPlan            memory.Plan `json:"memory_plan"`
	MemoryFits            bool        `json:"memory_fits"`
	InferenceFits         bool        `json:"inference_fits"`
	Training              TrainingFit `json:"training"`
	Embeddings            bool        `json:"embeddings,omitempty"`
	Rerank                bool        `json:"rerank,omitempty"`
	Notes                 []string    `json:"notes,omitempty"`
}

// TrainingFit describes rough training feasibility for local Apple hardware.
type TrainingFit struct {
	LoRAFeasible            bool     `json:"lora_feasible"`
	FullFineTuneFeasible    bool     `json:"full_fine_tune_feasible"`
	RecommendedLoRARank     int      `json:"recommended_lora_rank,omitempty"`
	EstimatedLoRABytes      uint64   `json:"estimated_lora_bytes,omitempty"`
	EstimatedOptimizerBytes uint64   `json:"estimated_optimizer_bytes,omitempty"`
	Notes                   []string `json:"notes,omitempty"`
}

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
	snapshots := core.PathGlob(core.PathJoin(path, "snapshots", "*", "config.json"))
	slices.Sort(snapshots)
	if len(snapshots) > 0 {
		return core.PathDir(snapshots[0])
	}
	if core.HasSuffix(core.Lower(path), "config.json") {
		return core.PathDir(path)
	}
	return path
}

func localModelID(inputPath, root string) string {
	for _, path := range []string{root, inputPath} {
		for current := path; current != "" && current != "."; current = core.PathDir(current) {
			base := core.PathBase(current)
			if core.HasPrefix(base, "models--") {
				return core.Replace(core.TrimPrefix(base, "models--"), "--", "/")
			}
			parent := core.PathDir(current)
			if parent == current {
				break
			}
		}
	}
	return core.PathBase(root)
}

// localModelFiles patterns: precomputed so the make([]string,...) literal
// is not re-built per call. The path glob payload is the dominant cost
// (one Path call + one glob syscall per pattern) so this is presentation
// rather than a perf hot spot, but it keeps the call site allocation-free
// for the iteration list itself.
var localModelFilePatterns = []string{
	"*.safetensors",
	"*.gguf",
	"*.bin",
	"tokenizer.json",
	"tokenizer_config.json",
}

func localModelFiles(root string) []ModelFile {
	// Pre-size: a typical pack has 1-4 safetensors shards + tokenizer.json
	// + tokenizer_config.json. 8 is a comfortable initial capacity that
	// avoids growslice for almost every real model.
	files := make([]ModelFile, 0, 8)
	for _, pattern := range localModelFilePatterns {
		for _, path := range core.PathGlob(core.PathJoin(root, pattern)) {
			info := core.Stat(path)
			var size uint64
			if info.OK {
				size = uint64(info.Value.(core.FsFileInfo).Size())
			}
			files = append(files, ModelFile{Name: core.PathBase(path), Size: size})
		}
	}
	slices.SortFunc(files, func(a, b ModelFile) int {
		if a.filename() < b.filename() {
			return -1
		}
		if a.filename() > b.filename() {
			return 1
		}
		return 0
	})
	return files
}

func planFit(entry fitEntry, cfg FitConfig) FitPlan {
	meta := entry.meta
	config := meta.Config.normalized()
	modelID := firstNonEmpty(meta.ID, meta.ModelID)
	arch := config.architecture()
	contextLimit := config.contextLength()
	quantBits, quantGroup := config.quantization()
	quantType := config.quantizationType()
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
	if quantBits == 0 {
		quantBits = inferQuantBits(meta.Files)
	}

	pack := mp.ModelPack{
		Architecture:          arch,
		SupportedArchitecture: archSupported(arch),
		QuantBits:             quantBits,
		QuantGroup:            quantGroup,
		QuantType:             quantType,
		QuantFamily:           quantFamily,
		ContextLength:         contextLimit,
		WeightBytes:           weightBytes,
	}
	resolveArchitectureProfile(&pack)
	memoryPlan := memory.NewPlan(memory.Input{Device: cfg.Device, Pack: &pack})
	if cfg.ContextHint > 0 && cfg.ContextHint < memoryPlan.ContextLength {
		memoryPlan.ContextLength = cfg.ContextHint
	}
	kvBytes := uint64(0)
	if usesGenerationKVCache(&pack, arch) {
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
		SupportedArchitecture: archSupported(arch),
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
		Embeddings:            pack.Embedding != nil,
		Rerank:                pack.Rerank != nil,
	}
	plan.NativeLoadable = plan.SupportedArchitecture && archNativeRuntime(arch) && format != ""
	plan.MemoryFits = weightBytes > 0 && (limit == 0 || totalBytes <= limit)
	plan.InferenceFits = plan.NativeLoadable && plan.MemoryFits
	plan.Training = estimateTrainingFit(config, plan, limit, cfg.LoRARank)
	plan.Notes = fitNotes(plan, limit)
	return plan
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

func inferQuantBits(files []ModelFile) int {
	if len(files) == 0 {
		return 0
	}
	// Reusable scratch buffer for the lowered form. Most filenames are
	// already lowercase ("model-q4_k_m.gguf") so the hot path skips the
	// allocation entirely; only mixed-case names pay for one lowering.
	// Scratch is reused across iterations: the previous lowered string is
	// not referenced past its switch block, so overwriting is safe.
	var scratch []byte
	for _, file := range files {
		name := file.filename()
		var lowered string
		if hasASCIIUpper(name) {
			scratch = appendLowerASCII(scratch[:0], name)
			lowered = core.AsString(scratch)
		} else {
			lowered = name
		}
		switch {
		case core.Contains(lowered, "q2"):
			return 2
		case core.Contains(lowered, "q3"):
			return 3
		case core.Contains(lowered, "q4") || core.Contains(lowered, "4bit") || core.Contains(lowered, "4-bit"):
			return 4
		case core.Contains(lowered, "q5"):
			return 5
		case core.Contains(lowered, "q6"):
			return 6
		case core.Contains(lowered, "q8") || core.Contains(lowered, "8bit") || core.Contains(lowered, "8-bit"):
			return 8
		case core.Contains(lowered, "bf16") || core.Contains(lowered, "fp16") || core.Contains(lowered, "f16"):
			return 16
		}
	}
	return 0
}

// hasASCIIUpper reports whether s contains any ASCII uppercase byte.
// Pure scan, no allocations — gate before paying for the lowering buffer.
func hasASCIIUpper(s string) bool {
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c >= 'A' && c <= 'Z' {
			return true
		}
	}
	return false
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
	if !fit.LoRAFeasible {
		fit.Notes = append(fit.Notes, "LoRA training estimate exceeds local working-set budget")
	}
	if plan.QuantBits > 0 && plan.QuantBits < 16 {
		fit.Notes = append(fit.Notes, "full fine-tune requires dense trainable weights; quantized pack is LoRA-only")
	}
	return fit
}

func fitNotes(plan FitPlan, memoryLimit uint64) []string {
	var notes []string
	if !plan.SupportedArchitecture {
		notes = append(notes, "architecture is not currently supported by native go-mlx loaders")
	}
	if plan.SupportedArchitecture && !archNativeRuntime(plan.Architecture) {
		notes = append(notes, "architecture is recognized, but native runtime kernels are not implemented yet")
	}
	if plan.WeightBytes == 0 {
		notes = append(notes, "weight byte size is unknown")
	}
	if memoryLimit > 0 && plan.ExpectedTotalBytes > memoryLimit {
		notes = append(notes, "estimated model+KV memory exceeds local working-set budget")
	}
	if plan.ContextLimit > 0 && plan.ContextRecommendation < plan.ContextLimit {
		notes = append(notes, "context recommendation is capped by local machine class")
	}
	if plan.QuantBits > 0 && plan.MemoryPlan.PreferredQuantization > 0 && plan.QuantBits < plan.MemoryPlan.PreferredQuantization {
		notes = append(notes, "model quantization is below machine-class preference")
	}
	return notes
}

func (config ModelConfig) normalized() ModelConfig {
	if config.TextConfig == nil {
		return config
	}
	text := *config.TextConfig
	if text.ModelType == "" {
		text.ModelType = config.ModelType
	}
	if len(text.Architectures) == 0 && len(config.Architectures) > 0 {
		// core.SliceClone — explicit zero-copy substrate primitive that
		// produces a backing array sized to len(src) only. The previous
		// append([]string(nil), src...) form went through the runtime
		// growslice path which over-allocates capacity for further appends
		// we never make.
		text.Architectures = core.SliceClone(config.Architectures)
	}
	return text
}

func (config ModelConfig) architecture() string {
	config = config.normalized()
	for _, arch := range config.Architectures {
		if modelType := architectureFromTransformersName(arch); modelType == "bert_rerank" {
			return modelType
		}
	}
	if config.ModelType != "" {
		return normalizeKnownArchitecture(config.ModelType)
	}
	for _, arch := range config.Architectures {
		if modelType := architectureFromTransformersName(arch); modelType != "" {
			return modelType
		}
	}
	return ""
}

func (config ModelConfig) contextLength() int {
	config = config.normalized()
	return firstPositive(config.ContextLength, config.MaxPositionEmbeddings)
}

func (config ModelConfig) quantization() (bits, group int) {
	config = config.normalized()
	quant := config.QuantizationConfig
	if quant == nil {
		quant = config.Quantization
	}
	if quant == nil {
		return 0, 0
	}
	return quant.Bits, quant.GroupSize
}

func (config ModelConfig) quantizationType() string {
	config = config.normalized()
	quant := config.QuantizationConfig
	if quant == nil {
		quant = config.Quantization
	}
	if quant == nil {
		return ""
	}
	return quant.Type
}

func (file ModelFile) filename() string {
	return firstNonEmpty(file.Name, file.RFilename)
}

func (file ModelFile) byteSize() uint64 {
	if file.Size > 0 {
		return file.Size
	}
	return file.SizeBytes
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

// info := mlx.InferJANG(meta)
func InferJANG(meta ModelMetadata) *jang.Info {
	// Build the lowercase "id tag1 tag2 file1 file2" haystack in one pass.
	// Previously each tag + file Concat allocated a fresh string and Lower
	// allocated again — 4+ allocs per tag/file. Now: one buf, lowercase
	// inline, zero-copy return.
	id := firstNonEmpty(meta.ID, meta.ModelID)
	size := len(id)
	for _, tag := range meta.Tags {
		size += 1 + len(tag)
	}
	for _, file := range meta.Files {
		size += 1 + len(file.filename())
	}
	buf := make([]byte, 0, size)
	buf = appendLowerASCII(buf, id)
	for _, tag := range meta.Tags {
		buf = append(buf, ' ')
		buf = appendLowerASCII(buf, tag)
	}
	for _, file := range meta.Files {
		buf = append(buf, ' ')
		buf = appendLowerASCII(buf, file.filename())
	}
	needle := core.AsString(buf)

	switch {
	case core.Contains(needle, "jangtq"):
		info := &jang.Info{
			Profile:          "JANGTQ",
			WeightFormat:     "mxtq",
			Method:           "affine+mxtq",
			GroupSize:        jangGroupSize(meta),
			BitsDefault:      2,
			RoutedExpertBits: 2,
		}
		info.Packed = jang.BuildPackedProfile(info)
		return info
	case core.Contains(needle, "jang"):
		profile := inferJANGProfileName(needle)
		info := &jang.Info{
			Profile:     profile,
			GroupSize:   jangGroupSize(meta),
			BitsDefault: firstPositive(jang.ProfileBits(profile), 0),
		}
		info.Packed = jang.BuildPackedProfile(info)
		return info
	default:
		return nil
	}
}

// appendLowerASCII appends s to dst with ASCII A-Z mapped to a-z. Non-ASCII
// bytes pass through unchanged (consistent with the previous core.Lower
// surface for our domain: model IDs, tags, filenames are all ASCII).
func appendLowerASCII(dst []byte, s string) []byte {
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c >= 'A' && c <= 'Z' {
			c += 'a' - 'A'
		}
		dst = append(dst, c)
	}
	return dst
}

func jangGroupSize(meta ModelMetadata) int {
	if quant := meta.Config.QuantizationConfig; quant != nil && quant.GroupSize > 0 {
		return quant.GroupSize
	}
	if quant := meta.Config.Quantization; quant != nil && quant.GroupSize > 0 {
		return quant.GroupSize
	}
	return 64
}

func inferJANGProfileName(value string) string {
	for _, profile := range []string{"jang_1l", "jang_2s", "jang_2l", "jang_3l", "jang_4k", "jang_4m"} {
		if core.Contains(value, profile) {
			return core.Upper(profile)
		}
	}
	return "JANG"
}

type modelConfigProbe struct {
	ModelType             string   `json:"model_type"`
	VocabSize             int      `json:"vocab_size"`
	HiddenSize            int      `json:"hidden_size"`
	NumHiddenLayers       int      `json:"num_hidden_layers"`
	MaxPositionEmbeddings int      `json:"max_position_embeddings"`
	Architectures         []string `json:"architectures"`
	NumLabels             int      `json:"num_labels"`
	TextConfig            struct {
		ModelType             string `json:"model_type"`
		VocabSize             int    `json:"vocab_size"`
		HiddenSize            int    `json:"hidden_size"`
		NumHiddenLayers       int    `json:"num_hidden_layers"`
		MaxPositionEmbeddings int    `json:"max_position_embeddings"`
	} `json:"text_config"`
	Quantization *struct {
		Bits      int `json:"bits"`
		GroupSize int `json:"group_size"`
	} `json:"quantization"`
	QuantizationConfig *struct {
		Bits      int `json:"bits"`
		GroupSize int `json:"group_size"`
	} `json:"quantization_config"`
}

func readModelConfig(dir string) (*modelConfigProbe, error) {
	read := core.ReadFile(core.PathJoin(dir, "config.json"))
	if !read.OK {
		return nil, read.Value.(error)
	}
	var config modelConfigProbe
	if result := core.JSONUnmarshal(read.Value.([]byte), &config); !result.OK {
		return nil, result.Value.(error)
	}
	return &config, nil
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if core.Trim(value) != "" {
			return value
		}
	}
	return ""
}

func firstPositive(values ...int) int {
	for _, value := range values {
		if value > 0 {
			return value
		}
	}
	return 0
}

func (probe *modelConfigProbe) architecture() string {
	if probe == nil {
		return ""
	}
	for _, architecture := range probe.Architectures {
		if modelType := architectureFromTransformersName(architecture); modelType == "bert_rerank" {
			return modelType
		}
	}
	if probe.ModelType != "" {
		return normalizeKnownArchitecture(probe.ModelType)
	}
	if probe.TextConfig.ModelType != "" {
		return normalizeKnownArchitecture(probe.TextConfig.ModelType)
	}
	for _, architecture := range probe.Architectures {
		if modelType := architectureFromTransformersName(architecture); modelType != "" {
			return modelType
		}
	}
	return ""
}

func (probe *modelConfigProbe) numLayers() int {
	if probe == nil {
		return 0
	}
	if probe.NumHiddenLayers > 0 {
		return probe.NumHiddenLayers
	}
	return probe.TextConfig.NumHiddenLayers
}

func (probe *modelConfigProbe) vocabSize() int {
	if probe == nil {
		return 0
	}
	if probe.VocabSize > 0 {
		return probe.VocabSize
	}
	return probe.TextConfig.VocabSize
}

func (probe *modelConfigProbe) hiddenSize() int {
	if probe == nil {
		return 0
	}
	if probe.HiddenSize > 0 {
		return probe.HiddenSize
	}
	return probe.TextConfig.HiddenSize
}

func (probe *modelConfigProbe) contextLength() int {
	if probe == nil {
		return 0
	}
	if probe.MaxPositionEmbeddings > 0 {
		return probe.MaxPositionEmbeddings
	}
	return probe.TextConfig.MaxPositionEmbeddings
}

func (probe *modelConfigProbe) quantBits() int {
	if probe == nil {
		return 0
	}
	if probe.Quantization != nil {
		return probe.Quantization.Bits
	}
	if probe.QuantizationConfig != nil {
		return probe.QuantizationConfig.Bits
	}
	return 0
}

func (probe *modelConfigProbe) quantGroup() int {
	if probe == nil {
		return 0
	}
	if probe.Quantization != nil {
		return probe.Quantization.GroupSize
	}
	if probe.QuantizationConfig != nil {
		return probe.QuantizationConfig.GroupSize
	}
	return 0
}

func normalizeKnownArchitecture(value string) string {
	// Skip Trim+Lower+Replace when the input is already in canonical form
	// (no leading/trailing whitespace, no uppercase, no '-'). Most callers
	// (ModelConfig.architecture for HF model_type, repeat lookups) hit this.
	if !needsNormalisation(value) {
		return matchKnownArchitecture(value)
	}
	return matchKnownArchitecture(normaliseArchString(value))
}

// normaliseArchString trims surrounding whitespace, lowercases ASCII, and
// rewrites '-' to '_' in a single pass. Replaces the old
// Lower(Trim(...))+Replace(...) chain that allocated twice and walked the
// string three times.
func normaliseArchString(s string) string {
	// Find trim bounds.
	start, end := 0, len(s)
	for start < end {
		c := s[start]
		if c != ' ' && c != '\t' && c != '\n' && c != '\r' {
			break
		}
		start++
	}
	for end > start {
		c := s[end-1]
		if c != ' ' && c != '\t' && c != '\n' && c != '\r' {
			break
		}
		end--
	}
	if start == end {
		return ""
	}
	buf := make([]byte, end-start)
	for i := start; i < end; i++ {
		c := s[i]
		if c == '-' {
			c = '_'
		} else if c >= 'A' && c <= 'Z' {
			c += 'a' - 'A'
		}
		buf[i-start] = c
	}
	return core.AsString(buf)
}

// needsNormalisation reports whether normalizeKnownArchitecture has any
// transformation work to do — true if value contains whitespace, '-', or
// ASCII uppercase. Pure scan, no allocations.
func needsNormalisation(value string) bool {
	for i := 0; i < len(value); i++ {
		c := value[i]
		if c == '-' || c == ' ' || c == '\t' || c == '\n' || c == '\r' || (c >= 'A' && c <= 'Z') {
			return true
		}
	}
	return false
}

// matchKnownArchitecture is the bare switch table — pulled out so both the
// fast and slow paths share it without duplication.
func matchKnownArchitecture(value string) string {
	switch value {
	case "qwen3_5":
		return "qwen3_next"
	case "minimaxm2", "minimax_m2":
		return "minimax_m2"
	case "mixtral":
		return "mixtral"
	case "mistral":
		return "mistral"
	case "phi", "phi3", "phi4":
		return "phi"
	case "deepseek", "deepseek_v3", "deepseek_r1":
		return "deepseek"
	case "gptoss", "gpt_oss", "gpt_oss_model":
		return "gpt_oss"
	case "bert":
		return "bert"
	case "bert_rerank", "bert_cross_encoder":
		return "bert_rerank"
	default:
		return value
	}
}

func architectureFromTransformersName(architecture string) string {
	// Case-sensitive fast path first — the canonical HF transformers class
	// names are PascalCase ("Qwen3ForCausalLM"). Avoids the Lower+Replace
	// allocs for the common path.
	switch {
	case core.Contains(architecture, "Gemma4"):
		return "gemma4_text"
	case core.Contains(architecture, "Gemma3"):
		return "gemma3"
	case core.Contains(architecture, "Gemma2"):
		return "gemma2"
	case core.Contains(architecture, "Qwen3"):
		// Qwen3 hits — disambiguate MoE / Next via compact form only here.
		if compact := lowerNoSep(architecture); core.Contains(compact, "qwen3moe") {
			return "qwen3_moe"
		} else if core.Contains(compact, "qwen3next") {
			return "qwen3_next"
		}
		return "qwen3"
	case core.Contains(architecture, "Qwen2"):
		return "qwen2"
	case core.Contains(architecture, "Llama"):
		return "llama"
	case core.Contains(architecture, "MiniMaxM2"):
		return "minimax_m2"
	case core.Contains(architecture, "Mixtral"):
		return "mixtral"
	case core.Contains(architecture, "Mistral"):
		return "mistral"
	case core.Contains(architecture, "Phi"):
		return "phi"
	case core.Contains(architecture, "Deepseek") || core.Contains(architecture, "DeepSeek"):
		return "deepseek"
	case core.Contains(architecture, "GptOss") || core.Contains(architecture, "GPTOSS"):
		return "gpt_oss"
	case core.Contains(architecture, "Bert") || core.Contains(architecture, "Roberta") || core.Contains(architecture, "Deberta"):
		// Bert / Roberta / Deberta family — disambiguate rerank via compact.
		compact := lowerNoSep(architecture)
		if core.Contains(compact, "bertforsequenceclassification") || core.Contains(compact, "robertaforsequenceclassification") || core.Contains(compact, "xlmrobertaforsequenceclassification") || core.Contains(compact, "debertav2forsequenceclassification") {
			return "bert_rerank"
		}
		if core.Contains(architecture, "Bert") {
			return "bert"
		}
		return ""
	default:
		// Unknown PascalCase shape — fall back to compact lower form so a
		// few stragglers like "qwen3_moe" or "bert_for_sequence_classification"
		// still classify when callers feed snake_case identifiers.
		compact := lowerNoSep(architecture)
		switch {
		case core.Contains(compact, "bertforsequenceclassification") || core.Contains(compact, "robertaforsequenceclassification") || core.Contains(compact, "xlmrobertaforsequenceclassification") || core.Contains(compact, "debertav2forsequenceclassification"):
			return "bert_rerank"
		case core.Contains(compact, "qwen3moe"):
			return "qwen3_moe"
		case core.Contains(compact, "qwen3next"):
			return "qwen3_next"
		}
		return ""
	}
}

// lowerNoSep returns architecture lowercased with "_" and "-" removed.
// Pure helper used by the slow paths of architectureFromTransformersName —
// kept out of line so the fast PascalCase path costs zero allocations.
func lowerNoSep(s string) string {
	if s == "" {
		return ""
	}
	// Single pass over bytes: skip "_"/"-" and lowercase ASCII inline.
	buf := make([]byte, 0, len(s))
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c == '_' || c == '-' {
			continue
		}
		if c >= 'A' && c <= 'Z' {
			c += 'a' - 'A'
		}
		buf = append(buf, c)
	}
	return core.AsString(buf)
}

func indexString(s, substr string) int {
	if substr == "" {
		return 0
	}
	if len(substr) > len(s) {
		return -1
	}
	for i := range len(s) - len(substr) + 1 {
		if s[i:i+len(substr)] == substr {
			return i
		}
	}
	return -1
}

func archSupported(architecture string) bool {
	_, ok := profile.LookupArchitectureProfile(architecture)
	return ok
}

func archNativeRuntime(architecture string) bool {
	p, ok := profile.LookupArchitectureProfile(architecture)
	return ok && p.NativeRuntime
}

func usesGenerationKVCache(pack *mp.ModelPack, architecture string) bool {
	if pack != nil {
		if pack.Embedding != nil || pack.Rerank != nil {
			return false
		}
		if pack.Architecture != "" {
			architecture = pack.Architecture
		}
		if pack.ArchitectureProfile != nil && (pack.ArchitectureProfile.Embeddings || pack.ArchitectureProfile.Rerank) {
			return false
		}
	}
	if p, ok := profile.LookupArchitectureProfile(architecture); ok && (p.Embeddings || p.Rerank) {
		return false
	}
	return true
}

func resolveArchitectureProfile(pack *mp.ModelPack) {
	if pack == nil || pack.Architecture == "" {
		return
	}
	if pack.ArchitectureProfile != nil {
		return
	}
	if resolved, ok := profile.LookupArchitectureProfile(pack.Architecture); ok {
		pack.ArchitectureProfile = &resolved
	}
}
