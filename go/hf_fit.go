// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"slices"

	core "dappco.re/go"
)

const (
	HFModelSourceRemote = "huggingface"
	HFModelSourceLocal  = "local"

	defaultHuggingFaceBaseURL = "https://huggingface.co"
)

// HFModelSource provides optional Hugging Face metadata lookup/search.
type HFModelSource interface {
	SearchModels(context.Context, string, int) ([]HFModelMetadata, error)
	ModelMetadata(context.Context, string) (HFModelMetadata, error)
}

// HuggingFaceModelSourceConfig configures the optional HF Hub metadata source.
type HuggingFaceModelSourceConfig struct {
	BaseURL   string
	Token     string
	UserAgent string
	Client    *core.HTTPClient
}

// HuggingFaceModelSource reads model metadata from the Hugging Face Hub API.
type HuggingFaceModelSource struct {
	baseURL   string
	token     string
	userAgent string
	client    *core.HTTPClient
}

// NewHuggingFaceModelSource creates a network-backed HF metadata source.
func NewHuggingFaceModelSource(cfg HuggingFaceModelSourceConfig) *HuggingFaceModelSource {
	baseURL := core.TrimSuffix(cfg.BaseURL, "/")
	if baseURL == "" {
		baseURL = defaultHuggingFaceBaseURL
	}
	client := cfg.Client
	if client == nil {
		client = &core.HTTPClient{}
	}
	return &HuggingFaceModelSource{
		baseURL:   baseURL,
		token:     cfg.Token,
		userAgent: firstNonEmpty(cfg.UserAgent, "go-mlx"),
		client:    client,
	}
}

// SearchModels queries HF model metadata. Network use is explicit via this source.
func (s *HuggingFaceModelSource) SearchModels(ctx context.Context, query string, limit int) ([]HFModelMetadata, error) {
	if s == nil {
		return nil, core.NewError("mlx: nil HuggingFaceModelSource")
	}
	if limit <= 0 {
		limit = 10
	}
	values := core.URLValues{
		"search": []string{query},
		"limit":  []string{core.Itoa(limit)},
		"full":   []string{"true"},
	}
	var models []HFModelMetadata
	target := core.Concat(s.baseURL, "/api/models?", values.Encode())
	if err := s.getJSON(ctx, target, &models); err != nil {
		return nil, err
	}
	return models, nil
}

// ModelMetadata returns detailed HF metadata for one model id.
func (s *HuggingFaceModelSource) ModelMetadata(ctx context.Context, modelID string) (HFModelMetadata, error) {
	if s == nil {
		return HFModelMetadata{}, core.NewError("mlx: nil HuggingFaceModelSource")
	}
	target := core.Concat(s.baseURL, "/api/models/", core.URLPathEscape(modelID))
	var meta HFModelMetadata
	if err := s.getJSON(ctx, target, &meta); err != nil {
		return HFModelMetadata{}, err
	}
	if meta.ID == "" && meta.ModelID == "" {
		meta.ID = modelID
	}
	return meta, nil
}

func (s *HuggingFaceModelSource) getJSON(ctx context.Context, target string, out any) error {
	reqResult := core.NewHTTPRequestContext(ctx, "GET", target, nil)
	if !reqResult.OK {
		return core.E("HuggingFaceModelSource", "build request", hfFitResultError(reqResult))
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
		return core.E("HuggingFaceModelSource", "GET metadata", err)
	}
	read := core.ReadAll(resp.Body)
	if !read.OK {
		return core.E("HuggingFaceModelSource", "read response", hfFitResultError(read))
	}
	body, ok := read.Value.(string)
	if !ok {
		return core.E("HuggingFaceModelSource", "read response", core.NewError("unexpected response body shape"))
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return core.NewError(core.Sprintf("mlx: HF metadata request failed: %d %s", resp.StatusCode, core.Trim(body)))
	}
	if result := core.JSONUnmarshal([]byte(body), out); !result.OK {
		return core.E("HuggingFaceModelSource", "parse response", hfFitResultError(result))
	}
	return nil
}

// HFModelFitConfig controls model discovery and local fit planning.
type HFModelFitConfig struct {
	Query       string
	ModelIDs    []string
	LocalPaths  []string
	MaxResults  int
	Device      DeviceInfo
	Source      HFModelSource
	LoRARank    int
	KVBytes     int
	ContextHint int
}

// HFModelMetadata is the subset of Hugging Face/local metadata needed for fit planning.
type HFModelMetadata struct {
	ID          string        `json:"id,omitempty"`
	ModelID     string        `json:"modelId,omitempty"`
	Tags        []string      `json:"tags,omitempty"`
	PipelineTag string        `json:"pipeline_tag,omitempty"`
	Config      HFModelConfig `json:"config,omitempty"`
	Files       []HFModelFile `json:"siblings,omitempty"`
}

// HFModelFile describes one model repository file.
type HFModelFile struct {
	Name      string `json:"name,omitempty"`
	RFilename string `json:"rfilename,omitempty"`
	Size      uint64 `json:"size,omitempty"`
	SizeBytes uint64 `json:"sizeBytes,omitempty"`
}

// HFModelConfig mirrors common transformer config fields exposed by HF.
type HFModelConfig struct {
	ModelType             string                `json:"model_type,omitempty"`
	Architectures         []string              `json:"architectures,omitempty"`
	VocabSize             int                   `json:"vocab_size,omitempty"`
	HiddenSize            int                   `json:"hidden_size,omitempty"`
	IntermediateSize      int                   `json:"intermediate_size,omitempty"`
	NumHiddenLayers       int                   `json:"num_hidden_layers,omitempty"`
	NumAttentionHeads     int                   `json:"num_attention_heads,omitempty"`
	NumKeyValueHeads      int                   `json:"num_key_value_heads,omitempty"`
	HeadDim               int                   `json:"head_dim,omitempty"`
	MaxPositionEmbeddings int                   `json:"max_position_embeddings,omitempty"`
	ContextLength         int                   `json:"context_length,omitempty"`
	Quantization          *HFQuantizationConfig `json:"quantization,omitempty"`
	QuantizationConfig    *HFQuantizationConfig `json:"quantization_config,omitempty"`
	TextConfig            *HFModelConfig        `json:"text_config,omitempty"`
}

// HFQuantizationConfig captures quantization metadata when present.
type HFQuantizationConfig struct {
	Bits      int    `json:"bits,omitempty"`
	GroupSize int    `json:"group_size,omitempty"`
	Type      string `json:"type,omitempty"`
}

// HFModelFitReport is the top-level library output for HF/local model fit planning.
type HFModelFitReport struct {
	Query       string           `json:"query,omitempty"`
	Device      DeviceInfo       `json:"device"`
	DeviceClass MemoryClass      `json:"device_class"`
	MemoryPlan  MemoryPlan       `json:"memory_plan"`
	Models      []HFModelFitPlan `json:"models"`
}

// HFModelFitPlan is one model's local Apple fit estimate.
type HFModelFitPlan struct {
	ModelID               string        `json:"model_id,omitempty"`
	LocalPath             string        `json:"local_path,omitempty"`
	Source                string        `json:"source"`
	Architecture          string        `json:"architecture,omitempty"`
	SupportedArchitecture bool          `json:"supported_architecture"`
	NativeLoadable        bool          `json:"native_loadable"`
	WeightFormat          string        `json:"weight_format,omitempty"`
	QuantBits             int           `json:"quant_bits,omitempty"`
	QuantGroup            int           `json:"quant_group,omitempty"`
	WeightBytes           uint64        `json:"weight_bytes,omitempty"`
	ExpectedKVBytes       uint64        `json:"expected_kv_bytes,omitempty"`
	ExpectedRuntimeBytes  uint64        `json:"expected_runtime_bytes,omitempty"`
	ExpectedTotalBytes    uint64        `json:"expected_total_bytes,omitempty"`
	ContextLimit          int           `json:"context_limit,omitempty"`
	ContextRecommendation int           `json:"context_recommendation,omitempty"`
	MemoryPlan            MemoryPlan    `json:"memory_plan"`
	InferenceFits         bool          `json:"inference_fits"`
	Training              HFTrainingFit `json:"training"`
	Notes                 []string      `json:"notes,omitempty"`
}

// HFTrainingFit describes rough training feasibility for local Apple hardware.
type HFTrainingFit struct {
	LoRAFeasible            bool     `json:"lora_feasible"`
	FullFineTuneFeasible    bool     `json:"full_fine_tune_feasible"`
	RecommendedLoRARank     int      `json:"recommended_lora_rank,omitempty"`
	EstimatedLoRABytes      uint64   `json:"estimated_lora_bytes,omitempty"`
	EstimatedOptimizerBytes uint64   `json:"estimated_optimizer_bytes,omitempty"`
	Notes                   []string `json:"notes,omitempty"`
}

// PlanHFModelFits discovers HF/local metadata and estimates local Apple fit.
func PlanHFModelFits(ctx context.Context, cfg HFModelFitConfig) (*HFModelFitReport, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if cfg.Device.MemorySize == 0 && cfg.Device.MaxRecommendedWorkingSetSize == 0 {
		cfg.Device = GetDeviceInfo()
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

	entries, err := collectHFModelFitEntries(ctx, cfg)
	if err != nil {
		return nil, err
	}
	if len(entries) == 0 {
		return nil, core.NewError("mlx: no model metadata available for fit planning")
	}

	basePlan := PlanMemory(MemoryPlanInput{Device: cfg.Device})
	report := &HFModelFitReport{
		Query:       cfg.Query,
		Device:      cfg.Device,
		DeviceClass: basePlan.MachineClass,
		MemoryPlan:  basePlan,
		Models:      make([]HFModelFitPlan, 0, len(entries)),
	}
	for _, entry := range entries {
		report.Models = append(report.Models, planHFModelFit(entry, cfg))
	}
	slices.SortFunc(report.Models, func(a, b HFModelFitPlan) int {
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

type hfFitEntry struct {
	meta      HFModelMetadata
	source    string
	localPath string
}

func collectHFModelFitEntries(ctx context.Context, cfg HFModelFitConfig) ([]hfFitEntry, error) {
	var entries []hfFitEntry
	for _, path := range cfg.LocalPaths {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		meta, root, err := inspectLocalHFModelMetadata(path)
		if err != nil {
			return nil, err
		}
		entries = append(entries, hfFitEntry{meta: meta, source: HFModelSourceLocal, localPath: root})
	}
	if cfg.Query != "" {
		if cfg.Source == nil {
			return nil, core.NewError("mlx: HF metadata source is required for query search")
		}
		found, err := cfg.Source.SearchModels(ctx, cfg.Query, cfg.MaxResults)
		if err != nil {
			return nil, err
		}
		for _, meta := range found {
			entries = append(entries, hfFitEntry{meta: meta, source: HFModelSourceRemote})
		}
	}
	for _, id := range cfg.ModelIDs {
		if cfg.Source == nil {
			return nil, core.NewError("mlx: HF metadata source is required for model id lookup")
		}
		meta, err := cfg.Source.ModelMetadata(ctx, id)
		if err != nil {
			return nil, err
		}
		if meta.ID == "" && meta.ModelID == "" {
			meta.ID = id
		}
		entries = append(entries, hfFitEntry{meta: meta, source: HFModelSourceRemote})
	}
	return entries, nil
}

func inspectLocalHFModelMetadata(path string) (HFModelMetadata, string, error) {
	root := resolveLocalHFMetadataRoot(path)
	read := core.ReadFile(core.PathJoin(root, "config.json"))
	if !read.OK {
		return HFModelMetadata{}, root, core.E("PlanHFModelFits", "read local config.json", hfFitResultError(read))
	}
	var config HFModelConfig
	if result := core.JSONUnmarshal(read.Value.([]byte), &config); !result.OK {
		return HFModelMetadata{}, root, core.E("PlanHFModelFits", "parse local config.json", hfFitResultError(result))
	}
	files := localHFModelFiles(root)
	return HFModelMetadata{
		ID:     localHFModelID(path, root),
		Config: config,
		Files:  files,
	}, root, nil
}

func resolveLocalHFMetadataRoot(path string) string {
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

func localHFModelID(inputPath, root string) string {
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

func localHFModelFiles(root string) []HFModelFile {
	var files []HFModelFile
	for _, pattern := range []string{"*.safetensors", "*.gguf", "*.bin", "tokenizer.json", "tokenizer_config.json"} {
		for _, path := range core.PathGlob(core.PathJoin(root, pattern)) {
			info := core.Stat(path)
			var size uint64
			if info.OK {
				size = uint64(info.Value.(core.FsFileInfo).Size())
			}
			files = append(files, HFModelFile{Name: core.PathBase(path), Size: size})
		}
	}
	slices.SortFunc(files, func(a, b HFModelFile) int {
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

func planHFModelFit(entry hfFitEntry, cfg HFModelFitConfig) HFModelFitPlan {
	meta := entry.meta
	config := meta.Config.normalized()
	modelID := firstNonEmpty(meta.ID, meta.ModelID)
	arch := config.architecture()
	contextLimit := config.contextLength()
	quantBits, quantGroup := config.quantization()
	format, weightBytes := hfWeightFormatAndBytes(meta.Files)
	if quantBits == 0 {
		quantBits = inferHFQuantBits(meta.Files)
	}

	pack := ModelPack{
		Architecture:          arch,
		SupportedArchitecture: modelPackSupportedArchitecture(arch),
		QuantBits:             quantBits,
		QuantGroup:            quantGroup,
		ContextLength:         contextLimit,
	}
	memoryPlan := PlanMemory(MemoryPlanInput{Device: cfg.Device, Pack: &pack})
	if cfg.ContextHint > 0 && cfg.ContextHint < memoryPlan.ContextLength {
		memoryPlan.ContextLength = cfg.ContextHint
	}
	kvBytes := estimateHFModelKVBytes(config, memoryPlan.ContextLength, memoryPlan.BatchSize, cfg.KVBytes)
	runtimeBytes := estimateRuntimeOverheadBytes(weightBytes)
	totalBytes := weightBytes + kvBytes + runtimeBytes
	limit := memoryPlan.MemoryLimitBytes
	if limit == 0 {
		limit = cfg.Device.MaxRecommendedWorkingSetSize
	}
	if limit == 0 {
		limit = cfg.Device.MemorySize
	}

	plan := HFModelFitPlan{
		ModelID:               modelID,
		LocalPath:             entry.localPath,
		Source:                entry.source,
		Architecture:          arch,
		SupportedArchitecture: modelPackSupportedArchitecture(arch),
		WeightFormat:          format,
		QuantBits:             quantBits,
		QuantGroup:            quantGroup,
		WeightBytes:           weightBytes,
		ExpectedKVBytes:       kvBytes,
		ExpectedRuntimeBytes:  runtimeBytes,
		ExpectedTotalBytes:    totalBytes,
		ContextLimit:          contextLimit,
		ContextRecommendation: memoryPlan.ContextLength,
		MemoryPlan:            memoryPlan,
	}
	plan.NativeLoadable = plan.SupportedArchitecture && format != ""
	plan.InferenceFits = plan.NativeLoadable && weightBytes > 0 && (limit == 0 || totalBytes <= limit)
	plan.Training = estimateHFTrainingFit(config, plan, limit, cfg.LoRARank)
	plan.Notes = hfFitNotes(plan, limit)
	return plan
}

func hfWeightFormatAndBytes(files []HFModelFile) (string, uint64) {
	var format string
	var total uint64
	for _, file := range files {
		name := core.Lower(file.filename())
		switch {
		case core.HasSuffix(name, ".safetensors"):
			if format == "" {
				format = string(ModelPackFormatSafetensors)
			} else if format != string(ModelPackFormatSafetensors) {
				format = string(ModelPackFormatMixed)
			}
			total += file.byteSize()
		case core.HasSuffix(name, ".gguf"):
			if format == "" {
				format = string(ModelPackFormatGGUF)
			} else if format != string(ModelPackFormatGGUF) {
				format = string(ModelPackFormatMixed)
			}
			total += file.byteSize()
		case core.HasSuffix(name, ".bin"):
			if format == "" {
				format = "bin"
			}
			total += file.byteSize()
		}
	}
	return format, total
}

func inferHFQuantBits(files []HFModelFile) int {
	for _, file := range files {
		name := core.Lower(file.filename())
		switch {
		case core.Contains(name, "q2"):
			return 2
		case core.Contains(name, "q3"):
			return 3
		case core.Contains(name, "q4") || core.Contains(name, "4bit") || core.Contains(name, "4-bit"):
			return 4
		case core.Contains(name, "q5"):
			return 5
		case core.Contains(name, "q6"):
			return 6
		case core.Contains(name, "q8") || core.Contains(name, "8bit") || core.Contains(name, "8-bit"):
			return 8
		case core.Contains(name, "bf16") || core.Contains(name, "fp16") || core.Contains(name, "f16"):
			return 16
		}
	}
	return 0
}

func estimateHFModelKVBytes(config HFModelConfig, contextLength, batchSize, bytesPerElement int) uint64 {
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
	if overhead < MemoryGiB {
		return MemoryGiB
	}
	return overhead
}

func estimateHFTrainingFit(config HFModelConfig, plan HFModelFitPlan, memoryLimit uint64, rank int) HFTrainingFit {
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
	fit := HFTrainingFit{
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

func hfFitNotes(plan HFModelFitPlan, memoryLimit uint64) []string {
	var notes []string
	if !plan.SupportedArchitecture {
		notes = append(notes, "architecture is not currently supported by native go-mlx loaders")
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

func (config HFModelConfig) normalized() HFModelConfig {
	if config.TextConfig == nil {
		return config
	}
	text := *config.TextConfig
	if text.ModelType == "" {
		text.ModelType = config.ModelType
	}
	if len(text.Architectures) == 0 {
		text.Architectures = append([]string(nil), config.Architectures...)
	}
	return text
}

func (config HFModelConfig) architecture() string {
	config = config.normalized()
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

func (config HFModelConfig) contextLength() int {
	config = config.normalized()
	return firstPositive(config.ContextLength, config.MaxPositionEmbeddings)
}

func (config HFModelConfig) quantization() (bits, group int) {
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

func (file HFModelFile) filename() string {
	return firstNonEmpty(file.Name, file.RFilename)
}

func (file HFModelFile) byteSize() uint64 {
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

func hfFitResultError(result core.Result) error {
	if result.OK {
		return nil
	}
	if err, ok := result.Value.(error); ok {
		return err
	}
	return core.NewError("core result failed")
}
