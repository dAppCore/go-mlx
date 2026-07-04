// SPDX-Licence-Identifier: EUPL-1.2

package hf

import (
	"context"
	"strconv"

	core "dappco.re/go"
	sharedhf "dappco.re/go/inference/hf"
	"dappco.re/go/inference/memory"
	mp "dappco.re/go/inference/modelpack"
	"dappco.re/go/inference/profile"
)

const (
	// SourceRemote and SourceLocal are aliased onto the shared package's
	// constants so go-mlx and dappco.re/go/inference/hf never drift on the
	// tag values FitPlan.Source reports.
	SourceRemote = sharedhf.SourceRemote
	SourceLocal  = sharedhf.SourceLocal

	defaultBaseURL = "https://huggingface.co"
)

// ModelSource provides optional Hugging Face metadata lookup/search. Aliased
// onto the shared interface — RemoteSource (below) and any fixture that
// implements it satisfy both go-mlx's and dappco.re/go/inference/hf's
// ModelSource identically.
type ModelSource = sharedhf.ModelSource

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
	authValue string // pre-built "Bearer <token>"; empty when no token
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
	// Pre-build the Authorization header value once at constructor time.
	// Every getJSON call previously paid for core.Concat("Bearer ", token)
	// — an allocation per request. The token is immutable after
	// construction, so the formatted value is too.
	var authValue string
	if cfg.Token != "" {
		authValue = core.Concat("Bearer ", cfg.Token)
	}
	return &RemoteSource{
		baseURL:   baseURL,
		token:     cfg.Token,
		userAgent: firstNonEmpty(cfg.UserAgent, "go-mlx"),
		authValue: authValue,
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
	// Build the query string directly via Concat — the previous form
	// allocated a URLValues map plus three []string{...} entries, then
	// url.Values.Encode() did a sorted string build. The HF /api/models
	// endpoint doesn't care about parameter order, so a direct Concat is
	// equivalent on the wire and saves four small allocations.
	var models []ModelMetadata
	target := core.Concat(
		s.baseURL,
		"/api/models?full=true&limit=",
		strconv.Itoa(limit),
		"&search=",
		core.URLEncode(query),
	)
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
	if s.authValue != "" {
		// authValue is pre-built at constructor time; skips the per-call
		// core.Concat("Bearer ", s.token) allocation.
		req.Header.Set("Authorization", s.authValue)
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
		// Avoid core.Sprintf — its fmt machinery is hot-path heavy for
		// what is just an int + string assembly. strconv.Itoa+Concat is
		// roughly 4x cheaper for this error message shape.
		return core.NewError(core.Concat(
			"mlx: HF metadata request failed: ",
			strconv.Itoa(resp.StatusCode),
			" ",
			core.Trim(body),
		))
	}
	// JSONUnmarshalString takes a string and zero-copies it to []byte via
	// AsBytes — json.Unmarshal treats the buffer as read-only and copies
	// strings into the target via SetString. Saves the []byte(body) copy
	// that allocated a duplicate of the entire response body on every call.
	if result := core.JSONUnmarshalString(body, out); !result.OK {
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

// ModelMetadata is the subset of Hugging Face/local metadata needed for fit
// planning — aliased onto the engine-agnostic type in
// dappco.re/go/inference/hf so every LEM Engine (mlx, rocm, cpu) shares one
// wire shape for Hub + local-cache metadata.
type ModelMetadata = sharedhf.ModelMetadata

// ModelFile describes one model repository file.
type ModelFile = sharedhf.ModelFile

// ModelConfig mirrors common transformer config fields exposed by HF. The
// shared type is deliberately free of architecture-support/quantisation
// interpretation logic — that stays local to each engine (below), built on
// top of this data.
type ModelConfig = sharedhf.ModelConfig

// QuantizationConfig captures quantization metadata when present.
type QuantizationConfig = sharedhf.QuantizationConfig

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

// normalizeModelConfig lifts a nested text_config (Gemma-4 unified/assistant
// wrappers, or any model with a text_config block) so downstream accessors
// read the real architecture/context rather than the outer wrapper.
//
// A plain function rather than a ModelConfig method: ModelConfig is aliased
// onto dappco.re/go/inference/hf's type (above), and Go forbids attaching new
// methods to a type defined in another package.
func normalizeModelConfig(config ModelConfig) ModelConfig {
	if config.TextConfig == nil {
		return config
	}
	text := *config.TextConfig
	if isGemma4AssistantConfig(config) {
		text.ModelType = "gemma4_assistant"
	} else if isGemma4UnifiedConfig(config) {
		text.ModelType = "gemma4_unified"
	} else if text.ModelType == "" {
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

// modelConfigArchitecture is the normalize-then-read variant of
// configArchitecture, for callers holding a possibly-unnormalised config.
func modelConfigArchitecture(config ModelConfig) string {
	config = normalizeModelConfig(config)
	return configArchitecture(&config)
}

// modelConfigContextLength is the normalize-then-read context-length accessor.
func modelConfigContextLength(config ModelConfig) int {
	config = normalizeModelConfig(config)
	return firstPositive(config.ContextLength, config.MaxPositionEmbeddings)
}

// modelConfigQuantization is the normalize-then-read quantisation accessor.
func modelConfigQuantization(config ModelConfig) (bits, group int) {
	config = normalizeModelConfig(config)
	quant := config.QuantizationConfig
	if quant == nil {
		quant = config.Quantization
	}
	if quant == nil {
		return 0, 0
	}
	return quant.Bits, quant.GroupSize
}

// modelConfigQuantizationType is the normalize-then-read quant-type accessor.
func modelConfigQuantizationType(config ModelConfig) string {
	config = normalizeModelConfig(config)
	quant := config.QuantizationConfig
	if quant == nil {
		quant = config.Quantization
	}
	if quant == nil {
		return ""
	}
	return quant.Type
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
	// hasNonWhitespace avoids the core.Trim allocation that the previous
	// implementation paid every time the input had any leading/trailing
	// whitespace. We only care whether the trimmed form is non-empty —
	// not what it contains — so a single byte scan is sufficient.
	for _, value := range values {
		if hasNonWhitespace(value) {
			return value
		}
	}
	return ""
}

func hasNonWhitespace(s string) bool {
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c != ' ' && c != '\t' && c != '\n' && c != '\r' && c != '\v' && c != '\f' {
			return true
		}
	}
	return false
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
		if modelType := profile.ArchitectureFromTransformersName(architecture); modelType == "bert_rerank" {
			return modelType
		}
	}
	if probe.ModelType != "" {
		return profile.NormalizeArchitecture(probe.ModelType)
	}
	if probe.TextConfig.ModelType != "" {
		return profile.NormalizeArchitecture(probe.TextConfig.ModelType)
	}
	for _, architecture := range probe.Architectures {
		if modelType := profile.ArchitectureFromTransformersName(architecture); modelType != "" {
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
	_, ok := profile.LookupArchitectureProfileRef(architecture)
	return ok
}

func archNativeRuntime(architecture string) bool {
	p, ok := profile.LookupArchitectureProfileRef(architecture)
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
	if p, ok := profile.LookupArchitectureProfileRef(architecture); ok && (p.Embeddings || p.Rerank) {
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
	if resolved, ok := profile.LookupArchitectureProfileRef(pack.Architecture); ok {
		pack.ArchitectureProfile = resolved
	}
}
