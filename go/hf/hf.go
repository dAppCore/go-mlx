// SPDX-Licence-Identifier: EUPL-1.2

package hf

import (
	"context"
	"slices"
	"strconv"

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
	if quantBits == 0 {
		quantBits = inferQuantBits(meta.Files)
	}

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
		Embeddings:            pack.Embedding != nil,
		Rerank:                pack.Rerank != nil,
	}
	plan.NativeLoadable = supportedArch && nativeRuntime && format != ""
	plan.MemoryFits = weightBytes > 0 && (limit == 0 || totalBytes <= limit)
	plan.InferenceFits = plan.NativeLoadable && plan.MemoryFits
	plan.Training = estimateTrainingFit(config, plan, limit, cfg.LoRARank)
	plan.Notes = fitNotes(plan, limit, nativeRuntime)
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
	if archProfileOK && archProfile != nil && (archProfile.Embeddings || archProfile.Rerank) {
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

func fitNotes(plan FitPlan, memoryLimit uint64, nativeRuntime bool) []string {
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
	quantBelowPref := plan.QuantBits > 0 && plan.MemoryPlan.PreferredQuantization > 0 && plan.QuantBits < plan.MemoryPlan.PreferredQuantization
	count := 0
	if unsupported {
		count++
	}
	if notNative {
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
	if quantBelowPref {
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
	if unknownBytes {
		notes = append(notes, "weight byte size is unknown")
	}
	if overBudget {
		notes = append(notes, "estimated model+KV memory exceeds local working-set budget")
	}
	if contextCapped {
		notes = append(notes, "context recommendation is capped by local machine class")
	}
	if quantBelowPref {
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
	return configArchitecture(&config)
}

// configArchitecture is the already-normalised, pointer-receiver variant
// for callers that have already done the normalize. Avoids the second
// normalize value-copy of ~96-byte ModelConfig.
func configArchitecture(config *ModelConfig) string {
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
	// Fast-path classify before any heap work. inferJANGNeedlePresent
	// scans the id / tags / filenames in-place for "jang" and "jangtq"
	// tokens. The miss path (the dominant case across HF metadata)
	// returns jangNone in zero allocs. The JANGTQ branch needs only the
	// QuantizationConfig group size — no haystack scan — so we skip the
	// lowercase-buffer build entirely for those packs.
	id := firstNonEmpty(meta.ID, meta.ModelID)
	presence := inferJANGNeedlePresent(id, meta.Tags, meta.Files)
	switch presence {
	case jangNone:
		return nil
	case jangTQ:
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
	}
	// jangBasic — need to scan the haystack for a specific profile name
	// (jang_1l, jang_2s, etc.). Build the lowercase "id tag1 tag2
	// file1 file2" haystack in one pass; the buffer is the only
	// allocation specific to this branch.
	size := len(id)
	for _, tag := range meta.Tags {
		size += 1 + len(tag)
	}
	for _, file := range meta.Files {
		// Upper bound — max(Name, RFilename). Avoids the firstNonEmpty
		// scan here while still preventing growslice in the append loop.
		nameLen := len(file.Name)
		if len(file.RFilename) > nameLen {
			nameLen = len(file.RFilename)
		}
		size += 1 + nameLen
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
	profile := inferJANGProfileName(needle)
	info := &jang.Info{
		Profile:     profile,
		GroupSize:   jangGroupSize(meta),
		BitsDefault: firstPositive(jang.ProfileBits(profile), 0),
	}
	info.Packed = jang.BuildPackedProfile(info)
	return info
}

// JANG token-presence states. Returned by inferJANGNeedlePresent so
// InferJANG can skip the lowercase-haystack build for the JANGTQ branch
// (which doesn't need a haystack scan past detection).
type jangPresence uint8

const (
	jangNone   jangPresence = 0
	jangBasic  jangPresence = 1 // "jang" present, "jangtq" not
	jangTQ     jangPresence = 2 // "jangtq" present (implies "jang")
)

// inferJANGNeedlePresent classifies the strongest JANG token present in
// the id / tags / filenames in a single pass per component. Pure scan,
// no allocations — used to gate the lowercase-buffer build inside
// InferJANG. jangNone (the dominant case across HF metadata) returns in
// zero allocs after a tight byte scan. jangTQ short-circuits the
// haystack build downstream because the JANGTQ branch only needs the
// QuantizationConfig group size, not a needle scan.
func inferJANGNeedlePresent(id string, tags []string, files []ModelFile) jangPresence {
	state := scanJANGFold(id)
	if state == jangTQ {
		return jangTQ
	}
	for _, tag := range tags {
		s := scanJANGFold(tag)
		if s == jangTQ {
			return jangTQ
		}
		if s > state {
			state = s
		}
	}
	for _, file := range files {
		s := scanJANGFold(file.Name)
		if s == jangTQ {
			return jangTQ
		}
		if s > state {
			state = s
		}
		s = scanJANGFold(file.RFilename)
		if s == jangTQ {
			return jangTQ
		}
		if s > state {
			state = s
		}
	}
	return state
}

// scanJANGFold reports the strongest JANG token present in s — jangTQ
// when "jangtq" is found, jangBasic when only "jang" is found, jangNone
// otherwise. Single ASCII byte scan with case folding inline. Per
// starting position 'j', try the longer 6-byte "jangtq" match first;
// fall back to 4-byte "jang". Returns early on jangTQ.
func scanJANGFold(s string) jangPresence {
	if len(s) < 4 {
		return jangNone
	}
	state := jangNone
	last4 := len(s) - 4
	for i := 0; i <= last4; i++ {
		c0 := s[i]
		if c0 >= 'A' && c0 <= 'Z' {
			c0 += 'a' - 'A'
		}
		if c0 != 'j' {
			continue
		}
		c1 := s[i+1]
		if c1 >= 'A' && c1 <= 'Z' {
			c1 += 'a' - 'A'
		}
		if c1 != 'a' {
			continue
		}
		c2 := s[i+2]
		if c2 >= 'A' && c2 <= 'Z' {
			c2 += 'a' - 'A'
		}
		if c2 != 'n' {
			continue
		}
		c3 := s[i+3]
		if c3 >= 'A' && c3 <= 'Z' {
			c3 += 'a' - 'A'
		}
		if c3 != 'g' {
			continue
		}
		// "jang" matched at i. Probe for the "tq" extension if there's
		// room — jangtq is the strongest match.
		if i+6 <= len(s) {
			c4 := s[i+4]
			if c4 >= 'A' && c4 <= 'Z' {
				c4 += 'a' - 'A'
			}
			if c4 == 't' {
				c5 := s[i+5]
				if c5 >= 'A' && c5 <= 'Z' {
					c5 += 'a' - 'A'
				}
				if c5 == 'q' {
					return jangTQ
				}
			}
		}
		state = jangBasic
	}
	return state
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

// jangProfileLookup parallels needle/value forms with their UPPER variants.
// Hoisted out of inferJANGProfileName so the literal slice and the
// per-match core.Upper allocation are paid once at init, not per call.
var jangProfileLookup = [...]struct{ Lower, Upper string }{
	{"jang_1l", "JANG_1L"},
	{"jang_2s", "JANG_2S"},
	{"jang_2l", "JANG_2L"},
	{"jang_3l", "JANG_3L"},
	{"jang_4k", "JANG_4K"},
	{"jang_4m", "JANG_4M"},
}

func inferJANGProfileName(value string) string {
	for i := range jangProfileLookup {
		if core.Contains(value, jangProfileLookup[i].Lower) {
			return jangProfileLookup[i].Upper
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
	// Folded-compare against the known canonical names BEFORE allocating
	// the lowered buffer. The known arms all return string literals, so
	// when the input maps to one of them we never need a normalised copy.
	// Only fall through to normaliseArchString for the passthrough case
	// (input doesn't match any arm), where we have to return the lowered
	// form to preserve current semantics.
	if matched := matchKnownArchitectureFolded(value); matched != "" {
		return matched
	}
	return matchKnownArchitecture(normaliseArchString(value))
}

// matchKnownArchitectureFolded reports the canonical name for value when
// its case+dash-folded form matches one of the known architecture keys.
// Returns "" when no arm matches — caller must then allocate the lowered
// form via normaliseArchString. Walks value once per candidate target
// with ASCII case folding and '-'→'_' rewriting inline; no allocations.
func matchKnownArchitectureFolded(value string) string {
	// Trim leading/trailing ASCII whitespace.
	start, end := 0, len(value)
	for start < end {
		c := value[start]
		if c != ' ' && c != '\t' && c != '\n' && c != '\r' {
			break
		}
		start++
	}
	for end > start {
		c := value[end-1]
		if c != ' ' && c != '\t' && c != '\n' && c != '\r' {
			break
		}
		end--
	}
	if start == end {
		return ""
	}
	// Each target { folded-key, canonical-result }. Mirror the
	// matchKnownArchitecture switch arms one-for-one.
	switch {
	case eqFolded(value, start, end, "qwen3_5"):
		return "qwen3_next"
	case eqFolded(value, start, end, "minimaxm2"),
		eqFolded(value, start, end, "minimax_m2"):
		return "minimax_m2"
	case eqFolded(value, start, end, "mixtral"):
		return "mixtral"
	case eqFolded(value, start, end, "mistral"):
		return "mistral"
	case eqFolded(value, start, end, "phi"),
		eqFolded(value, start, end, "phi3"),
		eqFolded(value, start, end, "phi4"):
		return "phi"
	case eqFolded(value, start, end, "deepseek"),
		eqFolded(value, start, end, "deepseek_v3"),
		eqFolded(value, start, end, "deepseek_r1"):
		return "deepseek"
	case eqFolded(value, start, end, "gptoss"),
		eqFolded(value, start, end, "gpt_oss"),
		eqFolded(value, start, end, "gpt_oss_model"):
		return "gpt_oss"
	case eqFolded(value, start, end, "bert"):
		return "bert"
	case eqFolded(value, start, end, "bert_rerank"),
		eqFolded(value, start, end, "bert_cross_encoder"):
		return "bert_rerank"
	}
	return ""
}

// eqFolded reports whether value[start:end] equals target after ASCII
// case folding and '-'→'_' rewriting. target must already be lowercased
// and use '_' separators. Pure byte scan, no allocations.
func eqFolded(value string, start, end int, target string) bool {
	if end-start != len(target) {
		return false
	}
	for i := 0; i < len(target); i++ {
		c := value[start+i]
		if c == '-' {
			c = '_'
		} else if c >= 'A' && c <= 'Z' {
			c += 'a' - 'A'
		}
		if c != target[i] {
			return false
		}
	}
	return true
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
	//
	// Dispatch via the first character so we run at most 3 Contains per
	// call (the family check + any disambiguation), instead of walking up
	// to 11 sequential Contains for less-common families like Bert. Most
	// transformer class names share a single first character per family
	// (Gemma*, Qwen*, Phi*, Bert*, etc.), so a first-byte switch is a
	// reliable family selector.
	if len(architecture) == 0 {
		return ""
	}
	switch architecture[0] {
	case 'G':
		switch {
		case core.Contains(architecture, "Gemma4"):
			return "gemma4_text"
		case core.Contains(architecture, "Gemma3"):
			return "gemma3"
		case core.Contains(architecture, "Gemma2"):
			return "gemma2"
		case core.Contains(architecture, "GptOss") || core.Contains(architecture, "GPTOSS"):
			return "gpt_oss"
		}
	case 'Q':
		switch {
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
		}
	case 'L':
		if core.Contains(architecture, "Llama") {
			return "llama"
		}
	case 'M':
		switch {
		case core.Contains(architecture, "MiniMaxM2"):
			return "minimax_m2"
		case core.Contains(architecture, "Mixtral"):
			return "mixtral"
		case core.Contains(architecture, "Mistral"):
			return "mistral"
		}
	case 'P':
		if core.Contains(architecture, "Phi") {
			return "phi"
		}
	case 'D':
		switch {
		case core.Contains(architecture, "Deepseek") || core.Contains(architecture, "DeepSeek"):
			return "deepseek"
		case core.Contains(architecture, "Deberta"):
			// Deberta family — disambiguate rerank via compact.
			compact := lowerNoSep(architecture)
			if core.Contains(compact, "debertav2forsequenceclassification") {
				return "bert_rerank"
			}
		}
	case 'B':
		if core.Contains(architecture, "Bert") {
			// Bert family — disambiguate rerank via compact.
			compact := lowerNoSep(architecture)
			if core.Contains(compact, "bertforsequenceclassification") {
				return "bert_rerank"
			}
			return "bert"
		}
	case 'R':
		if core.Contains(architecture, "Roberta") {
			compact := lowerNoSep(architecture)
			if core.Contains(compact, "robertaforsequenceclassification") {
				return "bert_rerank"
			}
		}
	case 'X':
		// xlm-roberta is the only family starting with X we classify.
		compact := lowerNoSep(architecture)
		if core.Contains(compact, "xlmrobertaforsequenceclassification") {
			return "bert_rerank"
		}
	}
	// Unknown first-character shape — the only patterns the compact form
	// matches all start with 'b' (bert/roberta/xlmroberta/debertav2) or
	// 'q' (qwen3moe/qwen3next). If the input has neither (case-
	// insensitively), the compact form can't match anything — return ""
	// without paying for lowerNoSep's allocation.
	if !hasASCIIByteFold(architecture, 'b') && !hasASCIIByteFold(architecture, 'q') {
		return ""
	}
	// Fall back to compact lower form so a few stragglers like
	// "qwen3_moe" or "bert_for_sequence_classification" still
	// classify when callers feed snake_case identifiers.
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

// hasASCIIByteFold reports whether s contains b or B (where b is the
// lowercase form). Pure byte scan, no allocations.
func hasASCIIByteFold(s string, lower byte) bool {
	upper := lower &^ 0x20 // upper-case form
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c == lower || c == upper {
			return true
		}
	}
	return false
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
