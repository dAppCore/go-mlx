// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"sort"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/quant/codebook"
	"dappco.re/go/inference/quant/jang"
	"dappco.re/go/mlx/profile"
)

// ModelPackFormat names the model weight container found in a pack.
type ModelPackFormat string

const (
	ModelPackFormatMissing     ModelPackFormat = "missing"
	ModelPackFormatSafetensors ModelPackFormat = "safetensors"
	ModelPackFormatGGUF        ModelPackFormat = "gguf"
	ModelPackFormatMixed       ModelPackFormat = "mixed"
)

// ModelPackChatTemplateSource records where chat formatting came from.
type ModelPackChatTemplateSource string

const (
	ModelPackChatTemplateNone   ModelPackChatTemplateSource = ""
	ModelPackChatTemplateFile   ModelPackChatTemplateSource = "tokenizer_config.json"
	ModelPackChatTemplateJinja  ModelPackChatTemplateSource = "chat_template.jinja"
	ModelPackChatTemplateNative ModelPackChatTemplateSource = "native"
)

// ModelPackIssueSeverity classifies a validation issue.
type ModelPackIssueSeverity string

const (
	ModelPackIssueError   ModelPackIssueSeverity = "error"
	ModelPackIssueWarning ModelPackIssueSeverity = "warning"
)

// ModelPackIssueCode is a stable machine-readable pack validation code.
type ModelPackIssueCode string

const (
	ModelPackIssueMissingConfig           ModelPackIssueCode = "missing_config"
	ModelPackIssueInvalidConfig           ModelPackIssueCode = "invalid_config"
	ModelPackIssueMissingWeights          ModelPackIssueCode = "missing_weights"
	ModelPackIssueMultipleGGUF            ModelPackIssueCode = "multiple_gguf"
	ModelPackIssueMixedWeightFormats      ModelPackIssueCode = "mixed_weight_formats"
	ModelPackIssueInvalidGGUF             ModelPackIssueCode = "invalid_gguf"
	ModelPackIssueMissingTokenizer        ModelPackIssueCode = "missing_tokenizer"
	ModelPackIssueInvalidTokenizer        ModelPackIssueCode = "invalid_tokenizer"
	ModelPackIssueUnsupportedArchitecture ModelPackIssueCode = "unsupported_architecture"
	ModelPackIssueUnsupportedRuntime      ModelPackIssueCode = "unsupported_runtime"
	ModelPackIssueMissingArchitecture     ModelPackIssueCode = "missing_architecture"
	ModelPackIssueMissingChatTemplate     ModelPackIssueCode = "missing_chat_template"
	ModelPackIssueQuantizationMismatch    ModelPackIssueCode = "quantization_mismatch"
	ModelPackIssueContextTooLarge         ModelPackIssueCode = "context_too_large"
	ModelPackIssueMiniMaxM2LayerSkeleton  ModelPackIssueCode = "minimax_m2_layer_skeleton"
	ModelPackIssueUnsupportedCodebook     ModelPackIssueCode = "unsupported_codebook"
)

// ModelPackIssue describes one pack validation finding.
type ModelPackIssue struct {
	Severity ModelPackIssueSeverity `json:"severity"`
	Code     ModelPackIssueCode     `json:"code"`
	Message  string                 `json:"message"`
	Path     string                 `json:"path,omitempty"`
}

// ModelEmbeddingProfile records metadata for encoder-style embedding packs.
type ModelEmbeddingProfile struct {
	Dimension         int    `json:"dimension,omitempty"`
	Pooling           string `json:"pooling,omitempty"`
	Normalize         bool   `json:"normalize,omitempty"`
	MaxSequenceLength int    `json:"max_sequence_length,omitempty"`
	Source            string `json:"source,omitempty"`
}

// ModelRerankProfile records metadata for cross-encoder rerank packs.
type ModelRerankProfile struct {
	Method            string `json:"method,omitempty"`
	MaxSequenceLength int    `json:"max_sequence_length,omitempty"`
	Source            string `json:"source,omitempty"`
}

// ModelPack summarises whether a local model directory is natively loadable.
type ModelPack struct {
	Path                     string                         `json:"path"`
	Root                     string                         `json:"root"`
	Format                   ModelPackFormat                `json:"format"`
	ConfigPath               string                         `json:"config_path,omitempty"`
	WeightFiles              []string                       `json:"weight_files,omitempty"`
	TokenizerPath            string                         `json:"tokenizer_path,omitempty"`
	TokenizerConfigPath      string                         `json:"tokenizer_config_path,omitempty"`
	Architecture             string                         `json:"architecture,omitempty"`
	SupportedArchitecture    bool                           `json:"supported_architecture"`
	NativeLoadable           bool                           `json:"native_loadable"`
	RequiresPythonConversion bool                           `json:"requires_python_conversion"`
	HasTokenizer             bool                           `json:"has_tokenizer"`
	HasChatTemplate          bool                           `json:"has_chat_template"`
	ChatTemplateSource       ModelPackChatTemplateSource    `json:"chat_template_source,omitempty"`
	ChatTemplate             string                         `json:"chat_template,omitempty"`
	QuantBits                int                            `json:"quant_bits,omitempty"`
	QuantGroup               int                            `json:"quant_group,omitempty"`
	QuantType                string                         `json:"quant_type,omitempty"`
	QuantFamily              string                         `json:"quant_family,omitempty"`
	Quantization             *GGUFQuantizationInfo          `json:"quantization,omitempty"`
	JANG                     *jang.Info          `json:"jang,omitempty"`
	PackedQuantization       *jang.PackedProfile `json:"packed_quantization,omitempty"`
	Codebook                 *codebook.Profile   `json:"codebook,omitempty"`
	MiniMaxM2                *MiniMaxM2TensorPlan           `json:"minimax_m2,omitempty"`
	MiniMaxM2LayerSkeleton   *MiniMaxM2LayerForwardSkeleton `json:"minimax_m2_layer_skeleton,omitempty"`
	ArchitectureProfile      *profile.ModelArchitectureProfile      `json:"architecture_profile,omitempty"`
	Embedding                *ModelEmbeddingProfile         `json:"embedding,omitempty"`
	Rerank                   *ModelRerankProfile            `json:"rerank,omitempty"`
	Capabilities             []inference.Capability         `json:"capabilities,omitempty"`
	WeightBytes              uint64                         `json:"weight_bytes,omitempty"`
	ContextLength            int                            `json:"context_length,omitempty"`
	NumLayers                int                            `json:"num_layers,omitempty"`
	HiddenSize               int                            `json:"hidden_size,omitempty"`
	VocabSize                int                            `json:"vocab_size,omitempty"`
	GGUF                     *GGUFInfo                      `json:"gguf,omitempty"`
	Issues                   []ModelPackIssue               `json:"issues,omitempty"`
	OK                       bool                           `json:"valid"`
}

// Valid reports whether the pack has no error-severity validation issues.
func (pack ModelPack) Valid() bool { return pack.OK }

// HasIssue reports whether a validation issue code is present.
func (pack ModelPack) HasIssue(code ModelPackIssueCode) bool {
	for _, issue := range pack.Issues {
		if issue.Code == code {
			return true
		}
	}
	return false
}

// ModelPackConfig configures pack validation.
type ModelPackConfig struct {
	ExpectedQuantBits   int
	MaxContextLength    int
	RequireChatTemplate bool
}

// ModelPackOption configures model-pack inspection.
type ModelPackOption func(*ModelPackConfig)

// WithPackQuantization requires a specific quantization width when metadata exposes one.
func WithPackQuantization(bits int) ModelPackOption {
	return func(cfg *ModelPackConfig) { cfg.ExpectedQuantBits = bits }
}

// WithPackMaxContextLength rejects packs whose declared context exceeds n.
func WithPackMaxContextLength(n int) ModelPackOption {
	return func(cfg *ModelPackConfig) { cfg.MaxContextLength = n }
}

// WithPackRequireChatTemplate controls whether a chat template is mandatory.
func WithPackRequireChatTemplate(required bool) ModelPackOption {
	return func(cfg *ModelPackConfig) { cfg.RequireChatTemplate = required }
}

func applyModelPackOptions(opts []ModelPackOption) ModelPackConfig {
	cfg := ModelPackConfig{RequireChatTemplate: true}
	for _, opt := range opts {
		opt(&cfg)
	}
	return cfg
}

// InspectModelPack validates a local model directory or GGUF file without loading weights.
func InspectModelPack(modelPath string, opts ...ModelPackOption) (ModelPack, error) {
	cfg := applyModelPackOptions(opts)
	resolvedPath := modelPath
	if abs := core.PathAbs(modelPath); abs.OK {
		resolvedPath = abs.Value.(string)
	}
	stat := core.Stat(resolvedPath)
	if !stat.OK {
		return ModelPack{}, stat.Value.(error)
	}

	root := resolvedPath
	if !stat.Value.(core.FsFileInfo).IsDir() {
		root = core.PathDir(resolvedPath)
	}
	pack := ModelPack{
		Path: resolvedPath,
		Root: root,
	}

	config, configErr := inspectModelPackConfig(&pack, root)
	inspectModelPackWeights(&pack, resolvedPath, root)
	if pack.Format == ModelPackFormatGGUF && len(pack.WeightFiles) == 1 {
		inspectModelPackGGUF(&pack, pack.WeightFiles[0])
	}
	if configErr == nil && config != nil {
		applyModelPackConfigMetadata(&pack, config)
	}
	inspectModelPackJANG(&pack, root)
	inspectModelPackCodebook(&pack, root)
	inspectModelPackTokenizer(&pack, root)
	inspectModelPackChatTemplate(&pack, root, cfg)
	inspectModelPackArchitecture(&pack)
	inspectModelPackTaskProfiles(&pack, root)
	inspectModelPackMiniMaxM2(&pack)
	inspectModelPackPolicy(&pack, cfg)
	finalizeModelPack(&pack)
	return pack, nil
}

// ValidateModelPack returns an error when InspectModelPack finds validation issues.
func ValidateModelPack(modelPath string, opts ...ModelPackOption) (ModelPack, error) {
	pack, err := InspectModelPack(modelPath, opts...)
	if err != nil {
		return pack, err
	}
	if pack.Valid() {
		return pack, nil
	}
	return pack, core.NewError("mlx: invalid model pack: " + pack.issueSummary())
}

func inspectModelPackConfig(pack *ModelPack, root string) (*modelConfigProbe, error) {
	configPath := core.PathJoin(root, "config.json")
	config, err := readModelConfig(root)
	if err != nil {
		code := ModelPackIssueMissingConfig
		message := "config.json is required for native go-mlx loading"
		if !core.IsNotExist(err) {
			code = ModelPackIssueInvalidConfig
			message = "config.json could not be parsed"
		}
		pack.addIssue(ModelPackIssueError, code, message, configPath)
		return nil, err
	}
	pack.ConfigPath = configPath
	return config, nil
}

func inspectModelPackWeights(pack *ModelPack, resolvedPath, root string) {
	lowerPath := core.Lower(resolvedPath)
	var safetensors []string
	var ggufs []string
	if core.HasSuffix(lowerPath, ".safetensors") {
		safetensors = []string{resolvedPath}
	} else if core.HasSuffix(lowerPath, ".gguf") {
		ggufs = []string{resolvedPath}
	} else {
		safetensors = core.PathGlob(core.PathJoin(root, "*.safetensors"))
		ggufs = core.PathGlob(core.PathJoin(root, "*.gguf"))
	}
	sort.Strings(safetensors)
	sort.Strings(ggufs)
	for _, path := range append(append([]string(nil), safetensors...), ggufs...) {
		if info := core.Stat(path); info.OK {
			pack.WeightBytes += uint64(info.Value.(core.FsFileInfo).Size())
		}
	}

	switch {
	case len(safetensors) > 0 && len(ggufs) > 0:
		pack.Format = ModelPackFormatMixed
		pack.WeightFiles = append(append([]string(nil), safetensors...), ggufs...)
		pack.addIssue(ModelPackIssueError, ModelPackIssueMixedWeightFormats, "model pack contains both safetensors and GGUF weights", root)
	case len(safetensors) > 0:
		pack.Format = ModelPackFormatSafetensors
		pack.WeightFiles = append([]string(nil), safetensors...)
	case len(ggufs) == 1:
		pack.Format = ModelPackFormatGGUF
		pack.WeightFiles = append([]string(nil), ggufs...)
	case len(ggufs) > 1:
		pack.Format = ModelPackFormatGGUF
		pack.WeightFiles = append([]string(nil), ggufs...)
		pack.addIssue(ModelPackIssueError, ModelPackIssueMultipleGGUF, "model pack contains multiple GGUF files; native loading expects one", root)
	default:
		pack.Format = ModelPackFormatMissing
		pack.addIssue(ModelPackIssueError, ModelPackIssueMissingWeights, "no .safetensors or .gguf weights found", root)
	}
}

func inspectModelPackGGUF(pack *ModelPack, path string) {
	info, err := ReadGGUFInfo(path)
	if err != nil {
		pack.addIssue(ModelPackIssueError, ModelPackIssueInvalidGGUF, err.Error(), path)
		return
	}
	pack.GGUF = &info
	if pack.Architecture == "" {
		pack.Architecture = info.Architecture
	}
	pack.QuantBits = firstPositive(pack.QuantBits, info.QuantBits)
	pack.QuantGroup = firstPositive(pack.QuantGroup, info.QuantGroup)
	pack.QuantType = firstNonEmpty(pack.QuantType, info.QuantType)
	pack.QuantFamily = firstNonEmpty(pack.QuantFamily, info.QuantFamily)
	pack.Quantization = cloneGGUFQuantizationInfo(info.Quantization)
	pack.ContextLength = firstPositive(pack.ContextLength, info.ContextLength)
	pack.NumLayers = firstPositive(pack.NumLayers, info.NumLayers)
	pack.HiddenSize = firstPositive(pack.HiddenSize, info.HiddenSize)
	pack.VocabSize = firstPositive(pack.VocabSize, info.VocabSize)
	if !info.Valid() {
		pack.addIssue(ModelPackIssueError, ModelPackIssueInvalidGGUF, "GGUF tensor metadata failed validation: "+ggufValidationSummary(info.ValidationIssues), path)
	}
}

func applyModelPackConfigMetadata(pack *ModelPack, config *modelConfigProbe) {
	pack.Architecture = firstNonEmpty(pack.Architecture, config.architecture())
	pack.QuantBits = firstPositive(pack.QuantBits, config.quantBits())
	pack.QuantGroup = firstPositive(pack.QuantGroup, config.quantGroup())
	pack.ContextLength = firstPositive(pack.ContextLength, config.contextLength())
	pack.NumLayers = firstPositive(pack.NumLayers, config.numLayers())
	pack.HiddenSize = firstPositive(pack.HiddenSize, config.hiddenSize())
	pack.VocabSize = firstPositive(pack.VocabSize, config.vocabSize())
}

func inspectModelPackJANG(pack *ModelPack, root string) {
	info, err := jang.ReadConfig(root)
	if err != nil {
		pack.addIssue(ModelPackIssueWarning, ModelPackIssueQuantizationMismatch, "jang_config.json could not be parsed: "+err.Error(), core.PathJoin(root, "jang_config.json"))
		return
	}
	if info == nil {
		return
	}
	pack.JANG = info
	pack.PackedQuantization = jang.ClonePackedProfile(info.Packed)
	if info.SourceArchitecture != "" && pack.Architecture == "" {
		pack.Architecture = info.SourceArchitecture
	}
	if info.BitsDefault > 0 {
		pack.QuantBits = info.BitsDefault
	}
	if info.GroupSize > 0 {
		pack.QuantGroup = info.GroupSize
	}
	if info.Packed != nil {
		pack.QuantType = info.Packed.Type
	}
	pack.QuantFamily = "jang"
	pack.Quantization = &GGUFQuantizationInfo{
		Type:      pack.QuantType,
		Family:    pack.QuantFamily,
		Bits:      pack.QuantBits,
		GroupSize: pack.QuantGroup,
		Mixed:     true,
	}
}

func inspectModelPackCodebook(pack *ModelPack, root string) {
	profile, err := codebook.ReadProfile(root)
	if err != nil {
		pack.addIssue(ModelPackIssueError, ModelPackIssueUnsupportedCodebook, "codebook_config.json could not be parsed: "+err.Error(), core.PathJoin(root, "codebook_config.json"))
		return
	}
	if profile == nil {
		return
	}
	pack.Codebook = codebook.CloneProfile(profile)
	pack.QuantType = codebook.FormatVQ
	pack.QuantFamily = codebook.Type
	pack.QuantBits = firstPositive(pack.QuantBits, profile.IndexBits)
	pack.Quantization = &GGUFQuantizationInfo{
		Type:   pack.QuantType,
		Family: pack.QuantFamily,
		Bits:   pack.QuantBits,
		Mixed:  true,
	}
	pack.addIssue(ModelPackIssueError, ModelPackIssueUnsupportedCodebook, "codebook/VQ tensor matvec is available, but full codebook-quantized model loading is not implemented yet", core.PathJoin(root, "codebook_config.json"))
}

func cloneGGUFQuantizationInfo(info GGUFQuantizationInfo) *GGUFQuantizationInfo {
	if info.Type == "" && info.Family == "" && info.Bits == 0 && len(info.TensorTypes) == 0 {
		return nil
	}
	cloned := info
	cloned.TensorTypes = append([]GGUFTensorTypeSummary(nil), info.TensorTypes...)
	return &cloned
}

func ggufValidationSummary(issues []GGUFValidationIssue) string {
	if len(issues) == 0 {
		return "unknown validation failure"
	}
	parts := make([]string, 0, len(issues))
	for _, issue := range issues {
		if issue.Tensor != "" {
			parts = append(parts, core.Concat(issue.Code, ":", issue.Tensor))
			continue
		}
		parts = append(parts, issue.Code)
	}
	return core.Join(", ", parts...)
}

func inspectModelPackTokenizer(pack *ModelPack, root string) {
	tokenizerPath := core.PathJoin(root, "tokenizer.json")
	stat := core.Stat(tokenizerPath)
	if !stat.OK {
		pack.addIssue(ModelPackIssueError, ModelPackIssueMissingTokenizer, "tokenizer.json is required", tokenizerPath)
		return
	}
	if _, err := LoadTokenizer(tokenizerPath); err != nil {
		pack.addIssue(ModelPackIssueError, ModelPackIssueInvalidTokenizer, err.Error(), tokenizerPath)
		return
	}
	pack.TokenizerPath = tokenizerPath
	pack.HasTokenizer = true
}

func inspectModelPackChatTemplate(pack *ModelPack, root string, cfg ModelPackConfig) {
	tokenizerConfigPath := core.PathJoin(root, "tokenizer_config.json")
	if template, ok, err := readTokenizerChatTemplate(tokenizerConfigPath); ok {
		pack.TokenizerConfigPath = tokenizerConfigPath
		pack.ChatTemplate = template
		pack.ChatTemplateSource = ModelPackChatTemplateFile
		pack.HasChatTemplate = true
		return
	} else if err != nil {
		pack.addIssue(ModelPackIssueWarning, ModelPackIssueMissingChatTemplate, err.Error(), tokenizerConfigPath)
	}

	jinjaPath := core.PathJoin(root, "chat_template.jinja")
	if template, ok, err := readJinjaChatTemplate(jinjaPath); ok {
		pack.TokenizerConfigPath = jinjaPath
		pack.ChatTemplate = template
		pack.ChatTemplateSource = ModelPackChatTemplateJinja
		pack.HasChatTemplate = true
		return
	} else if err != nil {
		pack.addIssue(ModelPackIssueWarning, ModelPackIssueMissingChatTemplate, err.Error(), jinjaPath)
	}

	if template := nativeChatTemplateName(pack.Architecture); template != "" {
		pack.ChatTemplate = template
		pack.ChatTemplateSource = ModelPackChatTemplateNative
		pack.HasChatTemplate = true
		return
	}
	if !modelPackRequiresChatTemplate(pack.Architecture) {
		return
	}
	if cfg.RequireChatTemplate {
		pack.addIssue(ModelPackIssueError, ModelPackIssueMissingChatTemplate, "no tokenizer_config.json chat_template or native chat template is available", root)
	}
}

func readTokenizerChatTemplate(path string) (string, bool, error) {
	read := core.ReadFile(path)
	if !read.OK {
		if core.IsNotExist(read.Value.(error)) {
			return "", false, nil
		}
		return "", false, read.Value.(error)
	}
	var config struct {
		ChatTemplate any `json:"chat_template"`
	}
	if result := core.JSONUnmarshal(read.Value.([]byte), &config); !result.OK {
		return "", false, result.Value.(error)
	}
	switch template := config.ChatTemplate.(type) {
	case string:
		template = core.Trim(template)
		return template, template != "", nil
	case []any:
		if len(template) > 0 {
			return "named_chat_templates", true, nil
		}
	}
	return "", false, nil
}

func readJinjaChatTemplate(path string) (string, bool, error) {
	read := core.ReadFile(path)
	if !read.OK {
		if core.IsNotExist(read.Value.(error)) {
			return "", false, nil
		}
		return "", false, read.Value.(error)
	}
	template := core.Trim(string(read.Value.([]byte)))
	return template, template != "", nil
}

func inspectModelPackArchitecture(pack *ModelPack) {
	if pack.Architecture == "" {
		pack.addIssue(ModelPackIssueError, ModelPackIssueMissingArchitecture, "model architecture could not be determined", pack.ConfigPath)
		return
	}
	if profile, ok := profile.LookupArchitectureProfile(pack.Architecture); ok {
		pack.Architecture = profile.ID
		pack.ArchitectureProfile = &profile
	}
	pack.SupportedArchitecture = modelPackSupportedArchitecture(pack.Architecture)
	if !pack.SupportedArchitecture {
		pack.addIssue(ModelPackIssueError, ModelPackIssueUnsupportedArchitecture, "architecture is not supported by native go-mlx loaders: "+pack.Architecture, pack.ConfigPath)
		return
	}
	if !modelPackNativeRuntimeSupported(pack.Architecture) {
		pack.addIssue(ModelPackIssueWarning, ModelPackIssueUnsupportedRuntime, modelPackUnsupportedRuntimeMessage(pack.Architecture), pack.ConfigPath)
	}
}

func modelPackUnsupportedRuntimeMessage(architecture string) string {
	if profile, ok := profile.LookupArchitectureProfile(architecture); ok {
		switch {
		case profile.Embeddings:
			return "architecture is recognized, but native embedding encoder loading is not implemented yet: " + architecture
		case profile.Rerank:
			return "architecture is recognized, but native rerank scorer loading is not implemented yet: " + architecture
		case profile.MoE:
			return "architecture is recognized, but sparse expert runtime loading is not implemented yet: " + architecture
		}
	}
	return "architecture is recognized, but native runtime loading is not implemented yet: " + architecture
}

func inspectModelPackTaskProfiles(pack *ModelPack, root string) {
	if pack == nil {
		return
	}
	arch := pack.ArchitectureProfile
	if arch == nil && pack.Architecture != "" {
		if resolved, ok := profile.LookupArchitectureProfile(pack.Architecture); ok {
			pack.ArchitectureProfile = &resolved
			arch = &resolved
		}
	}
	if arch == nil {
		return
	}
	if arch.Embeddings {
		embedding := inspectModelPackEmbeddingProfile(pack, root)
		pack.Embedding = &embedding
	}
	if arch.Rerank {
		rerank := inspectModelPackRerankProfile(pack, root)
		pack.Rerank = &rerank
	}
	pack.Capabilities = modelPackCapabilities(pack)
}

func inspectModelPackEmbeddingProfile(pack *ModelPack, root string) ModelEmbeddingProfile {
	profile := ModelEmbeddingProfile{
		Dimension:         pack.HiddenSize,
		Pooling:           "cls",
		MaxSequenceLength: pack.ContextLength,
		Source:            "transformers",
	}
	if root == "" {
		return profile
	}
	if maxSeq, ok := readSentenceBertMaxSequence(root); ok {
		profile.MaxSequenceLength = firstPositive(maxSeq, profile.MaxSequenceLength)
		profile.Source = "sentence-transformers"
	}
	if pooling, ok := readSentenceTransformerPooling(root); ok {
		profile.Pooling = pooling
		profile.Source = "sentence-transformers"
	}
	if normalize, ok := readSentenceTransformerNormalize(root); ok {
		profile.Normalize = normalize
		profile.Source = "sentence-transformers"
	}
	return profile
}

func inspectModelPackRerankProfile(pack *ModelPack, root string) ModelRerankProfile {
	profile := ModelRerankProfile{
		Method:            "cross-encoder",
		MaxSequenceLength: pack.ContextLength,
		Source:            "transformers",
	}
	if root != "" {
		if maxSeq, ok := readSentenceBertMaxSequence(root); ok {
			profile.MaxSequenceLength = firstPositive(maxSeq, profile.MaxSequenceLength)
			profile.Source = "sentence-transformers"
		}
	}
	return profile
}

func readSentenceBertMaxSequence(root string) (int, bool) {
	read := core.ReadFile(core.PathJoin(root, "sentence_bert_config.json"))
	if !read.OK {
		return 0, false
	}
	var config struct {
		MaxSequenceLength int `json:"max_seq_length"`
	}
	if result := core.JSONUnmarshal(read.Value.([]byte), &config); !result.OK {
		return 0, false
	}
	return config.MaxSequenceLength, config.MaxSequenceLength > 0
}

func readSentenceTransformerPooling(root string) (string, bool) {
	paths := core.PathGlob(core.PathJoin(root, "*_Pooling", "config.json"))
	sort.Strings(paths)
	for _, path := range paths {
		read := core.ReadFile(path)
		if !read.OK {
			continue
		}
		var config struct {
			CLS          bool `json:"pooling_mode_cls_token"`
			Mean         bool `json:"pooling_mode_mean_tokens"`
			Max          bool `json:"pooling_mode_max_tokens"`
			WeightedMean bool `json:"pooling_mode_weightedmean_tokens"`
		}
		if result := core.JSONUnmarshal(read.Value.([]byte), &config); !result.OK {
			continue
		}
		switch {
		case config.Mean:
			return "mean", true
		case config.CLS:
			return "cls", true
		case config.Max:
			return "max", true
		case config.WeightedMean:
			return "weighted_mean", true
		}
	}
	return "", false
}

func readSentenceTransformerNormalize(root string) (bool, bool) {
	read := core.ReadFile(core.PathJoin(root, "modules.json"))
	if !read.OK {
		return false, false
	}
	var modules []struct {
		Type string `json:"type"`
		Path string `json:"path"`
	}
	if result := core.JSONUnmarshal(read.Value.([]byte), &modules); !result.OK {
		return false, false
	}
	for _, module := range modules {
		if core.Contains(core.Lower(module.Type), "normalize") || core.Contains(core.Lower(module.Path), "normalize") {
			return true, true
		}
	}
	return false, true
}

func modelPackCapabilities(pack *ModelPack) []inference.Capability {
	if pack == nil {
		return nil
	}
	var capabilities []inference.Capability
	if pack.Embedding != nil {
		capabilities = append(capabilities, modelPackAlgorithmCapability(inference.CapabilityEmbeddings, pack.Architecture))
	}
	if pack.Rerank != nil {
		capabilities = append(capabilities, modelPackAlgorithmCapability(inference.CapabilityRerank, pack.Architecture))
	}
	if pack.ArchitectureProfile != nil && pack.ArchitectureProfile.MoE {
		capabilities = append(capabilities,
			modelPackAlgorithmCapability(inference.CapabilityMoERouting, pack.Architecture),
			modelPackAlgorithmCapability(inference.CapabilityMoELazyExperts, pack.Architecture),
		)
	}
	if pack.Codebook != nil {
		capabilities = append(capabilities, modelPackAlgorithmCapability(inference.CapabilityCodebookVQ, pack.Architecture))
	}
	return capabilities
}

func modelPackAlgorithmCapability(id inference.CapabilityID, architecture string) inference.Capability {
	if profile, ok := profile.LookupAlgorithmProfile(id); ok {
		capability := profile.Capability()
		if capability.Labels == nil {
			capability.Labels = map[string]string{}
		}
		if architecture != "" {
			capability.Labels["architecture"] = architecture
		}
		return capability
	}
	capability := inference.PlannedCapability(id, inference.CapabilityGroupModel, "model-pack metadata is available; native kernels are pending")
	if architecture != "" {
		capability.Labels = map[string]string{"architecture": architecture}
	}
	return capability
}

func modelPackUsesGenerationKVCache(pack *ModelPack, architecture string) bool {
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
	if profile, ok := profile.LookupArchitectureProfile(architecture); ok && (profile.Embeddings || profile.Rerank) {
		return false
	}
	return true
}

func inspectModelPackMiniMaxM2(pack *ModelPack) {
	if pack.Architecture != "minimax_m2" || pack.ConfigPath == "" {
		return
	}
	read := core.ReadFile(pack.ConfigPath)
	if !read.OK {
		pack.addIssue(ModelPackIssueWarning, ModelPackIssueInvalidConfig, "MiniMax M2 config could not be read: "+read.Value.(error).Error(), pack.ConfigPath)
		return
	}
	cfg, err := ParseMiniMaxM2Config(read.Value.([]byte))
	if err != nil {
		pack.addIssue(ModelPackIssueWarning, ModelPackIssueInvalidConfig, "MiniMax M2 config could not be parsed: "+err.Error(), pack.ConfigPath)
		return
	}
	plan, err := BuildMiniMaxM2TensorPlan(cfg, pack.JANG)
	if err != nil {
		pack.addIssue(ModelPackIssueWarning, ModelPackIssueUnsupportedRuntime, "MiniMax M2 tensor plan could not be built: "+err.Error(), pack.ConfigPath)
		return
	}
	pack.MiniMaxM2 = &plan
	if pack.Format != ModelPackFormatSafetensors || len(pack.WeightFiles) == 0 {
		return
	}
	skeleton, err := BuildMiniMaxM2LayerForwardSkeletonFromSafetensors(plan, pack.WeightFiles, 0)
	if err != nil {
		pack.addIssue(ModelPackIssueWarning, ModelPackIssueMiniMaxM2LayerSkeleton, "MiniMax M2 first-layer skeleton could not be validated: "+err.Error(), pack.Root)
		return
	}
	pack.MiniMaxM2LayerSkeleton = &skeleton
}

func inspectModelPackPolicy(pack *ModelPack, cfg ModelPackConfig) {
	if cfg.ExpectedQuantBits > 0 && pack.QuantBits != cfg.ExpectedQuantBits {
		pack.addIssue(ModelPackIssueError, ModelPackIssueQuantizationMismatch, core.Sprintf("quantization is %d-bit, expected %d-bit", pack.QuantBits, cfg.ExpectedQuantBits), pack.Root)
	}
	if cfg.MaxContextLength > 0 && pack.ContextLength > cfg.MaxContextLength {
		pack.addIssue(ModelPackIssueError, ModelPackIssueContextTooLarge, core.Sprintf("context length %d exceeds limit %d", pack.ContextLength, cfg.MaxContextLength), pack.Root)
	}
}

func finalizeModelPack(pack *ModelPack) {
	chatOK := pack.HasChatTemplate || !modelPackRequiresChatTemplate(pack.Architecture)
	pack.NativeLoadable = pack.SupportedArchitecture &&
		modelPackNativeRuntimeSupported(pack.Architecture) &&
		pack.ConfigPath != "" &&
		pack.HasTokenizer &&
		chatOK &&
		(pack.Format == ModelPackFormatSafetensors || pack.Format == ModelPackFormatGGUF) &&
		!pack.HasErrorIssue()
	pack.RequiresPythonConversion = !pack.NativeLoadable
	pack.OK = !pack.HasErrorIssue()
}

func modelPackSupportedArchitecture(architecture string) bool {
	_, ok := profile.LookupArchitectureProfile(architecture)
	return ok
}

func modelPackNativeRuntimeSupported(architecture string) bool {
	profile, ok := profile.LookupArchitectureProfile(architecture)
	return ok && profile.NativeRuntime
}

func nativeChatTemplateName(architecture string) string {
	if profile, ok := profile.LookupArchitectureProfile(architecture); ok {
		return profile.ChatTemplate
	}
	return ""
}

func modelPackRequiresChatTemplate(architecture string) bool {
	profile, ok := profile.LookupArchitectureProfile(architecture)
	return !ok || profile.RequiresChatTemplate
}

func (pack *ModelPack) addIssue(severity ModelPackIssueSeverity, code ModelPackIssueCode, message, path string) {
	pack.Issues = append(pack.Issues, ModelPackIssue{
		Severity: severity,
		Code:     code,
		Message:  message,
		Path:     path,
	})
}

// HasErrorIssue reports whether any issue has error severity.
func (pack ModelPack) HasErrorIssue() bool {
	for _, issue := range pack.Issues {
		if issue.Severity == ModelPackIssueError {
			return true
		}
	}
	return false
}

func (pack ModelPack) issueSummary() string {
	if len(pack.Issues) == 0 {
		return "unknown"
	}
	builder := core.NewBuilder()
	for i, issue := range pack.Issues {
		if issue.Severity != ModelPackIssueError {
			continue
		}
		if builder.Len() > 0 {
			builder.WriteString(", ")
		}
		builder.WriteString(string(issue.Code))
		if i == len(pack.Issues)-1 {
			continue
		}
	}
	if builder.Len() == 0 {
		return "unknown"
	}
	return builder.String()
}
