// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"sort"

	core "dappco.re/go"
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
	ModelPackIssueMissingArchitecture     ModelPackIssueCode = "missing_architecture"
	ModelPackIssueMissingChatTemplate     ModelPackIssueCode = "missing_chat_template"
	ModelPackIssueQuantizationMismatch    ModelPackIssueCode = "quantization_mismatch"
	ModelPackIssueContextTooLarge         ModelPackIssueCode = "context_too_large"
)

// ModelPackIssue describes one pack validation finding.
type ModelPackIssue struct {
	Severity ModelPackIssueSeverity `json:"severity"`
	Code     ModelPackIssueCode     `json:"code"`
	Message  string                 `json:"message"`
	Path     string                 `json:"path,omitempty"`
}

// ModelPack summarises whether a local model directory is natively loadable.
type ModelPack struct {
	Path                     string                      `json:"path"`
	Root                     string                      `json:"root"`
	Format                   ModelPackFormat             `json:"format"`
	ConfigPath               string                      `json:"config_path,omitempty"`
	WeightFiles              []string                    `json:"weight_files,omitempty"`
	TokenizerPath            string                      `json:"tokenizer_path,omitempty"`
	TokenizerConfigPath      string                      `json:"tokenizer_config_path,omitempty"`
	Architecture             string                      `json:"architecture,omitempty"`
	SupportedArchitecture    bool                        `json:"supported_architecture"`
	NativeLoadable           bool                        `json:"native_loadable"`
	RequiresPythonConversion bool                        `json:"requires_python_conversion"`
	HasTokenizer             bool                        `json:"has_tokenizer"`
	HasChatTemplate          bool                        `json:"has_chat_template"`
	ChatTemplateSource       ModelPackChatTemplateSource `json:"chat_template_source,omitempty"`
	ChatTemplate             string                      `json:"chat_template,omitempty"`
	QuantBits                int                         `json:"quant_bits,omitempty"`
	QuantGroup               int                         `json:"quant_group,omitempty"`
	ContextLength            int                         `json:"context_length,omitempty"`
	NumLayers                int                         `json:"num_layers,omitempty"`
	HiddenSize               int                         `json:"hidden_size,omitempty"`
	VocabSize                int                         `json:"vocab_size,omitempty"`
	GGUF                     *GGUFInfo                   `json:"gguf,omitempty"`
	Issues                   []ModelPackIssue            `json:"issues,omitempty"`
	OK                       bool                        `json:"valid"`
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
	inspectModelPackTokenizer(&pack, root)
	inspectModelPackChatTemplate(&pack, root, cfg)
	inspectModelPackArchitecture(&pack)
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
	pack.ContextLength = firstPositive(pack.ContextLength, info.ContextLength)
	pack.NumLayers = firstPositive(pack.NumLayers, info.NumLayers)
	pack.HiddenSize = firstPositive(pack.HiddenSize, info.HiddenSize)
	pack.VocabSize = firstPositive(pack.VocabSize, info.VocabSize)
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

	if template := nativeChatTemplateName(pack.Architecture); template != "" {
		pack.ChatTemplate = template
		pack.ChatTemplateSource = ModelPackChatTemplateNative
		pack.HasChatTemplate = true
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

func inspectModelPackArchitecture(pack *ModelPack) {
	if pack.Architecture == "" {
		pack.addIssue(ModelPackIssueError, ModelPackIssueMissingArchitecture, "model architecture could not be determined", pack.ConfigPath)
		return
	}
	pack.SupportedArchitecture = modelPackSupportedArchitecture(pack.Architecture)
	if !pack.SupportedArchitecture {
		pack.addIssue(ModelPackIssueError, ModelPackIssueUnsupportedArchitecture, "architecture is not supported by native go-mlx loaders: "+pack.Architecture, pack.ConfigPath)
	}
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
	pack.NativeLoadable = pack.SupportedArchitecture &&
		pack.ConfigPath != "" &&
		pack.HasTokenizer &&
		pack.HasChatTemplate &&
		(pack.Format == ModelPackFormatSafetensors || pack.Format == ModelPackFormatGGUF) &&
		!pack.HasErrorIssue()
	pack.RequiresPythonConversion = !pack.NativeLoadable
	pack.OK = !pack.HasErrorIssue()
}

func modelPackSupportedArchitecture(architecture string) bool {
	switch architecture {
	case "gemma2", "gemma3", "gemma3_text", "gemma4", "gemma4_text", "qwen2", "qwen3", "llama":
		return true
	default:
		return false
	}
}

func nativeChatTemplateName(architecture string) string {
	switch architecture {
	case "gemma2", "gemma3", "gemma3_text", "gemma4", "gemma4_text":
		return "gemma"
	case "qwen2", "qwen3":
		return "qwen"
	case "llama":
		return "llama"
	default:
		return ""
	}
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
