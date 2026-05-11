// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"sort"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/quant/codebook"
	"dappco.re/go/inference/quant/jang"
	mp "dappco.re/go/mlx/pack"
	"dappco.re/go/mlx/profile"
)

// InspectModelPack validates a local model directory or GGUF file without loading weights.
func InspectModelPack(modelPath string, opts ...mp.ModelPackOption) (mp.ModelPack, error) {
	cfg := mp.ApplyOptions(opts)
	resolvedPath := modelPath
	if abs := core.PathAbs(modelPath); abs.OK {
		resolvedPath = abs.Value.(string)
	}
	stat := core.Stat(resolvedPath)
	if !stat.OK {
		return mp.ModelPack{}, stat.Value.(error)
	}

	root := resolvedPath
	if !stat.Value.(core.FsFileInfo).IsDir() {
		root = core.PathDir(resolvedPath)
	}
	pack := mp.ModelPack{
		Path: resolvedPath,
		Root: root,
	}

	config, configErr := inspectModelPackConfig(&pack, root)
	inspectModelPackWeights(&pack, resolvedPath, root)
	if pack.Format == mp.ModelPackFormatGGUF && len(pack.WeightFiles) == 1 {
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
func ValidateModelPack(modelPath string, opts ...mp.ModelPackOption) (mp.ModelPack, error) {
	pack, err := InspectModelPack(modelPath, opts...)
	if err != nil {
		return pack, err
	}
	if pack.Valid() {
		return pack, nil
	}
	return pack, core.NewError("mlx: invalid model pack: " + pack.IssueSummary())
}

func inspectModelPackConfig(pack *mp.ModelPack, root string) (*modelConfigProbe, error) {
	configPath := core.PathJoin(root, "config.json")
	config, err := readModelConfig(root)
	if err != nil {
		code := mp.ModelPackIssueMissingConfig
		message := "config.json is required for native go-mlx loading"
		if !core.IsNotExist(err) {
			code = mp.ModelPackIssueInvalidConfig
			message = "config.json could not be parsed"
		}
		pack.AddIssue(mp.ModelPackIssueError, code, message, configPath)
		return nil, err
	}
	pack.ConfigPath = configPath
	return config, nil
}

func inspectModelPackWeights(pack *mp.ModelPack, resolvedPath, root string) {
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
		pack.Format = mp.ModelPackFormatMixed
		pack.WeightFiles = append(append([]string(nil), safetensors...), ggufs...)
		pack.AddIssue(mp.ModelPackIssueError, mp.ModelPackIssueMixedWeightFormats, "model pack contains both safetensors and GGUF weights", root)
	case len(safetensors) > 0:
		pack.Format = mp.ModelPackFormatSafetensors
		pack.WeightFiles = append([]string(nil), safetensors...)
	case len(ggufs) == 1:
		pack.Format = mp.ModelPackFormatGGUF
		pack.WeightFiles = append([]string(nil), ggufs...)
	case len(ggufs) > 1:
		pack.Format = mp.ModelPackFormatGGUF
		pack.WeightFiles = append([]string(nil), ggufs...)
		pack.AddIssue(mp.ModelPackIssueError, mp.ModelPackIssueMultipleGGUF, "model pack contains multiple GGUF files; native loading expects one", root)
	default:
		pack.Format = mp.ModelPackFormatMissing
		pack.AddIssue(mp.ModelPackIssueError, mp.ModelPackIssueMissingWeights, "no .safetensors or .gguf weights found", root)
	}
}

func inspectModelPackGGUF(pack *mp.ModelPack, path string) {
	info, err := ReadGGUFInfo(path)
	if err != nil {
		pack.AddIssue(mp.ModelPackIssueError, mp.ModelPackIssueInvalidGGUF, err.Error(), path)
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
		pack.AddIssue(mp.ModelPackIssueError, mp.ModelPackIssueInvalidGGUF, "GGUF tensor metadata failed validation: "+ggufValidationSummary(info.ValidationIssues), path)
	}
}

func applyModelPackConfigMetadata(pack *mp.ModelPack, config *modelConfigProbe) {
	pack.Architecture = firstNonEmpty(pack.Architecture, config.architecture())
	pack.QuantBits = firstPositive(pack.QuantBits, config.quantBits())
	pack.QuantGroup = firstPositive(pack.QuantGroup, config.quantGroup())
	pack.ContextLength = firstPositive(pack.ContextLength, config.contextLength())
	pack.NumLayers = firstPositive(pack.NumLayers, config.numLayers())
	pack.HiddenSize = firstPositive(pack.HiddenSize, config.hiddenSize())
	pack.VocabSize = firstPositive(pack.VocabSize, config.vocabSize())
}

func inspectModelPackJANG(pack *mp.ModelPack, root string) {
	info, err := jang.ReadConfig(root)
	if err != nil {
		pack.AddIssue(mp.ModelPackIssueWarning, mp.ModelPackIssueQuantizationMismatch, "jang_config.json could not be parsed: "+err.Error(), core.PathJoin(root, "jang_config.json"))
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

func inspectModelPackCodebook(pack *mp.ModelPack, root string) {
	profile, err := codebook.ReadProfile(root)
	if err != nil {
		pack.AddIssue(mp.ModelPackIssueError, mp.ModelPackIssueUnsupportedCodebook, "codebook_config.json could not be parsed: "+err.Error(), core.PathJoin(root, "codebook_config.json"))
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
	pack.AddIssue(mp.ModelPackIssueError, mp.ModelPackIssueUnsupportedCodebook, "codebook/VQ tensor matvec is available, but full codebook-quantized model loading is not implemented yet", core.PathJoin(root, "codebook_config.json"))
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

func inspectModelPackTokenizer(pack *mp.ModelPack, root string) {
	tokenizerPath := core.PathJoin(root, "tokenizer.json")
	stat := core.Stat(tokenizerPath)
	if !stat.OK {
		pack.AddIssue(mp.ModelPackIssueError, mp.ModelPackIssueMissingTokenizer, "tokenizer.json is required", tokenizerPath)
		return
	}
	if _, err := LoadTokenizer(tokenizerPath); err != nil {
		pack.AddIssue(mp.ModelPackIssueError, mp.ModelPackIssueInvalidTokenizer, err.Error(), tokenizerPath)
		return
	}
	pack.TokenizerPath = tokenizerPath
	pack.HasTokenizer = true
}

func inspectModelPackChatTemplate(pack *mp.ModelPack, root string, cfg mp.ModelPackConfig) {
	tokenizerConfigPath := core.PathJoin(root, "tokenizer_config.json")
	if template, ok, err := readTokenizerChatTemplate(tokenizerConfigPath); ok {
		pack.TokenizerConfigPath = tokenizerConfigPath
		pack.ChatTemplate = template
		pack.ChatTemplateSource = mp.ModelPackChatTemplateFile
		pack.HasChatTemplate = true
		return
	} else if err != nil {
		pack.AddIssue(mp.ModelPackIssueWarning, mp.ModelPackIssueMissingChatTemplate, err.Error(), tokenizerConfigPath)
	}

	jinjaPath := core.PathJoin(root, "chat_template.jinja")
	if template, ok, err := readJinjaChatTemplate(jinjaPath); ok {
		pack.TokenizerConfigPath = jinjaPath
		pack.ChatTemplate = template
		pack.ChatTemplateSource = mp.ModelPackChatTemplateJinja
		pack.HasChatTemplate = true
		return
	} else if err != nil {
		pack.AddIssue(mp.ModelPackIssueWarning, mp.ModelPackIssueMissingChatTemplate, err.Error(), jinjaPath)
	}

	if template := nativeChatTemplateName(pack.Architecture); template != "" {
		pack.ChatTemplate = template
		pack.ChatTemplateSource = mp.ModelPackChatTemplateNative
		pack.HasChatTemplate = true
		return
	}
	if !modelPackRequiresChatTemplate(pack.Architecture) {
		return
	}
	if cfg.RequireChatTemplate {
		pack.AddIssue(mp.ModelPackIssueError, mp.ModelPackIssueMissingChatTemplate, "no tokenizer_config.json chat_template or native chat template is available", root)
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

func inspectModelPackArchitecture(pack *mp.ModelPack) {
	if pack.Architecture == "" {
		pack.AddIssue(mp.ModelPackIssueError, mp.ModelPackIssueMissingArchitecture, "model architecture could not be determined", pack.ConfigPath)
		return
	}
	if profile, ok := profile.LookupArchitectureProfile(pack.Architecture); ok {
		pack.Architecture = profile.ID
		pack.ArchitectureProfile = &profile
	}
	pack.SupportedArchitecture = modelPackSupportedArchitecture(pack.Architecture)
	if !pack.SupportedArchitecture {
		pack.AddIssue(mp.ModelPackIssueError, mp.ModelPackIssueUnsupportedArchitecture, "architecture is not supported by native go-mlx loaders: "+pack.Architecture, pack.ConfigPath)
		return
	}
	if !modelPackNativeRuntimeSupported(pack.Architecture) {
		pack.AddIssue(mp.ModelPackIssueWarning, mp.ModelPackIssueUnsupportedRuntime, modelPackUnsupportedRuntimeMessage(pack.Architecture), pack.ConfigPath)
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

func inspectModelPackTaskProfiles(pack *mp.ModelPack, root string) {
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

func inspectModelPackEmbeddingProfile(pack *mp.ModelPack, root string) mp.ModelEmbeddingProfile {
	profile := mp.ModelEmbeddingProfile{
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

func inspectModelPackRerankProfile(pack *mp.ModelPack, root string) mp.ModelRerankProfile {
	profile := mp.ModelRerankProfile{
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

func modelPackCapabilities(pack *mp.ModelPack) []inference.Capability {
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

func modelPackUsesGenerationKVCache(pack *mp.ModelPack, architecture string) bool {
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

func inspectModelPackMiniMaxM2(pack *mp.ModelPack) {
	if pack.Architecture != "minimax_m2" || pack.ConfigPath == "" {
		return
	}
	read := core.ReadFile(pack.ConfigPath)
	if !read.OK {
		pack.AddIssue(mp.ModelPackIssueWarning, mp.ModelPackIssueInvalidConfig, "MiniMax M2 config could not be read: "+read.Value.(error).Error(), pack.ConfigPath)
		return
	}
	cfg, err := ParseMiniMaxM2Config(read.Value.([]byte))
	if err != nil {
		pack.AddIssue(mp.ModelPackIssueWarning, mp.ModelPackIssueInvalidConfig, "MiniMax M2 config could not be parsed: "+err.Error(), pack.ConfigPath)
		return
	}
	plan, err := BuildMiniMaxM2TensorPlan(cfg, pack.JANG)
	if err != nil {
		pack.AddIssue(mp.ModelPackIssueWarning, mp.ModelPackIssueUnsupportedRuntime, "MiniMax M2 tensor plan could not be built: "+err.Error(), pack.ConfigPath)
		return
	}
	pack.MiniMaxM2 = &plan
	if pack.Format != mp.ModelPackFormatSafetensors || len(pack.WeightFiles) == 0 {
		return
	}
	skeleton, err := BuildMiniMaxM2LayerForwardSkeletonFromSafetensors(plan, pack.WeightFiles, 0)
	if err != nil {
		pack.AddIssue(mp.ModelPackIssueWarning, mp.ModelPackIssueMiniMaxM2LayerSkeleton, "MiniMax M2 first-layer skeleton could not be validated: "+err.Error(), pack.Root)
		return
	}
	pack.MiniMaxM2LayerSkeleton = &skeleton
}

func inspectModelPackPolicy(pack *mp.ModelPack, cfg mp.ModelPackConfig) {
	if cfg.ExpectedQuantBits > 0 && pack.QuantBits != cfg.ExpectedQuantBits {
		pack.AddIssue(mp.ModelPackIssueError, mp.ModelPackIssueQuantizationMismatch, core.Sprintf("quantization is %d-bit, expected %d-bit", pack.QuantBits, cfg.ExpectedQuantBits), pack.Root)
	}
	if cfg.MaxContextLength > 0 && pack.ContextLength > cfg.MaxContextLength {
		pack.AddIssue(mp.ModelPackIssueError, mp.ModelPackIssueContextTooLarge, core.Sprintf("context length %d exceeds limit %d", pack.ContextLength, cfg.MaxContextLength), pack.Root)
	}
}

func finalizeModelPack(pack *mp.ModelPack) {
	chatOK := pack.HasChatTemplate || !modelPackRequiresChatTemplate(pack.Architecture)
	pack.NativeLoadable = pack.SupportedArchitecture &&
		modelPackNativeRuntimeSupported(pack.Architecture) &&
		pack.ConfigPath != "" &&
		pack.HasTokenizer &&
		chatOK &&
		(pack.Format == mp.ModelPackFormatSafetensors || pack.Format == mp.ModelPackFormatGGUF) &&
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

