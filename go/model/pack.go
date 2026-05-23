// SPDX-Licence-Identifier: EUPL-1.2

// Package model holds model-pack inspection and validation utilities that
// operate on local directories or GGUF files without loading weights.
package model

import (
	"sort"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/quant/codebook"
	"dappco.re/go/inference/quant/jang"
	"dappco.re/go/mlx/gguf"
	"dappco.re/go/mlx/model/minimax/m2"
	mp "dappco.re/go/mlx/pack"
	"dappco.re/go/mlx/profile"
)

// Inspect validates a local model directory or GGUF file without loading weights.
//
//	pack, err := model.Inspect(modelPath)
func Inspect(modelPath string, opts ...mp.ModelPackOption) (mp.ModelPack, error) {
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
	// The dir index is opportunistic — populated by inspectModelPackWeights
	// from its single glob, then consumed by downstream NotExist probes
	// to avoid spurious open()/Result allocations. Stays empty (and
	// therefore inert) when the caller hands us a single-file path.
	var dir modelPackDirIndex
	inspectModelPackWeights(&pack, resolvedPath, root, &dir)
	if pack.Format == mp.ModelPackFormatGGUF && len(pack.WeightFiles) == 1 {
		inspectModelPackGGUF(&pack, pack.WeightFiles[0])
	}
	if configErr == nil && config != nil {
		applyModelPackConfigMetadata(&pack, config)
	}
	inspectModelPackJANG(&pack, root, &dir)
	inspectModelPackCodebook(&pack, root, &dir)
	inspectModelPackTokenizer(&pack, root)
	// Architecture resolution happens BEFORE chat-template inspection so
	// the latter can read pack.ArchitectureProfile directly instead of
	// re-entering profile.LookupArchitectureProfile twice (one each for
	// nativeChatTemplateName + modelPackRequiresChatTemplate). The
	// canonical ID written into pack.Architecture is what subsequent
	// stages already expect anyway.
	inspectModelPackArchitecture(&pack)
	inspectModelPackChatTemplate(&pack, root, cfg, &dir)
	inspectModelPackTaskProfiles(&pack, root, &dir)
	inspectModelPackMiniMaxM2(&pack)
	inspectModelPackPolicy(&pack, cfg)
	finalizeModelPack(&pack)
	return pack, nil
}

// firstNonEmpty returns the first non-empty string after trimming whitespace.
func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if core.Trim(value) != "" {
			return value
		}
	}
	return ""
}

// firstPositive returns the first positive value from a list.
func firstPositive(values ...int) int {
	for _, value := range values {
		if value > 0 {
			return value
		}
	}
	return 0
}

// Validate returns an error when Inspect finds validation issues.
//
//	pack, err := model.Validate(modelPath)
func Validate(modelPath string, opts ...mp.ModelPackOption) (mp.ModelPack, error) {
	pack, err := Inspect(modelPath, opts...)
	if err != nil {
		return pack, err
	}
	if pack.Valid() {
		return pack, nil
	}
	return pack, core.NewError("model: invalid model pack: " + pack.IssueSummary())
}

func inspectModelPackConfig(pack *mp.ModelPack, root string) (*modelConfigProbe, error) {
	configPath := core.PathJoin(root, "config.json")
	// Pass the joined path in directly — readModelConfig would rebuild
	// the same string via filepath.Join, so reuse what we just minted.
	config, err := readModelConfigAt(configPath)
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

// modelPackDirIndex caches presence of the specific optional-config
// filenames the inspect pipeline probes downstream — built from the
// same single PathGlob the weight inspector already runs, so this is
// opportunistic and adds no extra syscall. The index records exactly
// the six basenames we'd otherwise ReadFile-then-IsNotExist for, in
// fixed bool fields, so populating + querying is zero-alloc.
//
// The `populated` flag lets callers distinguish "no listing available"
// (single-file resolvedPath) from "listed but file absent" — the
// former falls through to the regular ReadFile probe so semantics for
// the single-file entry path stay unchanged.
type modelPackDirIndex struct {
	populated         bool
	jangConfig        bool
	codebookConfig    bool
	tokenizerConfig   bool
	chatTemplateJinja bool
	sentenceBert      bool
	modulesJSON       bool
}

// has reports whether the named direct child of root is present in the
// pre-fetched listing. Returns true if the index is empty (no listing
// available) so callers fall through to the existing ReadFile probe —
// the precise root-stat is preserved in that path. The name argument
// is one of the six recognised optional-config filenames; anything
// else returns true (let the caller perform the normal probe).
func (d *modelPackDirIndex) has(name string) bool {
	if d == nil || !d.populated {
		return true
	}
	switch name {
	case "jang_config.json":
		return d.jangConfig
	case "codebook_config.json":
		return d.codebookConfig
	case "tokenizer_config.json":
		return d.tokenizerConfig
	case "chat_template.jinja":
		return d.chatTemplateJinja
	case "sentence_bert_config.json":
		return d.sentenceBert
	case "modules.json":
		return d.modulesJSON
	}
	return true
}

// record marks the matching field when basename is one of the
// recognised optional-config filenames; otherwise it's a no-op.
func (d *modelPackDirIndex) record(basename string) {
	if d == nil {
		return
	}
	switch basename {
	case "jang_config.json":
		d.jangConfig = true
	case "codebook_config.json":
		d.codebookConfig = true
	case "tokenizer_config.json":
		d.tokenizerConfig = true
	case "chat_template.jinja":
		d.chatTemplateJinja = true
	case "sentence_bert_config.json":
		d.sentenceBert = true
	case "modules.json":
		d.modulesJSON = true
	}
}

func inspectModelPackWeights(pack *mp.ModelPack, resolvedPath, root string, dir *modelPackDirIndex) {
	var safetensors []string
	var ggufs []string
	switch {
	case hasASCIIInsensitiveSuffix(resolvedPath, ".safetensors"):
		safetensors = []string{resolvedPath}
	case hasASCIIInsensitiveSuffix(resolvedPath, ".gguf"):
		ggufs = []string{resolvedPath}
	default:
		// One directory walk classifies both extensions instead of two
		// passes via `*.safetensors` + `*.gguf`. filepath.Glob opens
		// the directory and readdirs every entry regardless of pattern,
		// so calling it twice doubled the syscall/alloc surface for a
		// directory that typically holds 5-10 files. The single `*`
		// pattern lets us bucket in one pass — and the basenames of
		// non-weight entries become a presence index for the four
		// optional-config probes downstream (jang_config.json,
		// codebook_config.json, tokenizer_config.json,
		// chat_template.jinja). Those four ReadFile calls cost two
		// allocs each for NotExist on the common safetensors model
		// pack; the dir index lets us skip the syscall when the file
		// can't be there.
		entries := core.PathGlob(core.PathJoin(root, "*"))
		if dir != nil {
			dir.populated = true
		}
		for _, path := range entries {
			dir.record(core.PathBase(path))
			switch {
			case hasASCIIInsensitiveSuffix(path, ".safetensors"):
				safetensors = append(safetensors, path)
			case hasASCIIInsensitiveSuffix(path, ".gguf"):
				ggufs = append(ggufs, path)
			}
		}
	}
	sort.Strings(safetensors)
	sort.Strings(ggufs)
	for _, path := range safetensors {
		if info := core.Stat(path); info.OK {
			pack.WeightBytes += uint64(info.Value.(core.FsFileInfo).Size())
		}
	}
	for _, path := range ggufs {
		if info := core.Stat(path); info.OK {
			pack.WeightBytes += uint64(info.Value.(core.FsFileInfo).Size())
		}
	}

	// safetensors / ggufs are freshly minted: PathGlob returns a new
	// filepath.Glob slice, and the single-path cases assign a fresh
	// []string{resolvedPath} above. No prior reference exists, so we
	// hand the slice straight to pack.WeightFiles without cloning.
	switch {
	case len(safetensors) > 0 && len(ggufs) > 0:
		pack.Format = mp.ModelPackFormatMixed
		merged := make([]string, 0, len(safetensors)+len(ggufs))
		merged = append(merged, safetensors...)
		merged = append(merged, ggufs...)
		pack.WeightFiles = merged
		pack.AddIssue(mp.ModelPackIssueError, mp.ModelPackIssueMixedWeightFormats, "model pack contains both safetensors and GGUF weights", root)
	case len(safetensors) > 0:
		pack.Format = mp.ModelPackFormatSafetensors
		pack.WeightFiles = safetensors
	case len(ggufs) == 1:
		pack.Format = mp.ModelPackFormatGGUF
		pack.WeightFiles = ggufs
	case len(ggufs) > 1:
		pack.Format = mp.ModelPackFormatGGUF
		pack.WeightFiles = ggufs
		pack.AddIssue(mp.ModelPackIssueError, mp.ModelPackIssueMultipleGGUF, "model pack contains multiple GGUF files; native loading expects one", root)
	default:
		pack.Format = mp.ModelPackFormatMissing
		pack.AddIssue(mp.ModelPackIssueError, mp.ModelPackIssueMissingWeights, "no .safetensors or .gguf weights found", root)
	}
}

// containsASCIIInsensitive reports whether s contains substr, treating
// A-Z and a-z as equal. substr MUST already be lowercase ASCII (the
// caller passes a fixed string literal like "normalize"). Avoids
// allocating a lowered copy of s — the substr lengths in this package
// are short (≤ 12 bytes) so the naive byte-walk is fine.
//
//	containsASCIIInsensitive("Sentence/Normalize", "normalize")  // → true
func containsASCIIInsensitive(s, substr string) bool {
	if len(substr) == 0 {
		return true
	}
	if len(s) < len(substr) {
		return false
	}
	last := len(s) - len(substr)
	for i := 0; i <= last; i++ {
		matched := true
		for j := 0; j < len(substr); j++ {
			a := s[i+j]
			if a >= 'A' && a <= 'Z' {
				a += 'a' - 'A'
			}
			if a != substr[j] {
				matched = false
				break
			}
		}
		if matched {
			return true
		}
	}
	return false
}

// hasASCIIInsensitiveSuffix reports whether s ends with suffix, treating
// A-Z and a-z as equal. Avoids allocating a lowered copy of s when the
// only thing we need is a 4-12 byte extension match.
func hasASCIIInsensitiveSuffix(s, suffix string) bool {
	if len(s) < len(suffix) {
		return false
	}
	tail := s[len(s)-len(suffix):]
	for i := 0; i < len(suffix); i++ {
		a, b := tail[i], suffix[i]
		if a >= 'A' && a <= 'Z' {
			a += 'a' - 'A'
		}
		if b >= 'A' && b <= 'Z' {
			b += 'a' - 'A'
		}
		if a != b {
			return false
		}
	}
	return true
}

func inspectModelPackGGUF(pack *mp.ModelPack, path string) {
	info, err := gguf.ReadInfo(path)
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
		pack.AddIssue(mp.ModelPackIssueError, mp.ModelPackIssueInvalidGGUF, "GGUF tensor metadata failed validation: "+gguf.ValidationSummary(info.ValidationIssues), path)
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

func inspectModelPackJANG(pack *mp.ModelPack, root string, dir *modelPackDirIndex) {
	if !dir.has("jang_config.json") {
		return
	}
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
	pack.Quantization = &gguf.QuantizationInfo{
		Type:      pack.QuantType,
		Family:    pack.QuantFamily,
		Bits:      pack.QuantBits,
		GroupSize: pack.QuantGroup,
		Mixed:     true,
	}
}

func inspectModelPackCodebook(pack *mp.ModelPack, root string, dir *modelPackDirIndex) {
	if !dir.has("codebook_config.json") {
		return
	}
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
	pack.Quantization = &gguf.QuantizationInfo{
		Type:   pack.QuantType,
		Family: pack.QuantFamily,
		Bits:   pack.QuantBits,
		Mixed:  true,
	}
	pack.AddIssue(mp.ModelPackIssueError, mp.ModelPackIssueUnsupportedCodebook, "codebook/VQ tensor matvec is available, but full codebook-quantized model loading is not implemented yet", core.PathJoin(root, "codebook_config.json"))
}

func cloneGGUFQuantizationInfo(info gguf.QuantizationInfo) *gguf.QuantizationInfo {
	if info.Type == "" && info.Family == "" && info.Bits == 0 && len(info.TensorTypes) == 0 {
		return nil
	}
	cloned := info
	cloned.TensorTypes = core.SliceClone(info.TensorTypes)
	return &cloned
}

func inspectModelPackTokenizer(pack *mp.ModelPack, root string) {
	tokenizerPath := core.PathJoin(root, "tokenizer.json")
	// Single I/O round-trip: ReadFile already surfaces a stat-shaped
	// "does not exist" via core.IsNotExist, so the prior explicit Stat
	// was a duplicate syscall (and a duplicate Result alloc) on every
	// Inspect.
	read := core.ReadFile(tokenizerPath)
	if !read.OK {
		err := read.Value.(error)
		if core.IsNotExist(err) {
			pack.AddIssue(mp.ModelPackIssueError, mp.ModelPackIssueMissingTokenizer, "tokenizer.json is required", tokenizerPath)
			return
		}
		pack.AddIssue(mp.ModelPackIssueError, mp.ModelPackIssueInvalidTokenizer, err.Error(), tokenizerPath)
		return
	}
	// We only need to confirm tokenizer.json parses; the contents
	// aren't read here. Unmarshalling into an empty struct skips
	// allocating a map[string]any tree for a multi-MB tokenizer.
	var probe struct{}
	if result := core.JSONUnmarshal(read.Value.([]byte), &probe); !result.OK {
		pack.AddIssue(mp.ModelPackIssueError, mp.ModelPackIssueInvalidTokenizer, result.Value.(error).Error(), tokenizerPath)
		return
	}
	pack.TokenizerPath = tokenizerPath
	pack.HasTokenizer = true
}

func inspectModelPackChatTemplate(pack *mp.ModelPack, root string, cfg mp.ModelPackConfig, dir *modelPackDirIndex) {
	if dir.has("tokenizer_config.json") {
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
	}

	if dir.has("chat_template.jinja") {
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
	}

	// inspectModelPackArchitecture has already resolved
	// pack.ArchitectureProfile when the architecture is known; consult
	// it directly so we don't re-enter profile.LookupArchitectureProfile
	// once for the native template and again for the requires-template
	// predicate.
	archProfile := pack.ArchitectureProfile
	if archProfile != nil && archProfile.ChatTemplate != "" {
		pack.ChatTemplate = archProfile.ChatTemplate
		pack.ChatTemplateSource = mp.ModelPackChatTemplateNative
		pack.HasChatTemplate = true
		return
	}
	requiresTemplate := true
	if archProfile != nil {
		requiresTemplate = archProfile.RequiresChatTemplate
	}
	if !requiresTemplate {
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
	// chat_template is usually a single Jinja string but can also be a
	// list of {name, template} dicts. Defer the decode via RawMessage
	// so we don't pay the any-decoding cost — the common path is a
	// single string which only needs a string-unmarshal afterwards.
	var config struct {
		ChatTemplate core.RawMessage `json:"chat_template"`
	}
	if result := core.JSONUnmarshal(read.Value.([]byte), &config); !result.OK {
		return "", false, result.Value.(error)
	}
	raw := config.ChatTemplate
	if len(raw) == 0 || core.AsString(raw) == "null" {
		return "", false, nil
	}
	switch raw[0] {
	case '"':
		var template string
		if result := core.JSONUnmarshal(raw, &template); !result.OK {
			return "", false, result.Value.(error)
		}
		template = core.Trim(template)
		return template, template != "", nil
	case '[':
		// Non-empty arrays start with '[' followed by something other
		// than ']'. The whitespace shapes JSON allows are space/tab/
		// newline/carriage-return per RFC 8259.
		for i := 1; i < len(raw); i++ {
			c := raw[i]
			if c == ' ' || c == '\t' || c == '\n' || c == '\r' {
				continue
			}
			if c == ']' {
				return "", false, nil
			}
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
	template := core.Trim(core.AsString(read.Value.([]byte)))
	return template, template != "", nil
}

func inspectModelPackArchitecture(pack *mp.ModelPack) {
	if pack.Architecture == "" {
		pack.AddIssue(mp.ModelPackIssueError, mp.ModelPackIssueMissingArchitecture, "model architecture could not be determined", pack.ConfigPath)
		return
	}
	resolved, ok := profile.LookupArchitectureProfileRef(pack.Architecture)
	if ok {
		pack.Architecture = resolved.ID
		pack.ArchitectureProfile = resolved
	}
	pack.SupportedArchitecture = ok
	if !pack.SupportedArchitecture {
		pack.AddIssue(mp.ModelPackIssueError, mp.ModelPackIssueUnsupportedArchitecture, "architecture is not supported by native go-mlx loaders: "+pack.Architecture, pack.ConfigPath)
		return
	}
	if !resolved.NativeRuntime {
		// The unsupported-runtime message specialises on the resolved
		// profile we already hold; pass it in directly so we don't
		// re-enter profile.LookupArchitectureProfile (full trim, alias
		// scan, clone) just to read the same shape.
		pack.AddIssue(mp.ModelPackIssueWarning, mp.ModelPackIssueUnsupportedRuntime, modelPackUnsupportedRuntimeMessageFor(resolved, pack.Architecture), pack.ConfigPath)
	}
}

// modelPackUnsupportedRuntimeMessage retains the lookup-by-name shape
// for external callers; in-package consumers route through
// modelPackUnsupportedRuntimeMessageFor with a profile they already
// own to skip the redundant LookupArchitectureProfile.
func modelPackUnsupportedRuntimeMessage(architecture string) string {
	if profile, ok := profile.LookupArchitectureProfileRef(architecture); ok {
		return modelPackUnsupportedRuntimeMessageFor(profile, architecture)
	}
	return "architecture is recognized, but native runtime loading is not implemented yet: " + architecture
}

func modelPackUnsupportedRuntimeMessageFor(profile *profile.ModelArchitectureProfile, architecture string) string {
	if profile != nil {
		switch {
		case profile.ID == "qwen3_6":
			return "architecture is recognized, but native hybrid linear-attention loading is not implemented yet; use mlx_lm fallback: " + architecture
		case profile.ID == "qwen3_6_moe":
			return "architecture is recognized, but native hybrid linear-attention and sparse expert loading are not implemented yet; use mlx_lm fallback: " + architecture
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

func inspectModelPackTaskProfiles(pack *mp.ModelPack, root string, dir *modelPackDirIndex) {
	if pack == nil {
		return
	}
	arch := pack.ArchitectureProfile
	if arch == nil && pack.Architecture != "" {
		if resolved, ok := profile.LookupArchitectureProfileRef(pack.Architecture); ok {
			pack.ArchitectureProfile = resolved
			arch = resolved
		}
	}
	if arch == nil {
		return
	}
	if arch.Embeddings {
		embedding := inspectModelPackEmbeddingProfile(pack, root, dir)
		pack.Embedding = &embedding
	}
	if arch.Rerank {
		rerank := inspectModelPackRerankProfile(pack, root, dir)
		pack.Rerank = &rerank
	}
	pack.Capabilities = modelPackCapabilities(pack)
}

func inspectModelPackEmbeddingProfile(pack *mp.ModelPack, root string, dir *modelPackDirIndex) mp.ModelEmbeddingProfile {
	profile := mp.ModelEmbeddingProfile{
		Dimension:         pack.HiddenSize,
		Pooling:           "cls",
		MaxSequenceLength: pack.ContextLength,
		Source:            "transformers",
	}
	if root == "" {
		return profile
	}
	if maxSeq, ok := readSentenceBertMaxSequence(root, dir); ok {
		profile.MaxSequenceLength = firstPositive(maxSeq, profile.MaxSequenceLength)
		profile.Source = "sentence-transformers"
	}
	if pooling, ok := readSentenceTransformerPooling(root); ok {
		profile.Pooling = pooling
		profile.Source = "sentence-transformers"
	}
	if normalize, ok := readSentenceTransformerNormalize(root, dir); ok {
		profile.Normalize = normalize
		profile.Source = "sentence-transformers"
	}
	return profile
}

func inspectModelPackRerankProfile(pack *mp.ModelPack, root string, dir *modelPackDirIndex) mp.ModelRerankProfile {
	profile := mp.ModelRerankProfile{
		Method:            "cross-encoder",
		MaxSequenceLength: pack.ContextLength,
		Source:            "transformers",
	}
	if root != "" {
		if maxSeq, ok := readSentenceBertMaxSequence(root, dir); ok {
			profile.MaxSequenceLength = firstPositive(maxSeq, profile.MaxSequenceLength)
			profile.Source = "sentence-transformers"
		}
	}
	return profile
}

func readSentenceBertMaxSequence(root string, dir *modelPackDirIndex) (int, bool) {
	if !dir.has("sentence_bert_config.json") {
		return 0, false
	}
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

func readSentenceTransformerNormalize(root string, dir *modelPackDirIndex) (bool, bool) {
	if !dir.has("modules.json") {
		return false, false
	}
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
	// Test "normalize" insensitively against Type+Path without
	// allocating a lowered copy per field. modules.json typically
	// carries 1-4 entries; the per-call Lower allocs (one per field,
	// two per row) compound on every Inspect against a
	// sentence-transformers model.
	for _, module := range modules {
		if containsASCIIInsensitive(module.Type, "normalize") || containsASCIIInsensitive(module.Path, "normalize") {
			return true, true
		}
	}
	return false, true
}

func modelPackCapabilities(pack *mp.ModelPack) []inference.Capability {
	if pack == nil {
		return nil
	}
	// Tally first so we can size the slice exactly — capabilities is
	// short (typically 0-2 entries) but the per-grow alloc pattern
	// fires for every Inspect call on a MoE or embedding model. One
	// upfront make beats up to four geometric-growth reallocations.
	hasEmbedding := pack.Embedding != nil
	hasRerank := pack.Rerank != nil
	hasMoE := pack.ArchitectureProfile != nil && pack.ArchitectureProfile.MoE
	hasCodebook := pack.Codebook != nil
	count := 0
	if hasEmbedding {
		count++
	}
	if hasRerank {
		count++
	}
	if hasMoE {
		count += 2
	}
	if hasCodebook {
		count++
	}
	if count == 0 {
		return nil
	}
	capabilities := make([]inference.Capability, 0, count)
	if hasEmbedding {
		capabilities = append(capabilities, modelPackAlgorithmCapability(inference.CapabilityEmbeddings, pack.Architecture))
	}
	if hasRerank {
		capabilities = append(capabilities, modelPackAlgorithmCapability(inference.CapabilityRerank, pack.Architecture))
	}
	if hasMoE {
		capabilities = append(capabilities,
			modelPackAlgorithmCapability(inference.CapabilityMoERouting, pack.Architecture),
			modelPackAlgorithmCapability(inference.CapabilityMoELazyExperts, pack.Architecture),
		)
	}
	if hasCodebook {
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
	if profile, ok := profile.LookupArchitectureProfileRef(architecture); ok && (profile.Embeddings || profile.Rerank) {
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
	cfg, err := m2.ParseConfig(read.Value.([]byte))
	if err != nil {
		pack.AddIssue(mp.ModelPackIssueWarning, mp.ModelPackIssueInvalidConfig, "MiniMax M2 config could not be parsed: "+err.Error(), pack.ConfigPath)
		return
	}
	plan, err := m2.BuildTensorPlan(cfg, pack.JANG)
	if err != nil {
		pack.AddIssue(mp.ModelPackIssueWarning, mp.ModelPackIssueUnsupportedRuntime, "MiniMax M2 tensor plan could not be built: "+err.Error(), pack.ConfigPath)
		return
	}
	pack.MiniMaxM2 = &plan
	if pack.Format != mp.ModelPackFormatSafetensors || len(pack.WeightFiles) == 0 {
		return
	}
	skeleton, err := m2.BuildLayerForwardSkeleton(plan, pack.WeightFiles, 0)
	if err != nil {
		pack.AddIssue(mp.ModelPackIssueWarning, mp.ModelPackIssueMiniMaxM2LayerSkeleton, "MiniMax M2 first-layer skeleton could not be validated: "+err.Error(), pack.Root)
		return
	}
	pack.MiniMaxM2LayerSkeleton = &skeleton
}

func inspectModelPackPolicy(pack *mp.ModelPack, cfg mp.ModelPackConfig) {
	if cfg.ExpectedQuantBits > 0 && pack.QuantBits != cfg.ExpectedQuantBits {
		pack.AddIssue(mp.ModelPackIssueError, mp.ModelPackIssueQuantizationMismatch,
			core.Concat("quantization is ", core.Itoa(pack.QuantBits), "-bit, expected ", core.Itoa(cfg.ExpectedQuantBits), "-bit"),
			pack.Root)
	}
	if cfg.MaxContextLength > 0 && pack.ContextLength > cfg.MaxContextLength {
		pack.AddIssue(mp.ModelPackIssueError, mp.ModelPackIssueContextTooLarge,
			core.Concat("context length ", core.Itoa(pack.ContextLength), " exceeds limit ", core.Itoa(cfg.MaxContextLength)),
			pack.Root)
	}
}

func finalizeModelPack(pack *mp.ModelPack) {
	// pack.ArchitectureProfile is populated by inspectModelPackArchitecture
	// when the architecture id is known; consult it directly so we don't
	// re-enter profile.LookupArchitectureProfile twice per finalize.
	requiresChat := true
	nativeRuntime := false
	if pack.ArchitectureProfile != nil {
		requiresChat = pack.ArchitectureProfile.RequiresChatTemplate
		nativeRuntime = pack.ArchitectureProfile.NativeRuntime
	}
	chatOK := pack.HasChatTemplate || !requiresChat
	// HasErrorIssue scans pack.Issues for any error-severity entry —
	// cache it once so NativeLoadable + OK share one walk instead of
	// duplicating the scan for every finalize call.
	hasError := pack.HasErrorIssue()
	pack.NativeLoadable = pack.SupportedArchitecture &&
		nativeRuntime &&
		pack.ConfigPath != "" &&
		pack.HasTokenizer &&
		chatOK &&
		(pack.Format == mp.ModelPackFormatSafetensors || pack.Format == mp.ModelPackFormatGGUF) &&
		!hasError
	pack.RequiresPythonConversion = !pack.NativeLoadable
	pack.OK = !hasError
}

// SupportsArchitecture reports whether the named architecture has a known
// profile registered in dappco.re/go/mlx/profile.
//
//	if model.SupportsArchitecture("qwen3") { ... }
func SupportsArchitecture(architecture string) bool {
	_, ok := profile.LookupArchitectureProfileRef(architecture)
	return ok
}

func modelPackSupportedArchitecture(architecture string) bool {
	return SupportsArchitecture(architecture)
}

func modelPackNativeRuntimeSupported(architecture string) bool {
	profile, ok := profile.LookupArchitectureProfileRef(architecture)
	return ok && profile.NativeRuntime
}

func nativeChatTemplateName(architecture string) string {
	if profile, ok := profile.LookupArchitectureProfileRef(architecture); ok {
		return profile.ChatTemplate
	}
	return ""
}

func modelPackRequiresChatTemplate(architecture string) bool {
	profile, ok := profile.LookupArchitectureProfileRef(architecture)
	return !ok || profile.RequiresChatTemplate
}
