// SPDX-Licence-Identifier: EUPL-1.2

// Package model holds model-pack inspection and validation utilities that
// operate on local directories or GGUF files without loading weights.
package model

import (
	core "dappco.re/go"
	"dappco.re/go/inference"
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
	inspectModelPackAutoRound(&pack, root, &dir)
	inspectModelPackCodebook(&pack, root, &dir)
	inspectModelPackTokenizer(&pack, root, &dir)
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
			base := core.PathBase(path)
			dir.record(base)
			// Capture the first sentence-transformers pooling subdir
			// (e.g. "1_Pooling") so readSentenceTransformerPooling can
			// read its config.json directly rather than re-globbing the
			// tree. The weight glob already listed it; the suffix match
			// is the same shape the rest of this file uses for
			// extensions.
			if dir != nil && dir.poolingDir == "" && hasASCIIInsensitiveSuffix(base, "_Pooling") {
				dir.poolingDir = base
			}
			switch {
			case hasASCIIInsensitiveSuffix(path, ".safetensors"):
				safetensors = append(safetensors, path)
			case hasASCIIInsensitiveSuffix(path, ".gguf"):
				ggufs = append(ggufs, path)
			}
		}
	}
	// PathGlob returns lexically sorted results (filepath.Glob spec),
	// and the single-file entry paths above each hand us a 1-element
	// slice. Bucketing preserves the sorted order so the explicit
	// sort.Strings calls were redundant — drop them to skip the
	// pdqsort interface boxing on every Inspect.
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
	pack.NumKVHeads = firstPositive(pack.NumKVHeads, config.numKeyValueHeads())
	pack.HeadDim = firstPositive(pack.HeadDim, config.headDim())
	pack.VocabSize = firstPositive(pack.VocabSize, config.vocabSize())
}

func inspectModelPackTokenizer(pack *mp.ModelPack, root string, dir *modelPackDirIndex) {
	tokenizerPath := core.PathJoin(root, "tokenizer.json")
	// Skip the syscall + Result alloc when the directory listing the
	// weight inspector already gathered shows no tokenizer.json — the
	// MissingTokenizer issue path is the same shape either way, just
	// without an open()-returns-ENOENT round trip on every Inspect of
	// a weights-only or partial-download model pack.
	if !dir.has("tokenizer.json") {
		pack.AddIssue(mp.ModelPackIssueError, mp.ModelPackIssueMissingTokenizer, "tokenizer.json is required", tokenizerPath)
		return
	}
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
	// aren't read here. Unmarshalling into an empty struct already
	// skipped the map[string]any tree, but the stdlib still built a
	// fresh per-call scanner + decode state to fill zero fields — pure
	// overhead on every Inspect. tokenizerJSONValidObject reproduces the
	// exact json.Unmarshal(&struct{}{}) accept/reject boundary (validity
	// + top-level object-or-null) with zero allocations via the pooled
	// core.JSONValid scan. The issue carries a static message rather than
	// the stdlib SyntaxError text — the accept/reject + InvalidTokenizer
	// issue-code boundary is unchanged, matching the package's existing
	// parseConfigProbeStrict precedent.
	if !tokenizerJSONValidObject(read.Value.([]byte)) {
		pack.AddIssue(mp.ModelPackIssueError, mp.ModelPackIssueInvalidTokenizer, "tokenizer.json is not valid JSON", tokenizerPath)
		return
	}
	pack.TokenizerPath = tokenizerPath
	pack.HasTokenizer = true
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

func modelPackUnsupportedRuntimeMessageFor(profile *profile.ModelArchitectureProfile, architecture string) string {
	if profile != nil {
		switch {
		case profile.ID == "gemma4_assistant":
			return "Gemma 4 assistant is an attached MTP drafter; use LoadSpeculativePair or LoadGemma4AssistantPair with a Gemma 4 target: " + architecture
		case profile.ID == "qwen3_6":
			return "architecture is recognized, but native hybrid linear-attention loading is not implemented yet: " + architecture
		case profile.ID == "qwen3_6_moe":
			return "architecture is recognized, but native hybrid linear-attention and sparse expert loading are not implemented yet: " + architecture
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
	hasAutoRound := pack.AutoRound != nil
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
	if hasAutoRound {
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
	if hasAutoRound {
		capabilities = append(capabilities, modelPackAlgorithmCapability(inference.CapabilityQuantization, pack.Architecture))
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
