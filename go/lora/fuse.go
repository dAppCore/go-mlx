// SPDX-Licence-Identifier: EUPL-1.2

package lora

import (
	"context"
	core "dappco.re/go"
	"dappco.re/go/mlx/pack"
	"dappco.re/go/mlx/pkg/metal"
	"dappco.re/go/mlx/profile"
	"slices"
	"strings"
)

const (
	// FuseProvenanceFile is the basename written into fused model packs.
	FuseProvenanceFile = "adapter_provenance.json"
	fuseOutputWeights  = "model.safetensors"
)

// Sentinel errors returned by fuse validation and orchestration paths.
// Hoisted to package vars so each guard returns the shared instance
// instead of allocating a fresh *core.Err per call — relevant both for
// the always-fired validation guards in prepareFuse and the per-fuse
// integrity checks downstream.
var (
	errFuseSourceRootRequired   = core.NewError("mlx: source pack root is required")
	errFuseAdapterPathRequired  = core.NewError("mlx: LoRA adapter path is required")
	errFuseOutputPathRequired   = core.NewError("mlx: fused model output path is required")
	errFuseOutputNotPackDir     = core.NewError("mlx: fused output path must be a model-pack directory")
	errFuseRequiresSafetensors  = core.NewError("mlx: LoRA pack fusion currently requires safetensors base weights")
	errFuseRankRequired         = core.NewError("mlx: LoRA adapter rank is required for fusion")
	errFuseScaleRequired        = core.NewError("mlx: LoRA adapter scale is required for fusion")
	errFuseOutputSameAsSource   = core.NewError("mlx: fused output path must differ from source model path")
	errFuseOutputContainsWeight = core.NewError("mlx: fused output path already contains model weights")
	errFuseNoAdapterSafetensors = core.NewError("mlx: no adapter safetensors found")
	errFuseNoLoRATensorPairs    = core.NewError("mlx: no LoRA tensor pairs found")
	errFuseNoBaseWeightFiles    = core.NewError("mlx: no base weight files available for LoRA fusion")
)

// FuseOptions configures pack-level LoRA fusion.
//
// SourcePack must be a validated, safetensors-format model pack; callers
// validate via mlx.ValidateModelPack before invoking lora.FuseIntoPack.
// Splitting validation out of the lora package keeps lora free of the
// mlx-root cycle.
type FuseOptions struct {
	SourcePack  pack.ModelPack    `json:"source_pack"`
	AdapterPath string            `json:"adapter_path"`
	OutputPath  string            `json:"output_path"`
	Labels      map[string]string `json:"labels,omitempty"`
}

// FuseResult reports the paths and identity of a fused model pack.
//
// Callers re-validate the output via mlx.ValidateModelPack(OutputPath)
// when they need the populated pack.ModelPack for downstream use.
type FuseResult struct {
	OutputPath      string      `json:"output_path"`
	WeightPath      string      `json:"weight_path"`
	WeightFiles     []string    `json:"weight_files,omitempty"`
	ProvenancePath  string      `json:"provenance_path"`
	Adapter         AdapterInfo `json:"adapter"`
	FusedWeights    int         `json:"fused_weights"`
	FusedWeightKeys []string    `json:"fused_weight_keys,omitempty"`
}

// FuseProvenance records how a fused pack was produced. Written into
// adapter_provenance.json next to the fused weights.
type FuseProvenance struct {
	Version         int               `json:"version"`
	SourceModel     pack.ModelPack    `json:"source_model"`
	Adapter         AdapterInfo       `json:"adapter"`
	OutputWeight    string            `json:"output_weight"`
	OutputWeights   []string          `json:"output_weights,omitempty"`
	FusedWeightKeys []string          `json:"fused_weight_keys"`
	Labels          map[string]string `json:"labels,omitempty"`
}

type fusePrepared struct {
	Model   pack.ModelPack
	Adapter AdapterInfo
	Output  string
}

func prepareFuse(ctx context.Context, opts FuseOptions) (fusePrepared, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return fusePrepared{}, err
	}
	if opts.SourcePack.Root == "" {
		return fusePrepared{}, errFuseSourceRootRequired
	}
	if opts.AdapterPath == "" {
		return fusePrepared{}, errFuseAdapterPathRequired
	}
	if opts.OutputPath == "" {
		return fusePrepared{}, errFuseOutputPathRequired
	}
	// Case-fold only the trailing suffix bytes for the .safetensors /
	// .gguf shape check — the previous form called core.Lower on the
	// full output path twice (once each via HasSuffix on the lowered
	// copy), allocating whenever the path contained uppercase ASCII
	// anywhere (most paths do — tmp dirs, app bundles, drive letters).
	// hasSafetensorsSuffixFold + hasGgufSuffixFold scan only the last
	// 12/5 bytes, never alloc, and short-circuit on length mismatch.
	if hasSafetensorsSuffixFold(opts.OutputPath) || hasGgufSuffixFold(opts.OutputPath) {
		return fusePrepared{}, errFuseOutputNotPackDir
	}
	if opts.SourcePack.Format != pack.ModelPackFormatSafetensors {
		return fusePrepared{}, errFuseRequiresSafetensors
	}

	adapter, err := Inspect(opts.AdapterPath, opts.AdapterPath)
	if err != nil {
		return fusePrepared{}, core.E("lora.FuseIntoPack", "inspect LoRA adapter", err)
	}
	if adapter.Rank <= 0 {
		return fusePrepared{}, errFuseRankRequired
	}
	if adapter.Scale == 0 && adapter.Alpha == 0 {
		adapter.Alpha = float32(adapter.Rank) * 2
		adapter.Scale = adapter.Alpha / float32(adapter.Rank)
	}
	if adapter.Scale == 0 {
		return fusePrepared{}, errFuseScaleRequired
	}

	output := opts.OutputPath
	if abs := core.PathAbs(output); abs.OK {
		output = abs.Value.(string)
	}
	if samePath(opts.SourcePack.Root, output) {
		return fusePrepared{}, errFuseOutputSameAsSource
	}
	if err := ensureEmptyFuseWeightDestination(output); err != nil {
		return fusePrepared{}, err
	}
	if result := core.MkdirAll(output, 0o755); !result.OK {
		return fusePrepared{}, core.E("lora.FuseIntoPack", "create fused model directory", resultError(result))
	}
	if err := copyModelPackMetadata(opts.SourcePack.Root, output); err != nil {
		return fusePrepared{}, err
	}

	return fusePrepared{
		Model:   opts.SourcePack,
		Adapter: adapter,
		Output:  output,
	}, nil
}

func ensureEmptyFuseWeightDestination(output string) error {
	if stat := core.Stat(output); !stat.OK {
		if core.IsNotExist(stat.Value.(error)) {
			return nil
		}
		return core.E("lora.FuseIntoPack", "inspect output path", resultError(stat))
	}
	// Probe each weight pattern independently and short-circuit on the
	// first non-empty match. The previous form appended both glob results
	// into a fresh slice unconditionally, paying for the second glob +
	// the concat alloc even when the first run already proved the
	// destination is dirty. Real fuse paths fire this once per call;
	// shaving the second glob's Readdir trip is the win.
	//
	// Build the glob pattern with a direct concat instead of core.PathJoin
	// (filepath.Join → filepath.Clean), which always allocates an internal
	// lazybuf even when the inputs are already canonical. output came from
	// PathAbs + MkdirAll so it's clean by construction.
	if len(core.PathGlob(joinDirChildPattern(output, "*.safetensors"))) > 0 {
		return errFuseOutputContainsWeight
	}
	if len(core.PathGlob(joinDirChildPattern(output, "*.gguf"))) > 0 {
		return errFuseOutputContainsWeight
	}
	return nil
}

func samePath(a, b string) bool {
	// Fast path: identical strings cannot resolve to different absolutes,
	// so skip the two PathAbs round-trips when the raw inputs already
	// match. The fuse-self-fuse guard in prepareFuse fires this once per
	// call and the SameAbsolute bench covers the equality path.
	if a == b {
		return true
	}
	// Both inputs already absolute + canonical short-circuit. PathAbs
	// calls filepath.Abs which calls filepath.Clean — Clean allocates a
	// fresh byte buffer even when no cleaning is needed (the routine
	// always builds a "lazybuf" working buffer). When both inputs look
	// canonical (start with '/', no double-slashes, no ".." or "." path
	// segments, no trailing '/'), their absolute forms equal themselves,
	// and string inequality already proves they differ. The fuse
	// DistinctRelative bench covers this exact shape and the previous
	// path paid for two filepath.Abs+Clean trips returning fresh strings
	// only to compare them — two allocs / call.
	if isCleanAbsolute(a) && isCleanAbsolute(b) {
		return false
	}
	absA := a
	if resolved := core.PathAbs(a); resolved.OK {
		absA = resolved.Value.(string)
	}
	absB := b
	if resolved := core.PathAbs(b); resolved.OK {
		absB = resolved.Value.(string)
	}
	return absA == absB
}

// isCleanAbsolute reports whether p is a Unix absolute path with no
// segments that require filepath.Clean to canonicalise — no //,
// no /./ or trailing /., no /../ or trailing /.., and no trailing /.
// Matches the canonical-form invariant filepath.Clean produces.
func isCleanAbsolute(p string) bool {
	if len(p) == 0 || p[0] != '/' {
		return false
	}
	if len(p) > 1 && p[len(p)-1] == '/' {
		return false
	}
	for i := 0; i < len(p); i++ {
		if p[i] != '/' {
			continue
		}
		// Probe the segment that follows this '/'.
		switch {
		case i+1 < len(p) && p[i+1] == '/':
			return false
		case i+1 == len(p)-1 && p[i+1] == '.':
			return false
		case i+1 < len(p)-1 && p[i+1] == '.' && p[i+2] == '/':
			return false
		case i+2 == len(p)-1 && p[i+1] == '.' && p[i+2] == '.':
			return false
		case i+2 < len(p)-1 && p[i+1] == '.' && p[i+2] == '.' && p[i+3] == '/':
			return false
		}
	}
	return true
}

func copyModelPackMetadata(sourceRoot, outputRoot string) error {
	patterns := [...]string{"*.json", "*.model", "*.txt"}
	// Real qwen3 packs ship 6-8 metadata files, gemma4 closer to 10;
	// presize the dedup set so the dominant first-pattern fill avoids
	// the runtime map-growth cycle. Switch the patterns slice literal to
	// a fixed-size array so the loop iterates without the throwaway
	// per-call slice-header alloc.
	seen := make(map[string]struct{}, 12)
	for _, pattern := range patterns {
		// joinDirChildPattern skips the filepath.Clean trip core.PathJoin
		// would take — sourceRoot and outputRoot are already-canonical
		// directory paths (PathAbs + MkdirAll output), so the only
		// normalisation needed is the trailing-slash collapse rule.
		// Per-pattern + per-file path joins were ~30% of the metadata-
		// copy alloc count for a typical 8-file qwen3 metadata set.
		for _, sourcePath := range core.PathGlob(joinDirChildPattern(sourceRoot, pattern)) {
			name := core.PathBase(sourcePath)
			if _, ok := seen[name]; ok {
				continue
			}
			seen[name] = struct{}{}
			if isModelWeightMetadataCopySkip(name) {
				continue
			}
			if err := copyLocalFile(sourcePath, joinDirChildPattern(outputRoot, name)); err != nil {
				return err
			}
		}
	}
	return nil
}

func isModelWeightMetadataCopySkip(name string) bool {
	// Contains(".safetensors") is a strict superset of HasSuffix(".safetensors"):
	// any name ending in .safetensors necessarily contains the substring. The
	// previous HasSuffix terms were dead under the OR — drop them and let the
	// Contains checks carry both the suffix and the .safetensors.index.json
	// case the copy filter is meant to skip.
	//
	// Use case-fold-in-place compares (containsAsciiLowerFold +
	// strings.EqualFold) to avoid the core.Lower copy that fires whenever
	// the input contains uppercase ASCII (e.g. MODEL.GGUF). core.Lower
	// drops to strings.ToLower for uppercase input, which allocates a fresh
	// string per call — wasted on the dominant lowercase tokenizer/config
	// files we copy because we only need to compare, not normalise.
	if strings.EqualFold(name, FuseProvenanceFile) {
		return true
	}
	if containsAsciiLowerFold(name, ".safetensors") {
		return true
	}
	if containsAsciiLowerFold(name, ".gguf") {
		return true
	}
	return false
}

// containsAsciiLowerFold reports whether s contains sub, comparing
// ASCII A-Z in s case-insensitively against the all-lowercase sub.
// The caller MUST pass sub already in lowercase ASCII — this keeps the
// per-byte fold to one branch (s only) and skips the alloc strings.Lower
// would make for uppercase input.
func containsAsciiLowerFold(s, sub string) bool {
	n := len(s) - len(sub)
	if n < 0 {
		return false
	}
	for i := 0; i <= n; i++ {
		match := true
		for j := 0; j < len(sub); j++ {
			c := s[i+j]
			if c >= 'A' && c <= 'Z' {
				c += 'a' - 'A'
			}
			if c != sub[j] {
				match = false
				break
			}
		}
		if match {
			return true
		}
	}
	return false
}

func copyLocalFile(sourcePath, destinationPath string) error {
	read := core.ReadFile(sourcePath)
	if !read.OK {
		return core.E("lora.FuseIntoPack", "read "+sourcePath, resultError(read))
	}
	if result := core.WriteFile(destinationPath, read.Value.([]byte), 0o644); !result.OK {
		return core.E("lora.FuseIntoPack", "write "+destinationPath, resultError(result))
	}
	return nil
}

func fuseAdapterWeightFiles(path string) ([]string, error) {
	// HasSuffix on the lowered path allocates whenever the temp-dir or
	// caller path contains uppercase ASCII (every macOS bench tempdir
	// hits this — the bench reported 2 allocs for the single-file
	// path, one of which was core.Lower's case-fold copy). Case-fold
	// only the trailing 12 bytes that form the suffix candidate — that
	// covers the .Safetensors / .SAFETENSORS variants the previous
	// code admitted without paying for a full-path scan + alloc.
	if hasSafetensorsSuffixFold(path) {
		return []string{path}, nil
	}
	// joinDirChildPattern (direct concat) skips the filepath.Clean trip
	// core.PathJoin would take — path is the adapter directory the caller
	// passed in, treated as already-canonical (Inspect feeds the same
	// path through the directory branch without normalisation).
	matches := core.PathGlob(joinDirChildPattern(path, "*.safetensors"))
	slices.Sort(matches)
	if len(matches) == 0 {
		return nil, errFuseNoAdapterSafetensors
	}
	return matches, nil
}

// hasSafetensorsSuffixFold case-folds only the trailing 12-byte
// .safetensors candidate window, so paths with uppercase elsewhere
// (e.g. macOS /private/var/folders/.../T/... tempdirs) don't trigger
// a full-path Lower copy. Mirrors core.HasSuffix's semantics for the
// .safetensors / .Safetensors / .SAFETENSORS triple.
const safetensorsSuffix = ".safetensors"

func hasSafetensorsSuffixFold(path string) bool {
	if len(path) < len(safetensorsSuffix) {
		return false
	}
	tail := path[len(path)-len(safetensorsSuffix):]
	for i := range len(safetensorsSuffix) {
		c := tail[i]
		if c >= 'A' && c <= 'Z' {
			c += 'a' - 'A'
		}
		if c != safetensorsSuffix[i] {
			return false
		}
	}
	return true
}

// hasGgufSuffixFold mirrors hasSafetensorsSuffixFold for the .gguf
// 5-byte tail check used by prepareFuse to reject output paths that
// point at a weight file instead of a pack directory.
const ggufSuffix = ".gguf"

func hasGgufSuffixFold(path string) bool {
	if len(path) < len(ggufSuffix) {
		return false
	}
	tail := path[len(path)-len(ggufSuffix):]
	for i := range len(ggufSuffix) {
		c := tail[i]
		if c >= 'A' && c <= 'Z' {
			c += 'a' - 'A'
		}
		if c != ggufSuffix[i] {
			return false
		}
	}
	return true
}

func fusePairName(weightName string) (string, string, bool) {
	// The 8-variant table splits cleanly along ".weight"-tail: 4 variants
	// end in ".weight" (so the second-to-last segment is ".lora_X"), and
	// 4 are bare ".lora_X" tails. Probe the .weight tail once to halve
	// the candidate set, then dispatch on the kind byte ('a','A','b','B').
	// Worst case drops from 8 HasSuffix scans (the non-LoRA miss hit ~22ns)
	// to one HasSuffix + one byte read + one TrimSuffix. The kind byte
	// is the byte immediately preceding the chosen tail.
	if core.HasSuffix(weightName, ".weight") {
		// Layout: ...lora_<X>.weight — kind byte at len-8 ('.weight' is
		// 7 chars, the byte before that is the X).
		head := len(weightName) - len(".lora_X.weight")
		if head < 0 {
			return "", "", false
		}
		if weightName[head:head+6] != ".lora_" {
			return "", "", false
		}
		switch weightName[head+6] {
		case 'a', 'A':
			return weightName[:head], "a", true
		case 'b', 'B':
			return weightName[:head], "b", true
		}
		return "", "", false
	}
	// Bare ".lora_X" tail.
	head := len(weightName) - len(".lora_X")
	if head < 0 {
		return "", "", false
	}
	if weightName[head:head+6] != ".lora_" {
		return "", "", false
	}
	switch weightName[head+6] {
	case 'a', 'A':
		return weightName[:head], "a", true
	case 'b', 'B':
		return weightName[:head], "b", true
	}
	return "", "", false
}

func fuseBaseWeightKey(pairName string) string {
	return pairName + ".weight"
}

func fuseBaseWeightKeyForArchitecture(pairName string, architecture string) string {
	if fuseGemma4Architecture(architecture) {
		if canonical, ok := fuseGemma4PairName(pairName); ok {
			return canonical + ".weight"
		}
	}
	return fuseBaseWeightKey(pairName)
}

type fuseBaseWeightMatch struct {
	Key          string
	CanonicalKey string
	Quantized    bool
	ScaleKey     string
	BiasesKey    string
	SidecarKeys  []string
}

func fuseBaseWeightIndexForArchitecture(baseWeights map[string]*metal.Array, architecture string) map[string]fuseBaseWeightMatch {
	if !fuseGemma4Architecture(architecture) {
		return nil
	}
	keys := make([]string, 0, len(baseWeights))
	for key := range baseWeights {
		keys = append(keys, key)
	}
	slices.Sort(keys)

	index := make(map[string]fuseBaseWeightMatch, len(keys))
	for _, key := range keys {
		if baseWeights[key] == nil {
			continue
		}
		canonical, ok := metal.Gemma4CanonicalWeightName(key)
		if !ok || !core.HasSuffix(canonical, ".weight") {
			continue
		}
		existing, exists := index[canonical]
		if !exists || key == canonical || (existing.Key != canonical && key < existing.Key) {
			index[canonical] = fuseBaseWeightMatch{
				Key:          key,
				CanonicalKey: canonical,
			}
		}
	}
	for canonical, match := range index {
		match.ScaleKey, match.BiasesKey, match.SidecarKeys = fuseBaseWeightSidecars(baseWeights, match.Key, canonical)
		match.Quantized = match.ScaleKey != ""
		index[canonical] = match
	}
	return index
}

func fuseBaseWeightMatchForArchitecture(baseWeights map[string]*metal.Array, baseIndex map[string]fuseBaseWeightMatch, pairName string, architecture string) (fuseBaseWeightMatch, bool) {
	baseKey := fuseBaseWeightKeyForArchitecture(pairName, architecture)
	if match, ok := baseIndex[baseKey]; ok {
		return match, true
	}
	if baseWeights[baseKey] == nil {
		return fuseBaseWeightMatch{}, false
	}
	scaleKey, biasesKey, sidecarKeys := fuseBaseWeightSidecars(baseWeights, baseKey, baseKey)
	return fuseBaseWeightMatch{
		Key:          baseKey,
		CanonicalKey: baseKey,
		Quantized:    scaleKey != "",
		ScaleKey:     scaleKey,
		BiasesKey:    biasesKey,
		SidecarKeys:  sidecarKeys,
	}, true
}

func fuseBaseWeightSidecars(baseWeights map[string]*metal.Array, key string, canonical string) (string, string, []string) {
	var scaleKey string
	var biasKey string
	var sidecarKeys []string
	prefixes := make([]string, 0, 2)
	if prefix, ok := fuseBaseWeightPrefix(key); ok {
		prefixes = append(prefixes, prefix)
	}
	if canonical != key {
		if prefix, ok := fuseBaseWeightPrefix(canonical); ok {
			prefixes = append(prefixes, prefix)
		}
	}
	for i, prefix := range prefixes {
		duplicate := false
		for _, previous := range prefixes[:i] {
			if previous == prefix {
				duplicate = true
				break
			}
		}
		if duplicate {
			continue
		}
		scalesKey := prefix + ".scales"
		if _, ok := baseWeights[scalesKey]; ok {
			if scaleKey == "" {
				scaleKey = scalesKey
			}
			sidecarKeys = append(sidecarKeys, scalesKey)
		}
		biasesKey := prefix + ".biases"
		if _, ok := baseWeights[biasesKey]; ok {
			if biasKey == "" {
				biasKey = biasesKey
			}
			sidecarKeys = append(sidecarKeys, biasesKey)
		}
	}
	return scaleKey, biasKey, sidecarKeys
}

func fuseBaseWeightPrefix(key string) (string, bool) {
	if !core.HasSuffix(key, ".weight") {
		return "", false
	}
	return core.TrimSuffix(key, ".weight"), true
}

func fuseQuantizedTargetMetadata(model pack.ModelPack, match fuseBaseWeightMatch) (int, int, string, error) {
	groupSize := model.QuantGroup
	bits := model.QuantBits
	if groupSize <= 0 || bits <= 0 {
		return 0, 0, "", fuseQuantizedBaseTargetMetadataError(match)
	}
	return groupSize, bits, metal.NormalizeQuantizationMode(model.QuantType), nil
}

func fuseQuantizedBaseTargetMetadataError(match fuseBaseWeightMatch) error {
	message := "mlx: LoRA pack fusion cannot dequantize base target without quantization metadata: " + match.Key
	if match.CanonicalKey != "" && match.CanonicalKey != match.Key {
		message += " (canonical " + match.CanonicalKey + ")"
	}
	return core.NewError(message)
}

func fuseGemma4Architecture(architecture string) bool {
	switch profile.ArchitectureID(architecture) {
	case "gemma4", "gemma4_text", "gemma4_unified":
		return true
	default:
		return false
	}
}

func fuseGemma4PairName(pairName string) (string, bool) {
	if pairName == "" {
		return "", false
	}
	parts := core.Split(pairName, ".")
	if len(parts) >= 2 {
		target := parts[len(parts)-2] + "." + parts[len(parts)-1]
		if canonical, ok := metal.Gemma4LoRATargetPath(target); ok {
			return fuseJoinCanonicalTarget(parts[:len(parts)-2], canonical), true
		}
	}
	if canonical, ok := metal.Gemma4LoRATargetPath(parts[len(parts)-1]); ok {
		return fuseJoinCanonicalTarget(parts[:len(parts)-1], canonical), true
	}
	return "", false
}

func fuseJoinCanonicalTarget(prefix []string, canonical string) string {
	if len(prefix) == 0 {
		return canonical
	}
	target := core.Split(canonical, ".")
	parts := make([]string, 0, len(prefix)+len(target))
	parts = append(parts, prefix...)
	parts = append(parts, target...)
	return core.Join(".", parts...)
}

func writeFuseProvenance(path string, provenance FuseProvenance) error {
	slices.Sort(provenance.FusedWeightKeys)
	data := core.JSONMarshal(provenance)
	if !data.OK {
		return core.E("lora.FuseIntoPack", "marshal adapter provenance", resultError(data))
	}
	if result := core.WriteFile(path, data.Value.([]byte), 0o644); !result.OK {
		return core.E("lora.FuseIntoPack", "write adapter provenance", resultError(result))
	}
	return nil
}

type fusePair struct {
	MatrixA *metal.Array
	MatrixB *metal.Array
}

// FuseIntoPack merges a LoRA adapter into dense safetensors base weights
// and writes a go-mlx-loadable model pack. Callers validate
// opts.SourcePack with mlx.ValidateModelPack before invoking, and
// validate the OutputPath after the call returns.
//
//	src, err := mlx.ValidateModelPack(path)
//	res, err := lora.FuseIntoPack(ctx, lora.FuseOptions{SourcePack: src, AdapterPath: a, OutputPath: o})
//	out, err := mlx.ValidateModelPack(res.OutputPath)
func FuseIntoPack(ctx context.Context, opts FuseOptions) (*FuseResult, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	prepared, err := prepareFuse(ctx, opts)
	if err != nil {
		return nil, err
	}

	adapterWeights, err := loadFuseAdapterWeights(opts.AdapterPath)
	if err != nil {
		return nil, err
	}
	defer freeMetalMap(adapterWeights)

	pairs, err := buildFusePairs(adapterWeights)
	if err != nil {
		return nil, err
	}

	weightFiles, fusedKeys, err := fuseModelWeightFiles(ctx, prepared.Model.WeightFiles, prepared.Output, pairs, prepared.Adapter.Scale, prepared.Model)
	if err != nil {
		return nil, err
	}

	// prepared.Output is canonical (PathAbs + MkdirAll); skip the
	// filepath.Clean trip core.PathJoin would take and concat directly.
	provenancePath := joinDirChildPattern(prepared.Output, FuseProvenanceFile)
	// outputWeightFileNames maps PathBase across every weight shard; the
	// first basename is also written into the provenance OutputWeight
	// scalar. Build the slice once and reuse its first entry instead of
	// running core.PathBase a second time on weightFiles[0].
	outputWeightNames := outputWeightFileNames(weightFiles)
	if err := writeFuseProvenance(provenancePath, FuseProvenance{
		Version:         1,
		SourceModel:     prepared.Model,
		Adapter:         prepared.Adapter,
		OutputWeight:    outputWeightNames[0],
		OutputWeights:   outputWeightNames,
		FusedWeightKeys: fusedKeys,
		Labels:          opts.Labels,
	}); err != nil {
		return nil, err
	}

	return &FuseResult{
		OutputPath:      prepared.Output,
		WeightPath:      weightFiles[0],
		WeightFiles:     weightFiles,
		ProvenancePath:  provenancePath,
		Adapter:         prepared.Adapter,
		FusedWeights:    len(fusedKeys),
		FusedWeightKeys: fusedKeys,
	}, nil
}

func loadFuseAdapterWeights(path string) (map[string]*metal.Array, error) {
	paths, err := fuseAdapterWeightFiles(path)
	if err != nil {
		return nil, err
	}
	weights := make(map[string]*metal.Array)
	for _, path := range paths {
		loaded, err := metal.LoadAllSafetensors(path)
		if err != nil {
			freeMetalMap(weights)
			return nil, core.E("lora.FuseIntoPack", "load adapter weights "+core.PathBase(path), err)
		}
		for name, tensor := range loaded {
			if previous := weights[name]; previous != nil {
				metal.Free(previous)
			}
			weights[name] = tensor
		}
	}
	return weights, nil
}

func buildFusePairs(weights map[string]*metal.Array) (map[string]fusePair, error) {
	// Each fusePair binds exactly one lora_a + one lora_b tensor, so the
	// final map size is at most len(weights)/2; presize to that ceiling
	// to skip the runtime map-growth cycles a default-sized map would
	// take while filling. Real qwen3 fuses populate 200-400 entries.
	pairs := make(map[string]fusePair, len(weights)/2)
	for name, tensor := range weights {
		pairName, suffix, ok := fusePairName(name)
		if !ok {
			continue
		}
		pair := pairs[pairName]
		switch suffix {
		case "a":
			pair.MatrixA = tensor
		case "b":
			pair.MatrixB = tensor
		}
		pairs[pairName] = pair
	}
	if len(pairs) == 0 {
		return nil, errFuseNoLoRATensorPairs
	}
	for name, pair := range pairs {
		if pair.MatrixA == nil || pair.MatrixB == nil {
			return nil, core.NewError("mlx: incomplete LoRA tensor pair: " + name)
		}
	}
	return pairs, nil
}

func fuseModelWeightFiles(ctx context.Context, sourceFiles []string, outputRoot string, pairs map[string]fusePair, scale float32, model pack.ModelPack) ([]string, []string, error) {
	if len(sourceFiles) == 0 {
		return nil, nil, errFuseNoBaseWeightFiles
	}

	// Worst-case every pair gets fused; presize to len(pairs) so
	// the dominant fill phase avoids the runtime map-growth path.
	fusedPairs := make(map[string]struct{}, len(pairs))
	weightFiles := make([]string, 0, len(sourceFiles))
	fusedKeys := make([]string, 0, len(pairs))
	// Hoist the sharded-mode decision out of the loop — len(sourceFiles)
	// is loop-invariant, so the per-iter outputName branch was reading
	// it on every shard. Single-shard fuses keep the canonical
	// fuseOutputWeights basename; multi-shard fuses preserve the
	// source-file basename for round-tripping.
	multiShard := len(sourceFiles) > 1
	for _, sourceFile := range sourceFiles {
		if err := ctx.Err(); err != nil {
			return nil, nil, err
		}
		baseWeights, err := metal.LoadAllSafetensors(sourceFile)
		if err != nil {
			return nil, nil, core.E("lora.FuseIntoPack", "load base weights "+core.PathBase(sourceFile), err)
		}

		shardFusedKeys, err := fuseWeightPairs(ctx, baseWeights, pairs, fusedPairs, scale, model)
		if err != nil {
			freeMetalMap(baseWeights)
			return nil, nil, err
		}
		fusedKeys = append(fusedKeys, shardFusedKeys...)

		outputName := fuseOutputWeights
		if multiShard {
			outputName = core.PathBase(sourceFile)
		}
		// outputRoot is canonical (PathAbs + MkdirAll); skip the
		// filepath.Clean trip and concat directly.
		weightPath := joinDirChildPattern(outputRoot, outputName)
		if err := metal.SaveSafetensors(weightPath, baseWeights); err != nil {
			freeMetalMap(baseWeights)
			return nil, nil, core.E("lora.FuseIntoPack", "save fused safetensors", err)
		}
		freeMetalMap(baseWeights)
		weightFiles = append(weightFiles, weightPath)
	}

	for name := range pairs {
		if _, ok := fusedPairs[name]; ok {
			continue
		}
		return nil, nil, core.NewError("mlx: base weight not found for LoRA target: " + fuseBaseWeightKeyForArchitecture(name, model.Architecture))
	}
	return weightFiles, fusedKeys, nil
}

func fuseWeightPairs(ctx context.Context, baseWeights map[string]*metal.Array, pairs map[string]fusePair, fusedPairs map[string]struct{}, scale float32, model pack.ModelPack) ([]string, error) {
	names := make([]string, 0, len(pairs))
	for name := range pairs {
		names = append(names, name)
	}
	slices.Sort(names)
	baseIndex := fuseBaseWeightIndexForArchitecture(baseWeights, model.Architecture)

	fusedKeys := make([]string, 0, len(names))
	for _, name := range names {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if _, ok := fusedPairs[name]; ok {
			continue
		}
		baseMatch, ok := fuseBaseWeightMatchForArchitecture(baseWeights, baseIndex, name, model.Architecture)
		if !ok {
			continue
		}
		base := baseWeights[baseMatch.Key]

		pair := pairs[name]
		delta := metal.Matmul(pair.MatrixB, pair.MatrixA)
		scaled := metal.MulScalar(delta, scale)
		baseForFuse := base
		if baseMatch.Quantized {
			groupSize, bits, mode, err := fuseQuantizedTargetMetadata(model, baseMatch)
			if err != nil {
				metal.Free(delta, scaled)
				return nil, err
			}
			baseForFuse = metal.DequantizeMode(base, baseWeights[baseMatch.ScaleKey], baseWeights[baseMatch.BiasesKey], groupSize, bits, mode)
		}
		fused := metal.Add(baseForFuse, scaled)
		metal.Materialize(fused)
		metal.Free(delta, scaled)
		if baseForFuse != base {
			metal.Free(baseForFuse)
		}
		metal.Free(base)
		baseWeights[baseMatch.Key] = fused
		for _, sidecarKey := range baseMatch.SidecarKeys {
			if sidecar := baseWeights[sidecarKey]; sidecar != nil {
				metal.Free(sidecar)
			}
			delete(baseWeights, sidecarKey)
		}
		fusedKeys = append(fusedKeys, baseMatch.Key)
		fusedPairs[name] = struct{}{}
	}
	return fusedKeys, nil
}

func outputWeightFileNames(paths []string) []string {
	names := make([]string, 0, len(paths))
	for _, path := range paths {
		names = append(names, core.PathBase(path))
	}
	return names
}

func freeMetalMap(weights map[string]*metal.Array) {
	for _, tensor := range weights {
		metal.Free(tensor)
	}
}
