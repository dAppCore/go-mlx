// SPDX-Licence-Identifier: EUPL-1.2

package hf

import (
	"context"
	"testing"

	core "dappco.re/go"
	sharedhf "dappco.re/go/inference/hf"
	"dappco.re/go/inference/memory"
	mp "dappco.re/go/inference/modelpack"
	"dappco.re/go/inference/profile"
)

// TestFirstNonEmpty_AllEmpty_Bad covers firstNonEmpty's exhausted-loop return:
// when every candidate is empty or whitespace-only, the helper falls through
// to "". hasNonWhitespace must reject each blank entry before the final return.
func TestFirstNonEmpty_AllEmpty_Bad(t *testing.T) {
	if got := firstNonEmpty(); got != "" {
		t.Fatalf("firstNonEmpty() = %q, want empty (no candidates)", got)
	}
	if got := firstNonEmpty("", "   ", "\t\n", "\v\f\r"); got != "" {
		t.Fatalf("firstNonEmpty(all-whitespace) = %q, want empty", got)
	}
	// First non-whitespace candidate still wins (sanity on the positive arm).
	if got := firstNonEmpty("", " \t", "real"); got != "real" {
		t.Fatalf("firstNonEmpty(...real) = %q, want real", got)
	}
}

// TestFirstPositive_AllNonPositive_Bad covers firstPositive's exhausted-loop
// return: zero and negative candidates are all skipped, so the helper returns
// 0 rather than a stray non-positive value.
func TestFirstPositive_AllNonPositive_Bad(t *testing.T) {
	if got := firstPositive(); got != 0 {
		t.Fatalf("firstPositive() = %d, want 0 (no candidates)", got)
	}
	if got := firstPositive(0, -1, -99); got != 0 {
		t.Fatalf("firstPositive(all non-positive) = %d, want 0", got)
	}
	if got := firstPositive(0, -1, 7); got != 7 {
		t.Fatalf("firstPositive(...7) = %d, want 7", got)
	}
}

// TestModelConfigNormalized_TextConfigInheritsModelType_Ugly covers the
// normalized() branch where a nested text_config carries no model_type and the
// config is neither a Gemma-4 unified nor assistant shape: the outer
// model_type is copied down into the promoted text config (hf.go:277).
func TestModelConfigNormalized_TextConfigInheritsModelType_Ugly(t *testing.T) {
	cfg := ModelConfig{
		ModelType: "qwen3",
		TextConfig: &ModelConfig{
			// No ModelType, no Architectures — forces the inherit branch.
			HiddenSize:            1536,
			MaxPositionEmbeddings: 32768,
		},
	}
	if got := modelConfigArchitecture(cfg); got != "qwen3" {
		t.Fatalf("architecture() = %q, want qwen3 (outer model_type inherited into text_config)", got)
	}
	if got := modelConfigContextLength(cfg); got != 32768 {
		t.Fatalf("contextLength() = %d, want 32768 (from promoted text_config)", got)
	}
}

// TestModelConfigProbe_Architecture_Fallbacks_Ugly covers the lower tiers of
// probe.architecture that the synthetic-config tests don't all reach: the
// text_config.model_type fallback (hf.go:417), the architectures-name loop
// fallback when no model_type is present anywhere (hf.go:420), and the empty
// return when nothing resolves (hf.go:425).
func TestModelConfigProbe_Architecture_Fallbacks_Ugly(t *testing.T) {
	// Only text_config.model_type is set → normalised from there.
	dir := t.TempDir()
	writeModelPackFile(t, core.PathJoin(dir, "config.json"), `{
		"text_config": {"model_type": "qwen3"}
	}`)
	probe, err := readModelConfig(dir)
	if err != nil {
		t.Fatalf("readModelConfig() error = %v", err)
	}
	if got := probe.architecture(); got != "qwen3" {
		t.Fatalf("architecture() = %q, want qwen3 (from text_config.model_type)", got)
	}

	// No model_type anywhere, only a transformers architecture name → the
	// architectures loop resolves it.
	archDir := t.TempDir()
	writeModelPackFile(t, core.PathJoin(archDir, "config.json"), `{
		"architectures": ["LlamaForCausalLM"]
	}`)
	archProbe, err := readModelConfig(archDir)
	if err != nil {
		t.Fatalf("readModelConfig() error = %v", err)
	}
	if got := archProbe.architecture(); got != "llama" {
		t.Fatalf("architecture() = %q, want llama (from architectures name)", got)
	}

	// Nothing resolvable: no model_type, no recognised architecture name → "".
	emptyDir := t.TempDir()
	writeModelPackFile(t, core.PathJoin(emptyDir, "config.json"), `{
		"architectures": ["TotallyMadeUpForCausalLM"]
	}`)
	emptyProbe, err := readModelConfig(emptyDir)
	if err != nil {
		t.Fatalf("readModelConfig() error = %v", err)
	}
	if got := emptyProbe.architecture(); got != "" {
		t.Fatalf("architecture() = %q, want empty (nothing resolvable)", got)
	}
}

// TestModelConfigProbe_QuantAccessors_NoBlock_Bad covers the return-0 paths of
// probe.quantBits and probe.quantGroup when neither a `quantization` nor a
// `quantization_config` block is present (hf.go:478, hf.go:491).
func TestModelConfigProbe_QuantAccessors_NoBlock_Bad(t *testing.T) {
	dir := t.TempDir()
	writeModelPackFile(t, core.PathJoin(dir, "config.json"), `{
		"model_type": "qwen3",
		"hidden_size": 1024
	}`)
	probe, err := readModelConfig(dir)
	if err != nil {
		t.Fatalf("readModelConfig() error = %v", err)
	}
	if got := probe.quantBits(); got != 0 {
		t.Fatalf("quantBits() = %d, want 0 (no quant block)", got)
	}
	if got := probe.quantGroup(); got != 0 {
		t.Fatalf("quantGroup() = %d, want 0 (no quant block)", got)
	}
}

// scanJANGFold / inferJANGNeedlePresent / inferJANGProfileName white-box
// coverage moved with the implementation to dappco.re/go/inference/hf
// (hf_jang.go here is now a thin InferJANG delegate — see
// TestInferJANG_BasicProfileFromTag_Ugly below for the local black-box
// coverage of that delegation, and external/go-inference/go/hf/hf_test.go
// for the upstream branch coverage of the needle scan itself).

// TestInferJANG_BasicProfileFromTag_Ugly drives InferJANG through the jangBasic
// haystack-build branch where the "jang" signal lives in a tag rather than the
// id, the profile name is the generic "JANG" (no jang_Nx token present), and the
// group size falls back to 64. This exercises the lowercase-haystack assembly
// (id + tags + filenames) plus the firstPositive(ProfileBits, 0) read.
func TestInferJANG_BasicProfileFromTag_Ugly(t *testing.T) {
	meta := ModelMetadata{
		ID:    "vendor/plain-name",
		Tags:  []string{"mlx", "jang"},
		Files: []ModelFile{{Name: "model.safetensors"}, {RFilename: "tokenizer.json"}},
		// No QuantizationConfig group size → 64 default.
	}
	info := InferJANG(meta)
	if info == nil {
		t.Fatal("InferJANG(tag jang) = nil, want a generic JANG profile")
	}
	if info.Profile != "JANG" {
		t.Fatalf("Profile = %q, want JANG (no specific jang_Nx token)", info.Profile)
	}
	if info.GroupSize != 64 {
		t.Fatalf("GroupSize = %d, want 64 default", info.GroupSize)
	}
}

// TestPlanFits_NilContext_Good covers PlanFits' ctx==nil guard (hf_fit.go:18):
// passing a nil context must default to a background context rather than panic,
// and still produce a plan for a supported local model.
func TestPlanFits_NilContext_Good(t *testing.T) {
	cacheRoot := core.PathJoin(t.TempDir(), "models--org--name")
	dir := core.PathJoin(cacheRoot, "snapshots", "snap")
	if result := core.MkdirAll(dir, 0o755); !result.OK {
		t.Fatalf("mkdir %s: %v", dir, result.Value)
	}
	writeModelPackFile(t, core.PathJoin(dir, "config.json"), `{
		"model_type": "qwen3",
		"hidden_size": 1024,
		"num_hidden_layers": 28,
		"num_attention_heads": 16,
		"num_key_value_heads": 8,
		"max_position_embeddings": 40960
	}`)
	writeModelPackFile(t, core.PathJoin(dir, "model.safetensors"), "stub")

	//nolint:staticcheck // deliberately passing a nil context to exercise the guard.
	report, err := PlanFits(nil, FitConfig{
		LocalPaths: []string{cacheRoot},
		Device:     memory.DeviceInfo{MemorySize: 96 * memory.GiB, MaxRecommendedWorkingSetSize: 86 * memory.GiB},
	})
	if err != nil {
		t.Fatalf("PlanFits(nil ctx) error = %v", err)
	}
	if len(report.Models) != 1 {
		t.Fatalf("models = %d, want 1", len(report.Models))
	}
}

// TestPlanFits_SortComparator_Ugly drives PlanFits' result-ordering comparator
// (hf_fit.go:50-64) through every arm. The input deliberately lists the two
// inference-fitting models LAST so the sort must move them left past the
// non-fitting ones — that generates the comparator's argument pairs in both
// (fitting, non-fitting) and (non-fitting, fitting) orders, covering the
// InferenceFits -1 and +1 returns. Two non-fitting models with different
// weight bytes exercise the ExpectedTotalBytes -1/+1 ordering, and two fitting
// models with identical weight bytes exercise the equal-total return-0 arm. The
// device is sized so the two supported models fit and the two unsupported don't.
func TestPlanFits_SortComparator_Ugly(t *testing.T) {
	tmp := t.TempDir()

	mkModel := func(slug, configJSON, weights string) string {
		root := core.PathJoin(tmp, slug)
		dir := core.PathJoin(root, "snapshots", "s")
		if result := core.MkdirAll(dir, 0o755); !result.OK {
			t.Fatalf("mkdir %s: %v", dir, result.Value)
		}
		writeModelPackFile(t, core.PathJoin(dir, "config.json"), configJSON)
		writeModelPackFile(t, core.PathJoin(dir, "model.safetensors"), weights)
		return root
	}

	const supportedCfg = `{
		"model_type": "qwen3",
		"hidden_size": 1024,
		"num_hidden_layers": 28,
		"num_attention_heads": 16,
		"num_key_value_heads": 8,
		"max_position_embeddings": 40960
	}`

	// Two unsupported models with DIFFERENT weight bytes → byte ordering arm.
	unsupSmall := mkModel("models--org--unsup-small", `{
		"model_type": "totally_made_up_arch_a",
		"hidden_size": 2048,
		"num_hidden_layers": 24
	}`, "small")
	unsupBig := mkModel("models--org--unsup-big", `{
		"model_type": "totally_made_up_arch_b",
		"hidden_size": 4096,
		"num_hidden_layers": 32
	}`, "a-much-longer-weight-blob-than-the-small-unsupported-one")

	// Two supported models that fit, with IDENTICAL weight bytes → the equal
	// ExpectedTotalBytes return-0 arm fires when they are compared to each other.
	fitsA := mkModel("models--org--fits-a", supportedCfg, "identical")
	fitsB := mkModel("models--org--fits-b", supportedCfg, "identical")

	// List the fitting models LAST so the sort has to move them forward.
	report, err := PlanFits(context.Background(), FitConfig{
		LocalPaths: []string{unsupSmall, unsupBig, fitsA, fitsB},
		Device:     memory.DeviceInfo{MemorySize: 16 * memory.GiB, MaxRecommendedWorkingSetSize: 13 * memory.GiB},
	})
	if err != nil {
		t.Fatalf("PlanFits() error = %v", err)
	}
	if len(report.Models) != 4 {
		t.Fatalf("models = %d, want 4", len(report.Models))
	}
	// The two inference-fitting models must sort to the front.
	if !report.Models[0].InferenceFits || !report.Models[1].InferenceFits {
		t.Fatalf("first two models must be the fitting pair, got fits=[%v,%v]",
			report.Models[0].InferenceFits, report.Models[1].InferenceFits)
	}
	if report.Models[2].InferenceFits || report.Models[3].InferenceFits {
		t.Fatalf("last two models must be the non-fitting pair, got fits=[%v,%v]",
			report.Models[2].InferenceFits, report.Models[3].InferenceFits)
	}
	// Among the non-fitting tail, ExpectedTotalBytes is non-decreasing.
	tail := report.Models[2:]
	for i := 1; i < len(tail); i++ {
		if tail[i].ExpectedTotalBytes < tail[i-1].ExpectedTotalBytes {
			t.Fatalf("non-fitting tail not ordered by total bytes: %d before %d",
				tail[i-1].ExpectedTotalBytes, tail[i].ExpectedTotalBytes)
		}
	}
}

// TestPlanFits_ZeroDeviceMemoryLimitFallback_Ugly covers planFit's memory-limit
// fallback chain (hf_fit.go:342-347): a fully zero-valued device makes
// memoryPlan.MemoryLimitBytes 0, so planFit falls through
// MaxRecommendedWorkingSetSize (also 0) to MemorySize (also 0), leaving the
// limit unconstrained. The plan must still be produced; with no working-set
// budget MemoryFits stays true for a model with known weight bytes.
func TestPlanFits_ZeroDeviceMemoryLimitFallback_Ugly(t *testing.T) {
	cacheRoot := core.PathJoin(t.TempDir(), "models--org--name")
	dir := core.PathJoin(cacheRoot, "snapshots", "s")
	if result := core.MkdirAll(dir, 0o755); !result.OK {
		t.Fatalf("mkdir %s: %v", dir, result.Value)
	}
	writeModelPackFile(t, core.PathJoin(dir, "config.json"), `{
		"model_type": "qwen3",
		"hidden_size": 1024,
		"num_hidden_layers": 28,
		"num_attention_heads": 16,
		"num_key_value_heads": 8,
		"max_position_embeddings": 40960
	}`)
	writeModelPackFile(t, core.PathJoin(dir, "model.safetensors"), "weights")

	report, err := PlanFits(context.Background(), FitConfig{
		LocalPaths: []string{cacheRoot},
		Device:     memory.DeviceInfo{}, // zero device → MemoryLimitBytes 0
	})
	if err != nil {
		t.Fatalf("PlanFits(zero device) error = %v", err)
	}
	if len(report.Models) != 1 {
		t.Fatalf("models = %d, want 1", len(report.Models))
	}
	plan := report.Models[0]
	if plan.WeightBytes == 0 {
		t.Fatal("WeightBytes = 0, want a real weight estimate")
	}
	// limit resolved to 0 → the (limit == 0) arm of MemoryFits keeps it true.
	if !plan.MemoryFits {
		t.Fatalf("MemoryFits = false, want true (no working-set budget to exceed): %+v", plan)
	}
}

// TestCollectFitEntries_ModelMetadataError_Bad covers collectFitEntries'
// ModelMetadata-error propagation (hf_fit.go:111): an unknown id the injected
// source rejects surfaces as a PlanFits error rather than a partial report.
func TestCollectFitEntries_ModelMetadataError_Bad(t *testing.T) {
	source := &fakeHFModelSource{byID: map[string]ModelMetadata{}} // rejects every id
	_, err := PlanFits(context.Background(), FitConfig{
		ModelIDs: []string{"org/unknown"},
		Device:   memory.DeviceInfo{MemorySize: 16 * memory.GiB},
		Source:   source,
	})
	if err == nil {
		t.Fatal("PlanFits(unknown id) error = nil, want ModelMetadata error propagated")
	}
}

// TestCollectFitEntries_ModelMetadataIDFallback_Good covers the id-fill branch
// in collectFitEntries (hf_fit.go:114): when a model-id lookup returns metadata
// with neither id nor modelId, the requested id is stamped onto the entry so the
// resulting plan still carries a ModelID.
func TestCollectFitEntries_ModelMetadataIDFallback_Good(t *testing.T) {
	source := &fakeHFModelSource{
		byID: map[string]ModelMetadata{
			// Value carries no ID / ModelID at all.
			"org/no-id-model": {
				Config: ModelConfig{
					ModelType:             "qwen3",
					HiddenSize:            1024,
					NumHiddenLayers:       28,
					NumAttentionHeads:     16,
					NumKeyValueHeads:      8,
					MaxPositionEmbeddings: 40960,
				},
				Files: []ModelFile{{Name: "model.safetensors", Size: 420 * 1024 * 1024}},
			},
		},
	}
	report, err := PlanFits(context.Background(), FitConfig{
		ModelIDs: []string{"org/no-id-model"},
		Device:   memory.DeviceInfo{MemorySize: 96 * memory.GiB, MaxRecommendedWorkingSetSize: 86 * memory.GiB},
		Source:   source,
	})
	if err != nil {
		t.Fatalf("PlanFits() error = %v", err)
	}
	if len(report.Models) != 1 {
		t.Fatalf("models = %d, want 1", len(report.Models))
	}
	if report.Models[0].ModelID != "org/no-id-model" {
		t.Fatalf("ModelID = %q, want org/no-id-model (requested id filled in)", report.Models[0].ModelID)
	}
}

// TestResolveLocalMetadataRoot_NonDirSnapshotEntry_Ugly covers the !IsDir
// continue inside dappco.re/go/inference/hf's ResolveLocalMetadataRoot: a
// snapshots directory holding a stray file alongside the real snapshot
// directory must skip the file and pick the directory entry. Real HF Hub
// caches do grow stray entries (.lock/.no_exist markers) alongside
// snapshots/<rev>/, so this guards go-mlx's consumption of the shared
// resolver against that real-world shape — the shared package's own test
// suite doesn't name this exact scenario.
func TestResolveLocalMetadataRoot_NonDirSnapshotEntry_Ugly(t *testing.T) {
	cacheRoot := core.PathJoin(t.TempDir(), "models--org--name")
	snapshots := core.PathJoin(cacheRoot, "snapshots")
	realSnap := core.PathJoin(snapshots, "real")
	if result := core.MkdirAll(realSnap, 0o755); !result.OK {
		t.Fatalf("mkdir %s: %v", realSnap, result.Value)
	}
	// A stray non-directory entry directly under snapshots/ — must be skipped.
	writeModelPackFile(t, core.PathJoin(snapshots, "stray.txt"), "ignore me")
	writeModelPackFile(t, core.PathJoin(realSnap, "config.json"), `{"model_type":"qwen3"}`)

	got := sharedhf.ResolveLocalMetadataRoot(cacheRoot)
	if got != realSnap {
		t.Fatalf("ResolveLocalMetadataRoot = %q, want %q (stray file skipped)", got, realSnap)
	}
}

// TestPackUsesKVCache_EmbeddingRerankPack_Bad covers the early-false arm of
// packUsesKVCache (hf_fit.go:389): a pack carrying an Embedding or Rerank
// profile object never uses a generation KV cache, regardless of the
// architecture-profile flags.
func TestPackUsesKVCache_EmbeddingRerankPack_Bad(t *testing.T) {
	embed := &mp.ModelPack{Embedding: &mp.ModelEmbeddingProfile{}}
	if packUsesKVCache(embed, false, nil) {
		t.Fatal("packUsesKVCache(embedding pack) = true, want false")
	}
	rerank := &mp.ModelPack{Rerank: &mp.ModelRerankProfile{}}
	if packUsesKVCache(rerank, false, nil) {
		t.Fatal("packUsesKVCache(rerank pack) = true, want false")
	}
	// A plain generation pack with a generation profile keeps the KV cache.
	genProfile := &profile.ModelArchitectureProfile{Generation: true}
	if !packUsesKVCache(&mp.ModelPack{}, true, genProfile) {
		t.Fatal("packUsesKVCache(generation profile) = false, want true")
	}
	// A non-generation profile forces false.
	encoderProfile := &profile.ModelArchitectureProfile{Embeddings: true}
	if packUsesKVCache(&mp.ModelPack{}, true, encoderProfile) {
		t.Fatal("packUsesKVCache(embedding profile) = true, want false")
	}
}

// TestEstimateModelKVBytes_NoGeometry_Bad covers estimateModelKVBytes'
// perToken<=0 return (hf_fit.go:491): a config with positive layers and context
// but no head/hidden geometry to size a token yields a zero estimate rather than
// a bogus number.
func TestEstimateModelKVBytes_NoGeometry_Bad(t *testing.T) {
	// Layers + context are positive, but hidden/heads/headDim are all zero, so
	// neither perToken branch produces a positive size.
	cfg := ModelConfig{NumHiddenLayers: 4}
	if got := estimateModelKVBytes(cfg, 2048, 1, 2); got != 0 {
		t.Fatalf("estimateModelKVBytes(no geometry) = %d, want 0", got)
	}
}

// TestEstimateTrainingFit_NoGeometry_Ugly covers estimateTrainingFit's
// targets=0 branch (hf_fit.go:516): when hidden or layers is non-positive the
// LoRA target count collapses to zero, so the estimated LoRA/optimizer byte
// figures are zero while the rank still defaults.
func TestEstimateTrainingFit_NoGeometry_Ugly(t *testing.T) {
	plan := FitPlan{
		NativeLoadable:     true,
		InferenceFits:      true,
		QuantBits:          16,
		WeightBytes:        100,
		ExpectedTotalBytes: 120,
	}
	// hidden_size 0 → targets=0 → loraParams 0 → zero LoRA/optimizer bytes.
	fit := estimateTrainingFit(ModelConfig{NumHiddenLayers: 2}, plan, 0, 8)
	if fit.RecommendedLoRARank != 8 {
		t.Fatalf("RecommendedLoRARank = %d, want 8", fit.RecommendedLoRARank)
	}
	if fit.EstimatedLoRABytes != 0 || fit.EstimatedOptimizerBytes != 0 {
		t.Fatalf("estimates = lora:%d opt:%d, want 0/0 (no geometry → no targets)",
			fit.EstimatedLoRABytes, fit.EstimatedOptimizerBytes)
	}
}

// TestFitNotes_NotNativeArchitecture_Ugly covers the notNative branch of
// fitNotes (hf_fit.go:581 count, hf_fit.go:603 note append): a recognised but
// non-native-runtime architecture attaches the "native runtime kernels are not
// implemented yet" advisory. fitNotes is package-internal so a hand-built
// FitPlan drives it directly with nativeRuntime=false.
func TestFitNotes_NotNativeArchitecture_Ugly(t *testing.T) {
	plan := FitPlan{SupportedArchitecture: true, WeightBytes: 100}
	notes := fitNotes(plan, 0, false /* nativeRuntime */, false /* nonStandaloneNative */)
	found := false
	for _, n := range notes {
		if core.Contains(n, "native runtime kernels are not implemented yet") {
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("fitNotes(supported, not native) = %v, want the not-native advisory", notes)
	}
}

// TestFitResultError_ErrorValue_Good covers fitResultError's error-typed value
// arm (hf_fit.go:680): a failed Result whose Value is a real error returns that
// error verbatim rather than the generic fallback, and an OK result returns nil.
func TestFitResultError_ErrorValue_Good(t *testing.T) {
	sentinel := core.NewError("boom")
	if got := fitResultError(core.Result{Value: sentinel, OK: false}); got != sentinel {
		t.Fatalf("fitResultError(error value) = %v, want the wrapped error verbatim", got)
	}
	if got := fitResultError(core.Result{OK: true}); got != nil {
		t.Fatalf("fitResultError(ok) = %v, want nil", got)
	}
}

// TestGetJSON_RequestBuildError_Bad covers getJSON's request-build failure arm
// (hf.go:118): a BaseURL carrying a control character produces a malformed
// target URL that NewHTTPRequestContext rejects, so SearchModels and
// ModelMetadata surface a build-request error before any network call. No
// network — the request never leaves the constructor.
func TestGetJSON_RequestBuildError_Bad(t *testing.T) {
	// The DEL control byte (0x7f) is invalid in a URL; http.NewRequest rejects
	// the assembled target. The slash keeps it past the empty-BaseURL default.
	source := NewRemoteSource(RemoteConfig{BaseURL: "http://example.com/\x7f"})
	if _, err := source.SearchModels(context.Background(), "qwen", 1); err == nil {
		t.Fatal("SearchModels(control-char URL) error = nil, want a build-request error")
	}
	if _, err := source.ModelMetadata(context.Background(), "org/model"); err == nil {
		t.Fatal("ModelMetadata(control-char URL) error = nil, want a build-request error")
	}
}
