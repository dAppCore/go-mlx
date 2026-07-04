// SPDX-Licence-Identifier: EUPL-1.2

package lora

import (
	"context"
	core "dappco.re/go"
	"dappco.re/go/mlx/internal/metaltest"
	pack "dappco.re/go/inference/modelpack"
	"dappco.re/go/mlx/pkg/metal"
	"math"
	"testing"
)

func writeFuseTestFile(t *testing.T, path string, data string) {
	t.Helper()
	if result := core.WriteFile(path, []byte(data), 0o644); !result.OK {
		t.Fatalf("write %s: %v", path, result.Value)
	}
}

func TestFusePairName_Good(t *testing.T) {
	pair, suffix, ok := fusePairName("model.layers.0.self_attn.q_proj.lora_a")
	if !ok || pair != "model.layers.0.self_attn.q_proj" || suffix != "a" {
		t.Fatalf("pair=%q suffix=%q ok=%v, want q_proj/a/true", pair, suffix, ok)
	}
	if got := fuseBaseWeightKey(pair); got != "model.layers.0.self_attn.q_proj.weight" {
		t.Fatalf("base weight key = %q", got)
	}

	pair, suffix, ok = fusePairName("model.layers.0.self_attn.q_proj.lora_B.weight")
	if !ok || pair != "model.layers.0.self_attn.q_proj" || suffix != "b" {
		t.Fatalf("PEFT pair=%q suffix=%q ok=%v, want q_proj/b/true", pair, suffix, ok)
	}

	for _, name := range []string{
		"layer.lora_a.weight",
		"layer.lora_A.weight",
		"layer.lora_A",
		"layer.lora_b.weight",
		"layer.lora_B",
	} {
		pair, suffix, ok := fusePairName(name)
		if !ok || pair != "layer" || (suffix != "a" && suffix != "b") {
			t.Fatalf("fusePairName(%q) = pair:%q suffix:%q ok:%v", name, pair, suffix, ok)
		}
	}
	if pair, suffix, ok := fusePairName("layer.weight"); ok || pair != "" || suffix != "" {
		t.Fatalf("fusePairName(non-lora) = pair:%q suffix:%q ok:%v", pair, suffix, ok)
	}
}

func TestFuseBaseWeightKey_GenericSuffixTargetsStayModelLocal_Good(t *testing.T) {
	if got := fuseBaseWeightKey("model.layers.0.q_proj"); got != "model.layers.0.q_proj.weight" {
		t.Fatalf("generic base weight key = %q, want model-local q_proj path", got)
	}
}

func TestFuseBaseWeightKeyForArchitecture_Gemma4SuffixTargets_Good(t *testing.T) {
	tests := map[string]string{
		"model.layers.0.q_proj":               "model.layers.0.self_attn.q_proj.weight",
		"model.layers.0.k_proj":               "model.layers.0.self_attn.k_proj.weight",
		"model.layers.0.v_proj":               "model.layers.0.self_attn.v_proj.weight",
		"model.layers.0.o_proj":               "model.layers.0.self_attn.o_proj.weight",
		"model.layers.0.gate_proj":            "model.layers.0.mlp.gate_proj.weight",
		"model.layers.0.up_proj":              "model.layers.0.mlp.up_proj.weight",
		"model.layers.0.down_proj":            "model.layers.0.mlp.down_proj.weight",
		"model.layers.0.router.proj":          "model.layers.0.router.proj.weight",
		"model.layers.0.per_layer_input_gate": "model.layers.0.per_layer_input_gate.weight",
	}
	for pairName, want := range tests {
		if got := fuseBaseWeightKeyForArchitecture(pairName, "gemma4_text"); got != want {
			t.Fatalf("gemma4 base weight key for %q = %q, want %q", pairName, got, want)
		}
	}
	if got := fuseBaseWeightKeyForArchitecture("model.layers.0.q_proj", "qwen3"); got != "model.layers.0.q_proj.weight" {
		t.Fatalf("qwen3 base weight key = %q, want generic suffix path", got)
	}
	if got := fuseBaseWeightKeyForArchitecture("model.layers.0.q_proj", "Gemma4AssistantForCausalLM"); got != "model.layers.0.q_proj.weight" {
		t.Fatalf("gemma4 assistant base weight key = %q, want attached drafter to keep generic suffix path", got)
	}
	for _, architecture := range []string{
		"gemma4",
		"gemma4_text",
		"gemma4_unified",
		"gemma4_unified_text",
		"Gemma4ForConditionalGeneration",
		"Gemma4UnifiedForConditionalGeneration",
		"Gemma4ForCausalLM",
		"Gemma4TextForCausalLM",
	} {
		if got := fuseBaseWeightKeyForArchitecture("model.layers.0.q_proj", architecture); got != "model.layers.0.self_attn.q_proj.weight" {
			t.Fatalf("gemma4 base weight key for architecture %q = %q, want canonical q_proj key", architecture, got)
		}
	}
}

func TestPrepareFuse_OutputMustBePackDirectory_Bad(t *testing.T) {
	_, err := prepareFuse(context.Background(), FuseOptions{
		SourcePack:  pack.ModelPack{Root: "/tmp/source", Format: pack.ModelPackFormatSafetensors},
		AdapterPath: "/tmp/adapter",
		OutputPath:  "/tmp/fused.safetensors",
	})
	if err == nil {
		t.Fatal("expected output directory error")
	}
	if !core.Contains(err.Error(), "directory") {
		t.Fatalf("error = %v, want directory context", err)
	}
}

func TestPrepareFuse_ValidationErrors_Bad(t *testing.T) {
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := prepareFuse(cancelled, FuseOptions{}); err != context.Canceled {
		t.Fatalf("prepareFuse(cancelled) = %v, want context.Canceled", err)
	}
	if _, err := prepareFuse(context.Background(), FuseOptions{}); err == nil {
		t.Fatal("expected missing source pack error")
	}
	if _, err := prepareFuse(context.Background(), FuseOptions{SourcePack: pack.ModelPack{Root: "/tmp/model", Format: pack.ModelPackFormatSafetensors}}); err == nil {
		t.Fatal("expected missing adapter path error")
	}
	if _, err := prepareFuse(context.Background(), FuseOptions{SourcePack: pack.ModelPack{Root: "/tmp/model", Format: pack.ModelPackFormatSafetensors}, AdapterPath: "/tmp/adapter"}); err == nil {
		t.Fatal("expected missing output path error")
	}
}

func TestPrepareFuse_MissingAdapterRank_Bad(t *testing.T) {
	source := t.TempDir()
	adapter := t.TempDir()
	output := core.PathJoin(t.TempDir(), "fused")
	writeFuseTestFile(t, core.PathJoin(source, "config.json"), `{"model_type":"qwen3"}`)
	writeFuseTestFile(t, core.PathJoin(adapter, "adapter_config.json"), `{"target_modules":["q_proj"]}`)
	writeFuseTestFile(t, core.PathJoin(adapter, "adapter.safetensors"), "stub")

	_, err := prepareFuse(context.Background(), FuseOptions{
		SourcePack:  pack.ModelPack{Root: source, Path: source, Format: pack.ModelPackFormatSafetensors},
		AdapterPath: adapter,
		OutputPath:  output,
	})
	if err != errFuseRankRequired {
		t.Fatalf("prepareFuse() error = %v, want errFuseRankRequired", err)
	}
}

func TestPrepareFuse_RankOnlyAdapterDefaultsScale_Good(t *testing.T) {
	source := t.TempDir()
	adapter := t.TempDir()
	output := core.PathJoin(t.TempDir(), "fused")
	writeFuseTestFile(t, core.PathJoin(source, "config.json"), `{"model_type":"qwen3"}`)
	writeFuseTestFile(t, core.PathJoin(adapter, "adapter_config.json"), `{"rank":4,"target_modules":["q_proj"]}`)
	writeFuseTestFile(t, core.PathJoin(adapter, "adapter.safetensors"), "stub")

	prepared, err := prepareFuse(context.Background(), FuseOptions{
		SourcePack:  pack.ModelPack{Root: source, Path: source, Format: pack.ModelPackFormatSafetensors},
		AdapterPath: adapter,
		OutputPath:  output,
	})
	if err != nil {
		t.Fatalf("prepareFuse() error = %v", err)
	}
	if prepared.Adapter.Rank != 4 || prepared.Adapter.Alpha != 8 || prepared.Adapter.Scale != 2 {
		t.Fatalf("adapter metadata = %+v, want rank 4 with default alpha 8 scale 2", prepared.Adapter)
	}
}

func TestFuseDestinationAndMetadata_Good(t *testing.T) {
	base := t.TempDir()
	output := core.PathJoin(t.TempDir(), "fused")
	if result := core.MkdirAll(output, 0o755); !result.OK {
		t.Fatalf("mkdir output: %v", result.Value)
	}
	files := map[string]string{
		"config.json":              `{"model_type":"qwen3"}`,
		"tokenizer.json":           `{"model":{"type":"BPE"}}`,
		"adapter_provenance.json":  `{"skip":true}`,
		"model.safetensors.index":  "skip",
		"notes.txt":                "keep",
		"tokenizer.model":          "keep model",
		"ignored.gguf":             "skip",
		"ignored.safetensors":      "skip",
		"model.safetensors.index2": "skip because contains",
	}
	for name, content := range files {
		writeFuseTestFile(t, core.PathJoin(base, name), content)
	}

	if err := copyModelPackMetadata(base, output); err != nil {
		t.Fatalf("copyModelPackMetadata: %v", err)
	}
	for _, name := range []string{"config.json", "tokenizer.json", "notes.txt", "tokenizer.model"} {
		if stat := core.Stat(core.PathJoin(output, name)); !stat.OK {
			t.Fatalf("%s was not copied: %v", name, stat.Value)
		}
	}
	for _, name := range []string{"adapter_provenance.json", "ignored.gguf", "ignored.safetensors", "model.safetensors.index"} {
		if stat := core.Stat(core.PathJoin(output, name)); stat.OK {
			t.Fatalf("%s should not have been copied", name)
		}
	}
	if err := ensureEmptyFuseWeightDestination(core.PathJoin(t.TempDir(), "missing")); err != nil {
		t.Fatalf("missing destination should be accepted: %v", err)
	}
	if !samePath(base, base) {
		t.Fatal("samePath(base, base) = false, want true")
	}
}

func TestFuseDestinationAndMetadata_Bad(t *testing.T) {
	dir := t.TempDir()
	if result := core.WriteFile(core.PathJoin(dir, "model.safetensors"), []byte("weights"), 0o644); !result.OK {
		t.Fatalf("write weights: %v", result.Value)
	}
	if err := ensureEmptyFuseWeightDestination(dir); err == nil || !core.Contains(err.Error(), "already contains") {
		t.Fatalf("ensureEmptyFuseWeightDestination() error = %v", err)
	}
	if !isModelWeightMetadataCopySkip("MODEL.GGUF") || !isModelWeightMetadataCopySkip("adapter_provenance.json") {
		t.Fatal("expected model weight metadata files to be skipped")
	}
	if isModelWeightMetadataCopySkip("tokenizer.json") {
		t.Fatal("tokenizer.json should not be skipped")
	}
	if err := copyLocalFile(core.PathJoin(dir, "missing.json"), core.PathJoin(dir, "out.json")); err == nil {
		t.Fatal("expected copyLocalFile missing source error")
	}
}

func TestFuseAdapterWeightFiles_Good(t *testing.T) {
	dir := t.TempDir()
	a := core.PathJoin(dir, "b.safetensors")
	b := core.PathJoin(dir, "a.safetensors")
	for _, path := range []string{a, b} {
		if result := core.WriteFile(path, []byte("weights"), 0o644); !result.OK {
			t.Fatalf("write adapter weight: %v", result.Value)
		}
	}
	files, err := fuseAdapterWeightFiles(dir)
	if err != nil {
		t.Fatalf("fuseAdapterWeightFiles(dir): %v", err)
	}
	if len(files) != 2 || files[0] != b || files[1] != a {
		t.Fatalf("adapter files = %+v, want sorted", files)
	}
	files, err = fuseAdapterWeightFiles(a)
	if err != nil {
		t.Fatalf("fuseAdapterWeightFiles(file): %v", err)
	}
	if len(files) != 1 || files[0] != a {
		t.Fatalf("adapter file result = %+v, want %q", files, a)
	}
	if _, err := fuseAdapterWeightFiles(core.PathJoin(t.TempDir(), "empty")); err == nil {
		t.Fatal("expected no adapter safetensors error")
	}
}

func TestWriteFuseProvenance_Ugly(t *testing.T) {
	path := core.PathJoin(t.TempDir(), FuseProvenanceFile)
	err := writeFuseProvenance(path, FuseProvenance{
		Version:         1,
		OutputWeight:    "model.safetensors",
		FusedWeightKeys: []string{"z.weight", "a.weight"},
		Labels:          map[string]string{"run": "probe"},
	})
	if err != nil {
		t.Fatalf("writeFuseProvenance() error = %v", err)
	}
	read := core.ReadFile(path)
	if !read.OK {
		t.Fatalf("ReadFile provenance: %v", read.Value)
	}
	text := string(read.Value.([]byte))
	if !core.Contains(text, "model.safetensors") || !core.Contains(text, "probe") {
		t.Fatalf("provenance missing expected fields: %s", text)
	}
	parts := core.Split(text, "a.weight")
	if len(parts) < 2 || !core.Contains(parts[1], "z.weight") {
		t.Fatalf("fused keys are not sorted: %s", text)
	}
}

func TestSamePath_Branches_Ugly(t *testing.T) {
	// Ugly: samePath has three branches the existing equality test never
	// separates. None need metal or fault injection.
	//   Good (identity): two identical strings short-circuit to true.
	//   Bad (clean-absolute fast reject): two distinct clean-absolute paths
	//     skip the PathAbs round-trips and report false on string inequality.
	//   Ugly (PathAbs fallback): a non-clean pair (contains "..") cannot use
	//     the fast reject, so both inputs are resolved via PathAbs and then
	//     compared — "/x/../x" resolves to "/x", matching the literal "/x".
	if !samePath("/models/lora", "/models/lora") {
		t.Fatal("samePath(identical) = false, want true")
	}
	if samePath("/models/a", "/models/b") {
		t.Fatal("samePath(distinct clean-absolute) = true, want false")
	}
	if !samePath("/models/lora/../lora", "/models/lora") {
		t.Fatal("samePath(non-clean pair resolving equal) = false, want true via PathAbs fallback")
	}
}

func TestIsCleanAbsolute_RejectBranches_Ugly(t *testing.T) {
	// Ugly: isCleanAbsolute is the canonical-form predicate that lets samePath
	// skip filepath.Clean. Each reject branch corresponds to a shape
	// filepath.Clean would have to rewrite. Drive each directly — one
	// crafted string per rejecting clause, plus the accept case.
	rejects := map[string]string{
		"empty":          "",
		"relative":       "models/lora",
		"trailingSlash":  "/models/lora/",
		"doubleSlash":    "/models//lora",
		"trailingDot":    "/models/.",
		"midDot":         "/models/./lora",
		"trailingDotDot": "/models/..",
		"midDotDot":      "/models/../lora",
	}
	for name, p := range rejects {
		t.Run(name, func(t *testing.T) {
			if isCleanAbsolute(p) {
				t.Fatalf("isCleanAbsolute(%q) = true, want false (Clean would rewrite it)", p)
			}
		})
	}
	for _, p := range []string{"/", "/models", "/models/lora/adapter.safetensors"} {
		if !isCleanAbsolute(p) {
			t.Fatalf("isCleanAbsolute(%q) = false, want true (already canonical)", p)
		}
	}
}

func TestEnsureEmptyFuseWeightDestination_GgufBranch_Bad(t *testing.T) {
	// Bad: ensureEmptyFuseWeightDestination rejects a destination that already
	// holds weights. The existing test covers the *.safetensors arm; this
	// drives the second probe — a directory containing only a *.gguf file —
	// so the gguf glob branch (L176-178) fires.
	dir := t.TempDir()
	if result := core.WriteFile(core.PathJoin(dir, "model.gguf"), []byte("weights"), 0o644); !result.OK {
		t.Fatalf("write gguf: %v", result.Value)
	}
	err := ensureEmptyFuseWeightDestination(dir)
	if err == nil || !core.Contains(err.Error(), "already contains") {
		t.Fatalf("ensureEmptyFuseWeightDestination(dir with .gguf) error = %v, want already-contains", err)
	}
}

func TestFusePairName_BareTailRejects_Bad(t *testing.T) {
	// Bad: the bare ".lora_X" tail branch (no ".weight" suffix) has reject
	// arms the happy-path test never reaches — a name too short to hold the
	// tail, a same-length name whose tail is not ".lora_", and a well-formed
	// ".lora_X" whose kind byte is neither a/A nor b/B. All must report
	// (\"\", \"\", false). Pure string logic, no metal.
	for _, name := range []string{
		"lora_a",                     // shorter than ".lora_X" → head < 0
		"model.attn",                 // length-ok tail but not ".lora_"
		"model.q_proj.lora_c",        // valid ".lora_" prefix, unknown kind 'c'
		"model.q_proj.lora_c.weight", // ".weight" branch, unknown kind 'c'
	} {
		if pair, suffix, ok := fusePairName(name); ok || pair != "" || suffix != "" {
			t.Fatalf("fusePairName(%q) = pair:%q suffix:%q ok:%v, want empty/false", name, pair, suffix, ok)
		}
	}
}

func TestSuffixFolds_Branches_Ugly(t *testing.T) {
	// Ugly: the case-folding suffix predicates short-circuit on length and
	// fold ASCII case in the tail window. prepareFuse only ever feeds
	// hasGgufSuffixFold non-.gguf paths, so its accept + uppercase-fold arms
	// are otherwise unexercised. Drive both predicates across short, mixed,
	// and rejecting inputs directly.
	if hasSafetensorsSuffixFold("short") {
		t.Fatal("hasSafetensorsSuffixFold(short) = true, want false (length guard)")
	}
	if !hasSafetensorsSuffixFold("/m/A.SAFETENSORS") {
		t.Fatal("hasSafetensorsSuffixFold(uppercase) = false, want true (case-fold tail)")
	}
	if hasSafetensorsSuffixFold("/m/adapter.bin") {
		t.Fatal("hasSafetensorsSuffixFold(.bin) = true, want false")
	}

	if hasGgufSuffixFold(".gg") {
		t.Fatal("hasGgufSuffixFold(short) = true, want false (length guard)")
	}
	if !hasGgufSuffixFold("/m/MODEL.GGUF") {
		t.Fatal("hasGgufSuffixFold(uppercase) = false, want true (case-fold tail)")
	}
	if hasGgufSuffixFold("/m/model.json") {
		t.Fatal("hasGgufSuffixFold(.json) = true, want false")
	}
}

func TestFuseBaseWeightPrefix_NoWeightSuffix_Bad(t *testing.T) {
	// Bad: fuseBaseWeightPrefix returns ok=false for a key that does not end
	// in ".weight" (the sidecar-prefix resolver skips such keys). The
	// accept case is exercised by the fuse tests; this is the reject arm.
	if prefix, ok := fuseBaseWeightPrefix("model.layers.0.q_proj.scales"); ok || prefix != "" {
		t.Fatalf("fuseBaseWeightPrefix(non-.weight) = %q,%v, want \"\",false", prefix, ok)
	}
	if prefix, ok := fuseBaseWeightPrefix("model.layers.0.q_proj.weight"); !ok || prefix != "model.layers.0.q_proj" {
		t.Fatalf("fuseBaseWeightPrefix(.weight) = %q,%v, want trimmed prefix,true", prefix, ok)
	}
}

func requireFuseMetal(t *testing.T) {
	t.Helper()
	if !metaltest.RunMetalTests {
		t.Skip("build with -tags metal_runtime to enable native LoRA fuse tensor tests")
	}
	if !metal.MetalAvailable() {
		t.Skip("Metal runtime unavailable")
	}
}

func writeFuseSourcePack(t *testing.T, dir string, tensors map[string]*metal.Array) pack.ModelPack {
	t.Helper()
	writeFuseTestFile(t, core.PathJoin(dir, "config.json"), `{
		"model_type": "qwen3",
		"vocab_size": 151936,
		"hidden_size": 2,
		"num_hidden_layers": 1,
		"max_position_embeddings": 4096
	}`)
	writeFuseTestFile(t, core.PathJoin(dir, "tokenizer.json"), `{"model":{"type":"BPE"}}`)
	weightPath := core.PathJoin(dir, "model.safetensors")
	if err := metal.SaveSafetensors(weightPath, tensors); err != nil {
		t.Fatalf("SaveSafetensors source: %v", err)
	}
	return pack.ModelPack{
		Root:         dir,
		Path:         dir,
		Format:       pack.ModelPackFormatSafetensors,
		WeightFiles:  []string{weightPath},
		Architecture: "qwen3",
		ConfigPath:   core.PathJoin(dir, "config.json"),
	}
}

func writeGemma4FuseSourcePack(t *testing.T, dir string, tensors map[string]*metal.Array) pack.ModelPack {
	t.Helper()
	writeFuseTestFile(t, core.PathJoin(dir, "config.json"), `{
		"model_type": "gemma4_text",
		"vocab_size": 262144,
		"hidden_size": 2,
		"num_hidden_layers": 1,
		"max_position_embeddings": 262144
	}`)
	writeFuseTestFile(t, core.PathJoin(dir, "tokenizer.json"), `{"model":{"type":"BPE"}}`)
	weightPath := core.PathJoin(dir, "model.safetensors")
	if err := metal.SaveSafetensors(weightPath, tensors); err != nil {
		t.Fatalf("SaveSafetensors gemma4 source: %v", err)
	}
	return pack.ModelPack{
		Root:         dir,
		Path:         dir,
		Format:       pack.ModelPackFormatSafetensors,
		WeightFiles:  []string{weightPath},
		Architecture: "gemma4_text",
		ConfigPath:   core.PathJoin(dir, "config.json"),
	}
}

func writeFuseAdapter(t *testing.T, dir string, tensors map[string]*metal.Array) {
	t.Helper()
	writeFuseAdapterWithConfig(t, dir, `{
		"rank": 1,
		"alpha": 2,
		"lora_layers": ["self_attn.q_proj"]
	}`, tensors)
}

func writeFuseAdapterWithConfig(t *testing.T, dir string, config string, tensors map[string]*metal.Array) {
	t.Helper()
	writeFuseTestFile(t, core.PathJoin(dir, "adapter_config.json"), config)
	if err := metal.SaveSafetensors(core.PathJoin(dir, "adapter.safetensors"), tensors); err != nil {
		t.Fatalf("SaveSafetensors adapter: %v", err)
	}
}

func closeTensorMap(tensors map[string]*metal.Array) {
	for _, tensor := range tensors {
		metal.Free(tensor)
	}
}

func fuseTestPackedIn(inDim, bits int) int {
	return (inDim*bits + 31) / 32
}

func zeroUint32s(n int) []uint32 {
	return make([]uint32, n)
}

func float32Fill(n int, value float32) []float32 {
	values := make([]float32, n)
	for i := range values {
		values[i] = value
	}
	return values
}

func TestFuseIntoPack_DenseSafetensors_Good(t *testing.T) {
	requireFuseMetal(t)

	source := core.PathJoin(t.TempDir(), "source")
	adapter := core.PathJoin(t.TempDir(), "adapter")
	output := core.PathJoin(t.TempDir(), "fused")
	if result := core.MkdirAll(source, 0o755); !result.OK {
		t.Fatalf("MkdirAll source: %v", result.Value)
	}
	if result := core.MkdirAll(adapter, 0o755); !result.OK {
		t.Fatalf("MkdirAll adapter: %v", result.Value)
	}

	baseWeights := map[string]*metal.Array{
		"model.layers.0.self_attn.q_proj.weight": metal.FromValues([]float32{0, 0, 0, 0}, 2, 2),
		"model.layers.0.self_attn.k_proj.weight": metal.FromValues([]float32{10, 20, 30, 40}, 2, 2),
	}
	defer closeTensorMap(baseWeights)
	sourcePack := writeFuseSourcePack(t, source, baseWeights)

	adapterWeights := map[string]*metal.Array{
		"model.layers.0.self_attn.q_proj.lora_a": metal.FromValues([]float32{1, 2}, 1, 2),
		"model.layers.0.self_attn.q_proj.lora_b": metal.FromValues([]float32{3, 4}, 2, 1),
	}
	defer closeTensorMap(adapterWeights)
	writeFuseAdapter(t, adapter, adapterWeights)

	result, err := FuseIntoPack(context.Background(), FuseOptions{
		SourcePack:  sourcePack,
		AdapterPath: adapter,
		OutputPath:  output,
	})
	if err != nil {
		t.Fatalf("FuseIntoPack() error = %v", err)
	}
	if result.OutputPath != output {
		t.Fatalf("OutputPath = %q, want %q", result.OutputPath, output)
	}
	if result.Adapter.Rank != 1 || result.Adapter.Alpha != 2 || result.Adapter.Scale != 2 {
		t.Fatalf("adapter = %+v, want rank 1 alpha 2 scale 2", result.Adapter)
	}
	if result.FusedWeights != 1 {
		t.Fatalf("FusedWeights = %d, want 1", result.FusedWeights)
	}

	loaded, err := metal.LoadAllSafetensors(core.PathJoin(output, "model.safetensors"))
	if err != nil {
		t.Fatalf("LoadAllSafetensors fused: %v", err)
	}
	defer closeTensorMap(loaded)

	got := loaded["model.layers.0.self_attn.q_proj.weight"].Floats()
	want := []float32{6, 12, 8, 16}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 0.0001 {
			t.Fatalf("fused q_proj[%d] = %v, want %v; full=%v", i, got[i], want[i], got)
		}
	}

	unchanged := loaded["model.layers.0.self_attn.k_proj.weight"].Floats()
	for i, wantValue := range []float32{10, 20, 30, 40} {
		if unchanged[i] != wantValue {
			t.Fatalf("unmatched base weight changed: %v", unchanged)
		}
	}

	provenance := core.ReadFile(core.PathJoin(output, "adapter_provenance.json"))
	if !provenance.OK {
		t.Fatalf("read adapter provenance: %v", provenance.Value)
	}
	if !core.Contains(string(provenance.Value.([]byte)), "self_attn.q_proj") {
		t.Fatalf("adapter provenance missing target: %s", provenance.Value.([]byte))
	}
}

func TestFuseIntoPack_Gemma4SuffixTargetAliases_Good(t *testing.T) {
	requireFuseMetal(t)

	source := core.PathJoin(t.TempDir(), "source")
	adapter := core.PathJoin(t.TempDir(), "adapter")
	output := core.PathJoin(t.TempDir(), "fused")
	if result := core.MkdirAll(source, 0o755); !result.OK {
		t.Fatalf("MkdirAll source: %v", result.Value)
	}
	if result := core.MkdirAll(adapter, 0o755); !result.OK {
		t.Fatalf("MkdirAll adapter: %v", result.Value)
	}

	baseWeights := map[string]*metal.Array{
		"model.layers.0.self_attn.q_proj.weight": metal.FromValues([]float32{0, 0, 0, 0}, 2, 2),
		"model.layers.0.self_attn.k_proj.weight": metal.FromValues([]float32{10, 20, 30, 40}, 2, 2),
	}
	defer closeTensorMap(baseWeights)
	sourcePack := writeGemma4FuseSourcePack(t, source, baseWeights)

	adapterWeights := map[string]*metal.Array{
		"model.layers.0.q_proj.lora_A.weight": metal.FromValues([]float32{1, 2}, 1, 2),
		"model.layers.0.q_proj.lora_B.weight": metal.FromValues([]float32{3, 4}, 2, 1),
	}
	defer closeTensorMap(adapterWeights)
	writeFuseAdapterWithConfig(t, adapter, `{
		"r": 1,
		"lora_alpha": 2,
		"target_modules": ["q_proj"]
	}`, adapterWeights)

	result, err := FuseIntoPack(context.Background(), FuseOptions{
		SourcePack:  sourcePack,
		AdapterPath: adapter,
		OutputPath:  output,
	})
	if err != nil {
		t.Fatalf("FuseIntoPack() error = %v", err)
	}
	if result.FusedWeights != 1 {
		t.Fatalf("FusedWeights = %d, want 1", result.FusedWeights)
	}

	loaded, err := metal.LoadAllSafetensors(core.PathJoin(output, "model.safetensors"))
	if err != nil {
		t.Fatalf("LoadAllSafetensors fused: %v", err)
	}
	defer closeTensorMap(loaded)

	got := loaded["model.layers.0.self_attn.q_proj.weight"].Floats()
	want := []float32{6, 12, 8, 16}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 0.0001 {
			t.Fatalf("fused gemma4 q_proj[%d] = %v, want %v; full=%v", i, got[i], want[i], got)
		}
	}
	if len(result.FusedWeightKeys) != 1 || result.FusedWeightKeys[0] != "model.layers.0.self_attn.q_proj.weight" {
		t.Fatalf("FusedWeightKeys = %v, want canonical Gemma4 q_proj base key", result.FusedWeightKeys)
	}
}

func TestFuseIntoPack_Gemma4PrefixedDenseSource_Good(t *testing.T) {
	requireFuseMetal(t)

	source := core.PathJoin(t.TempDir(), "source")
	adapter := core.PathJoin(t.TempDir(), "adapter")
	output := core.PathJoin(t.TempDir(), "fused")
	if result := core.MkdirAll(source, 0o755); !result.OK {
		t.Fatalf("MkdirAll source: %v", result.Value)
	}
	if result := core.MkdirAll(adapter, 0o755); !result.OK {
		t.Fatalf("MkdirAll adapter: %v", result.Value)
	}

	baseKey := "language_model.model.layers.0.self_attn.q_proj.weight"
	baseWeights := map[string]*metal.Array{
		baseKey: metal.FromValues([]float32{0, 0, 0, 0}, 2, 2),
	}
	defer closeTensorMap(baseWeights)
	sourcePack := writeGemma4FuseSourcePack(t, source, baseWeights)

	adapterWeights := map[string]*metal.Array{
		"model.layers.0.q_proj.lora_A.weight": metal.FromValues([]float32{1, 2}, 1, 2),
		"model.layers.0.q_proj.lora_B.weight": metal.FromValues([]float32{3, 4}, 2, 1),
	}
	defer closeTensorMap(adapterWeights)
	writeFuseAdapterWithConfig(t, adapter, `{
		"r": 1,
		"lora_alpha": 2,
		"target_modules": ["q_proj"]
	}`, adapterWeights)

	result, err := FuseIntoPack(context.Background(), FuseOptions{
		SourcePack:  sourcePack,
		AdapterPath: adapter,
		OutputPath:  output,
	})
	if err != nil {
		t.Fatalf("FuseIntoPack() error = %v", err)
	}
	if len(result.FusedWeightKeys) != 1 || result.FusedWeightKeys[0] != baseKey {
		t.Fatalf("FusedWeightKeys = %v, want raw Gemma4 source key", result.FusedWeightKeys)
	}

	loaded, err := metal.LoadAllSafetensors(core.PathJoin(output, "model.safetensors"))
	if err != nil {
		t.Fatalf("LoadAllSafetensors fused: %v", err)
	}
	defer closeTensorMap(loaded)

	got := loaded[baseKey].Floats()
	want := []float32{6, 12, 8, 16}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 0.0001 {
			t.Fatalf("fused prefixed gemma4 q_proj[%d] = %v, want %v; full=%v", i, got[i], want[i], got)
		}
	}
	if _, exists := loaded["model.layers.0.self_attn.q_proj.weight"]; exists {
		t.Fatal("fuse should preserve the source safetensors key instead of adding a duplicate canonical key")
	}
}

func TestFuseIntoPack_Gemma4Q6BaseTargetDequantizesAndDropsSidecars_Good(t *testing.T) {
	requireFuseMetal(t)

	source := core.PathJoin(t.TempDir(), "source")
	adapter := core.PathJoin(t.TempDir(), "adapter")
	output := core.PathJoin(t.TempDir(), "fused")
	if result := core.MkdirAll(source, 0o755); !result.OK {
		t.Fatalf("MkdirAll source: %v", result.Value)
	}
	if result := core.MkdirAll(adapter, 0o755); !result.OK {
		t.Fatalf("MkdirAll adapter: %v", result.Value)
	}

	basePrefix := "language_model.model.layers.0.self_attn.q_proj"
	const (
		outDim    = 2
		inDim     = 64
		groupSize = 64
		bits      = 6
	)
	baseWeights := map[string]*metal.Array{
		basePrefix + ".weight": metal.FromValues(zeroUint32s(outDim*fuseTestPackedIn(inDim, bits)), outDim, fuseTestPackedIn(inDim, bits)),
		basePrefix + ".scales": metal.FromValues([]float32{1, 1}, outDim, inDim/groupSize),
		basePrefix + ".biases": metal.FromValues([]float32{0, 0}, outDim, inDim/groupSize),
	}
	defer closeTensorMap(baseWeights)
	sourcePack := writeGemma4FuseSourcePack(t, source, baseWeights)
	sourcePack.QuantBits = bits
	sourcePack.QuantGroup = groupSize

	adapterWeights := map[string]*metal.Array{
		"model.layers.0.q_proj.lora_A.weight": metal.FromValues(float32Fill(inDim, 1), 1, inDim),
		"model.layers.0.q_proj.lora_B.weight": metal.FromValues([]float32{3, 4}, outDim, 1),
	}
	defer closeTensorMap(adapterWeights)
	writeFuseAdapterWithConfig(t, adapter, `{
		"r": 1,
		"lora_alpha": 2,
		"target_modules": ["q_proj"]
	}`, adapterWeights)

	result, err := FuseIntoPack(context.Background(), FuseOptions{
		SourcePack:  sourcePack,
		AdapterPath: adapter,
		OutputPath:  output,
	})
	if err != nil {
		t.Fatalf("FuseIntoPack() error = %v", err)
	}
	if result.FusedWeights != 1 || len(result.FusedWeightKeys) != 1 || result.FusedWeightKeys[0] != basePrefix+".weight" {
		t.Fatalf("fuse result = %+v, want one raw q6 Gemma4 target", result)
	}

	loaded, err := metal.LoadAllSafetensors(core.PathJoin(output, "model.safetensors"))
	if err != nil {
		t.Fatalf("LoadAllSafetensors fused: %v", err)
	}
	defer closeTensorMap(loaded)
	if _, exists := loaded[basePrefix+".scales"]; exists {
		t.Fatal("fused q6 target retained .scales sidecar; output should load that target as dense")
	}
	if _, exists := loaded[basePrefix+".biases"]; exists {
		t.Fatal("fused q6 target retained .biases sidecar; output should load that target as dense")
	}
	if _, exists := loaded["model.layers.0.self_attn.q_proj.scales"]; exists {
		t.Fatal("fused q6 target retained canonical .scales sidecar alias")
	}
	if _, exists := loaded["model.layers.0.self_attn.q_proj.weight"]; exists {
		t.Fatal("fuse should preserve the source safetensors key instead of adding a duplicate canonical key")
	}
	fused := loaded[basePrefix+".weight"]
	if shape := fused.Shape(); len(shape) != 2 || shape[0] != outDim || shape[1] != inDim {
		t.Fatalf("fused dense shape = %v, want [%d %d]", shape, outDim, inDim)
	}
	got := fused.Floats()
	for i, value := range got[:inDim] {
		if math.Abs(float64(value-6)) > 0.0001 {
			t.Fatalf("fused first output row[%d] = %v, want 6", i, value)
		}
	}
	for i, value := range got[inDim:] {
		if math.Abs(float64(value-8)) > 0.0001 {
			t.Fatalf("fused second output row[%d] = %v, want 8", i, value)
		}
	}
}

func TestFuseIntoPack_QuantizedBaseTargetMissingMetadata_Bad(t *testing.T) {
	requireFuseMetal(t)

	source := core.PathJoin(t.TempDir(), "source")
	adapter := core.PathJoin(t.TempDir(), "adapter")
	output := core.PathJoin(t.TempDir(), "fused")
	if result := core.MkdirAll(source, 0o755); !result.OK {
		t.Fatalf("MkdirAll source: %v", result.Value)
	}
	if result := core.MkdirAll(adapter, 0o755); !result.OK {
		t.Fatalf("MkdirAll adapter: %v", result.Value)
	}

	baseWeights := map[string]*metal.Array{
		"model.layers.0.self_attn.q_proj.weight": metal.FromValues([]uint32{0}, 1, 1),
		"model.layers.0.self_attn.q_proj.scales": metal.FromValues([]float32{1}, 1, 1),
	}
	defer closeTensorMap(baseWeights)
	sourcePack := writeFuseSourcePack(t, source, baseWeights)

	adapterWeights := map[string]*metal.Array{
		"model.layers.0.self_attn.q_proj.lora_a": metal.FromValues([]float32{1}, 1, 1),
		"model.layers.0.self_attn.q_proj.lora_b": metal.FromValues([]float32{1}, 1, 1),
	}
	defer closeTensorMap(adapterWeights)
	writeFuseAdapter(t, adapter, adapterWeights)

	_, err := FuseIntoPack(context.Background(), FuseOptions{
		SourcePack:  sourcePack,
		AdapterPath: adapter,
		OutputPath:  output,
	})
	if err == nil {
		t.Fatal("expected missing quantization metadata error")
	}
	if !core.Contains(err.Error(), "cannot dequantize base target without quantization metadata") ||
		!core.Contains(err.Error(), "model.layers.0.self_attn.q_proj.weight") {
		t.Fatalf("error = %v, want explicit missing quantization metadata context", err)
	}
}

func TestFuseIntoPack_MissingBaseWeight_Bad(t *testing.T) {
	requireFuseMetal(t)

	source := core.PathJoin(t.TempDir(), "source")
	adapter := core.PathJoin(t.TempDir(), "adapter")
	output := core.PathJoin(t.TempDir(), "fused")
	if result := core.MkdirAll(source, 0o755); !result.OK {
		t.Fatalf("MkdirAll source: %v", result.Value)
	}
	if result := core.MkdirAll(adapter, 0o755); !result.OK {
		t.Fatalf("MkdirAll adapter: %v", result.Value)
	}

	baseWeights := map[string]*metal.Array{
		"model.layers.0.self_attn.k_proj.weight": metal.FromValues([]float32{1, 2, 3, 4}, 2, 2),
	}
	defer closeTensorMap(baseWeights)
	sourcePack := writeFuseSourcePack(t, source, baseWeights)

	adapterWeights := map[string]*metal.Array{
		"model.layers.0.self_attn.q_proj.lora_a": metal.FromValues([]float32{1, 2}, 1, 2),
		"model.layers.0.self_attn.q_proj.lora_b": metal.FromValues([]float32{3, 4}, 2, 1),
	}
	defer closeTensorMap(adapterWeights)
	writeFuseAdapter(t, adapter, adapterWeights)

	_, err := FuseIntoPack(context.Background(), FuseOptions{
		SourcePack:  sourcePack,
		AdapterPath: adapter,
		OutputPath:  output,
	})
	if err == nil {
		t.Fatal("expected missing base weight error")
	}
	if !core.Contains(err.Error(), "base weight") {
		t.Fatalf("error = %v, want base weight context", err)
	}
}

func TestFuseIntoPack_CopiesTokenizerConfig_Ugly(t *testing.T) {
	requireFuseMetal(t)

	source := core.PathJoin(t.TempDir(), "source")
	adapter := core.PathJoin(t.TempDir(), "adapter")
	output := core.PathJoin(t.TempDir(), "fused")
	if result := core.MkdirAll(source, 0o755); !result.OK {
		t.Fatalf("MkdirAll source: %v", result.Value)
	}
	if result := core.MkdirAll(adapter, 0o755); !result.OK {
		t.Fatalf("MkdirAll adapter: %v", result.Value)
	}

	baseWeights := map[string]*metal.Array{
		"model.layers.0.self_attn.q_proj.weight": metal.FromValues([]float32{1, 1, 1, 1}, 2, 2),
	}
	defer closeTensorMap(baseWeights)
	sourcePack := writeFuseSourcePack(t, source, baseWeights)
	writeFuseTestFile(t, core.PathJoin(source, "tokenizer_config.json"), `{"chat_template": "{{ messages }}"}`)

	adapterWeights := map[string]*metal.Array{
		"model.layers.0.self_attn.q_proj.lora_a": metal.FromValues([]float32{0, 0}, 1, 2),
		"model.layers.0.self_attn.q_proj.lora_b": metal.FromValues([]float32{0, 0}, 2, 1),
	}
	defer closeTensorMap(adapterWeights)
	writeFuseAdapter(t, adapter, adapterWeights)

	_, err := FuseIntoPack(context.Background(), FuseOptions{
		SourcePack:  sourcePack,
		AdapterPath: adapter,
		OutputPath:  output,
	})
	if err != nil {
		t.Fatalf("FuseIntoPack() error = %v", err)
	}
	copied := core.ReadFile(core.PathJoin(output, "tokenizer_config.json"))
	if !copied.OK {
		t.Fatalf("read copied tokenizer_config.json: %v", copied.Value)
	}
}

func TestBuildFusePairs_ValidationBranches_GoodBad(t *testing.T) {
	a := &metal.Array{}
	b := &metal.Array{}
	pairs, err := buildFusePairs(map[string]*metal.Array{
		"ignored.weight":                         {},
		"model.layers.0.mlp.down_proj.lora_A":    a,
		"model.layers.0.mlp.down_proj.lora_B":    b,
		"model.layers.0.self_attn.q_proj.weight": {},
	})
	if err != nil {
		t.Fatalf("buildFusePairs() error = %v", err)
	}
	pair := pairs["model.layers.0.mlp.down_proj"]
	if pair.MatrixA != a || pair.MatrixB != b {
		t.Fatalf("pair = %+v, want supplied A/B arrays", pair)
	}

	if _, err := buildFusePairs(map[string]*metal.Array{"plain.weight": {}}); err == nil {
		t.Fatal("expected no LoRA tensor pairs error")
	}
	if _, err := buildFusePairs(map[string]*metal.Array{"layer.lora_a": a}); err == nil {
		t.Fatal("expected incomplete LoRA tensor pair error")
	}
}

func TestFuseDarwinPureErrorBranches_Bad(t *testing.T) {
	if _, err := FuseIntoPack(context.Background(), FuseOptions{}); err == nil {
		t.Fatal("expected top-level fuse option validation error")
	}
	if _, err := loadFuseAdapterWeights(core.PathJoin(t.TempDir(), "empty-adapter")); err == nil {
		t.Fatal("expected missing adapter safetensors error")
	}
	if _, _, err := fuseModelWeightFiles(context.Background(), nil, t.TempDir(), nil, 1, pack.ModelPack{}); err == nil {
		t.Fatal("expected no base weight files error")
	}
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, _, err := fuseModelWeightFiles(cancelled, []string{core.PathJoin(t.TempDir(), "missing.safetensors")}, t.TempDir(), nil, 1, pack.ModelPack{}); err != context.Canceled {
		t.Fatalf("fuseModelWeightFiles(cancelled) = %v, want context.Canceled", err)
	}

	pairs := map[string]fusePair{
		"model.layers.0.self_attn.q_proj": {MatrixA: &metal.Array{}, MatrixB: &metal.Array{}},
	}
	fused, err := fuseWeightPairs(context.Background(), map[string]*metal.Array{}, pairs, map[string]struct{}{}, 1, pack.ModelPack{})
	if err != nil {
		t.Fatalf("fuseWeightPairs(missing base) error = %v", err)
	}
	if len(fused) != 0 {
		t.Fatalf("fused keys = %v, want none for missing base", fused)
	}
	if _, err := fuseWeightPairs(cancelled, map[string]*metal.Array{}, pairs, map[string]struct{}{}, 1, pack.ModelPack{}); err != context.Canceled {
		t.Fatalf("fuseWeightPairs(cancelled) = %v, want context.Canceled", err)
	}

	names := outputWeightFileNames([]string{"/tmp/a.safetensors", "/tmp/shard/b.safetensors"})
	if len(names) != 2 || names[0] != "a.safetensors" || names[1] != "b.safetensors" {
		t.Fatalf("outputWeightFileNames() = %v", names)
	}
	freeMetalMap(map[string]*metal.Array{"nil": nil})
}
