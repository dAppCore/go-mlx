// SPDX-Licence-Identifier: EUPL-1.2

// Second coverage-closing pass for adapter.go + fuse.go — drives the
// residual branches the primary suites and coverage_test.go leave
// uncovered: the prepareFuse validation + orchestration error legs
// (non-safetensors format, adapter-inspect failure, same-as-source,
// dirty destination, MkdirAll failure, metadata-copy failure), the
// copyModelPackMetadata directory-scan legs (unreadable source dir,
// sub-directory skip), the isModelWeightMetadataCopySkip .safetensors
// arm, the ensureEmptyFuseWeightDestination empty-existing-dir
// fall-through, and the metal-gated FuseIntoPack / loadFuseAdapterWeights
// / fuseModelWeightFiles fault legs (corrupt adapter weights, corrupt
// base weights, duplicate adapter tensor names, multi-shard output
// naming, SaveSafetensors + provenance write failures).

package lora

import (
	"context"
	"math"
	"testing"

	core "dappco.re/go"
	pack "dappco.re/go/inference/modelpack"
	"dappco.re/go/mlx/pkg/metal"
)

// --- fuse.go: prepareFuse format guard (errFuseRequiresSafetensors) ---

func TestPrepareFuse_NonSafetensorsFormat_Bad(t *testing.T) {
	// Bad: prepareFuse rejects a source pack whose format is not safetensors
	// (fuse.go L112-114). The output path passes the pack-directory shape
	// check (no .safetensors / .gguf suffix), so the format guard is the
	// first failing clause. No adapter inspection runs — validation fails
	// before the Inspect call.
	_, err := prepareFuse(context.Background(), FuseOptions{
		SourcePack:  pack.ModelPack{Root: "/tmp/source", Format: pack.ModelPackFormatGGUF},
		AdapterPath: "/tmp/adapter",
		OutputPath:  core.PathJoin(t.TempDir(), "fused"),
	})
	if err != errFuseRequiresSafetensors {
		t.Fatalf("prepareFuse(non-safetensors format) = %v, want errFuseRequiresSafetensors", err)
	}
}

// --- fuse.go: prepareFuse adapter-inspect wrap ---

func TestPrepareFuse_AdapterInspectFailure_Bad(t *testing.T) {
	// Bad: prepareFuse wraps an Inspect failure (fuse.go L117-119). An adapter
	// path with no adapter_config.json makes Inspect's ReadFile fail, and the
	// wrap carries the "inspect LoRA adapter" context. The source pack is a
	// valid safetensors-format pack so the earlier guards pass and the adapter
	// inspection is reached.
	source := t.TempDir()
	adapter := t.TempDir() // empty — no adapter_config.json
	_, err := prepareFuse(context.Background(), FuseOptions{
		SourcePack:  pack.ModelPack{Root: source, Path: source, Format: pack.ModelPackFormatSafetensors},
		AdapterPath: adapter,
		OutputPath:  core.PathJoin(t.TempDir(), "fused"),
	})
	if err == nil {
		t.Fatal("prepareFuse(missing adapter_config) = nil, want inspect failure")
	}
	if !core.Contains(err.Error(), "inspect LoRA adapter") {
		t.Fatalf("error = %v, want inspect-adapter context", err)
	}
}

// --- fuse.go: prepareFuse same-as-source guard ---

func TestPrepareFuse_OutputSameAsSource_Bad(t *testing.T) {
	// Bad: prepareFuse rejects an output path that resolves to the source pack
	// root (fuse.go L135-137) — fusing in-place would clobber the base
	// weights. Point OutputPath at the source root directory; samePath's
	// identity fast-path returns true and the guard fires. A complete adapter
	// (rank + safetensors) ensures the same-as-source check is reached after
	// the adapter metadata is validated.
	source := t.TempDir()
	adapter := t.TempDir()
	writeFuseTestFile(t, core.PathJoin(source, "config.json"), `{"model_type":"qwen3"}`)
	writeFuseTestFile(t, core.PathJoin(adapter, "adapter_config.json"), `{"rank":4,"alpha":8,"target_modules":["q_proj"]}`)
	writeFuseTestFile(t, core.PathJoin(adapter, "adapter.safetensors"), "stub")

	_, err := prepareFuse(context.Background(), FuseOptions{
		SourcePack:  pack.ModelPack{Root: source, Path: source, Format: pack.ModelPackFormatSafetensors},
		AdapterPath: adapter,
		OutputPath:  source,
	})
	if err != errFuseOutputSameAsSource {
		t.Fatalf("prepareFuse(output == source) = %v, want errFuseOutputSameAsSource", err)
	}
}

// --- fuse.go: prepareFuse dirty-destination propagation ---

func TestPrepareFuse_DirtyDestination_Bad(t *testing.T) {
	// Bad: prepareFuse propagates the ensureEmptyFuseWeightDestination error
	// (fuse.go L138-140) when the output directory already holds weights. An
	// output dir pre-seeded with a model.safetensors makes the destination
	// check return errFuseOutputContainsWeight, which prepareFuse returns
	// verbatim.
	source := t.TempDir()
	adapter := t.TempDir()
	output := core.PathJoin(t.TempDir(), "fused")
	if result := core.MkdirAll(output, 0o755); !result.OK {
		t.Fatalf("MkdirAll output: %s", result.Error())
	}
	if result := core.WriteFile(core.PathJoin(output, "model.safetensors"), []byte("weights"), 0o644); !result.OK {
		t.Fatalf("WriteFile output weights: %s", result.Error())
	}
	writeFuseTestFile(t, core.PathJoin(source, "config.json"), `{"model_type":"qwen3"}`)
	writeFuseTestFile(t, core.PathJoin(adapter, "adapter_config.json"), `{"rank":4,"alpha":8,"target_modules":["q_proj"]}`)
	writeFuseTestFile(t, core.PathJoin(adapter, "adapter.safetensors"), "stub")

	_, err := prepareFuse(context.Background(), FuseOptions{
		SourcePack:  pack.ModelPack{Root: source, Path: source, Format: pack.ModelPackFormatSafetensors},
		AdapterPath: adapter,
		OutputPath:  output,
	})
	if err != errFuseOutputContainsWeight {
		t.Fatalf("prepareFuse(dirty destination) = %v, want errFuseOutputContainsWeight", err)
	}
}

// --- fuse.go: prepareFuse MkdirAll failure ---

func TestPrepareFuse_MkdirAllFailure_Bad(t *testing.T) {
	// Bad: prepareFuse wraps a MkdirAll failure on the output directory
	// (fuse.go L141-143). An output path nested under a read-only parent
	// cannot be created (uid != 0 honours the 0o500 mode), so MkdirAll fails
	// after the same-as-source + dirty-destination checks pass (the not-exist
	// destination is accepted). The wrap carries the create-directory context.
	source := t.TempDir()
	adapter := t.TempDir()
	roParent := core.PathJoin(t.TempDir(), "ro")
	if result := core.MkdirAll(roParent, 0o755); !result.OK {
		t.Fatalf("MkdirAll roParent: %s", result.Error())
	}
	if result := core.Chmod(roParent, 0o500); !result.OK {
		t.Fatalf("Chmod roParent: %s", result.Error())
	}
	t.Cleanup(func() { core.Chmod(roParent, 0o755) })
	writeFuseTestFile(t, core.PathJoin(source, "config.json"), `{"model_type":"qwen3"}`)
	writeFuseTestFile(t, core.PathJoin(adapter, "adapter_config.json"), `{"rank":4,"alpha":8,"target_modules":["q_proj"]}`)
	writeFuseTestFile(t, core.PathJoin(adapter, "adapter.safetensors"), "stub")

	_, err := prepareFuse(context.Background(), FuseOptions{
		SourcePack:  pack.ModelPack{Root: source, Path: source, Format: pack.ModelPackFormatSafetensors},
		AdapterPath: adapter,
		OutputPath:  core.PathJoin(roParent, "fused"),
	})
	if err == nil {
		t.Fatal("prepareFuse(uncreatable output) = nil, want MkdirAll failure")
	}
	if !core.Contains(err.Error(), "create fused model directory") {
		t.Fatalf("error = %v, want create-directory context", err)
	}
}

// --- fuse.go: prepareFuse metadata-copy failure ---

func TestPrepareFuse_MetadataCopyFailure_Bad(t *testing.T) {
	// Bad: prepareFuse propagates a copyModelPackMetadata failure (fuse.go
	// L144-146). The source root holds a config.json the suffix filter accepts
	// but chmod'd 0o000 so its read fails inside copyLocalFile. MkdirAll has
	// already created the output dir, so the metadata copy is the first
	// failing step and its read-error context propagates out.
	source := t.TempDir()
	adapter := t.TempDir()
	output := core.PathJoin(t.TempDir(), "fused")
	blockedConfig := core.PathJoin(source, "config.json")
	if result := core.WriteFile(blockedConfig, []byte(`{"model_type":"qwen3"}`), 0o644); !result.OK {
		t.Fatalf("WriteFile source config: %s", result.Error())
	}
	if result := core.Chmod(blockedConfig, 0o000); !result.OK {
		t.Fatalf("Chmod source config: %s", result.Error())
	}
	t.Cleanup(func() { core.Chmod(blockedConfig, 0o644) })
	writeFuseTestFile(t, core.PathJoin(adapter, "adapter_config.json"), `{"rank":4,"alpha":8,"target_modules":["q_proj"]}`)
	writeFuseTestFile(t, core.PathJoin(adapter, "adapter.safetensors"), "stub")

	_, err := prepareFuse(context.Background(), FuseOptions{
		SourcePack:  pack.ModelPack{Root: source, Path: source, Format: pack.ModelPackFormatSafetensors},
		AdapterPath: adapter,
		OutputPath:  output,
	})
	if err == nil {
		t.Fatal("prepareFuse(unreadable source metadata) = nil, want metadata-copy failure")
	}
	if !core.Contains(err.Error(), "read "+blockedConfig) {
		t.Fatalf("error = %v, want propagated metadata read failure", err)
	}
}

// --- fuse.go: ensureEmptyFuseWeightDestination empty-existing-dir pass ---

func TestEnsureEmptyFuseWeightDestination_EmptyExistingDir_Good(t *testing.T) {
	// Good: ensureEmptyFuseWeightDestination accepts an existing directory that
	// holds no weight files (fuse.go L179 final return). The existing suites
	// cover the not-exist branch (returns at the IsNotExist check) and both
	// dirty-destination arms; this drives the stat-OK, no-weights fall-through
	// — an empty directory that already exists.
	dir := t.TempDir() // exists, empty
	if err := ensureEmptyFuseWeightDestination(dir); err != nil {
		t.Fatalf("ensureEmptyFuseWeightDestination(empty existing dir) = %v, want nil", err)
	}
}

// --- fuse.go: copyModelPackMetadata unreadable-source + sub-directory legs ---

func TestCopyModelPackMetadata_UnreadableSource_Good(t *testing.T) {
	// Good: copyModelPackMetadata treats an unreadable source directory as a
	// no-op (fuse.go L263-268) — a missing/unreadable metadata dir is not
	// fatal because the fuse still produced weights. A source path that does
	// not exist makes ReadDir fail, so the function returns nil without
	// copying anything.
	missing := core.PathJoin(t.TempDir(), "does-not-exist")
	output := t.TempDir()
	if err := copyModelPackMetadata(missing, output); err != nil {
		t.Fatalf("copyModelPackMetadata(missing source) = %v, want nil (non-fatal)", err)
	}
}

func TestCopyModelPackMetadata_SkipsSubdirectory_Ugly(t *testing.T) {
	// Ugly: copyModelPackMetadata skips directory entries (fuse.go L274-275
	// continue) so a metadata-suffixed sub-directory is never fed to
	// copyLocalFile. A source dir holding a real config.json plus a directory
	// named "nested.json" proves the file is copied while the same-suffixed
	// directory is skipped — no error, and the directory does not appear in
	// the output.
	source := t.TempDir()
	output := t.TempDir()
	writeFuseTestFile(t, core.PathJoin(source, "config.json"), `{"model_type":"qwen3"}`)
	// A sub-directory whose name carries a metadata suffix — must be skipped
	// by the IsDir() guard, not copied.
	if result := core.MkdirAll(core.PathJoin(source, "nested.json"), 0o755); !result.OK {
		t.Fatalf("MkdirAll nested.json dir: %s", result.Error())
	}

	if err := copyModelPackMetadata(source, output); err != nil {
		t.Fatalf("copyModelPackMetadata(with subdir) = %v, want nil", err)
	}
	if stat := core.Stat(core.PathJoin(output, "config.json")); !stat.OK {
		t.Fatalf("config.json was not copied: %v", stat.Value)
	}
	if stat := core.Stat(core.PathJoin(output, "nested.json")); stat.OK {
		t.Fatal("nested.json directory should have been skipped, not copied")
	}
}

// --- fuse.go: isModelWeightMetadataCopySkip .safetensors arm ---

func TestIsModelWeightMetadataCopySkip_SafetensorsArm_Bad(t *testing.T) {
	// Bad: the .safetensors substring arm (fuse.go L322-324) skips safetensors
	// weight + index files from the metadata copy. coverage_test.go covers the
	// .gguf arm and the provenance-name equality; the existing MODEL.GGUF case
	// short-circuits before this arm. Drive a lowercase .safetensors name and a
	// .safetensors.index.json name directly so the .safetensors fold path
	// (which the .gguf-only and provenance cases never reach) is exercised.
	for _, name := range []string{
		"model.safetensors",
		"model.safetensors.index.json",
		"weights-00001-of-00002.safetensors",
	} {
		if !isModelWeightMetadataCopySkip(name) {
			t.Fatalf("isModelWeightMetadataCopySkip(%q) = false, want true (.safetensors skip)", name)
		}
	}
	// A name merely containing "safetensors" without the leading dot is NOT
	// skipped — the substring is ".safetensors", anchoring the extension.
	if isModelWeightMetadataCopySkip("safetensors-notes.json") {
		t.Fatal("isModelWeightMetadataCopySkip(safetensors-notes.json) = true, want false")
	}
}

// --- fuse.go (metal-gated): loadFuseAdapterWeights load failure ---

func TestLoadFuseAdapterWeights_CorruptWeights_Bad(t *testing.T) {
	// Bad: loadFuseAdapterWeights wraps a LoadAllSafetensors failure (fuse.go
	// L742-744) and frees any tensors already loaded. An adapter directory
	// holding a single .safetensors file of garbage bytes makes the native
	// load fail with an invalid-header error, which the wrap carries with the
	// load-adapter-weights context. Metal-gated: the native loader must run.
	requireFuseMetal(t)
	dir := t.TempDir()
	if result := core.WriteFile(core.PathJoin(dir, "adapter.safetensors"), []byte("definitely not a safetensors header"), 0o644); !result.OK {
		t.Fatalf("WriteFile corrupt adapter: %s", result.Error())
	}
	_, err := loadFuseAdapterWeights(dir)
	if err == nil {
		t.Fatal("loadFuseAdapterWeights(corrupt) = nil, want load failure")
	}
	if !core.Contains(err.Error(), "load adapter weights") {
		t.Fatalf("error = %v, want load-adapter-weights context", err)
	}
}

// --- fuse.go (metal-gated): loadFuseAdapterWeights duplicate-name free ---

func TestLoadFuseAdapterWeights_DuplicateNameAcrossShards_Ugly(t *testing.T) {
	// Ugly: when two adapter shards define the same tensor name,
	// loadFuseAdapterWeights frees the earlier tensor before overwriting the
	// map entry (fuse.go L747-749) so the duplicate does not leak. Two
	// safetensors shards sorted so both carry the key "dup" force the second
	// load to hit the previous != nil free arm. The final map keeps exactly
	// one entry. Metal-gated: real tensors are loaded.
	requireFuseMetal(t)
	dir := t.TempDir()
	first := metal.FromValues([]float32{1}, 1, 1)
	second := metal.FromValues([]float32{2}, 1, 1)
	if err := metal.SaveSafetensors(core.PathJoin(dir, "adapter-00001.safetensors"), map[string]*metal.Array{"dup": first}); err != nil {
		t.Fatalf("SaveSafetensors shard 1: %v", err)
	}
	if err := metal.SaveSafetensors(core.PathJoin(dir, "adapter-00002.safetensors"), map[string]*metal.Array{"dup": second}); err != nil {
		t.Fatalf("SaveSafetensors shard 2: %v", err)
	}
	metal.Free(first, second)

	weights, err := loadFuseAdapterWeights(dir)
	if err != nil {
		t.Fatalf("loadFuseAdapterWeights(dup shards) error = %v", err)
	}
	defer freeMetalMap(weights)
	if len(weights) != 1 {
		t.Fatalf("loaded %d tensors, want 1 (duplicate name freed + overwritten)", len(weights))
	}
	if weights["dup"] == nil {
		t.Fatal("dup tensor missing after duplicate-name overwrite")
	}
}

// --- fuse.go (metal-gated): FuseIntoPack adapter-load failure leg ---

func TestFuseIntoPack_CorruptAdapterWeights_Bad(t *testing.T) {
	// Bad: FuseIntoPack returns the loadFuseAdapterWeights error (fuse.go
	// L688-690). A complete adapter_config.json (so prepareFuse succeeds) plus
	// a corrupt adapter.safetensors makes the post-prepare weight load fail.
	// The source pack is a valid dense safetensors pack so prepareFuse runs to
	// completion before the load is attempted. Metal-gated.
	requireFuseMetal(t)
	source := core.PathJoin(t.TempDir(), "source")
	adapter := core.PathJoin(t.TempDir(), "adapter")
	output := core.PathJoin(t.TempDir(), "fused")
	if result := core.MkdirAll(source, 0o755); !result.OK {
		t.Fatalf("MkdirAll source: %s", result.Error())
	}
	if result := core.MkdirAll(adapter, 0o755); !result.OK {
		t.Fatalf("MkdirAll adapter: %s", result.Error())
	}
	baseWeights := map[string]*metal.Array{
		"model.layers.0.self_attn.q_proj.weight": metal.FromValues([]float32{1, 1, 1, 1}, 2, 2),
	}
	defer closeTensorMap(baseWeights)
	sourcePack := writeFuseSourcePack(t, source, baseWeights)
	writeFuseTestFile(t, core.PathJoin(adapter, "adapter_config.json"), `{"rank":1,"alpha":2,"lora_layers":["self_attn.q_proj"]}`)
	if result := core.WriteFile(core.PathJoin(adapter, "adapter.safetensors"), []byte("garbage not safetensors"), 0o644); !result.OK {
		t.Fatalf("WriteFile corrupt adapter: %s", result.Error())
	}

	_, err := FuseIntoPack(context.Background(), FuseOptions{
		SourcePack:  sourcePack,
		AdapterPath: adapter,
		OutputPath:  output,
	})
	if err == nil {
		t.Fatal("FuseIntoPack(corrupt adapter) = nil, want adapter-load failure")
	}
	if !core.Contains(err.Error(), "load adapter weights") {
		t.Fatalf("error = %v, want load-adapter-weights context", err)
	}
}

// --- fuse.go (metal-gated): FuseIntoPack buildFusePairs failure leg ---

func TestFuseIntoPack_NoLoRATensorPairs_Bad(t *testing.T) {
	// Bad: FuseIntoPack returns the buildFusePairs error (fuse.go L694-696)
	// when the adapter safetensors load cleanly but hold no lora_a/lora_b
	// tensor pairs. An adapter whose only tensor is a non-LoRA weight makes
	// buildFusePairs return errFuseNoLoRATensorPairs. Metal-gated: the adapter
	// weights are really loaded before the pairing is attempted.
	requireFuseMetal(t)
	source := core.PathJoin(t.TempDir(), "source")
	adapter := core.PathJoin(t.TempDir(), "adapter")
	output := core.PathJoin(t.TempDir(), "fused")
	if result := core.MkdirAll(source, 0o755); !result.OK {
		t.Fatalf("MkdirAll source: %s", result.Error())
	}
	if result := core.MkdirAll(adapter, 0o755); !result.OK {
		t.Fatalf("MkdirAll adapter: %s", result.Error())
	}
	baseWeights := map[string]*metal.Array{
		"model.layers.0.self_attn.q_proj.weight": metal.FromValues([]float32{1, 1, 1, 1}, 2, 2),
	}
	defer closeTensorMap(baseWeights)
	sourcePack := writeFuseSourcePack(t, source, baseWeights)
	// Valid adapter config + a safetensors file with only a non-LoRA tensor.
	noPairWeights := map[string]*metal.Array{
		"model.layers.0.self_attn.q_proj.weight": metal.FromValues([]float32{1}, 1, 1),
	}
	defer closeTensorMap(noPairWeights)
	writeFuseAdapterWithConfig(t, adapter, `{"rank":1,"alpha":2,"lora_layers":["self_attn.q_proj"]}`, noPairWeights)

	_, err := FuseIntoPack(context.Background(), FuseOptions{
		SourcePack:  sourcePack,
		AdapterPath: adapter,
		OutputPath:  output,
	})
	if err != errFuseNoLoRATensorPairs {
		t.Fatalf("FuseIntoPack(no lora pairs) = %v, want errFuseNoLoRATensorPairs", err)
	}
}

// --- fuse.go (metal-gated): fuseModelWeightFiles base-load failure leg ---

func TestFuseModelWeightFiles_CorruptBaseWeights_Bad(t *testing.T) {
	// Bad: fuseModelWeightFiles wraps a LoadAllSafetensors failure on a base
	// weight shard (fuse.go L808-810). A source weight file of garbage bytes
	// makes the per-shard load fail, and the wrap carries the load-base-weights
	// context. Driven directly so the failure isolates this leg without
	// needing a full FuseIntoPack run. Metal-gated.
	requireFuseMetal(t)
	dir := t.TempDir()
	badShard := core.PathJoin(dir, "model.safetensors")
	if result := core.WriteFile(badShard, []byte("not a safetensors shard"), 0o644); !result.OK {
		t.Fatalf("WriteFile corrupt base shard: %s", result.Error())
	}
	pairs := map[string]fusePair{
		"model.layers.0.self_attn.q_proj": {MatrixA: metal.FromValues([]float32{1}, 1, 1), MatrixB: metal.FromValues([]float32{1}, 1, 1)},
	}
	defer func() {
		for _, p := range pairs {
			metal.Free(p.MatrixA, p.MatrixB)
		}
	}()
	_, _, err := fuseModelWeightFiles(context.Background(), []string{badShard}, t.TempDir(), pairs, 1, pack.ModelPack{})
	if err == nil {
		t.Fatal("fuseModelWeightFiles(corrupt base) = nil, want load failure")
	}
	if !core.Contains(err.Error(), "load base weights") {
		t.Fatalf("error = %v, want load-base-weights context", err)
	}
}

// --- fuse.go (metal-gated): fuseModelWeightFiles multi-shard naming + save failure ---

func TestFuseModelWeightFiles_MultiShardNaming_Good(t *testing.T) {
	// Good: a multi-shard base (len(sourceFiles) > 1) preserves each source
	// file's basename for the fused output instead of the canonical
	// model.safetensors name (fuse.go L820-822). Two real source shards, each
	// holding a fusable q_proj weight, produce two fused output files named
	// after their sources. Metal-gated: the matmul + add + save run per shard.
	requireFuseMetal(t)
	srcDir := t.TempDir()
	outDir := core.PathJoin(t.TempDir(), "out")
	if result := core.MkdirAll(outDir, 0o755); !result.OK {
		t.Fatalf("MkdirAll outDir: %s", result.Error())
	}
	shardA := core.PathJoin(srcDir, "model-00001-of-00002.safetensors")
	shardB := core.PathJoin(srcDir, "model-00002-of-00002.safetensors")
	if err := metal.SaveSafetensors(shardA, map[string]*metal.Array{
		"model.layers.0.self_attn.q_proj.weight": metal.FromValues([]float32{0, 0, 0, 0}, 2, 2),
	}); err != nil {
		t.Fatalf("SaveSafetensors shardA: %v", err)
	}
	if err := metal.SaveSafetensors(shardB, map[string]*metal.Array{
		"model.layers.1.self_attn.q_proj.weight": metal.FromValues([]float32{0, 0, 0, 0}, 2, 2),
	}); err != nil {
		t.Fatalf("SaveSafetensors shardB: %v", err)
	}
	pairs := map[string]fusePair{
		"model.layers.0.self_attn.q_proj": {MatrixA: metal.FromValues([]float32{1, 2}, 1, 2), MatrixB: metal.FromValues([]float32{3, 4}, 2, 1)},
		"model.layers.1.self_attn.q_proj": {MatrixA: metal.FromValues([]float32{1, 2}, 1, 2), MatrixB: metal.FromValues([]float32{3, 4}, 2, 1)},
	}
	defer func() {
		for _, p := range pairs {
			metal.Free(p.MatrixA, p.MatrixB)
		}
	}()

	weightFiles, fusedKeys, err := fuseModelWeightFiles(context.Background(), []string{shardA, shardB}, outDir, pairs, 2, pack.ModelPack{Architecture: "qwen3"})
	if err != nil {
		t.Fatalf("fuseModelWeightFiles(multi-shard) error = %v", err)
	}
	if len(weightFiles) != 2 {
		t.Fatalf("weightFiles = %v, want 2 shard outputs", weightFiles)
	}
	if core.PathBase(weightFiles[0]) != "model-00001-of-00002.safetensors" || core.PathBase(weightFiles[1]) != "model-00002-of-00002.safetensors" {
		t.Fatalf("multi-shard output names = %v, want source basenames preserved", weightFiles)
	}
	if len(fusedKeys) != 2 {
		t.Fatalf("fusedKeys = %v, want both shards fused", fusedKeys)
	}
}

func TestFuseModelWeightFiles_SaveFailure_Bad(t *testing.T) {
	// Bad: fuseModelWeightFiles wraps a SaveSafetensors failure (fuse.go
	// L826-829). A read-only output directory makes the native save of the
	// fused shard fail after the matmul + add succeed, and the wrap carries
	// the save-fused-safetensors context. Metal-gated: the fuse arithmetic
	// runs before the save is attempted.
	requireFuseMetal(t)
	srcDir := t.TempDir()
	shard := core.PathJoin(srcDir, "model.safetensors")
	if err := metal.SaveSafetensors(shard, map[string]*metal.Array{
		"model.layers.0.self_attn.q_proj.weight": metal.FromValues([]float32{0, 0, 0, 0}, 2, 2),
	}); err != nil {
		t.Fatalf("SaveSafetensors source shard: %v", err)
	}
	roOut := core.PathJoin(t.TempDir(), "ro")
	if result := core.MkdirAll(roOut, 0o755); !result.OK {
		t.Fatalf("MkdirAll roOut: %s", result.Error())
	}
	if result := core.Chmod(roOut, 0o500); !result.OK {
		t.Fatalf("Chmod roOut: %s", result.Error())
	}
	t.Cleanup(func() { core.Chmod(roOut, 0o755) })
	pairs := map[string]fusePair{
		"model.layers.0.self_attn.q_proj": {MatrixA: metal.FromValues([]float32{1, 2}, 1, 2), MatrixB: metal.FromValues([]float32{3, 4}, 2, 1)},
	}
	defer func() {
		for _, p := range pairs {
			metal.Free(p.MatrixA, p.MatrixB)
		}
	}()

	_, _, err := fuseModelWeightFiles(context.Background(), []string{shard}, roOut, pairs, 1, pack.ModelPack{Architecture: "qwen3"})
	if err == nil {
		t.Fatal("fuseModelWeightFiles(readonly output) = nil, want save failure")
	}
	if !core.Contains(err.Error(), "save fused safetensors") {
		t.Fatalf("error = %v, want save-fused-safetensors context", err)
	}
}

// --- fuse.go (metal-gated): FuseIntoPack provenance-write failure leg ---

func TestFuseIntoPack_ProvenanceWriteFailure_Bad(t *testing.T) {
	// Bad: FuseIntoPack returns the writeFuseProvenance error (fuse.go
	// L719-721). The fused weights save successfully into the output dir, but
	// a directory pre-created at the adapter_provenance.json path makes the
	// provenance WriteFile fail with EISDIR — the only way to fail the
	// provenance write while the weight save succeeds. Metal-gated: the full
	// fuse runs up to the provenance write.
	requireFuseMetal(t)
	source := core.PathJoin(t.TempDir(), "source")
	adapter := core.PathJoin(t.TempDir(), "adapter")
	output := core.PathJoin(t.TempDir(), "fused")
	if result := core.MkdirAll(source, 0o755); !result.OK {
		t.Fatalf("MkdirAll source: %s", result.Error())
	}
	if result := core.MkdirAll(adapter, 0o755); !result.OK {
		t.Fatalf("MkdirAll adapter: %s", result.Error())
	}
	// Pre-create the output dir and a *directory* where the provenance file
	// would be written. prepareFuse's MkdirAll(output) is idempotent, the
	// destination holds no weights so ensureEmpty passes, and the provenance
	// WriteFile later collides with the directory.
	if result := core.MkdirAll(core.PathJoin(output, FuseProvenanceFile), 0o755); !result.OK {
		t.Fatalf("MkdirAll provenance-collision dir: %s", result.Error())
	}
	baseWeights := map[string]*metal.Array{
		"model.layers.0.self_attn.q_proj.weight": metal.FromValues([]float32{0, 0, 0, 0}, 2, 2),
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
		t.Fatal("FuseIntoPack(provenance path is a dir) = nil, want provenance-write failure")
	}
	if !core.Contains(err.Error(), "write adapter provenance") {
		t.Fatalf("error = %v, want write-provenance context", err)
	}
}

// --- fuse.go: writeFuseProvenance marshal failure ---

func TestWriteFuseProvenance_MarshalFailure_Bad(t *testing.T) {
	// Bad: writeFuseProvenance wraps a JSONMarshal failure (fuse.go L656-658).
	// core.JSONMarshal is plain encoding/json.Marshal, which rejects NaN and
	// ±Inf floats with an UnsupportedValueError. AdapterInfo.Alpha is a
	// float32, so a NaN there makes the marshal of the embedded Adapter fail
	// before any write is attempted — the only reachable marshal error for an
	// otherwise JSON-safe FuseProvenance. (NaN is not the zero value, so the
	// omitempty tag does not drop it.) A writable temp path proves the failure
	// is the marshal, not the write.
	path := core.PathJoin(t.TempDir(), FuseProvenanceFile)
	err := writeFuseProvenance(path, FuseProvenance{
		Version: 1,
		Adapter: AdapterInfo{Alpha: float32(math.NaN())},
	})
	if err == nil {
		t.Fatal("writeFuseProvenance(NaN alpha) = nil, want marshal failure")
	}
	if !core.Contains(err.Error(), "marshal adapter provenance") {
		t.Fatalf("error = %v, want marshal-provenance context", err)
	}
	if stat := core.Stat(path); stat.OK {
		t.Fatal("provenance file was written despite the marshal failure")
	}
}
