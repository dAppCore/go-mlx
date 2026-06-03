// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for pure-CPU LoRA fuse helpers — name matching,
// destination preparation, provenance serialisation. The Metal-side
// matmul path is excluded; this file targets the orchestration scaffolding
// that runs on every fuse invocation regardless of base-weight size.
//
// Per AX-11 — fusePairName fires once per adapter weight name (a rank-16
// adapter touching all attention projections produces ~14 LoRA tensors per
// layer × 28 layers ≈ 400 pair-name lookups), copyModelPackMetadata
// scans the source pack metadata once per fuse, and writeFuseProvenance is
// the closing JSON marshal step.
//
// Run:    go test -bench='BenchmarkFuse' -benchmem -run='^$' ./go/lora

package lora

import (
	"testing"

	core "dappco.re/go"
)

// Sinks defeat compiler DCE. Keep these names distinct from the
// adapter-bench sinks in adapter_bench_test.go.
var (
	fuseBenchSinkString string
	fuseBenchSinkKind   string
	fuseBenchSinkBool   bool
	fuseBenchSinkBase   string
	fuseBenchSinkPaths  []string
	fuseBenchSinkErr    error
	fuseBenchSinkNames  []string
)

// --- fusePairName — the per-tensor suffix matcher.
// Every adapter weight name in the loaded map runs through this; the
// 8-variant suffix table means worst-case is 8 HasSuffix scans.

func BenchmarkFuse_FusePairName_LoraA_LowercaseDotWeight(b *testing.B) {
	name := "model.layers.12.self_attn.q_proj.lora_a.weight"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fuseBenchSinkString, fuseBenchSinkKind, fuseBenchSinkBool = fusePairName(name)
	}
}

func BenchmarkFuse_FusePairName_LoraB_UppercaseBare(b *testing.B) {
	name := "model.layers.12.self_attn.q_proj.lora_B"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fuseBenchSinkString, fuseBenchSinkKind, fuseBenchSinkBool = fusePairName(name)
	}
}

func BenchmarkFuse_FusePairName_LoraA_PEFTUppercaseDotWeight(b *testing.B) {
	name := "model.layers.12.self_attn.q_proj.lora_A.weight"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fuseBenchSinkString, fuseBenchSinkKind, fuseBenchSinkBool = fusePairName(name)
	}
}

// Worst-case: name that's not a LoRA tensor at all — must scan all 8
// suffix candidates before returning false. Real fuse runs hit this
// on every base-weight tensor that flows through buildFusePairs.
func BenchmarkFuse_FusePairName_NonLoraMiss(b *testing.B) {
	name := "model.layers.12.self_attn.q_proj.weight"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fuseBenchSinkString, fuseBenchSinkKind, fuseBenchSinkBool = fusePairName(name)
	}
}

// Sweep a representative qwen3-class adapter weight name set — proxy for
// the inner loop of buildFusePairs over a ~28-layer rank-8 adapter
// touching q/k/v/o + gate/up/down (so 14 lora_a + 14 lora_b per layer).
func BenchmarkFuse_FusePairName_Sweep_RepresentativeNames(b *testing.B) {
	names := []string{
		"model.layers.0.self_attn.q_proj.lora_a",
		"model.layers.0.self_attn.q_proj.lora_b",
		"model.layers.0.self_attn.k_proj.lora_A.weight",
		"model.layers.0.self_attn.k_proj.lora_B.weight",
		"model.layers.0.self_attn.v_proj.lora_a.weight",
		"model.layers.0.self_attn.v_proj.lora_b.weight",
		"model.layers.0.self_attn.o_proj.lora_A",
		"model.layers.0.self_attn.o_proj.lora_B",
		"model.layers.0.mlp.gate_proj.lora_a",
		"model.layers.0.mlp.gate_proj.lora_b",
		"model.layers.0.mlp.up_proj.lora_A.weight",
		"model.layers.0.mlp.up_proj.lora_B.weight",
		"model.layers.0.mlp.down_proj.lora_a.weight",
		"model.layers.0.mlp.down_proj.lora_b.weight",
		"model.layers.0.self_attn.q_proj.weight",
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, name := range names {
			fuseBenchSinkString, fuseBenchSinkKind, fuseBenchSinkBool = fusePairName(name)
		}
	}
}

// --- fuseBaseWeightKey — string concat helper used per fused pair ---

func BenchmarkFuse_FuseBaseWeightKey(b *testing.B) {
	pair := "model.layers.12.self_attn.q_proj"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fuseBenchSinkBase = fuseBaseWeightKey(pair)
	}
}

// --- isModelWeightMetadataCopySkip — the per-file decision when
// copying tokenizer / config metadata from source to fused pack.
// Hit count = number of *.json / *.model / *.txt files in source.

func BenchmarkFuse_IsModelWeightMetadataCopySkip_KeepJSON(b *testing.B) {
	name := "tokenizer.json"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fuseBenchSinkBool = isModelWeightMetadataCopySkip(name)
	}
}

func BenchmarkFuse_IsModelWeightMetadataCopySkip_SkipProvenance(b *testing.B) {
	name := "adapter_provenance.json"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fuseBenchSinkBool = isModelWeightMetadataCopySkip(name)
	}
}

func BenchmarkFuse_IsModelWeightMetadataCopySkip_SkipSafetensorsIndex(b *testing.B) {
	name := "model.safetensors.index.json"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fuseBenchSinkBool = isModelWeightMetadataCopySkip(name)
	}
}

// Uppercase input exercises the case-fold path. Pre-Wave10AC this fired
// strings.ToLower internally and allocated a fresh lowered copy per call;
// the case-fold-in-place containsAsciiLowerFold variant keeps the path
// alloc-free.
func BenchmarkFuse_IsModelWeightMetadataCopySkip_SkipUppercaseGGUF(b *testing.B) {
	name := "MODEL.GGUF"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fuseBenchSinkBool = isModelWeightMetadataCopySkip(name)
	}
}

// --- samePath — invariant check fired once per fuse but uses the
// PathAbs OS round-trip both sides; keep an eye on alloc churn.

func BenchmarkFuse_SamePath_DistinctRelative(b *testing.B) {
	a := "/tmp/source/model"
	c := "/tmp/fused/model"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fuseBenchSinkBool = samePath(a, c)
	}
}

func BenchmarkFuse_SamePath_SameAbsolute(b *testing.B) {
	a := "/tmp/source/model"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fuseBenchSinkBool = samePath(a, a)
	}
}

// --- ensureEmptyFuseWeightDestination — directory probe + glob check
// fired once per fuse. The Stat/Glob OS calls are the cost; this bench
// puts the destination in tmpfs to keep IO predictable.

func BenchmarkFuse_EnsureEmptyDestination_Missing(b *testing.B) {
	root := b.TempDir()
	// Build a path that does NOT exist — the IsNotExist short-circuit.
	missing := core.PathJoin(root, "fused-missing")
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fuseBenchSinkErr = ensureEmptyFuseWeightDestination(missing)
	}
}

func BenchmarkFuse_EnsureEmptyDestination_Empty(b *testing.B) {
	dir := b.TempDir()
	// Directory exists, contains no .safetensors / .gguf — exercises the
	// full Stat OK + Glob path.
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fuseBenchSinkErr = ensureEmptyFuseWeightDestination(dir)
	}
}

// --- fuseAdapterWeightFiles — directory-vs-single-file branch +
// sort. Hit once per fuse, but the slices.Sort + glob is non-trivial.

func BenchmarkFuse_FuseAdapterWeightFiles_DirSorted(b *testing.B) {
	dir := b.TempDir()
	// Out-of-order shards so the sort has work to do.
	for _, name := range []string{"c.safetensors", "a.safetensors", "b.safetensors", "d.safetensors"} {
		if result := core.WriteFile(core.PathJoin(dir, name), []byte("stub"), 0o600); !result.OK {
			b.Fatalf("write %s: %v", name, result.Value)
		}
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fuseBenchSinkPaths, fuseBenchSinkErr = fuseAdapterWeightFiles(dir)
	}
}

func BenchmarkFuse_FuseAdapterWeightFiles_SingleFile(b *testing.B) {
	dir := b.TempDir()
	path := core.PathJoin(dir, "adapter.safetensors")
	if result := core.WriteFile(path, []byte("stub"), 0o600); !result.OK {
		b.Fatalf("write file: %v", result.Value)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fuseBenchSinkPaths, fuseBenchSinkErr = fuseAdapterWeightFiles(path)
	}
}

// --- outputWeightFileNames — basename mapping helper. Fired once
// per fuse over the list of shard paths.

func BenchmarkFuse_OutputWeightFileNames(b *testing.B) {
	paths := []string{
		"/tmp/fused/model-00001-of-00004.safetensors",
		"/tmp/fused/model-00002-of-00004.safetensors",
		"/tmp/fused/model-00003-of-00004.safetensors",
		"/tmp/fused/model-00004-of-00004.safetensors",
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fuseBenchSinkNames = outputWeightFileNames(paths)
	}
}

// --- copyModelPackMetadata — the source pack scan + selective copy.
// Cost scales with metadata-file count in source root. Real qwen3
// packs ship ~6-8 metadata files; gemma4 closer to 10.

func BenchmarkFuse_CopyModelPackMetadata_TypicalSet(b *testing.B) {
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		source := b.TempDir()
		files := map[string]string{
			"config.json":             `{"model_type":"qwen3"}`,
			"tokenizer.json":          `{"model":{"type":"BPE"}}`,
			"tokenizer_config.json":   `{"chat_template":"qwen3"}`,
			"generation_config.json":  `{"max_new_tokens":256}`,
			"special_tokens_map.json": `{"bos_token":"<s>"}`,
			"vocab.json":              `{"<unk>":0}`,
			"merges.txt":              "stub merges",
			"tokenizer.model":         "stub model",
			// These should be skipped — exercises the skip-rule path.
			"adapter_provenance.json": `{"skip":true}`,
			"ignored.safetensors":     "skip",
		}
		for name, content := range files {
			if result := core.WriteFile(core.PathJoin(source, name), []byte(content), 0o600); !result.OK {
				b.Fatalf("write %s: %v", name, result.Value)
			}
		}
		output := b.TempDir()
		b.ReportAllocs()
		b.StartTimer()
		fuseBenchSinkErr = copyModelPackMetadata(source, output)
	}
}

// --- writeFuseProvenance — JSON marshal + sort + WriteFile.
// One-shot per fuse, but the FusedWeightKeys slice grows with the
// number of fused tensor sites (28 layers × 7 projections = ~200).

func BenchmarkFuse_WriteFuseProvenance_SmallFuseSet(b *testing.B) {
	dir := b.TempDir()
	path := core.PathJoin(dir, FuseProvenanceFile)
	provenance := FuseProvenance{
		Version:         1,
		OutputWeight:    "model.safetensors",
		FusedWeightKeys: []string{"model.layers.0.self_attn.q_proj.weight", "model.layers.0.self_attn.v_proj.weight"},
		Labels:          map[string]string{"run": "probe"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fuseBenchSinkErr = writeFuseProvenance(path, provenance)
	}
}

func BenchmarkFuse_WriteFuseProvenance_FullModelFuseSet(b *testing.B) {
	dir := b.TempDir()
	path := core.PathJoin(dir, FuseProvenanceFile)
	// 28 layers × 7 projections — proxy for a qwen3-class full fuse.
	projections := []string{"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"}
	keys := make([]string, 0, 28*len(projections))
	for layer := range 28 {
		for _, proj := range projections {
			keys = append(keys, "model.layers."+itoaFuseBench(layer)+"."+proj+".weight")
		}
	}
	provenance := FuseProvenance{
		Version:         1,
		OutputWeight:    "model.safetensors",
		FusedWeightKeys: keys,
		Labels:          map[string]string{"run": "probe", "arch": "qwen3"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fuseBenchSinkErr = writeFuseProvenance(path, provenance)
	}
}

// itoaFuseBench — minimal integer-to-string helper used during fixture
// build. Kept local to avoid pulling strconv into the bench file.
func itoaFuseBench(n int) string {
	if n == 0 {
		return "0"
	}
	var buf [20]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	return string(buf[i:])
}
