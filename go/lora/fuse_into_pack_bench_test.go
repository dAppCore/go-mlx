// SPDX-Licence-Identifier: EUPL-1.2

// End-to-end benchmark for the FuseIntoPack orchestration — the per-tensor
// fuse of a LoRA adapter into a model pack. The CPU-only fuse helpers
// (name matching, metadata copy, provenance) are benched in
// fuse_bench_test.go; this file drives the full pack-level fuse, which
// requires the native Metal runtime (Matmul/MulScalar/Add/Materialize +
// safetensors load/save).
//
// Per AX-11 — FuseIntoPack runs once per adapter-merge request. The lora
// package owns the orchestration scaffolding (pair build, base-weight index,
// per-shard fused-key accumulation); the tensor math + safetensors IO live
// in pkg/metal. This bench measures the orchestration allocs the lora
// package is responsible for, over a synthetic small adapter + base pack.
//
// The Gemma4 fixture is primary: it exercises
// fuseBaseWeightIndexForArchitecture + fuseBaseWeightSidecars (the
// canonical-name index + per-pair sidecar probe), which the qwen3 path
// skips entirely (the index builder returns nil for non-gemma4 archs).
//
// Run:  go test -tags metal_runtime -bench='BenchmarkFuseIntoPack' -benchmem -run='^$' ./go/lora
//       (set MLX_METALLIB_PATH to dist/lib/mlx.metallib)

package lora

import (
	"context"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/mlx/internal/metaltest"
	pack "dappco.re/go/inference/modelpack"
	"dappco.re/go/mlx/pkg/metal"
)

var fuseIntoPackBenchSinkResult *FuseResult

// requireFuseMetalBench mirrors requireFuseMetal for the *testing.B side —
// the test helper takes *testing.T, and the Skip/Skipf calls there are not
// reachable from a benchmark. Same gate: the native fuse path needs the
// metal_runtime build tag and an available Metal device.
func requireFuseMetalBench(b *testing.B) {
	b.Helper()
	if !metaltest.RunMetalTests {
		b.Skip("build with -tags metal_runtime to enable native LoRA fuse benchmarks")
	}
	if !metal.MetalAvailable() {
		b.Skip("Metal runtime unavailable")
	}
}

// fuseBenchAdapterTensors builds a small Gemma4-shaped LoRA adapter touching
// the four attention projections of one layer (q/k/v/o), rank 2. Eight LoRA
// tensors (4 projections × lora_A/lora_B) drive buildFusePairs over a
// representative-but-cheap pair set. Caller owns Free via closeBenchTensorMap.
func fuseBenchAdapterTensors() map[string]*metal.Array {
	const rank = 2
	proj := []string{"q_proj", "k_proj", "v_proj", "o_proj"}
	tensors := make(map[string]*metal.Array, len(proj)*2)
	for _, p := range proj {
		key := "model.layers.0." + p
		// lora_A: [rank, in], lora_B: [out, rank]; in=out=2 keeps the
		// Metal matmul tiny while still exercising the full fuse chain.
		tensors[key+".lora_A.weight"] = metal.FromValues([]float32{1, 0, 0, 1}, rank, 2)
		tensors[key+".lora_B.weight"] = metal.FromValues([]float32{1, 0, 0, 1}, 2, rank)
	}
	return tensors
}

// fuseBenchBaseTensors builds the dense Gemma4 base weights the adapter above
// fuses into — the canonical self_attn.<proj>.weight keys plus one unmatched
// MLP weight so the index/match path has a non-target to skip.
func fuseBenchBaseTensors() map[string]*metal.Array {
	proj := []string{"q_proj", "k_proj", "v_proj", "o_proj"}
	tensors := make(map[string]*metal.Array, len(proj)+1)
	for _, p := range proj {
		tensors["model.layers.0.self_attn."+p+".weight"] = metal.FromValues([]float32{1, 2, 3, 4}, 2, 2)
	}
	// An unmatched base weight — the fuse loop must walk past it.
	tensors["model.layers.0.mlp.gate_proj.weight"] = metal.FromValues([]float32{5, 6, 7, 8}, 2, 2)
	return tensors
}

func closeBenchTensorMap(tensors map[string]*metal.Array) {
	for _, t := range tensors {
		metal.Free(t)
	}
}

// writeGemma4FuseBenchSourcePack writes a Gemma4 source pack to dir and
// returns the ModelPack. Mirrors writeGemma4FuseSourcePack but takes a
// *testing.B (the test helper is *testing.T-only).
func writeGemma4FuseBenchSourcePack(b *testing.B, dir string, tensors map[string]*metal.Array) pack.ModelPack {
	b.Helper()
	cfg := `{
		"model_type": "gemma4_text",
		"vocab_size": 262144,
		"hidden_size": 2,
		"num_hidden_layers": 1,
		"max_position_embeddings": 262144
	}`
	if result := core.WriteFile(core.PathJoin(dir, "config.json"), []byte(cfg), 0o644); !result.OK {
		b.Fatalf("write config.json: %v", result.Value)
	}
	if result := core.WriteFile(core.PathJoin(dir, "tokenizer.json"), []byte(`{"model":{"type":"BPE"}}`), 0o644); !result.OK {
		b.Fatalf("write tokenizer.json: %v", result.Value)
	}
	weightPath := core.PathJoin(dir, "model.safetensors")
	if err := metal.SaveSafetensors(weightPath, tensors); err != nil {
		b.Fatalf("SaveSafetensors gemma4 bench source: %v", err)
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

func writeGemma4FuseBenchAdapter(b *testing.B, dir string, tensors map[string]*metal.Array) {
	b.Helper()
	cfg := `{
		"r": 2,
		"lora_alpha": 4,
		"target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]
	}`
	if result := core.WriteFile(core.PathJoin(dir, "adapter_config.json"), []byte(cfg), 0o644); !result.OK {
		b.Fatalf("write adapter_config.json: %v", result.Value)
	}
	if err := metal.SaveSafetensors(core.PathJoin(dir, "adapter.safetensors"), tensors); err != nil {
		b.Fatalf("SaveSafetensors bench adapter: %v", err)
	}
}

// BenchmarkFuseIntoPack_Gemma4SmallAdapter drives the full pack-level fuse:
// load adapter → build pairs → load base shard → fuse each pair (matmul +
// scale + add + materialize) → save fused safetensors → write provenance.
//
// Source pack + adapter are built once (read-only inputs). Each iteration
// gets a fresh output dir — prepareFuse rejects an output that already holds
// *.safetensors, so reusing one dir would make iteration 2+ fast-fail and
// report a falsely floored profile.
func BenchmarkFuseIntoPack_Gemma4SmallAdapter(b *testing.B) {
	requireFuseMetalBench(b)

	source := core.PathJoin(b.TempDir(), "source")
	adapter := core.PathJoin(b.TempDir(), "adapter")
	if result := core.MkdirAll(source, 0o755); !result.OK {
		b.Fatalf("MkdirAll source: %v", result.Value)
	}
	if result := core.MkdirAll(adapter, 0o755); !result.OK {
		b.Fatalf("MkdirAll adapter: %v", result.Value)
	}

	baseTensors := fuseBenchBaseTensors()
	sourcePack := writeGemma4FuseBenchSourcePack(b, source, baseTensors)
	closeBenchTensorMap(baseTensors)

	adapterTensors := fuseBenchAdapterTensors()
	writeGemma4FuseBenchAdapter(b, adapter, adapterTensors)
	closeBenchTensorMap(adapterTensors)

	ctx := context.Background()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		output := core.PathJoin(b.TempDir(), "fused")
		b.StartTimer()

		result, err := FuseIntoPack(ctx, FuseOptions{
			SourcePack:  sourcePack,
			AdapterPath: adapter,
			OutputPath:  output,
		})
		if err != nil {
			b.Fatalf("FuseIntoPack: %v", err)
		}
		fuseIntoPackBenchSinkResult = result
	}
}
