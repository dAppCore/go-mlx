// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && !nomlx

package lora

import (
	"context"
	"math"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/mlx/internal/metal"
	"dappco.re/go/mlx/pack"
)

func requireFuseMetal(t *testing.T) {
	t.Helper()
	if core.Getenv("GO_MLX_RUN_METAL_TESTS") != "1" {
		t.Skip("set GO_MLX_RUN_METAL_TESTS=1 to enable native LoRA fuse tensor tests")
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

func writeFuseAdapter(t *testing.T, dir string, tensors map[string]*metal.Array) {
	t.Helper()
	writeFuseTestFile(t, core.PathJoin(dir, "adapter_config.json"), `{
		"rank": 1,
		"alpha": 2,
		"lora_layers": ["self_attn.q_proj"]
	}`)
	if err := metal.SaveSafetensors(core.PathJoin(dir, "adapter.safetensors"), tensors); err != nil {
		t.Fatalf("SaveSafetensors adapter: %v", err)
	}
}

func closeTensorMap(tensors map[string]*metal.Array) {
	for _, tensor := range tensors {
		metal.Free(tensor)
	}
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
	if _, _, err := fuseModelWeightFiles(context.Background(), nil, t.TempDir(), nil, 1); err == nil {
		t.Fatal("expected no base weight files error")
	}
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, _, err := fuseModelWeightFiles(cancelled, []string{core.PathJoin(t.TempDir(), "missing.safetensors")}, t.TempDir(), nil, 1); err != context.Canceled {
		t.Fatalf("fuseModelWeightFiles(cancelled) = %v, want context.Canceled", err)
	}

	pairs := map[string]fusePair{
		"model.layers.0.self_attn.q_proj": {MatrixA: &metal.Array{}, MatrixB: &metal.Array{}},
	}
	fused, err := fuseWeightPairs(context.Background(), map[string]*metal.Array{}, pairs, map[string]struct{}{}, 1)
	if err != nil {
		t.Fatalf("fuseWeightPairs(missing base) error = %v", err)
	}
	if len(fused) != 0 {
		t.Fatalf("fused keys = %v, want none for missing base", fused)
	}
	if _, err := fuseWeightPairs(cancelled, map[string]*metal.Array{}, pairs, map[string]struct{}{}, 1); err != context.Canceled {
		t.Fatalf("fuseWeightPairs(cancelled) = %v, want context.Canceled", err)
	}

	names := outputWeightFileNames([]string{"/tmp/a.safetensors", "/tmp/shard/b.safetensors"})
	if len(names) != 2 || names[0] != "a.safetensors" || names[1] != "b.safetensors" {
		t.Fatalf("outputWeightFileNames() = %v", names)
	}
	freeMetalMap(map[string]*metal.Array{"nil": nil})
}
