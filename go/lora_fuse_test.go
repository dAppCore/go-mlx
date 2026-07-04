// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"dappco.re/go/mlx/internal/metaltest"
	"math"
	"testing"

	core "dappco.re/go"
	pack "dappco.re/go/inference/modelpack"
	"dappco.re/go/mlx/pkg/metal"
)

const localGemma4E2BQ6SmokeAdapter = "/private/tmp/go-mlx-self/gemma4-e2b-lora-smoke-adapter"

const loraFuseTestTokenizerJSON = `{
  "model": {
    "type": "BPE",
    "vocab": {
      "h": 0,
      "e": 1,
      "l": 2,
      "o": 3
    },
    "merges": ["h e", "l l"]
  }
}`

func TestFuseLoRAIntoModelPack_Gemma4SuffixTargetValidatesOutput_Good(t *testing.T) {
	requireLoRAFuseMetal(t)

	source := core.PathJoin(t.TempDir(), "gemma4-source")
	adapter := core.PathJoin(t.TempDir(), "adapter")
	output := core.PathJoin(t.TempDir(), "fused")
	for _, dir := range []string{source, adapter} {
		if result := core.MkdirAll(dir, 0o755); !result.OK {
			t.Fatalf("MkdirAll(%s): %v", dir, result.Value)
		}
	}

	writeModelPackFile(t, core.PathJoin(source, "config.json"), `{
		"architectures": ["Gemma4ForConditionalGeneration"],
		"model_type": "gemma4",
		"quantization": {"group_size": 64, "bits": 6, "mode": "affine"},
		"text_config": {
			"model_type": "gemma4_text",
			"vocab_size": 262144,
			"hidden_size": 64,
			"num_hidden_layers": 1,
			"max_position_embeddings": 131072
		}
	}`)
	writeModelPackFile(t, core.PathJoin(source, "tokenizer.json"), loraFuseTestTokenizerJSON)
	baseKey := "language_model.model.layers.0.self_attn.q_proj.weight"
	const (
		outDim    = 2
		inDim     = 64
		groupSize = 64
		bits      = 6
	)
	sourceWeights := map[string]*metal.Array{
		baseKey: metal.FromValues(loraFuseZeroUint32s(outDim*loraFusePackedIn(inDim, bits)), outDim, loraFusePackedIn(inDim, bits)),
		"language_model.model.layers.0.self_attn.q_proj.scales": metal.FromValues([]float32{1, 1}, outDim, inDim/groupSize),
		"language_model.model.layers.0.self_attn.q_proj.biases": metal.FromValues([]float32{0, 0}, outDim, inDim/groupSize),
		"model.layers.0.self_attn.k_proj.weight":                metal.FromValues(loraFuseFloat32Fill(outDim*inDim, 10), outDim, inDim),
	}
	defer freeLoRAFuseTensors(sourceWeights)
	if err := metal.SaveSafetensors(core.PathJoin(source, "model.safetensors"), sourceWeights); err != nil {
		t.Fatalf("SaveSafetensors source: %v", err)
	}

	writeModelPackFile(t, core.PathJoin(adapter, "adapter_config.json"), `{
		"r": 1,
		"lora_alpha": 2,
		"target_modules": ["q_proj"]
	}`)
	adapterWeights := map[string]*metal.Array{
		"model.layers.0.q_proj.lora_A.weight": metal.FromValues(loraFuseFloat32Fill(inDim, 1), 1, inDim),
		"model.layers.0.q_proj.lora_B.weight": metal.FromValues([]float32{3, 4}, outDim, 1),
	}
	defer freeLoRAFuseTensors(adapterWeights)
	if err := metal.SaveSafetensors(core.PathJoin(adapter, "adapter.safetensors"), adapterWeights); err != nil {
		t.Fatalf("SaveSafetensors adapter: %v", err)
	}

	result, err := FuseLoRAIntoModelPack(context.Background(), FuseLoRAOptions{
		ModelPath:   source,
		AdapterPath: adapter,
		OutputPath:  output,
	})
	if err != nil {
		t.Fatalf("FuseLoRAIntoModelPack() error = %v", err)
	}
	if !result.SourcePack.Valid() || !result.OutputPack.Valid() {
		t.Fatalf("source valid=%v output valid=%v source issues=%+v output issues=%+v", result.SourcePack.Valid(), result.OutputPack.Valid(), result.SourcePack.Issues, result.OutputPack.Issues)
	}
	if result.OutputPack.Architecture != "gemma4_text" || result.OutputPack.Format != pack.ModelPackFormatSafetensors {
		t.Fatalf("output pack architecture=%q format=%q", result.OutputPack.Architecture, result.OutputPack.Format)
	}
	if result.Adapter.Rank != 1 || result.Adapter.Alpha != 2 || result.Adapter.Scale != 2 {
		t.Fatalf("adapter = %+v, want PEFT rank=1 alpha=2 scale=2", result.Adapter)
	}
	if result.FusedWeights != 1 || len(result.FusedWeightKeys) != 1 || result.FusedWeightKeys[0] != baseKey {
		t.Fatalf("fused weights=%d keys=%v, want raw Gemma-4 q_proj source key", result.FusedWeights, result.FusedWeightKeys)
	}

	loaded, err := metal.LoadAllSafetensors(core.PathJoin(output, "model.safetensors"))
	if err != nil {
		t.Fatalf("LoadAllSafetensors fused: %v", err)
	}
	defer freeLoRAFuseTensors(loaded)
	fused := loaded[baseKey]
	if shape := fused.Shape(); len(shape) != 2 || shape[0] != outDim || shape[1] != inDim {
		t.Fatalf("fused q_proj shape = %v, want [%d %d]", shape, outDim, inDim)
	}
	got := fused.Floats()
	for i, value := range got[:inDim] {
		if math.Abs(float64(value-6)) > 0.0001 {
			t.Fatalf("fused q_proj first row[%d] = %v, want 6", i, value)
		}
	}
	for i, value := range got[inDim:] {
		if math.Abs(float64(value-8)) > 0.0001 {
			t.Fatalf("fused q_proj second row[%d] = %v, want 8", i, value)
		}
	}
	if _, exists := loaded["language_model.model.layers.0.self_attn.q_proj.scales"]; exists {
		t.Fatal("root fuse should drop q6 .scales for the fused dense target")
	}
	if _, exists := loaded["language_model.model.layers.0.self_attn.q_proj.biases"]; exists {
		t.Fatal("root fuse should drop q6 .biases for the fused dense target")
	}
	if _, exists := loaded["model.layers.0.self_attn.q_proj.weight"]; exists {
		t.Fatal("root fuse should preserve the raw Gemma-4 safetensors key instead of writing a duplicate canonical key")
	}
}

func TestFuseLoRAIntoModelPack_Gemma4Q6RealPackReloadGenerate_Good(t *testing.T) {
	modelPath := requireLocalGemma4E2BQ6SFTModel(t)
	adapterPath := requireLocalGemma4E2BQ6LoRAAdapter(t)
	output := core.PathJoin(t.TempDir(), "gemma4-e2b-q6-fused")

	result, err := FuseLoRAIntoModelPack(context.Background(), FuseLoRAOptions{
		ModelPath:   modelPath,
		AdapterPath: adapterPath,
		OutputPath:  output,
		Labels:      map[string]string{"test": t.Name(), "model": "gemma4-e2b-q6"},
	})
	if err != nil {
		t.Fatalf("FuseLoRAIntoModelPack(real Gemma-4 q6) error = %v", err)
	}
	if result.FusedWeights != 105 {
		t.Fatalf("FusedWeights = %d, want 105 q/v/o projections across 35 Gemma-4 layers; keys=%v", result.FusedWeights, result.FusedWeightKeys)
	}
	if result.OutputPack.Architecture != "gemma4_text" || result.OutputPack.QuantBits != 6 {
		t.Fatalf("output pack architecture=%q quant=%d, want gemma4_text q6", result.OutputPack.Architecture, result.OutputPack.QuantBits)
	}

	fused, err := LoadModel(
		result.OutputPath,
		WithExpectedQuantization(6),
		WithPromptCache(false),
	)
	if err != nil {
		t.Fatalf("LoadModel(fused Gemma-4 q6) error = %v", err)
	}
	t.Cleanup(func() { _ = fused.Close() })

	info := fused.Info()
	if info.Architecture != "gemma4_text" || info.QuantBits != 6 {
		t.Fatalf("fused model info architecture=%q quant=%d, want gemma4_text q6", info.Architecture, info.QuantBits)
	}
	if !info.Adapter.IsEmpty() {
		t.Fatalf("fused model adapter info = %+v, want no live adapter attached", info.Adapter)
	}

	text, err := fused.Generate("What should a retained State runner preserve?")
	if err != nil {
		t.Fatalf("Generate(fused Gemma-4 q6) error = %v", err)
	}
	metrics := fused.Metrics()
	if metrics.GeneratedTokens == 0 {
		t.Fatalf("fused generation produced no tokens; text=%q metrics=%+v", text, metrics)
	}
	t.Logf("fused Gemma-4 q6 reload/generate ok: fused_weights=%d generated_tokens=%d decode_tps=%.2f", result.FusedWeights, metrics.GeneratedTokens, metrics.DecodeTokensPerSec)
}

func TestFuseLoRAIntoModelPack_RejectsInvalidSourcePack_Bad(t *testing.T) {
	dir := t.TempDir()
	writeModelPackFile(t, core.PathJoin(dir, "config.json"), `{"model_type":"gemma4_text"}`)
	writeModelPackFile(t, core.PathJoin(dir, "model.safetensors"), "stub")

	_, err := FuseLoRAIntoModelPack(context.Background(), FuseLoRAOptions{
		ModelPath:   dir,
		AdapterPath: core.PathJoin(t.TempDir(), "adapter"),
		OutputPath:  core.PathJoin(t.TempDir(), "fused"),
	})
	if err == nil {
		t.Fatal("expected invalid source pack error")
	}
	if !core.Contains(err.Error(), "validate source model pack") || !core.Contains(err.Error(), string(pack.ModelPackIssueMissingTokenizer)) {
		t.Fatalf("error = %v, want source validation context and missing tokenizer issue", err)
	}
}

func requireLocalGemma4E2BQ6LoRAAdapter(t *testing.T) string {
	t.Helper()
	for _, path := range []string{
		core.PathJoin(localGemma4E2BQ6SmokeAdapter, "adapter_config.json"),
		core.PathJoin(localGemma4E2BQ6SmokeAdapter, "adapter.safetensors"),
	} {
		if result := core.Stat(path); !result.OK {
			t.Skip("local Gemma-4 E2B q6 LoRA adapter is not available")
		}
	}
	return localGemma4E2BQ6SmokeAdapter
}

func requireLoRAFuseMetal(t *testing.T) {
	t.Helper()
	if !metaltest.RunMetalTests {
		t.Skip("build with -tags metal_runtime to enable native LoRA fuse tensor tests")
	}
	if !MetalAvailable() {
		t.Skip("Metal runtime unavailable")
	}
}

func freeLoRAFuseTensors(tensors map[string]*metal.Array) {
	for _, tensor := range tensors {
		metal.Free(tensor)
	}
}

func loraFusePackedIn(inDim, bits int) int {
	return (inDim*bits + 31) / 32
}

func loraFuseZeroUint32s(n int) []uint32 {
	return make([]uint32, n)
}

func loraFuseFloat32Fill(n int, value float32) []float32 {
	values := make([]float32, n)
	for i := range values {
		values[i] = value
	}
	return values
}
