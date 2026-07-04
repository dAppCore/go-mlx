// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package mlx

import (
	"encoding/binary"
	"math"
	"os"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/native"
	"dappco.re/go/mlx/pkg/safetensors"
)

func TestNativeTextModelLoadLoRAFusesAndRestoresTokenModel_Good(t *testing.T) {
	if os.Getenv(native.MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}

	source := core.PathJoin(t.TempDir(), "source")
	adapter := core.PathJoin(t.TempDir(), "adapter")
	for _, dir := range []string{source, adapter} {
		if result := core.MkdirAll(dir, 0o755); !result.OK {
			t.Fatalf("MkdirAll(%s): %v", dir, result.Value)
		}
	}
	writeNativeLoRATestGemma4BF16Pack(t, source)
	writeNativeLoRATestAdapter(t, adapter)

	base, err := native.LoadTokenModelDir(source, 4)
	if err != nil {
		t.Fatalf("LoadTokenModelDir(base): %v", err)
	}
	m := &nativeTextModel{tm: base, modelPath: source, maxLen: 4, modelType: "gemma4"}
	before := m.tm

	loaded, err := m.LoadAdapter(adapter)
	if err != nil {
		t.Fatalf("LoadAdapter() error = %v", err)
	}
	if loaded.Path != adapter || loaded.Rank != 1 || loaded.Alpha != 2 || loaded.Labels["scale"] != "2" {
		t.Fatalf("LoadAdapter() = %+v, want loaded adapter identity", loaded)
	}
	if m.tm == before {
		t.Fatal("LoadAdapter() did not replace the native token model")
	}
	if got := m.ActiveAdapter(); got.Path != adapter || got.Rank != 1 || got.Alpha != 2 || got.Labels["scale"] != "2" {
		t.Fatalf("ActiveAdapter() after LoadAdapter = %+v, want loaded adapter identity", got)
	}

	adapted := m.tm
	if err := m.UnloadAdapter(); err != nil {
		t.Fatalf("UnloadAdapter() error = %v", err)
	}
	if m.tm == adapted {
		t.Fatal("UnloadAdapter() did not restore a base native token model")
	}
	if got := m.ActiveAdapter(); got.Path != "" || got.Hash != "" || got.Rank != 0 {
		t.Fatalf("ActiveAdapter() after UnloadAdapter = %+v, want empty adapter identity", got)
	}
}

func writeNativeLoRATestGemma4BF16Pack(t *testing.T, dir string) {
	t.Helper()
	writeModelPackFile(t, core.PathJoin(dir, "config.json"), `{
		"architectures": ["Gemma4ForConditionalGeneration"],
		"model_type": "gemma4",
		"vocab_size": 4,
		"hidden_size": 64,
		"num_hidden_layers": 1,
		"intermediate_size": 128,
		"num_attention_heads": 1,
		"num_key_value_heads": 1,
		"head_dim": 64,
		"rms_norm_eps": 0.000001,
		"rope_theta": 10000,
		"max_position_embeddings": 128,
		"sliding_window": 128,
		"layer_types": ["full_attention"]
	}`)
	writeModelPackFile(t, core.PathJoin(dir, "tokenizer.json"), loraFuseTestTokenizerJSON)

	tensors := map[string]safetensors.Tensor{}
	mkVec := func(name string, elems, salt int) {
		tensors[name] = safetensors.Tensor{Dtype: "BF16", Shape: []int{1, elems}, Data: nativeLoRATestBF16(elems, salt)}
	}
	mkMat := func(name string, rows, cols, salt int) {
		tensors[name] = safetensors.Tensor{Dtype: "BF16", Shape: []int{rows, cols}, Data: nativeLoRATestBF16(rows*cols, salt)}
	}
	const dModel, qDim, kvDim, dFF, vocab = 64, 64, 64, 128, 4
	mkMat("model.embed_tokens.weight", vocab, dModel, 1)
	mkVec("model.norm.weight", dModel, 2)
	p := "model.layers.0"
	mkVec(p+".input_layernorm.weight", dModel, 3)
	mkMat(p+".self_attn.q_proj.weight", qDim, dModel, 4)
	mkMat(p+".self_attn.k_proj.weight", kvDim, dModel, 5)
	mkMat(p+".self_attn.v_proj.weight", kvDim, dModel, 6)
	mkMat(p+".self_attn.o_proj.weight", dModel, qDim, 7)
	mkVec(p+".self_attn.q_norm.weight", qDim, 8)
	mkVec(p+".self_attn.k_norm.weight", kvDim, 9)
	mkVec(p+".post_attention_layernorm.weight", dModel, 10)
	mkVec(p+".pre_feedforward_layernorm.weight", dModel, 11)
	mkVec(p+".post_feedforward_layernorm.weight", dModel, 12)
	mkMat(p+".mlp.gate_proj.weight", dFF, dModel, 13)
	mkMat(p+".mlp.up_proj.weight", dFF, dModel, 14)
	mkMat(p+".mlp.down_proj.weight", dModel, dFF, 15)

	blob, err := safetensors.Encode(tensors)
	if err != nil {
		t.Fatalf("Encode source safetensors: %v", err)
	}
	if result := core.WriteFile(core.PathJoin(dir, "model.safetensors"), blob, 0o644); !result.OK {
		t.Fatalf("write source safetensors: %v", result.Value)
	}
}

func writeNativeLoRATestAdapter(t *testing.T, dir string) {
	t.Helper()
	writeModelPackFile(t, core.PathJoin(dir, "adapter_config.json"), `{
		"r": 1,
		"lora_alpha": 2,
		"target_modules": ["q_proj"]
	}`)
	weights := map[string]safetensors.Tensor{
		"model.layers.0.q_proj.lora_A.weight": nativeLoRATestF32Tensor([]int{1, 64}, nativeLoRATestFillF32(64, 0.25)),
		"model.layers.0.q_proj.lora_B.weight": nativeLoRATestF32Tensor([]int{64, 1}, nativeLoRATestFillF32(64, 0.5)),
	}
	blob, err := safetensors.Encode(weights)
	if err != nil {
		t.Fatalf("Encode adapter safetensors: %v", err)
	}
	if result := core.WriteFile(core.PathJoin(dir, "adapter.safetensors"), blob, 0o644); !result.OK {
		t.Fatalf("write adapter safetensors: %v", result.Value)
	}
}

func nativeLoRATestBF16(elems, salt int) []byte {
	values := make([]float32, elems)
	for i := range values {
		values[i] = float32((i*salt+7)%31-15) * 0.01
	}
	return nativeTextF32ToBF16(values)
}

func nativeLoRATestFillF32(elems int, value float32) []float32 {
	values := make([]float32, elems)
	for i := range values {
		values[i] = value
	}
	return values
}

func nativeLoRATestF32Tensor(shape []int, values []float32) safetensors.Tensor {
	data := make([]byte, len(values)*4)
	for i, v := range values {
		binary.LittleEndian.PutUint32(data[i*4:], math.Float32bits(v))
	}
	return safetensors.Tensor{Dtype: "F32", Shape: shape, Data: data}
}
