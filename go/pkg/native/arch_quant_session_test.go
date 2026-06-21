// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package native

import (
	"math"
	"os"
	"testing"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
	"dappco.re/go/mlx/pkg/metal"
	"dappco.re/go/mlx/pkg/model"
	g4 "dappco.re/go/mlx/pkg/model/gemma4"
	"dappco.re/go/mlx/pkg/safetensors"
)

// quantizeProj quantises a synthetic [outDim × inDim] bf16 weight via metal (the real affine
// packing the checkpoint stores) → (packed, scales, biases) bytes.
func quantizeProj(t *testing.T, outDim, inDim, gs, bits, salt int) (packed, scales, biases []byte) {
	t.Helper()
	f := make([]float32, outDim*inDim)
	for i := range f {
		f[i] = float32((i*salt+7)%101-50) * 0.02
	}
	arr := metal.FromRawBytes(toBF16Bytes(f), []int{outDim, inDim}, metal.DTypeBFloat16)
	wq, sc, bi, err := metal.Quantize(arr, gs, bits, "affine")
	if err != nil {
		metal.Free(arr)
		t.Fatalf("Quantize: %v", err)
	}
	metal.Materialize(wq, sc, bi)
	packed = append([]byte(nil), wq.RawBytes()...)
	scales = append([]byte(nil), sc.RawBytes()...)
	biases = append([]byte(nil), bi.RawBytes()...)
	metal.Free(arr, wq, sc, bi)
	return
}

// quantGemma4Tensors builds a full 4-bit gemma4 checkpoint's tensors with REAL quant weights
// (every projection + the embedding affine-packed via metal.Quantize, the norms bf16).
func quantGemma4Tensors(t *testing.T, arch model.Arch, gs, bits int) map[string]safetensors.Tensor {
	t.Helper()
	ts := map[string]safetensors.Tensor{}
	salt := 1
	mkNorm := func(name string, elems int) {
		f := make([]float32, elems)
		for i := range f {
			f[i] = float32((i*salt+13)%97-48) * 0.02
		}
		ts[name] = safetensors.Tensor{Dtype: "BF16", Shape: []int{elems}, Data: toBF16Bytes(f)}
		salt++
	}
	mkQuant := func(prefix string, outDim, inDim int) {
		p, s, b := quantizeProj(t, outDim, inDim, gs, bits, salt)
		salt++
		ts[prefix+".weight"] = safetensors.Tensor{Dtype: "U32", Shape: []int{outDim, inDim * bits / 32}, Data: p}
		ts[prefix+".scales"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{outDim, inDim / gs}, Data: s}
		ts[prefix+".biases"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{outDim, inDim / gs}, Data: b}
	}
	dModel, headDim, dFF, vocab := arch.Hidden, arch.HeadDim, arch.FF, arch.Vocab
	mkQuant("model.embed_tokens", vocab, dModel)
	mkNorm("model.norm.weight", dModel)
	for i := range arch.Layer {
		p := core.Sprintf("model.layers.%d", i)
		mkNorm(p+".input_layernorm.weight", dModel)
		mkNorm(p+".pre_feedforward_layernorm.weight", dModel)
		lhd := headDimOf(arch.Layer[i], headDim) // per-layer head dim (gemma4 full_attention > sliding)
		lqDim, lkvDim := arch.Heads*lhd, arch.KVHeads*lhd
		mkNorm(p+".self_attn.q_norm.weight", lhd)
		mkNorm(p+".self_attn.k_norm.weight", lhd)
		mkNorm(p+".post_attention_layernorm.weight", dModel)
		mkNorm(p+".post_feedforward_layernorm.weight", dModel)
		mkQuant(p+".self_attn.q_proj", lqDim, dModel)
		mkQuant(p+".self_attn.k_proj", lkvDim, dModel)
		mkQuant(p+".self_attn.v_proj", lkvDim, dModel)
		mkQuant(p+".self_attn.o_proj", dModel, lqDim)
		mkQuant(p+".mlp.gate_proj", dFF, dModel)
		mkQuant(p+".mlp.up_proj", dFF, dModel)
		mkQuant(p+".mlp.down_proj", dModel, dFF)
	}
	return ts
}

// TestLoadGemma4TokenModelDir gates the contract loader: a synthetic 4-bit gemma4 on
// disk loads via LoadTokenModelDir into a model.TokenModel that model.Generate
// drives to the SAME tokens as the model assembled in memory — the dir → contract
// path the no-cgo serve adapter (mlx.LoadNativeTextModel) builds on.
func TestLoadGemma4TokenModelDir(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const gs, bits = 32, 4
	const maxLen, n = 16, 4
	cfg := g4.Config{
		HiddenSize: 128, NumHiddenLayers: 2, IntermediateSize: 256,
		NumAttentionHeads: 2, NumKeyValueHeads: 1, HeadDim: 64, VocabSize: 32, RMSNormEps: 1e-6,
		Quantization: &g4.QuantConfig{GroupSize: gs, Bits: bits},
	}
	arch, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	ts := quantGemma4Tensors(t, arch, gs, bits)
	prompt := []int32{1, 5, 3}

	// in-memory reference: assemble (registry) + NewQuantTokenModel + model.Generate.
	lm, err := g4.Assemble(ts, arch)
	if err != nil {
		t.Fatalf("gemma4.Assemble: %v", err)
	}
	g, err := loadedToQuant(lm, gs, bits)
	if err != nil {
		t.Fatalf("loadedToQuant: %v", err)
	}
	refTM, err := NewQuantTokenModel(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewQuantTokenModel: %v", err)
	}
	want, err := model.Generate(refTM, prompt, n, -1)
	if err != nil {
		t.Fatalf("ref Generate: %v", err)
	}

	// on disk → LoadTokenModelDir → model.Generate.
	cj := core.JSONMarshal(cfg)
	if !cj.OK {
		t.Fatalf("marshal config")
	}
	dir := t.TempDir()
	if err := coreio.Local.Write(core.PathJoin(dir, "config.json"), string(cj.Value.([]byte))); err != nil {
		t.Fatalf("write config: %v", err)
	}
	blob, err := safetensors.Encode(ts)
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	if err := coreio.Local.Write(core.PathJoin(dir, "model.safetensors"), string(blob)); err != nil {
		t.Fatalf("write weights: %v", err)
	}
	tm, err := LoadTokenModelDir(dir, maxLen)
	if err != nil {
		t.Fatalf("LoadTokenModelDir: %v", err)
	}
	got, err := model.Generate(tm, prompt, n, -1)
	if err != nil {
		t.Fatalf("dir Generate: %v", err)
	}
	if len(got) != len(want) {
		t.Fatalf("dir-loaded %d tokens, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("dir-loaded token %d = %d, in-memory = %d (%v vs %v)", i, got[i], want[i], got, want)
		}
	}
	t.Logf("contract loader: LoadTokenModelDir ≡ in-memory NewQuantTokenModel = %v", got)
}

// TestLoadGemma4Quant4Dir gates the whole 4-bit load+session path: a synthetic 4-bit gemma4
// assembles into a quant session that generates; the FIRST generated token equals the gated
// whole-sequence quant chain (EmbedTokensQuant → DecodeForwardArchQuant → LMHeadQuant →
// greedy); and a config.json + weights written to a temp dir — single AND sharded — load to
// the same tokens. The model is all-global so the session's per-type RoPE coincides with
// DecodeForwardArchQuant's one base (a sliding model would legitimately diverge there).
func TestLoadGemma4Quant4Dir(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const gs, bits = 32, 4
	const maxLen, n = 16, 4
	cfg := g4.Config{
		HiddenSize: 128, NumHiddenLayers: 2, IntermediateSize: 256,
		NumAttentionHeads: 2, NumKeyValueHeads: 1, HeadDim: 64, VocabSize: 32, RMSNormEps: 1e-6,
		Quantization: &g4.QuantConfig{GroupSize: gs, Bits: bits},
	}
	arch, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	ts := quantGemma4Tensors(t, arch, gs, bits)
	prompt := []int32{1, 5, 3}

	// direct: assemble in memory (registry) → quant session → generate.
	lmDirect, err := g4.Assemble(ts, arch)
	if err != nil {
		t.Fatalf("gemma4.Assemble: %v", err)
	}
	gDirect, err := loadedToQuant(lmDirect, gs, bits)
	if err != nil {
		t.Fatalf("loadedToQuant: %v", err)
	}
	sd, err := NewArchQuantSession(gDirect, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchQuantSession: %v", err)
	}
	genDirect, err := sd.Generate(prompt, n, -1)
	if err != nil {
		t.Fatalf("direct Generate: %v", err)
	}
	if len(genDirect) != n {
		t.Fatalf("generated %d tokens, want %d", len(genDirect), n)
	}
	for i, id := range genDirect {
		if id < 0 || int(id) >= arch.Vocab {
			t.Fatalf("token %d = %d out of [0,%d)", i, id, arch.Vocab)
		}
	}

	// correctness: the first generated token ≡ the gated whole-seq quant chain.
	embs, err := EmbedTokensQuant(gDirect.Embed, gDirect.EmbedScales, gDirect.EmbedBiases, prompt, arch.Vocab, arch.Hidden, gs, bits, float32(math.Sqrt(float64(arch.Hidden))))
	if err != nil {
		t.Fatalf("EmbedTokensQuant: %v", err)
	}
	attnScale := arch.AttnScale // the model-declared scale (gemma4 1.0), matching the session
	hs, err := DecodeForwardArchQuant(embs, gDirect.Layers, arch.Layer, arch.Hidden, arch.Heads, arch.KVHeads, arch.HeadDim, maxLen, arch.FF, arch.SlidingWindow, arch.RopeBase, attnScale, arch.Eps, false)
	if err != nil {
		t.Fatalf("DecodeForwardArchQuant: %v", err)
	}
	logits, err := LMHeadQuant(hs[len(hs)-1], gDirect.FinalNorm, gDirect.LMHead, gDirect.LMHeadScales, gDirect.LMHeadBiases, arch.Hidden, arch.Vocab, gs, bits, arch.Eps, arch.SoftCap)
	if err != nil {
		t.Fatalf("LMHeadQuant: %v", err)
	}
	wantFirst, err := model.Greedy(logits, arch.Vocab)
	if err != nil {
		t.Fatalf("Greedy: %v", err)
	}
	if genDirect[0] != wantFirst {
		t.Fatalf("quant session first token %d != whole-seq quant chain %d", genDirect[0], wantFirst)
	}

	// dir round-trip: write config.json + weights, single AND sharded → LoadDir ≡ direct.
	cj := core.JSONMarshal(cfg)
	if !cj.OK {
		t.Fatalf("marshal config")
	}
	configJSON := cj.Value.([]byte)
	genFromDir := func(dir string) []int32 {
		if err := coreio.Local.Write(core.PathJoin(dir, "config.json"), string(configJSON)); err != nil {
			t.Fatalf("write config.json: %v", err)
		}
		s, err := LoadDir(dir, maxLen)
		if err != nil {
			t.Fatalf("LoadDir(%s): %v", dir, err)
		}
		out, err := s.Generate(prompt, n, -1)
		if err != nil {
			t.Fatalf("dir Generate: %v", err)
		}
		return out
	}

	single := t.TempDir()
	blob, err := safetensors.Encode(ts)
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	if err := coreio.Local.Write(core.PathJoin(single, "model.safetensors"), string(blob)); err != nil {
		t.Fatalf("write single: %v", err)
	}
	if got := genFromDir(single); !idsEqual(got, genDirect) {
		t.Fatalf("single-file dir %v != in-memory %v", got, genDirect)
	}

	sharded := t.TempDir()
	half1, half2 := map[string]safetensors.Tensor{}, map[string]safetensors.Tensor{}
	wm := map[string]string{}
	i := 0
	for name, tns := range ts {
		if i%2 == 0 {
			half1[name], wm[name] = tns, "model-00001-of-00002.safetensors"
		} else {
			half2[name], wm[name] = tns, "model-00002-of-00002.safetensors"
		}
		i++
	}
	b1, err := safetensors.Encode(half1)
	if err != nil {
		t.Fatalf("Encode shard1: %v", err)
	}
	b2, err := safetensors.Encode(half2)
	if err != nil {
		t.Fatalf("Encode shard2: %v", err)
	}
	if err := coreio.Local.Write(core.PathJoin(sharded, "model-00001-of-00002.safetensors"), string(b1)); err != nil {
		t.Fatalf("write shard1: %v", err)
	}
	if err := coreio.Local.Write(core.PathJoin(sharded, "model-00002-of-00002.safetensors"), string(b2)); err != nil {
		t.Fatalf("write shard2: %v", err)
	}
	idx := core.JSONMarshal(map[string]any{"weight_map": wm})
	if !idx.OK {
		t.Fatalf("marshal index")
	}
	if err := coreio.Local.Write(core.PathJoin(sharded, "model.safetensors.index.json"), string(idx.Value.([]byte))); err != nil {
		t.Fatalf("write index: %v", err)
	}
	if got := genFromDir(sharded); !idsEqual(got, genDirect) {
		t.Fatalf("sharded dir %v != in-memory %v", got, genDirect)
	}

	t.Logf("4-bit dir-load: assemble → quant session generates %v; first token ≡ whole-seq quant chain; single + sharded dirs ≡ in-memory (the path mlx-community 4-bit takes)", genDirect)
}
