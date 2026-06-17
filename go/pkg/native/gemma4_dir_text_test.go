// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package native

import (
	"os"
	"testing"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
	g4 "dappco.re/go/mlx/pkg/model/gemma4"
	"dappco.re/go/mlx/pkg/safetensors"
	"dappco.re/go/mlx/pkg/tokenizer"
)

// TestGenerateTextFromDir gates the one-call directory → text entry: a 4-bit checkpoint
// (config + weights + tokenizer.json) on disk generates text in a single GenerateTextFromDir
// call, equal to the manual LoadGemma4Dir → LoadTokenizer → GenerateText composition. It also
// proves LoadGemma4Dir picks the 4-bit path (the model is U32-packed; the bf16 assembler would
// reject it). This is the entry a real-model smoke uses.
func TestGenerateTextFromDir(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 102 // vocab covers the tokenizer max id (101)
	const gs, bits, maxLen, maxNew = 64, 4, 24, 5
	cfg := g4.Config{
		HiddenSize: dModel, NumHiddenLayers: 2, IntermediateSize: dFF,
		NumAttentionHeads: nHeads, NumKeyValueHeads: nKV, HeadDim: headDim, VocabSize: vocab, RMSNormEps: 1e-6,
		Quantization: &g4.QuantConfig{GroupSize: gs, Bits: bits},
	}
	arch, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	ts := quantGemma4Tensors(t, arch, gs, bits)

	dir := t.TempDir()
	cj := core.JSONMarshal(cfg)
	if !cj.OK {
		t.Fatalf("marshal config")
	}
	blob, err := safetensors.Encode(ts)
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	for name, content := range map[string]string{
		"config.json":       string(cj.Value.([]byte)),
		"model.safetensors": string(blob),
		"tokenizer.json":    textTestTokenizerJSON,
	} {
		if err := coreio.Local.Write(core.PathJoin(dir, name), content); err != nil {
			t.Fatalf("write %s: %v", name, err)
		}
	}

	const prompt = "hello"
	got, err := GenerateTextFromDir(dir, prompt, maxNew, maxLen)
	if err != nil {
		t.Fatalf("GenerateTextFromDir: %v", err)
	}

	// manual composition: LoadGemma4Dir → LoadTokenizer → GenerateText.
	sess, err := LoadGemma4Dir(dir, maxLen)
	if err != nil {
		t.Fatalf("LoadGemma4Dir: %v", err)
	}
	tok, err := tokenizer.LoadTokenizer(core.PathJoin(dir, "tokenizer.json"))
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}
	want, err := sess.GenerateText(tok, prompt, maxNew)
	if err != nil {
		t.Fatalf("manual GenerateText: %v", err)
	}
	if got != want {
		t.Fatalf("GenerateTextFromDir %q != manual composition %q", got, want)
	}
	t.Logf("one-call dir→text: %q → %q (≡ LoadGemma4Dir+LoadTokenizer+GenerateText); LoadGemma4Dir picked the 4-bit path", prompt, got)
}
