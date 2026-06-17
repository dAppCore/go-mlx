// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package native

import (
	"os"
	"testing"

	core "dappco.re/go"
	g4 "dappco.re/go/mlx/pkg/model/gemma4"
	"dappco.re/go/mlx/pkg/safetensors"
)

// TestLoadGemma4BF16Session gates the model-load pipe: a gemma4 config.json (bytes) + a
// safetensors blob (bytes) → a persistent session that generates IDENTICALLY to one built
// by assembling the same tensors directly. That proves the whole pipe — config JSON →
// Config → Arch, blob → Encode/Parse → tensors, assemble, session — wires up correctly.
func TestLoadGemma4BF16Session(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const headDim, vocab = 64, 32 // headDim 64 so the SDPA kernel exists
	const maxLen = 16
	cfg := g4.Config{
		HiddenSize: 128, NumHiddenLayers: 2, IntermediateSize: 256,
		NumAttentionHeads: 2, NumKeyValueHeads: 1, HeadDim: headDim, VocabSize: vocab, RMSNormEps: 1e-6,
	}
	arch, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	tensors, _ := gemma4Tensors(arch, false) // the assembler test's gemma4-named tensor set (tied head)
	prompt := []int32{1, 5, 3}
	const n = 4

	// direct: assemble the tensors → session → generate.
	gDirect, err := AssembleGemma4BF16(tensors, arch)
	if err != nil {
		t.Fatalf("AssembleGemma4BF16: %v", err)
	}
	sd, err := NewGemma4Session(gDirect, arch, maxLen)
	if err != nil {
		t.Fatalf("NewGemma4Session: %v", err)
	}
	genDirect, err := sd.Generate(prompt, n, -1)
	if err != nil {
		t.Fatalf("direct Generate: %v", err)
	}

	// load: config.json bytes + a safetensors blob → session → generate.
	cj := core.JSONMarshal(cfg)
	if !cj.OK {
		t.Fatalf("marshal config")
	}
	configJSON := cj.Value.([]byte)
	blob, err := safetensors.Encode(tensors)
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	sl, err := LoadGemma4BF16Session(configJSON, blob, maxLen)
	if err != nil {
		t.Fatalf("LoadGemma4BF16Session: %v", err)
	}
	genLoad, err := sl.Generate(prompt, n, -1)
	if err != nil {
		t.Fatalf("loaded Generate: %v", err)
	}

	if !idsEqual(genLoad, genDirect) {
		t.Fatalf("loaded session %v != directly-assembled %v (the load pipe diverged)", genLoad, genDirect)
	}
	if len(genLoad) != n {
		t.Fatalf("generated %d tokens, want %d", len(genLoad), n)
	}
	for i, id := range genLoad {
		if id < 0 || int(id) >= vocab {
			t.Fatalf("token %d = %d out of [0,%d)", i, id, vocab)
		}
	}

	// LoadGemma4BF16 (weights + arch) recovers the dims from the parsed config.
	_, arch2, err := LoadGemma4BF16(configJSON, blob)
	if err != nil {
		t.Fatalf("LoadGemma4BF16: %v", err)
	}
	if arch2.Hidden != arch.Hidden || arch2.HeadDim != arch.HeadDim || arch2.Vocab != arch.Vocab || len(arch2.Layer) != len(arch.Layer) {
		t.Fatalf("config round-trip dims wrong: %+v vs %+v", arch2, arch)
	}

	t.Logf("load pipe: config.json + safetensors blob → session → %v ≡ directly-assembled session — config→Arch + Encode/Parse + assemble + session wired end to end", genLoad)
}
