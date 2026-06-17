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

func mustEncode(t *testing.T, tensors map[string]safetensors.Tensor) []byte {
	t.Helper()
	blob, err := safetensors.Encode(tensors)
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	return blob
}

// TestLoadGemma4BF16Dir gates the on-disk directory path: a config.json + safetensors written
// to a temp dir — as BOTH a single model.safetensors AND a 2-shard index.json + shards —
// loads into a session generating IDENTICALLY to the in-memory pipe. Proves the thin dir/
// sharded I/O layer (safetensors.LoadDir) feeds the assembler unchanged: real gemma4
// checkpoints are always sharded, so this is the load path a real model actually takes.
func TestLoadGemma4BF16Dir(t *testing.T) {
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
	tensors, _ := gemma4Tensors(arch, false)
	prompt := []int32{1, 5, 3}
	const n = 4

	cj := core.JSONMarshal(cfg)
	if !cj.OK {
		t.Fatalf("marshal config")
	}
	configJSON := cj.Value.([]byte)

	// reference: the proven in-memory pipe.
	refSess, err := LoadGemma4BF16Session(configJSON, mustEncode(t, tensors), maxLen)
	if err != nil {
		t.Fatalf("LoadGemma4BF16Session: %v", err)
	}
	want, err := refSess.Generate(prompt, n, -1)
	if err != nil {
		t.Fatalf("ref Generate: %v", err)
	}

	// write config.json into dir, load it, generate.
	genFromDir := func(dir string) []int32 {
		if err := coreio.Local.Write(core.PathJoin(dir, "config.json"), string(configJSON)); err != nil {
			t.Fatalf("write config.json: %v", err)
		}
		s, err := LoadGemma4BF16Dir(dir, maxLen)
		if err != nil {
			t.Fatalf("LoadGemma4BF16Dir(%s): %v", dir, err)
		}
		out, err := s.Generate(prompt, n, -1)
		if err != nil {
			t.Fatalf("dir Generate: %v", err)
		}
		return out
	}

	// (a) single model.safetensors.
	single := t.TempDir()
	if err := coreio.Local.Write(core.PathJoin(single, "model.safetensors"), string(mustEncode(t, tensors))); err != nil {
		t.Fatalf("write single: %v", err)
	}
	if got := genFromDir(single); !idsEqual(got, want) {
		t.Fatalf("single-file dir %v != in-memory pipe %v", got, want)
	}

	// (b) two shards + index.json — split the gemma4 tensor set across two files.
	sharded := t.TempDir()
	half1, half2 := map[string]safetensors.Tensor{}, map[string]safetensors.Tensor{}
	wm := map[string]string{}
	i := 0
	for name, tns := range tensors {
		if i%2 == 0 {
			half1[name], wm[name] = tns, "model-00001-of-00002.safetensors"
		} else {
			half2[name], wm[name] = tns, "model-00002-of-00002.safetensors"
		}
		i++
	}
	if err := coreio.Local.Write(core.PathJoin(sharded, "model-00001-of-00002.safetensors"), string(mustEncode(t, half1))); err != nil {
		t.Fatalf("write shard1: %v", err)
	}
	if err := coreio.Local.Write(core.PathJoin(sharded, "model-00002-of-00002.safetensors"), string(mustEncode(t, half2))); err != nil {
		t.Fatalf("write shard2: %v", err)
	}
	idx := core.JSONMarshal(map[string]any{"weight_map": wm})
	if !idx.OK {
		t.Fatalf("marshal index")
	}
	if err := coreio.Local.Write(core.PathJoin(sharded, "model.safetensors.index.json"), string(idx.Value.([]byte))); err != nil {
		t.Fatalf("write index: %v", err)
	}
	if got := genFromDir(sharded); !idsEqual(got, want) {
		t.Fatalf("sharded dir %v != in-memory pipe %v", got, want)
	}

	t.Logf("dir-load: single + 2-shard checkpoints both → session ≡ in-memory pipe %v (the path a real sharded gemma4 takes)", want)
}
