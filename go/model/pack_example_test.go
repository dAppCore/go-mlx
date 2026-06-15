// SPDX-Licence-Identifier: EUPL-1.2

package model_test

import (
	"fmt"

	core "dappco.re/go"
	"dappco.re/go/mlx/model"
	mp "dappco.re/go/mlx/pack"
)

// ExampleSupportsArchitecture screens architecture names against the registered
// profiles in dappco.re/go/mlx/profile — the predicate the loader uses to decide
// whether a candidate model has a known runtime profile before any bytes are
// read. Names carrying a profile (gemma-4 text, qwen3) report true; an unknown
// name reports false. The match is case-insensitive, so an upper-cased alias
// resolves the same as its canonical form.
func ExampleSupportsArchitecture() {
	for _, arch := range []string{"gemma4_text", "qwen3", "QWEN3", "totally_unknown_arch"} {
		fmt.Printf("%-22s -> %v\n", arch, model.SupportsArchitecture(arch))
	}
	// Output:
	// gemma4_text            -> true
	// qwen3                  -> true
	// QWEN3                  -> true
	// totally_unknown_arch   -> false
}

// ExampleInspect reads a model directory and reports the pack metadata the
// loader needs — architecture, on-disk format, declared quantisation, and
// whether the native go-mlx runtime can load it as-is. Nothing is loaded into
// memory: Inspect reads config.json, the tokenizer presence, and the
// safetensors header index, never the weight tensors. The synthetic pack here
// is a minimal 4-bit Gemma 4 text model written to a temp directory.
func ExampleInspect() {
	dir := exampleWritePack("gemma4_text", true)
	defer core.RemoveAll(dir)

	pack, err := model.Inspect(dir, mp.WithPackQuantization(4))
	if err != nil {
		fmt.Println("inspect error:", err)
		return
	}

	fmt.Println("Architecture:  ", pack.Architecture)
	fmt.Println("Format:        ", pack.Format)
	fmt.Println("QuantBits:     ", pack.QuantBits)
	fmt.Println("QuantGroup:    ", pack.QuantGroup)
	fmt.Println("Supported:     ", pack.SupportedArchitecture)
	fmt.Println("NativeLoadable:", pack.NativeLoadable)
	// Output:
	// Architecture:   gemma4_text
	// Format:         safetensors
	// QuantBits:      4
	// QuantGroup:     64
	// Supported:      true
	// NativeLoadable: true
}

// ExampleValidate is Inspect plus a validity gate: it returns the same pack but
// surfaces an error when the pack carries any blocking issue, so a caller can
// fail loud instead of probing pack.Valid() by hand. A complete pack validates
// clean; the same pack with its tokenizer removed fails — go-mlx loading
// requires a tokenizer, so its absence is a blocking issue, not a warning.
func ExampleValidate() {
	good := exampleWritePack("gemma4_text", true)
	defer core.RemoveAll(good)
	if _, err := model.Validate(good, mp.WithPackQuantization(4)); err != nil {
		fmt.Println("complete pack:", err)
	} else {
		fmt.Println("complete pack: valid")
	}

	noTokenizer := exampleWritePack("gemma4_text", false)
	defer core.RemoveAll(noTokenizer)
	if _, err := model.Validate(noTokenizer, mp.WithPackQuantization(4)); err != nil {
		fmt.Println("missing tokenizer: invalid")
	} else {
		fmt.Println("missing tokenizer: valid")
	}
	// Output:
	// complete pack: valid
	// missing tokenizer: invalid
}

// exampleWritePack synthesises a minimal safetensors model pack in a fresh temp
// directory and returns its path: a 4-bit Gemma 4 text config.json, a stub
// weight shard, and (when withTokenizer) a tokenizer.json. No weights are real
// — Inspect only reads the safetensors header, never the tensor bytes. Setup
// failures panic: an example that cannot stage its own fixture has nothing to
// demonstrate.
func exampleWritePack(modelType string, withTokenizer bool) string {
	made := core.MkdirTemp("", "model-example-*")
	if !made.OK {
		panic(made.Value)
	}
	dir := made.Value.(string)

	write := func(name, body string) {
		if r := core.WriteFile(core.PathJoin(dir, name), []byte(body), 0o644); !r.OK {
			panic(r.Value)
		}
	}

	write("config.json", fmt.Sprintf(`{
		"model_type": %q,
		"vocab_size": 262208,
		"hidden_size": 2048,
		"num_hidden_layers": 26,
		"max_position_embeddings": 131072,
		"quantization_config": {"bits": 4, "group_size": 64}
	}`, modelType))
	write("model-00001-of-00001.safetensors", "stub")
	if withTokenizer {
		write("tokenizer.json", `{"model":{"type":"BPE","vocab":{},"merges":[]}}`)
	}
	return dir
}
