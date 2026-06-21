// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	core "dappco.re/go"
	coreio "dappco.re/go/io"
	"dappco.re/go/mlx/pkg/model"
	"dappco.re/go/mlx/pkg/model/mistral"
	"dappco.re/go/mlx/pkg/safetensors"
	"dappco.re/go/mlx/pkg/tokenizer"
)

// LoadMistralBF16 is the model-load pipe for a Ministral-3 checkpoint: a config.json (bytes) →
// mistral.Config → the backend-agnostic Arch, a safetensors blob → tensors, assembled onto the
// native bf16 structs. Returns the weights + the derived Arch, ready for NewArchSession (the
// shared session — Ministral is a gemma4-subset arch). Dense bf16 only (Base/Reasoning variants).
func LoadMistralBF16(configJSON, safetensorsBlob []byte) (*Gemma4BF16, model.Arch, error) {
	var cfg mistral.Config
	if r := core.JSONUnmarshal(configJSON, &cfg); !r.OK {
		return nil, model.Arch{}, core.NewError("native.LoadMistralBF16: config.json parse failed")
	}
	arch, err := cfg.Arch()
	if err != nil {
		return nil, model.Arch{}, err
	}
	tensors, err := safetensors.Parse(safetensorsBlob)
	if err != nil {
		return nil, model.Arch{}, err
	}
	g, err := AssembleMistralBF16(tensors, arch)
	if err != nil {
		return nil, model.Arch{}, err
	}
	return g, arch, nil
}

// LoadMistralBF16Dir loads a Ministral-3 checkpoint DIRECTORY into a persistent session — the
// one-call path from an on-disk HF checkpoint to a ready-to-Generate session. It reads
// <dir>/config.json + the safetensors weights (single or sharded, via safetensors.LoadDir; real
// Ministral-3 packs are sharded), parses the Mistral config, assembles and builds the session.
// Dense bf16 only — the Base and Reasoning variants. Loading a real multi-GB checkpoint is a
// deliberate, memory-heavy step (every shard's bytes stay resident).
func LoadMistralBF16Dir(dir string, maxLen int) (*ArchSession, error) {
	cfgStr, err := coreio.Local.Read(core.PathJoin(dir, "config.json"))
	if err != nil {
		return nil, core.E("native.LoadMistralBF16Dir", "read config.json", err)
	}
	var cfg mistral.Config
	if r := core.JSONUnmarshal([]byte(cfgStr), &cfg); !r.OK {
		return nil, core.NewError("native.LoadMistralBF16Dir: config.json parse failed")
	}
	arch, err := cfg.Arch()
	if err != nil {
		return nil, err
	}
	tensors, err := safetensors.LoadDir(dir)
	if err != nil {
		return nil, err
	}
	g, err := AssembleMistralBF16(tensors, arch)
	if err != nil {
		return nil, err
	}
	return NewArchSession(g, arch, maxLen)
}

// GenerateTextFromMistralDir is the one-call text-in/text-out path from an on-disk Ministral-3
// checkpoint: it loads the model (LoadMistralBF16Dir) and the tokenizer (<dir>/tokenizer.json)
// and generates up to maxNew tokens for prompt — the whole text → tokens → no-cgo decode → text
// path from a directory, in a single call. maxLen sizes the KV cache (prompt + maxNew must fit).
func GenerateTextFromMistralDir(dir, prompt string, maxNew, maxLen int) (string, error) {
	sess, err := LoadMistralBF16Dir(dir, maxLen)
	if err != nil {
		return "", err
	}
	tok, err := tokenizer.LoadTokenizer(core.PathJoin(dir, "tokenizer.json"))
	if err != nil {
		return "", core.E("native.GenerateTextFromMistralDir", "load tokenizer", err)
	}
	return sess.GenerateText(tok, prompt, maxNew)
}
