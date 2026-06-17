// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	core "dappco.re/go"
	coreio "dappco.re/go/io"
	g4 "dappco.re/go/mlx/pkg/model/gemma4"
	"dappco.re/go/mlx/pkg/safetensors"
)

// LoadGemma4BF16 is the model-load pipe for the no-cgo native stack: a gemma4 config.json
// (bytes) → Arch, a safetensors blob (bytes) → tensors, assembled onto the native weight
// structs. Returns the weights + the derived Arch, ready for GenerateGemma4BF16 or
// NewGemma4Session. Dense bf16 only (the assembler's scope). The caller supplies the bytes
// — reading them from a model directory (and merging sharded safetensors) is a thin I/O
// layer on top; loading a real multi-GB checkpoint is a deliberate, memory-heavy step.
func LoadGemma4BF16(configJSON, safetensorsBlob []byte) (*Gemma4BF16, g4.Arch, error) {
	var cfg g4.Config
	if r := core.JSONUnmarshal(configJSON, &cfg); !r.OK {
		return nil, g4.Arch{}, core.NewError("native.LoadGemma4BF16: config.json parse failed")
	}
	arch, err := cfg.Arch()
	if err != nil {
		return nil, g4.Arch{}, err
	}
	tensors, err := safetensors.Parse(safetensorsBlob)
	if err != nil {
		return nil, g4.Arch{}, err
	}
	g, err := AssembleGemma4BF16(tensors, arch)
	if err != nil {
		return nil, g4.Arch{}, err
	}
	return g, arch, nil
}

// LoadGemma4BF16Session loads a gemma4 (config + safetensors bytes) straight into a
// persistent decode session with maxLen cache rows — the one-call path from a checkpoint
// to a ready-to-Generate session.
func LoadGemma4BF16Session(configJSON, safetensorsBlob []byte, maxLen int) (*Gemma4Session, error) {
	g, arch, err := LoadGemma4BF16(configJSON, safetensorsBlob)
	if err != nil {
		return nil, err
	}
	return NewGemma4Session(g, arch, maxLen)
}

// LoadGemma4BF16Dir loads a gemma4 checkpoint DIRECTORY into a persistent session — the
// one-call path from an on-disk HF checkpoint to a ready-to-Generate session. It reads
// <dir>/config.json + the safetensors weights, handling BOTH layouts via safetensors.LoadDir:
// a single model.safetensors or a sharded model.safetensors.index.json + shards (real gemma4
// checkpoints are always sharded). Dense bf16 only (the assembler's scope). Loading a real
// multi-GB checkpoint is a deliberate, memory-heavy step — every shard's bytes stay resident.
func LoadGemma4BF16Dir(dir string, maxLen int) (*Gemma4Session, error) {
	cfgStr, err := coreio.Local.Read(core.PathJoin(dir, "config.json"))
	if err != nil {
		return nil, core.E("native.LoadGemma4BF16Dir", "read config.json", err)
	}
	var cfg g4.Config
	if r := core.JSONUnmarshal([]byte(cfgStr), &cfg); !r.OK {
		return nil, core.NewError("native.LoadGemma4BF16Dir: config.json parse failed")
	}
	arch, err := cfg.Arch()
	if err != nil {
		return nil, err
	}
	tensors, err := safetensors.LoadDir(dir)
	if err != nil {
		return nil, err
	}
	g, err := AssembleGemma4BF16(tensors, arch)
	if err != nil {
		return nil, err
	}
	return NewGemma4Session(g, arch, maxLen)
}

// LoadGemma4Quant4Dir loads a 4-bit gemma4 checkpoint DIRECTORY into a persistent session —
// the quant sibling of LoadGemma4BF16Dir. It reads <dir>/config.json (which must carry the mlx
// quantization block {group_size, bits}), the safetensors weights (single or sharded, via
// safetensors.LoadDir), assembles the quant model, and builds the session. This is the load
// path the served quants (mlx-community/gemma-4-*-4bit) actually take.
func LoadGemma4Quant4Dir(dir string, maxLen int) (*Gemma4Session, error) {
	cfgStr, err := coreio.Local.Read(core.PathJoin(dir, "config.json"))
	if err != nil {
		return nil, core.E("native.LoadGemma4Quant4Dir", "read config.json", err)
	}
	var cfg g4.Config
	if r := core.JSONUnmarshal([]byte(cfgStr), &cfg); !r.OK {
		return nil, core.NewError("native.LoadGemma4Quant4Dir: config.json parse failed")
	}
	if cfg.Quantization == nil || cfg.Quantization.GroupSize <= 0 || cfg.Quantization.Bits <= 0 {
		return nil, core.NewError("native.LoadGemma4Quant4Dir: config.json has no quantization {group_size, bits}")
	}
	arch, err := cfg.Arch()
	if err != nil {
		return nil, err
	}
	tensors, err := safetensors.LoadDir(dir)
	if err != nil {
		return nil, err
	}
	g, err := AssembleGemma4Quant(tensors, arch, cfg.Quantization.GroupSize, cfg.Quantization.Bits)
	if err != nil {
		return nil, err
	}
	return NewGemma4QuantSession(g, arch, maxLen)
}
