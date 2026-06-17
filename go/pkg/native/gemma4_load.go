// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	core "dappco.re/go"
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
