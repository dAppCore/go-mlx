// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"dappco.re/go"
)

// MLP is the feed-forward network shared by the dense transformer models on the
// metal SDK (Gemma 3, Gemma 4, and the native MLP decode kernels).
type MLP struct {
	GateProj *Linear
	UpProj   *Linear
	DownProj *Linear
}

// Forward runs the gated feed-forward network, preferring the native MLX matvec
// and GELU paths when available and falling back to the Go compute graph.
func (m *MLP) Forward(x *Array) *Array {
	if out, ok, err := nativeMLPMatVec(x, m); ok {
		if err == nil {
			return out
		}
		core.Error("mlx: native MLP matvec failed; falling back to Go graph", "error", err)
	}
	if out, ok, err := nativeMLPGELU(x, m); ok {
		if err == nil {
			return out
		}
		core.Error("mlx: native MLP GELU failed; falling back to Go graph", "error", err)
	}
	gateProj := m.GateProj.Forward(x)
	upProj := m.UpProj.Forward(x)
	activated := GeluGateMul(gateProj, upProj)
	Free(gateProj, upProj)
	result := m.DownProj.Forward(activated)
	Free(activated)
	return result
}

// SiLUMLP is the SiLU-gated SwiGLU feed-forward network: down(silu(gate(x)) *
// up(x)). It is the dense FFN for the Llama-family models (Qwen 2/3, Mistral,
// Mixtral, GPT-OSS, Kimi). It shares MLP's gate/up/down layout but gates with
// SiLU instead of GELU and runs the Go compute graph directly — distinct from
// MLP, which prefers the native GELU matvec kernels.
type SiLUMLP struct {
	GateProj *Linear
	UpProj   *Linear
	DownProj *Linear
}

// Forward computes SwiGLU: down(silu(gate(x)) * up(x)).
func (m *SiLUMLP) Forward(x *Array) *Array {
	gateProj := m.GateProj.Forward(x)
	upProj := m.UpProj.Forward(x)
	activated := SiluGateMul(gateProj, upProj)
	Free(gateProj, upProj)
	result := m.DownProj.Forward(activated)
	Free(activated)
	return result
}

// compiledGELU is retained for standalone GELU call sites.
var compiledGELU *CompiledFunc
// GELU fast-path toggles — in-code diagnostics, off by default, NEVER ambient
// env (an env-readable compute toggle is external control of the engine). Set a
// var locally (or in a test) to trial a path; a proven path graduates to a
// model-declared EngineFeatures, not an env var.
var (
	enableNativeGELUGateMul = false
	enableNativeMLPGELU     = false
	enableCompiledGELU      = false
)

func getCompiledGELU() *CompiledFunc {
	if compiledGELU == nil {
		compiledGELU = CompileShapeless(func(inputs []*Array) []*Array {
			return []*Array{geluApprox(inputs[0])}
		}, true)
	}
	return compiledGELU
}

func GeluGateMul(gate, up *Array) *Array {
	if enableNativeGELUGateMul {
		return GELUGateMul(gate, up)
	}
	activated := GeluActivation(gate)
	out := Mul(activated, up)
	Free(activated)
	return out
}

func GeluActivation(x *Array) *Array {
	if enableCompiledGELU {
		return getCompiledGELU().Call(x)[0]
	}
	return geluApprox(x)
}

func SiluGateMul(gate, up *Array) *Array {
	activated := SiLU(gate)
	out := Mul(activated, up)
	Free(activated)
	return out
}

// geluApprox computes GELU using the tanh approximation:
// 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
func geluApprox(x *Array) *Array {
	const sqrt2OverPi = 0.7978845608028654
	const coeff = 0.044715

	xSquared := Mul(x, x)
	x3 := Mul(xSquared, x)
	Free(xSquared)
	x3Scaled := MulScalar(x3, coeff)
	Free(x3)
	inner := Add(x, x3Scaled)
	Free(x3Scaled)
	scaled := MulScalar(inner, sqrt2OverPi)
	Free(inner)
	t := Tanh(scaled)
	Free(scaled)
	onePlusT := AddScalar(t, 1.0)
	Free(t)
	halfX := MulScalar(x, 0.5)
	result := Mul(halfX, onePlusT)
	Free(halfX, onePlusT)
	return result
}
