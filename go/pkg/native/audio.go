// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"

	core "dappco.re/go"
)

// audio.go ports the gemma4 Conformer audio tower to the no-cgo native path — the faithful
// translation of metal's audio_encoder.go, composed from native's existing kernels (on-device
// matmuls) plus host-side elementwise/conv/attention work (the tower runs once per audio clip at
// prefill, AX-11 not a perf target). Numerically equivalent to pkg/metal within the measured
// tolerance, not bit-identical. Engine-neutral: no model name; geometry arrives as AudioConfig.
// Shares the bf16↔fp32 + rmsNormVec + MatRowsBF16 helpers with vision.go.

// AudioConfig is the engine-neutral Conformer geometry the forward reads. ClipMin/ClipMax are the
// ±gradient-clipping clamp every module borrows (ClipMin==ClipMax ⇒ no clamp). Act is the FF/conv
// activation ("silu"/"swish"/""→SiLU, "relu", "gelu"/"gelu_pytorch_tanh").
type AudioConfig struct {
	Hidden     int
	FFInter    int
	Eps        float32
	Act        string
	FFResidual float32
	ClipMin    float32
	ClipMax    float32
}

// audioClamp clamps v to [min,max] in place (metal's gradient-clipping Clip); min==max ⇒ no-op.
func audioClamp(v []float32, min, max float32) {
	if min == max {
		return
	}
	for i := range v {
		if v[i] < min {
			v[i] = min
		} else if v[i] > max {
			v[i] = max
		}
	}
}

// audioActivate applies the Conformer activation, matching metal's gemma4AudioActivate.
func audioActivate(v []float32, act string) {
	switch act {
	case "relu":
		for i := range v {
			if v[i] < 0 {
				v[i] = 0
			}
		}
	case "gelu", "gelu_pytorch_tanh":
		for i := range v {
			v[i] = geluTanhScalar(v[i])
		}
	default: // silu / swish / ""
		for i := range v {
			v[i] = v[i] / (1 + float32(math.Exp(float64(-v[i]))))
		}
	}
}

// rmsRowsHost RMS-normalises each [axis] row of [rows,axis] fp32 in place-returning, scaling by w
// (nil ⇒ no scale) — the host sibling of RMSNormBF16, reusing rmsNormVec from vision.go.
func rmsRowsHost(m, w []float32, rows, axis int, eps float32) []float32 {
	o := make([]float32, len(m))
	for r := 0; r < rows; r++ {
		copy(o[r*axis:r*axis+axis], m[r*axis:r*axis+axis])
		rmsNormVec(o[r*axis:r*axis+axis], w, eps)
	}
	return o
}

// AudioFeedForwardWeights is one Conformer FeedForward's bf16 weight views: pre/post RMSNorm [hidden]
// and the two linears FFW1 [inter,hidden], FFW2 [hidden,inter]. (gemma4 audio FF linears carry no
// per-linear input/output clip — the FF-level gradient clamp is the active one.)
type AudioFeedForwardWeights struct {
	PreNorm, PostNorm []byte
	FFW1, FFW2        []byte
}

// AudioFeedForward runs one Conformer FeedForward block on [L, hidden] bf16 — the port of metal's
// Gemma4AudioFeedForward.Forward: clamp → RMSNorm(pre) → FFW1 → activation → FFW2 → clamp →
// RMSNorm(post) → ·residual → + x (the ORIGINAL, unclamped input). The two linears run on-device
// (MatRowsBF16); the clamps/activation/norms/residual are host fp32.
func AudioFeedForward(x []byte, w *AudioFeedForwardWeights, cfg AudioConfig) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if cfg.Hidden == 0 || cfg.FFInter == 0 {
		return nil, core.NewError("native.AudioFeedForward: cfg.Hidden and cfg.FFInter must be set")
	}
	L := len(x) / (cfg.Hidden * bf16Size)
	xf := bf16ToF32Slice(x)

	clamped := append([]float32(nil), xf...)
	audioClamp(clamped, cfg.ClipMin, cfg.ClipMax)
	pre := rmsRowsHost(clamped, bf16ToF32Slice(w.PreNorm), L, cfg.Hidden, cfg.Eps)

	up, err := MatRowsBF16(w.FFW1, f32ToBf16Slice(pre), L, cfg.FFInter, cfg.Hidden)
	if err != nil {
		return nil, err
	}
	act := bf16ToF32Slice(up)
	audioActivate(act, cfg.Act)
	down, err := MatRowsBF16(w.FFW2, f32ToBf16Slice(act), L, cfg.Hidden, cfg.FFInter)
	if err != nil {
		return nil, err
	}

	df := bf16ToF32Slice(down)
	audioClamp(df, cfg.ClipMin, cfg.ClipMax)
	post := rmsRowsHost(df, bf16ToF32Slice(w.PostNorm), L, cfg.Hidden, cfg.Eps)
	out := make([]float32, len(post))
	for i := range post {
		out[i] = post[i]*cfg.FFResidual + xf[i] // residual on the original input
	}
	return f32ToBf16Slice(out), nil
}
