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
	Channels   int // LightConv conv channels (== Hidden for gemma4 audio)
	KernelSize int // LightConv depthwise conv1d kernel
	Eps        float32
	Act        string
	FFResidual float32
	ClipMin    float32
	ClipMax    float32

	// Relative-position attention geometry (the chunked Conformer attention).
	NumHeads      int
	HeadDim       int
	ChunkSize     int
	PastHorizon   int // ContextLeft-1
	FutureHorizon int // ContextRight
	KScale        float32
	LogitCap      float32 // tanh soft-cap
	InvalidLogit  float32 // masked-position fill
}

func (c AudioConfig) audioContextSize() int { return c.ChunkSize + c.PastHorizon + c.FutureHorizon }

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

// clampBF16 is the byte-parity bf16 clamp to [min,max] — metal.Clip is a SELECT (no arithmetic), so
// the host comparison on bf16 values gives identical bytes: in-range elements keep their original
// bytes, clipped elements become bf16(min)/bf16(max). min==max ⇒ pass-through.
func clampBF16(b []byte, min, max float32) []byte {
	out := make([]byte, len(b))
	copy(out, b)
	if min == max {
		return out
	}
	for i := 0; i+1 < len(b); i += bf16Size {
		v := bf16ToF32(b[i], b[i+1])
		var h uint16
		switch {
		case v < min:
			h = f32ToBF16(min)
		case v > max:
			h = f32ToBF16(max)
		default:
			continue
		}
		out[i], out[i+1] = byte(h), byte(h>>8)
	}
	return out
}

// mulScalarBF16 multiplies every bf16 element by the f32 scalar s, rounding once to bf16 — the same
// bf16-in / f32-scalar / bf16-out computation as metal.MulScalar (verified eqBytes).
func mulScalarBF16(b []byte, s float32) []byte {
	out := make([]byte, len(b))
	for i := 0; i+1 < len(b); i += bf16Size {
		h := f32ToBF16(bf16ToF32(b[i], b[i+1]) * s)
		out[i], out[i+1] = byte(h), byte(h>>8)
	}
	return out
}

// audioActivateBF16 applies the Conformer activation as a byte-parity bf16 op, matching metal's
// gemma4AudioActivate (SiLU = Mul(x, Sigmoid(x)); ReLU = Maximum(x,0); GeLU = the tanh approx).
func audioActivateBF16(b []byte, act string) ([]byte, error) {
	switch act {
	case "relu":
		// metal ReLU = Maximum(x, 0); the byte-parity Maximum-bf16 wrapper is a follow-up.
		return nil, core.NewError("native.audioActivateBF16: relu byte-parity activation not yet ported")
	case "gelu", "gelu_pytorch_tanh":
		return GeluBF16(b)
	default: // silu / swish / ""
		return SiLUBF16(b)
	}
}

// AudioFeedForward runs one Conformer FeedForward block on [L, hidden] bf16 — the BYTE-IDENTICAL port
// of metal's Gemma4AudioFeedForward.Forward: clamp → RMSNorm(pre) → FFW1 → activation → FFW2 → clamp
// → RMSNorm(post) → ·residual → + x (the ORIGINAL input). Every step is a native byte-parity op
// (RMSNormBF16, MatRowsBF16==metal.Matmul, SiLUBF16, AddBF16) or a byte-identical select/scalar
// (clamp, mulScalar) — no host-fp32 reimplementation, so the bytes equal metal's.
func AudioFeedForward(x []byte, w *AudioFeedForwardWeights, cfg AudioConfig) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if cfg.Hidden == 0 || cfg.FFInter == 0 {
		return nil, core.NewError("native.AudioFeedForward: cfg.Hidden and cfg.FFInter must be set")
	}
	L := len(x) / (cfg.Hidden * bf16Size)

	pre, err := RMSNormBF16(clampBF16(x, cfg.ClipMin, cfg.ClipMax), w.PreNorm, L, cfg.Hidden, cfg.Eps)
	if err != nil {
		return nil, err
	}
	up, err := MatRowsBF16(w.FFW1, pre, L, cfg.FFInter, cfg.Hidden)
	if err != nil {
		return nil, err
	}
	act, err := audioActivateBF16(up, cfg.Act)
	if err != nil {
		return nil, err
	}
	down, err := MatRowsBF16(w.FFW2, act, L, cfg.Hidden, cfg.FFInter)
	if err != nil {
		return nil, err
	}
	post, err := RMSNormBF16(clampBF16(down, cfg.ClipMin, cfg.ClipMax), w.PostNorm, L, cfg.Hidden, cfg.Eps)
	if err != nil {
		return nil, err
	}
	return AddBF16(mulScalarBF16(post, cfg.FFResidual), x) // residual on the original input
}

// AudioLightConvWeights is one Conformer LightConv module's bf16 views: pre/conv RMSNorm, the GLU
// expand (LinearStart [2·channels, hidden]) and contract (LinearEnd [hidden, channels]) linears, and
// the depthwise conv1d weight [channels, kernel] (flattened from torch's [channels, kernel, 1]).
type AudioLightConvWeights struct {
	PreNorm, ConvNorm []byte
	LinearStart       []byte
	LinearEnd         []byte
	DepthwiseWeight   []byte
}

// AudioLightConv runs the Conformer GLU-conv module on [L, hidden] bf16 — the port of metal's
// Gemma4AudioLightConv.Forward: RMSNorm → LinearStart(h→2·channels) → GLU (gate·σ(gateIn)) → causal
// depthwise conv1d (left-pad kernel-1) → clamp → RMSNorm → activation → LinearEnd → + x. The two
// linears run on-device (MatRowsBF16); the GLU, the causal depthwise conv1d, the clamp/activation/
// norms are host fp32 (once-per-clip prefill).
func AudioLightConv(x []byte, w *AudioLightConvWeights, cfg AudioConfig) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	ch, K := cfg.Channels, cfg.KernelSize
	if cfg.Hidden == 0 || ch == 0 || K == 0 {
		return nil, core.NewError("native.AudioLightConv: cfg.Hidden, Channels, KernelSize must be set")
	}
	L := len(x) / (cfg.Hidden * bf16Size)
	xf := bf16ToF32Slice(x)

	pre := rmsRowsHost(xf, bf16ToF32Slice(w.PreNorm), L, cfg.Hidden, cfg.Eps)
	start, err := MatRowsBF16(w.LinearStart, f32ToBf16Slice(pre), L, 2*ch, cfg.Hidden)
	if err != nil {
		return nil, err
	}
	sf := bf16ToF32Slice(start) // [L, 2·ch]: first ch = gate, next ch = gateIn

	// GLU: gate · sigmoid(gateIn)
	glu := make([]float32, L*ch)
	for t := 0; t < L; t++ {
		for c := 0; c < ch; c++ {
			gate := sf[t*2*ch+c]
			gateIn := sf[t*2*ch+ch+c]
			glu[t*ch+c] = gate * (1 / (1 + float32(math.Exp(float64(-gateIn))))) // gate · σ(gateIn)
		}
	}

	// causal depthwise conv1d: out[t,c] = Σ_k glu[t-(K-1)+k, c]·dw[c,k], glu[<0]=0 (left-pad K-1).
	dw := bf16ToF32Slice(w.DepthwiseWeight) // [ch, K]
	conv := make([]float32, L*ch)
	for t := 0; t < L; t++ {
		for c := 0; c < ch; c++ {
			var acc float32
			for k := 0; k < K; k++ {
				src := t - (K - 1) + k
				if src >= 0 {
					acc += glu[src*ch+c] * dw[c*K+k]
				}
			}
			conv[t*ch+c] = acc
		}
	}
	audioClamp(conv, cfg.ClipMin, cfg.ClipMax)
	normed := rmsRowsHost(conv, bf16ToF32Slice(w.ConvNorm), L, ch, cfg.Eps)
	audioActivate(normed, cfg.Act)
	end, err := MatRowsBF16(w.LinearEnd, f32ToBf16Slice(normed), L, cfg.Hidden, ch)
	if err != nil {
		return nil, err
	}
	ef := bf16ToF32Slice(end)
	for i := range ef {
		ef[i] += xf[i]
	}
	return f32ToBf16Slice(ef), nil
}
