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

// reluBF16 is metal's ReLU (Maximum(x, 0)) as a byte-identical bf16 select: x≥0 keeps its bytes,
// x<0 becomes bf16 0. No arithmetic, so it equals metal byte-for-byte.
func reluBF16(b []byte) []byte {
	out := make([]byte, len(b))
	copy(out, b)
	for i := 0; i+1 < len(b); i += bf16Size {
		// bf16 sign bit is the top bit of the high byte; negative (and not -0) → 0.
		if b[i+1]&0x80 != 0 {
			out[i], out[i+1] = 0, 0
		}
	}
	return out
}

// AudioSubsampleWeights is the subsampler's bf16 views: two conv layers (weight [outC,3,3,inC] +
// scale-only LayerNorm weight/bias [outC]) and the input projection [hidden, F1·outC1].
type AudioSubsampleWeights struct {
	Conv0, Norm0W, Norm0B []byte
	Conv1, Norm1W, Norm1B []byte
	InputProj             []byte
}

// AudioSubsampleConfig is the subsampler geometry (B=1): mel input dims + the two conv output channel
// counts + the encoder width.
type AudioSubsampleConfig struct {
	Frames, MelBins int
	OutC0, OutC1    int
	Hidden          int
	Eps             float32
}

// convOut returns the strided-conv output length for (in, kernel 3, stride 2, pad 1).
func convOut(in int) int { return (in+2-3)/2 + 1 }

// AudioSubsample runs the gemma4 audio subsampler on log-mel features [frames, melBins] bf16 — the
// BYTE-IDENTICAL port of Gemma4AudioSubSampleConvProjection.Forward (B=1): reshape to NHWC C=1 →
// 2×(Conv2d 3×3 s2 p1 → scale-only LayerNorm over channels → ReLU) → flatten (F1·outC1) → InputProj.
// Returns [ceil(frames/4), hidden] bf16. Every op is a native byte-parity op (Conv2dBF16,
// LayerNormBF16, reluBF16, MatRowsBF16).
func AudioSubsample(features []byte, w *AudioSubsampleWeights, cfg AudioSubsampleConfig) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if len(features) != cfg.Frames*cfg.MelBins*bf16Size {
		return nil, core.NewError("native.AudioSubsample: len(features) must equal Frames*MelBins*2 bytes")
	}
	t0, f0 := convOut(cfg.Frames), convOut(cfg.MelBins)
	h0, err := Conv2dBF16(features, w.Conv0, 1, cfg.Frames, cfg.MelBins, 1, cfg.OutC0, 3, 3, 2, 2, 1, 1)
	if err != nil {
		return nil, err
	}
	if h0, err = LayerNormBF16(h0, w.Norm0W, w.Norm0B, t0*f0, cfg.OutC0, cfg.Eps); err != nil {
		return nil, err
	}
	h0 = reluBF16(h0)

	t1, f1 := convOut(t0), convOut(f0)
	h1, err := Conv2dBF16(h0, w.Conv1, 1, t0, f0, cfg.OutC0, cfg.OutC1, 3, 3, 2, 2, 1, 1)
	if err != nil {
		return nil, err
	}
	if h1, err = LayerNormBF16(h1, w.Norm1W, w.Norm1B, t1*f1, cfg.OutC1, cfg.Eps); err != nil {
		return nil, err
	}
	h1 = reluBF16(h1)

	// flatten [t1, f1, outC1] → [t1, f1·outC1] is a contiguous reinterpret; InputProj maps to hidden.
	return MatRowsBF16(w.InputProj, h1, t1, cfg.Hidden, f1*cfg.OutC1)
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

// sliceColsBF16 extracts columns [c0:c1) from each row of an [rows,cols] bf16 buffer — a byte-copy
// (byte-identical to metal.SliceAxis on the last axis).
func sliceColsBF16(b []byte, rows, cols, c0, c1 int) []byte {
	w := (c1 - c0) * bf16Size
	out := make([]byte, rows*w)
	for r := 0; r < rows; r++ {
		copy(out[r*w:r*w+w], b[(r*cols+c0)*bf16Size:(r*cols+c1)*bf16Size])
	}
	return out
}

// depthwiseConv1dBF16 is the causal depthwise conv1d over time, bf16: out[t,c] = Σ_k in[t-(K-1)+k,c]·
// dw[c,k] (left-pad K-1, in[<0]=0), fp32 accumulation rounded to bf16 — matching metal's
// PadAxis+Conv1d(groups=channels). in is [L,ch], dw is [ch,K], out is [L,ch].
func depthwiseConv1dBF16(in, dw []byte, L, ch, K int) []byte {
	inF, dwF := bf16ToF32Slice(in), bf16ToF32Slice(dw)
	out := make([]byte, L*ch*bf16Size)
	for t := 0; t < L; t++ {
		for c := 0; c < ch; c++ {
			var acc float32
			for k := 0; k < K; k++ {
				if src := t - (K - 1) + k; src >= 0 {
					acc += inF[src*ch+c] * dwF[c*K+k]
				}
			}
			h := f32ToBF16(acc)
			o := (t*ch + c) * bf16Size
			out[o], out[o+1] = byte(h), byte(h>>8)
		}
	}
	return out
}

// AudioLightConv runs the Conformer GLU-conv module on [L, hidden] bf16 — the BYTE-IDENTICAL port of
// metal's Gemma4AudioLightConv.Forward: RMSNorm → LinearStart(h→2·channels) → GLU (gate·σ(gateIn)) →
// causal depthwise conv1d (left-pad kernel-1) → clamp → RMSNorm → activation → LinearEnd → + x. Every
// step is a native byte-parity op (RMSNormBF16, MatRowsBF16, SigmoidBF16, MulBF16, SiLUBF16, AddBF16)
// or a byte-identical select/copy (slice, clamp) / fp32-accumulate conv.
func AudioLightConv(x []byte, w *AudioLightConvWeights, cfg AudioConfig) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	ch, K := cfg.Channels, cfg.KernelSize
	if cfg.Hidden == 0 || ch == 0 || K == 0 {
		return nil, core.NewError("native.AudioLightConv: cfg.Hidden, Channels, KernelSize must be set")
	}
	L := len(x) / (cfg.Hidden * bf16Size)

	pre, err := RMSNormBF16(x, w.PreNorm, L, cfg.Hidden, cfg.Eps)
	if err != nil {
		return nil, err
	}
	start, err := MatRowsBF16(w.LinearStart, pre, L, 2*ch, cfg.Hidden) // [L, 2·ch]
	if err != nil {
		return nil, err
	}
	// GLU: gate · sigmoid(gateIn) — gate = cols [0:ch], gateIn = cols [ch:2ch].
	sig, err := SigmoidBF16(sliceColsBF16(start, L, 2*ch, ch, 2*ch))
	if err != nil {
		return nil, err
	}
	glu, err := MulBF16(sliceColsBF16(start, L, 2*ch, 0, ch), sig)
	if err != nil {
		return nil, err
	}

	conv := clampBF16(depthwiseConv1dBF16(glu, w.DepthwiseWeight, L, ch, K), cfg.ClipMin, cfg.ClipMax)
	normed, err := RMSNormBF16(conv, w.ConvNorm, L, ch, cfg.Eps)
	if err != nil {
		return nil, err
	}
	act, err := audioActivateBF16(normed, cfg.Act)
	if err != nil {
		return nil, err
	}
	end, err := MatRowsBF16(w.LinearEnd, act, L, cfg.Hidden, ch)
	if err != nil {
		return nil, err
	}
	return AddBF16(end, x)
}
