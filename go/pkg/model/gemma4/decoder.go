// SPDX-Licence-Identifier: EUPL-1.2

package gemma4

// The compute seam for the ONE gemma4 decode.
//
// gemma4's decode logic — the order of norms, projections, RoPE, attention, the
// SwiGLU MLP, the residuals, the gemma4-specifics (per-attention-type head_dim,
// K==V shared projection, the no-scale value norm, the per-layer scalar) — is the
// SAME regardless of who computes it. Duplicating that sequence per backend is how
// pkg/native's decode drifted from the proven pkg/metal one (the "two decoders that
// disagree" bug). So the sequence lives ONCE here, in pkg/model/gemma4, and a
// backend supplies only the compute via this Decoder.
//
// Why a record-style interface and not value-in/value-out []byte primitives: native
// is the host-unconstrained successor to the (host-bound) metal engine — its speed
// comes from RECORDING a whole token's ops into one command buffer and committing
// once, never dispatch-and-wait per op. So a primitive here ENQUEUES an op into the
// backend's current batch over backend-opaque Buffer handles (no bytes cross the
// seam mid-decode); the backend owns buffer residency + when to Commit. The
// orchestration just sequences. Native records into its Metal command encoder; a
// graph backend records into its graph. One sequence, each backend's own batching,
// no drift.
//
// This is the TEXT decode primitive set (dense + the per-type/K==V/value-norm
// gemma4-specifics). MoE / per-layer-input / vision / audio extend it later; they
// are separate concerns, salvaged only when needed.

// Buffer is a backend-opaque handle to a resident compute buffer (native: a Metal
// buffer; a graph backend: its tensor). The orchestration passes these between
// primitives; it never inspects their contents (bytes only cross the seam at the
// edges, via Decoder.Upload / Read).
type Buffer any

// Decoder is the compute a backend provides for the shared gemma4 decode: each
// method RECORDS one op into the backend's current batch (it does not execute
// immediately). The orchestration in this package sequences them; the backend
// decides residency and flushes with Commit. All activations are the backend's
// own dtype — opaque here.
type Decoder interface {
	// Alloc reserves a resident scratch/cache buffer of nBytes (zeroed). Upload
	// stages host bytes into a fresh buffer; Read flushes and returns a buffer's
	// bytes (the only points data crosses the seam).
	Alloc(nBytes int) Buffer
	Upload(b []byte) Buffer
	Read(buf Buffer) ([]byte, error)

	// RMSNorm records a per-row RMS normalisation: out = (src / rms(src)) * weight,
	// over `rows` independent rows of `axisSize` each. Used for the layer norms
	// (rows=1), the per-head QK-norm (rows = head count), and — with a ones weight —
	// gemma4's no-scale value norm. src/out may alias (in-place). srcOff/outOff are
	// element offsets (e.g. a token's row within a growing cache).
	RMSNorm(out, src, weight Buffer, srcOff, outOff, rows, axisSize int, eps float32)

	// Proj records out = src @ Wᵀ for a row-major weight (the bf16 linear, single
	// query). QuantProj is the 4-bit sibling (packed/scales/biases, group/bits).
	Proj(out, src, weight Buffer, outOff, outDim, inDim int)
	QuantProj(out, src, packed, scales, biases Buffer, outOff, outDim, inDim, groupSize, bits int)

	// RoPE records a partial rotary embedding over `nHeads` heads of `headDim`,
	// rotating the first `rotaryDim` dims (tail passes through); pos is the position
	// buffer, freqs the explicit per-dim periods (YaRN) or nil for base-derived.
	// In-place at srcOff/outOff.
	RoPE(out, src, pos, freqs Buffer, srcOff, outOff, nHeads, headDim, rotaryDim int, base, scale float32)

	// SDPA records scaled dot-product attention of one query against the first `n`
	// rows of seq-major K/V caches (GQA: nKVHeads broadcast across nHeads), with the
	// K/V read starting at kvElemOff (the sliding-window start). scale is the
	// model-declared attention scale.
	SDPA(out, q, kCache, vCache Buffer, nHeads, nKVHeads, headDim, n int, scale float32, kvElemOff int)

	// SwiGLU records the gemma MLP activation: out = gelu_approx(gate) * up, over n
	// elements. Add / Mul record elementwise residual add and the per-layer scalar
	// multiply (Mul broadcasts a [1] scalar across n).
	SwiGLU(out, gate, up Buffer, n int)
	Add(out, a, b Buffer, n int)
	Mul(out, a, scalar Buffer, n int)

	// Commit flushes the recorded batch and waits — called once per token by the
	// orchestration (a MoE/PLE host-readback layer commits mid-token; the text decode
	// commits once at the end).
	Commit() error
}
