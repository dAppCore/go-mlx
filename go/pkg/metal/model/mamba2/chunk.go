// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package mamba2

import (
	core "dappco.re/go"
	metal "dappco.re/go/mlx/pkg/metal"
	flakernel "dappco.re/go/mlx/pkg/metal/model/internal/flakernel"
)

// chunk.go is the length-PARALLEL Mamba-2 selective scan — the state-space
// duality (SSD) chunked form that computes a whole chunk of outputs with
// matmuls + cumulative sums instead of the O(L) serial timestep loop in
// scan.go. It is the prefill speedup: SSDScan walks L timesteps one at a time
// (the correct shape for decode, L=1), whereas SSDScanChunked turns the
// intra-chunk decay into a single [B,H,L,L] decay-weighted score matrix and the
// inter-chunk carry into a per-chunk recurrence over L/chunk steps.
//
// The reference is github.com/state-spaces/mamba (ssd_minimal,
// ssd_minimal_discrete) and github.com/fla-org/flash-linear-attention
// (fla/ops/mamba2). Because A is one scalar per head, the discrete decay
// exp(Δ·A) is a scalar, so the segment-sum decay L_i − L_j is a scalar per
// (i,j) per head — there is NO matrix inverse or triangular solve here, only
// matmuls, a causal mask, and exp. (That is the SSM/scalar-decay case; the
// diagonal-plus-low-rank delta-rule families — RWKV-7, DeltaNet — need the WY /
// UT-transform triangular solve, which is the separate linalg-op work.)
//
// The four steps, per chunk c of length C (mamba_ssm ssd_minimal_discrete):
//
//	L      = exp(segsum(a))                       intra-chunk [C,C] causal decay
//	Y_diag = (C·Bᵀ ⊙ L) @ (Δ·X)                   within-chunk output
//	S_c    = Σ_j exp(L_C − L_j)·(Δ_j x_j) ⊗ B_j   chunk-end state contribution
//	Y_off  = exp(L_i)·(S_entry @ C_i)             carried-state read-out
//
// where a_t = Δ_t·A (scalar log-decay) and L_t = Σ_{s≤t} a_s (cumsum). The
// per-chunk states chain through the inter-chunk recurrence
// S_entry(c+1) = exp(L_C)·S_entry(c) + S_c, which is the same recurrence the
// decode path runs — just at chunk granularity rather than per token.

// DefaultChunkSize is the segment length the chunked scan re-bases the
// cumulative decay within. It bounds the magnitude of exp(L_i − L_j): the
// decay is re-anchored at every chunk boundary so exp never sees more than
// chunkSize accumulated steps, keeping the float32 score matrix well-scaled
// for the prefill lengths the decoder uses. 64 matches the FLA / mamba_ssm
// default and divides the common power-of-two prefill chunk sizes evenly.
const DefaultChunkSize = 64

// SSDScanChunked runs the length-parallel Mamba-2 selective scan for one chunk
// of inputs and returns the mixed output y [B, L, H, P] and the advanced SSM
// state [B, H, P, N] — the same signature and the same result as SSDScan, but
// computed with the chunked SSD matmul form instead of the serial timestep
// loop. prior is the carried state from the previous chunk (decode) or nil for
// a fresh sequence (prefill).
//
// chunkSize is the segment length the sequence is split into for the
// inter-chunk recurrence; pass 0 for DefaultChunkSize. A chunkSize >= L runs
// the whole sequence as a single parallel block (no inter-chunk loop). The
// result is identical to SSDScan to within float rounding; SSDScanChunked is
// the prefill-speed path, SSDScan stays the exact fallback and the decode (L=1)
// path.
//
//	y, next := mamba2.SSDScanChunked(in, prior, 0) // 0 ⇒ DefaultChunkSize
func SSDScanChunked(in ScanInput, prior *metal.Array, chunkSize int32) (*metal.Array, *metal.Array) {
	B := int32(in.X.Dim(0))
	L := int32(in.X.Dim(1))
	H := int32(in.X.Dim(2))
	P := int32(in.X.Dim(3))
	N := int32(in.B.Dim(3))
	dtype := in.X.Dtype()

	if chunkSize <= 0 {
		chunkSize = DefaultChunkSize
	}

	// Δ·X folded once for the whole sequence: the chunked form multiplies the
	// input by the discretisation step before the decay matmul, exactly as the
	// serial scan multiplies B_t by Δ_t. [B,L,H,P] · [B,L,H,1].
	dtCol := metal.Reshape(in.Dt, B, L, H, 1) // [B,L,H,1]
	dtX := metal.Mul(in.X, dtCol)             // [B,L,H,P] = Δ·X
	metal.Free(dtCol)

	// Per-head log-decay a_t = Δ_t·A over the whole sequence. [B,L,H].
	aSeq := metal.Mul(in.Dt, in.A) // [B,L,H]

	// State [B,H,P,N]; zero on prefill, the carried state on decode.
	var state *metal.Array
	if prior != nil && prior.Valid() {
		state = prior.Clone()
	} else {
		state = metal.Zeros([]int32{B, H, P, N}, dtype)
	}

	outputs := make([]*metal.Array, 0, (L+chunkSize-1)/chunkSize)
	for start := int32(0); start < L; start += chunkSize {
		end := start + chunkSize
		if end > L {
			end = L
		}
		cLen := end - start

		seg := segment{
			B: B, H: H, P: P, N: N, cLen: cLen, dtype: dtype,
			dtX:   metal.SliceAxis(dtX, 1, start, end),  // [B,cLen,H,P]
			a:     metal.SliceAxis(aSeq, 1, start, end), // [B,cLen,H]
			bProj: metal.SliceAxis(in.B, 1, start, end), // [B,cLen,H,N]
			cProj: metal.SliceAxis(in.C, 1, start, end), // [B,cLen,H,N]
		}
		y, next := seg.compute(state)
		metal.Free(seg.dtX, seg.a, seg.bProj, seg.cProj)
		metal.Free(state)
		state = next
		outputs = append(outputs, y) // [B,cLen,H,P]
	}
	metal.Free(dtX, aSeq)

	// D skip applied once over the whole concatenated output: y += D ⊙ X. D is
	// per-head [H] (one scalar per head), reshaped to [1,1,H,1] to broadcast
	// over batch, length and head_dim — matching the serial scan's per-head
	// skip (NOT a per-channel [..,P] skip).
	var y *metal.Array
	if len(outputs) == 1 {
		y = outputs[0]
	} else {
		y = metal.Concatenate(outputs, 1) // [B,L,H,P]
		metal.Free(outputs...)
	}
	if in.D != nil && in.D.Valid() {
		dHead := metal.Reshape(in.D, 1, 1, H, 1) // [1,1,H,1]
		dx := metal.Mul(in.X, dHead)             // [B,L,H,P] broadcast over B,L,P
		metal.Free(dHead)
		withSkip := metal.Add(y, dx)
		metal.Free(y, dx)
		y = withSkip
	}

	if err := metal.Eval(y, state); err != nil {
		core.Error("mamba2: SSD chunked scan eval failed", "error", err, "B", B, "L", L, "H", H, "P", P, "N", N, "chunk", chunkSize)
	}
	return y, state
}

// segment is the resolved inputs for one chunk of the SSD scan. dtX is the
// already-Δ-scaled input, a is the per-head log-decay, bProj/cProj the
// selection/read-out projections — all sliced to the chunk's [start,end) span.
type segment struct {
	dtX, a, bProj, cProj *metal.Array
	B, H, P, N, cLen     int32
	dtype                metal.DType
}

// compute runs the four SSD steps for one chunk against the carried entry state
// and returns the chunk output [B,cLen,H,P] and the advanced state [B,H,P,N].
//
// The chunk is laid out head-major ([B,H,cLen,·]) so the decay-weighted score
// matrix and the state matmuls are batched over (B,H) the way metal.Matmul
// contracts the trailing two axes.
func (s segment) compute(entry *metal.Array) (*metal.Array, *metal.Array) {
	cLen := s.cLen

	// Head-major views: [B,cLen,H,·] → [B,H,cLen,·].
	dtX := metal.Transpose4(s.dtX, 0, 2, 1, 3)     // [B,H,cLen,P]
	bHM := metal.Transpose4(s.bProj, 0, 2, 1, 3)   // [B,H,cLen,N]
	cHM := metal.Transpose4(s.cProj, 0, 2, 1, 3)   // [B,H,cLen,N]
	aHM := metal.Transpose(s.a, 0, 2, 1)           // [B,H,cLen]

	// Cumulative log-decay L_t within the chunk (re-based at the chunk start so
	// exp never accumulates beyond cLen steps). [B,H,cLen].
	cumL := metal.CumSum(aHM, 2, false, true) // inclusive forward cumsum over time
	metal.Free(aHM)

	// --- Step 1: intra-chunk diagonal output -------------------------------
	// scores[i,j] = C_i · B_jᵀ  →  C[B,H,cLen,N] @ B[B,H,N,cLen] = [B,H,cLen,cLen].
	bT := metal.Transpose4(bHM, 0, 1, 3, 2) // [B,H,N,cLen]
	scores := metal.Matmul(cHM, bT)         // [B,H,cLen,cLen]
	metal.Free(bT)

	// Segment-sum decay L_i − L_j, masked causal, exp'd. expL [B,H,cLen,1] as
	// the row factor exp(L_i); expNegL [B,H,1,cLen] as the column factor
	// exp(−L_j). Their product over the [cLen,cLen] grid is exp(L_i − L_j); the
	// lower-triangle keep-mask drops j > i.
	expL := metal.Exp(cumL)                              // [B,H,cLen]
	expLrow := metal.Reshape(expL, s.B, s.H, cLen, 1)    // [B,H,cLen,1] = exp(L_i)
	negL := metal.Negative(cumL)                         // −L_j
	expNegL := metal.Exp(negL)                           // [B,H,cLen]
	metal.Free(negL)
	expNegLcol := metal.Reshape(expNegL, s.B, s.H, 1, cLen) // [B,H,1,cLen] = exp(−L_j)
	decay := metal.Mul(expLrow, expNegLcol)                 // [B,H,cLen,cLen] = exp(L_i−L_j)
	metal.Free(expLrow, expNegLcol)

	keep := flakernel.LowerTriangle(cLen)        // [cLen,cLen] causal keep-mask
	keepB := metal.Reshape(keep, 1, 1, cLen, cLen) // [1,1,cLen,cLen]
	metal.Free(keep)
	decayMasked := metal.Mul(decay, keepB) // [B,H,cLen,cLen] causal decay
	metal.Free(decay, keepB)

	weighted := metal.Mul(scores, decayMasked) // [B,H,cLen,cLen]
	metal.Free(scores, decayMasked)
	yDiag := metal.Matmul(weighted, dtX) // [B,H,cLen,P] = (C·Bᵀ ⊙ L) @ (Δ·X)
	metal.Free(weighted)

	// --- Step 2 + 4: carried-state read-out (Y_off) ------------------------
	// y_off[i] = exp(L_i) · (state_entry @ C_iᵀ). state_entry [B,H,P,N] @
	// C[B,H,N,cLen] (= C_iᵀ stacked) gives [B,H,P,cLen]; transpose to
	// [B,H,cLen,P] and scale row i by exp(L_i).
	y := yDiag
	if entry != nil && entry.Valid() {
		cT := metal.Transpose4(cHM, 0, 1, 3, 2) // [B,H,N,cLen]
		off := metal.Matmul(entry, cT)          // [B,H,P,cLen]
		metal.Free(cT)
		offT := metal.Transpose4(off, 0, 1, 3, 2) // [B,H,cLen,P]
		metal.Free(off)
		offScaled := metal.Mul(offT, expLrowP(expL, s.B, s.H, cLen, s.P)) // [B,H,cLen,P]
		metal.Free(offT)
		ySum := metal.Add(yDiag, offScaled)
		metal.Free(yDiag, offScaled)
		y = ySum
	}
	metal.Free(expL)

	// --- Step 3: chunk-end state -------------------------------------------
	// S_c = Σ_j exp(L_C − L_j)·(Δ_j x_j) ⊗ B_j, then
	// state_exit = exp(L_C)·state_entry + S_c.
	newState := s.advanceState(entry, dtX, bHM, cumL)
	metal.Free(cumL, dtX, bHM, cHM)

	// Back to length-major [B,cLen,H,P] for concatenation across chunks.
	yLM := metal.Transpose4(y, 0, 2, 1, 3) // [B,cLen,H,P]
	metal.Free(y)
	return yLM, newState
}

// expLrowP broadcasts the per-position chunk decay exp(L_i) [B,H,cLen] to the
// [B,H,cLen,P] head_dim shape so it scales every channel of the carried-state
// read-out equally (the decay is scalar per head, shared across head_dim).
func expLrowP(expL *metal.Array, b, h, cLen, p int32) *metal.Array {
	row := metal.Reshape(expL, b, h, cLen, 1)        // [B,H,cLen,1]
	out := metal.BroadcastTo(row, []int32{b, h, cLen, p}) // [B,H,cLen,P]
	metal.Free(row)
	return out
}

// advanceState produces state_exit for the next chunk:
//
//	S_c        = Σ_j exp(L_C − L_j)·(Δ_j x_j) ⊗ B_j      [B,H,P,N]
//	state_exit = exp(L_C)·state_entry + S_c
//
// L_C is the last (cumulative) log-decay of the chunk; exp(L_C − L_j) is the
// scalar decay applied to each position's contribution as it ages to the chunk
// end. Δ_j x_j is dtX (already Δ-scaled), so the rank-1 add per position is
// (decay_j · dtX_j) ⊗ B_j, summed over the time axis via a matmul.
func (s segment) advanceState(entry, dtX, bHM, cumL *metal.Array) *metal.Array {
	cLen := s.cLen

	// L_C = last timestep of cumL → [B,H,1]; decay_j = exp(L_C − L_j) [B,H,cLen].
	lLast := metal.SliceAxis(cumL, 2, cLen-1, cLen) // [B,H,1]
	diff := metal.Subtract(lLast, cumL)             // [B,H,cLen] = L_C − L_j (broadcast L_C)
	metal.Free(lLast)
	decayEnd := metal.Exp(diff) // [B,H,cLen]
	metal.Free(diff)

	// Weight each position's Δ·x by its decay-to-end, then contract j with B_j:
	// (decayEnd ⊙ dtX)ᵀ [B,H,P,cLen] @ B[B,H,cLen,N] = Σ_j (decay·Δx)_j ⊗ B_j.
	decayCol := metal.Reshape(decayEnd, s.B, s.H, cLen, 1) // [B,H,cLen,1]
	metal.Free(decayEnd)
	weightedX := metal.Mul(dtX, decayCol) // [B,H,cLen,P] broadcast over P
	metal.Free(decayCol)
	wXT := metal.Transpose4(weightedX, 0, 1, 3, 2) // [B,H,P,cLen]
	metal.Free(weightedX)
	contrib := metal.Matmul(wXT, bHM) // [B,H,P,cLen] @ [B,H,cLen,N] = [B,H,P,N]
	metal.Free(wXT)

	if entry == nil || !entry.Valid() {
		return contrib
	}

	// state_exit = exp(L_C)·state_entry + S_c. exp(L_C) is scalar per head; it
	// scales the whole [P,N] entry-state block of that head equally.
	lLast2 := metal.SliceAxis(cumL, 2, cLen-1, cLen) // [B,H,1]
	expLC := metal.Exp(lLast2)                       // [B,H,1] = exp(L_C)
	metal.Free(lLast2)
	expLCmat := metal.Reshape(expLC, s.B, s.H, 1, 1) // [B,H,1,1] broadcast over P,N
	metal.Free(expLC)
	decayedEntry := metal.Mul(entry, expLCmat) // [B,H,P,N]
	metal.Free(expLCmat)
	out := metal.Add(decayedEntry, contrib)
	metal.Free(decayedEntry, contrib)
	return out
}
