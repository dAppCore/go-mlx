// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package moba

import (
	"math"
	"testing"

	"dappco.re/go/mlx/internal/metaltest"
	metal "dappco.re/go/mlx/pkg/metal"
)

// moba_bench_test.go measures the MoBA block-sparse attention kernel at a
// representative synthetic geometry — no model load (AX-11), the
// gla/retnet/mamba2/rwkv7 bench style. It closes the measurement gap: moba had
// no benchmarks, so the allocation behaviour of its mask-build path
// (blockScores → blockSelectMask → expandToTokens) was invisible while its
// siblings were swept.
//
// The benched composite is buildMask + attend — exactly Mixer.Forward lines
// 79-81 minus the four Q/K/V/O projections (which only touch the *metal.Linear
// weights, not the block math). A geometry-only &Mixer{BlockSize,TopK,Scale}
// suffices: buildMask reads B/H/L/D from the array dims, never the Linears or
// NumHeads/HeadDim. So this benches the real hot path with synthetic q/k/v and
// loads no weights.
//
// Two shapes matter, and they run different code:
//
//   - The chunked prefill path (L ≥ BlockSize) is the full MoBA kernel:
//     blockScores mean-pools K into nBlocks reps and dots the query (one
//     Slice4 + Reshape + Mean + Matmul), blockSelectMask does the boost →
//     TopK → threshold-slice → Where → future-forbid (the SliceAxis threshold
//     cut is the rank-4 slice that gave the sequential siblings their wins),
//     and expandToTokens broadcasts the block keep-mask out to [B,H,L,L] and
//     adds the tril causal. At L=256, BlockSize=64 → nBlocks=4.
//   - The single-token decode path (L=1) hits buildMask's L < BlockSize early
//     return: nBlocks would be 0, so MoBA degenerates to plain causal
//     attention over the one token (causalMask(1,1) → attend). This is the
//     honest per-token kernel today — moba's kernels carry no recurrent state
//     argument, and Forward's own comment notes decode-time cache concat
//     "is wired when #1's decoder integration lands; until then Forward
//     attends within the chunk it is handed." So unlike gla's GatedChunk decode
//     (which threads a prior state), there is no carried state to fabricate
//     here; the block-selection fixes (blockSelectMask) are not exercised by
//     this path — they only run when L ≥ BlockSize.
//
// Both forms are force-Eval'd inside the timed region: a lazy return measures
// graph build, not compute (the MLA-bench lesson, inherited via gla/retnet).
//
// Run: go test -tags metal_runtime -bench='MoBA' -benchmem -run='^$' \
//   -benchtime=200ms ./go/pkg/metal/model/moba/
// (needs MLX_METALLIB_PATH set to dist/lib/mlx.metallib)

const (
	mobaBenchH         = 4
	mobaBenchD         = 64
	mobaBenchL         = 256 // 4 × BlockSize(64) blocks — exercises block scoring + top-k
	mobaBenchBlockSize = 64
	mobaBenchTopK      = 3
)

func mobaBenchGate(b *testing.B) {
	if !metaltest.RunMetalTests || !metal.MetalAvailable() {
		b.Skip("build with -tags metal_runtime to enable Metal runtime benchmarks")
	}
}

func mobaBenchScale() float32 { return float32(1.0 / math.Sqrt(float64(mobaBenchD))) }

// mobaBenchQKV builds q/k/v [1,H,L,D] with cheap deterministic fills — values
// do not affect allocation counts or op timing.
func mobaBenchQKV(h, l, d int32) (q, k, v *metal.Array, free func()) {
	fill := func(seed float32) []float32 {
		s := make([]float32, h*l*d)
		for i := range s {
			s[i] = seed + 0.01*float32(i%13) - 0.02*float32(i%5)
		}
		return s
	}
	q = metal.FromValues(fill(0.3), 1, int(h), int(l), int(d))
	k = metal.FromValues(fill(-0.2), 1, int(h), int(l), int(d))
	v = metal.FromValues(fill(0.15), 1, int(h), int(l), int(d))
	free = func() { metal.Free(q, k, v) }
	return q, k, v, free
}

// BenchmarkMoBA_ChunkPrefill is the full block-sparse prefill: score blocks,
// top-k select (self-block forced in, future forbidden), expand to a token
// mask, single softmax attention over it. Exercises buildMask's L ≥ BlockSize
// branch (blockScores → blockSelectMask → expandToTokens) plus attend — the
// path where both the SliceAxis threshold cut and the allNegInf shape live.
func BenchmarkMoBA_ChunkPrefill(b *testing.B) {
	mobaBenchGate(b)
	m := &Mixer{BlockSize: mobaBenchBlockSize, TopK: mobaBenchTopK, Scale: mobaBenchScale()}
	q, k, v, free := mobaBenchQKV(mobaBenchH, mobaBenchL, mobaBenchD)
	defer free()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		mask := m.buildMask(q, k, mobaBenchL)
		out := attend(q, k, v, mask, m.Scale)
		if out == nil {
			b.Fatal("attend returned nil")
		}
		_ = metal.Eval(out)
		metal.Free(mask, out)
	}
}

// BenchmarkMoBA_DecodeStep is the single-token path (L=1): buildMask hits the
// L < BlockSize early return (causalMask(1,1)) and MoBA reduces to plain causal
// attention over the one token. The steady-state per-token kernel today — moba
// carries no recurrent state, so unlike the gla/retnet decode benches there is
// no prior state to thread, and the block-selection kernels do not run here.
func BenchmarkMoBA_DecodeStep(b *testing.B) {
	mobaBenchGate(b)
	m := &Mixer{BlockSize: mobaBenchBlockSize, TopK: mobaBenchTopK, Scale: mobaBenchScale()}
	q, k, v, free := mobaBenchQKV(mobaBenchH, 1, mobaBenchD)
	defer free()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		mask := m.buildMask(q, k, 1)
		out := attend(q, k, v, mask, m.Scale)
		if out == nil {
			b.Fatal("attend returned nil")
		}
		_ = metal.Eval(out)
		metal.Free(mask, out)
	}
}
