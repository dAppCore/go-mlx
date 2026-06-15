// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"testing"
)

// BenchmarkInspectAttentionCache_Realistic exercises the host-side
// inspectAttentionCache fan-out used by attention probes. Cache shape
// [1, 32, 1024, 128] = 4M float32 = 16MB — the per-call copy that the
// W11-R zero-copy view pattern eliminates.
func BenchmarkInspectAttentionCache_Realistic(b *testing.B) {
	cache := NewKVCache()
	// [1, 32 heads, 1024 tokens, 128 head_dim] = 4_194_304 float32 = 16 MB
	const heads, seqLen, headDim = 32, 1024, 128
	size := 1 * heads * seqLen * headDim
	data := make([]float32, size)
	for i := range data {
		data[i] = float32(i) * 0.0001
	}
	k := FromValues(data, 1, heads, seqLen, headDim)
	v := FromValues(data, 1, heads, seqLen, headDim)
	outK, outV := cache.Update(k, v, seqLen)
	Materialize(outK, outV)
	Detach(outK)
	Detach(outV)
	for b.Loop() {
		snapshot, ok := inspectAttentionCache(cache, seqLen)
		if !ok {
			b.Fatal("inspectAttentionCache returned not-ok")
		}
		if snapshot.NumHeads != heads {
			b.Fatalf("snapshot.NumHeads = %d, want %d", snapshot.NumHeads, heads)
		}
	}
}
