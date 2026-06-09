// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import "testing"

// The KV cache Update runs every layer every token: SliceUpdateInplace4 writes
// the new K/V into the buffer at offset, Slice4 returns the active [0:offset]
// region for attention (~4 ops/call, plus an occasional grow). These N-batched
// benches measure the real per-token cost below the sync floor; x34 layers is
// the cache contribution to the per-token budget that the component sweep left
// unmeasured.

func benchmarkKVCacheUpdate(b *testing.B, rotating bool, cap, n int) {
	const B, H, D = 1, 8, 256
	k := RandomUniform(-1, 1, []int32{B, H, 1, D}, DTypeFloat32)
	v := RandomUniform(-1, 1, []int32{B, H, 1, D}, DTypeFloat32)
	Materialize(k, v)
	defer Free(k, v)
	b.ReportAllocs()
	for b.Loop() {
		var c Cache
		if rotating {
			c = NewRotatingKVCache(cap)
		} else {
			c = NewKVCache()
		}
		outs := make([]*Array, 0, n*2)
		for range n {
			ak, av := c.Update(k, v, 1)
			outs = append(outs, ak, av)
		}
		if err := Eval(outs...); err != nil {
			b.Fatalf("Eval: %v", err)
		}
		Free(outs...)
		switch cc := c.(type) {
		case *KVCache:
			if cc.keys != nil {
				Free(cc.keys, cc.values)
			}
		case *RotatingKVCache:
			if cc.keys != nil {
				Free(cc.keys, cc.values)
			}
		}
	}
}

func BenchmarkKVCacheUpdate_Standard_Decode_Batched32(b *testing.B) {
	benchmarkKVCacheUpdate(b, false, 0, 32)
}
func BenchmarkKVCacheUpdate_Rotating_Cap512_Decode_Batched32(b *testing.B) {
	benchmarkKVCacheUpdate(b, true, 512, 32)
}
