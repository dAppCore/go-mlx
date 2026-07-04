// SPDX-Licence-Identifier: EUPL-1.2

package metal

import "testing"

// callNativeFixedAttentionAt runs the native fixed single-token attention at
// one geometry. Shapes: q [1,qH,1,D], caches [1,kvH,cap,D], k/v [1,kvH,1,D].
func callNativeFixedAttentionAt(t *testing.T, qHeads, kvHeads, capacity, headDim int32, offsetVal int) error {
	t.Helper()
	q := Zeros([]int32{1, qHeads, 1, headDim}, DTypeFloat32)
	kc := Zeros([]int32{1, kvHeads, capacity, headDim}, DTypeFloat32)
	vc := Zeros([]int32{1, kvHeads, capacity, headDim}, DTypeFloat32)
	k := Zeros([]int32{1, kvHeads, 1, headDim}, DTypeFloat32)
	v := Zeros([]int32{1, kvHeads, 1, headDim}, DTypeFloat32)
	offset := FromValue(offsetVal)
	defer Free(q, kc, vc, k, v, offset)
	out, nk, nv, ok, err := NativeFixedSingleTokenAttention(q, kc, vc, k, v, offset, nil, 1)
	if err != nil {
		Free(out, nk, nv)
		return err
	}
	if !ok {
		t.Fatalf("NativeFixedSingleTokenAttention(cap=%d) unsupported — probe shapes need updating", capacity)
	}
	evalErr := Eval(out, nk, nv)
	Free(out, nk, nv)
	return evalErr
}

// THE #91 probe: a capacity-4 native attention call must not poison a later
// capacity-2 call in the same process. If the second call broadcast-fails,
// the native compiled-singleton trace cache (decode_bridge.cpp) is replaying
// a stale-geometry graph — the cross-instance leak behind the red tagged
// suite (AttentionFixedCacheUsesNativeBridge failing after GreedyARSEquivalence).
func TestDecodeGeometryProbe_FixedAttentionCapacityIsolation_Good(t *testing.T) {
	requireMetalRuntime(t)
	if err := callNativeFixedAttentionAt(t, 1, 1, 4, 8, 3); err != nil {
		t.Fatalf("capacity-4 call: %v", err)
	}
	if err := callNativeFixedAttentionAt(t, 1, 1, 2, 8, 1); err != nil {
		t.Fatalf("capacity-2 call after capacity-4 poisoned the process: %v", err)
	}
}

// The #91 pair replicated minimally: the verify fixture's GQA geometry
// (q 2 heads × D4 over a 1-KV-head cap-4 cache, offsets walking 0..3 like a
// 12-token rotated generation's pre-cap phase) followed by the bridge test's
// geometry (1 head × D2, cap 4, offset 0). If the second call broadcast-fails
// (1,1,1,4)-vs-(1,1,1,2), the native compiled trace is corrupted across
// geometries with NO MTP machinery involved.
func TestDecodeGeometryProbe_VerifyThenBridgeGeometry_Good(t *testing.T) {
	requireMetalRuntime(t)
	for off := 0; off < 4; off++ {
		if err := callNativeFixedAttentionAt(t, 2, 1, 4, 4, off); err != nil {
			t.Fatalf("verify-geometry call (offset %d): %v", off, err)
		}
	}
	if err := callNativeFixedAttentionAt(t, 1, 1, 4, 2, 0); err != nil {
		t.Fatalf("bridge-geometry call after verify geometry poisoned the process: %v", err)
	}
}
