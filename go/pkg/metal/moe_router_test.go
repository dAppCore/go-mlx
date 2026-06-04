// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import "testing"

func TestMoERouterTopK(t *testing.T) {
	requireMetalRuntime(t)

	input := FromValues([]float32{1, 2, 3}, 1, 1, 3)
	defer Free(input)

	// Happy path: top-2 of 4 experts by router logit.
	routerWeight := FromValues([]float32{
		1, 0, 0,
		0, 2, 0,
		0, 0, 3,
		-1, 0, 0,
	}, 4, 3)
	defer Free(routerWeight)
	ids, weights, ok, err := moeRouterTopK(input, &MoERouter{Weight: routerWeight}, 2)
	if err != nil {
		t.Fatalf("moeRouterTopK() error = %v", err)
	}
	if !ok {
		t.Fatal("moeRouterTopK() ok = false, want true")
	}
	defer Free(ids, weights)
	if err := Eval(ids, weights); err != nil {
		t.Fatalf("Eval: %v", err)
	}
	gotIDs := ids.DataInt32()
	for i, want := range []int32{2, 1} {
		if gotIDs[i] != want {
			t.Fatalf("ids[%d] = %d, want %d", i, gotIDs[i], want)
		}
	}
	floatSliceApprox(t, weights.Floats(), []float32{0.9933072, 0.006692851})

	// Failure modes: nil router → not ok; topK=0 → diagnostic error. Both ok=false.
	if _, _, ok, err := moeRouterTopK(input, nil, 2); err != nil || ok {
		t.Fatalf("moeRouterTopK(nil router) = ok %v, err %v; want ok false, err nil", ok, err)
	}
	if _, _, ok, err := moeRouterTopK(input, &MoERouter{}, 0); err == nil || ok {
		t.Fatalf("moeRouterTopK(topK=0) = ok %v, err %v; want ok false with diagnostic error", ok, err)
	}
}
