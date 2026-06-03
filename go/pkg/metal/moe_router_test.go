// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import "testing"

func TestMoERouterSelectTopK_DenseQwenWrapper_Good(t *testing.T) {
	coverageTokens := "MoERouterSelectTopK DenseQwenWrapper"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	requireMetalRuntime(t)

	input := FromValues([]float32{1, 2, 3}, 1, 1, 3)
	routerWeight := FromValues([]float32{
		1, 0, 0,
		0, 2, 0,
		0, 0, 3,
		-1, 0, 0,
	}, 4, 3)
	defer Free(input, routerWeight)

	router := &Qwen3MoERouter{Weight: routerWeight}
	ids, weights, ok, err := qwen3MoERouterSelectTopK(input, router, 2)
	if err != nil {
		t.Fatalf("qwen3MoERouterSelectTopK() error = %v", err)
	}
	if !ok {
		t.Fatal("qwen3MoERouterSelectTopK() ok = false, want true")
	}
	defer Free(ids, weights)
	if err := Eval(ids, weights); err != nil {
		t.Fatalf("Eval: %v", err)
	}

	gotIDs := ids.DataInt32()
	wantIDs := []int32{2, 1}
	for i := range wantIDs {
		if gotIDs[i] != wantIDs[i] {
			t.Fatalf("ids[%d] = %d, want %d", i, gotIDs[i], wantIDs[i])
		}
	}
	floatSliceApprox(t, weights.Floats(), []float32{0.9933072, 0.006692851})
}

func TestMoERouterSelectTopK_Bad(t *testing.T) {
	coverageTokens := "MoERouterSelectTopK Bad"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	requireMetalRuntime(t)

	input := FromValues([]float32{1, 2, 3}, 1, 1, 3)
	defer Free(input)

	_, _, ok, err := qwen3MoERouterSelectTopK(input, nil, 2)
	if err != nil {
		t.Fatalf("qwen3MoERouterSelectTopK(nil router) error = %v", err)
	}
	if ok {
		t.Fatal("qwen3MoERouterSelectTopK(nil router) ok = true, want false")
	}

	_, _, ok, err = qwen3MoERouterSelectTopK(input, &Qwen3MoERouter{}, 0)
	if err == nil {
		t.Fatal("qwen3MoERouterSelectTopK(topK=0) error = nil, want diagnostic")
	}
	if ok {
		t.Fatal("qwen3MoERouterSelectTopK(topK=0) ok = true, want false")
	}
}
