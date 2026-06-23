// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"testing"
)

func perLayerProjUnbatchedRef(t testing.TB, projW, hidden, perLayer []byte, projScale float32, projNormW []byte, numLayers, pliDim, dModel int, eps float32) []byte {
	t.Helper()
	plDim := numLayers * pliDim
	must := func(b []byte, err error) []byte {
		if err != nil {
			t.Fatalf("perLayerProj unbatched op: %v", err)
		}
		return b
	}
	projected := must(MatVecBF16(projW, hidden, plDim, dModel))
	scaled := must(MulBF16(projected, bf16ConstBytes(plDim, projScale)))
	projNormed := must(RMSNormBF16(scaled, projNormW, numLayers, pliDim, eps))
	combined := must(AddBF16(projNormed, perLayer))
	return must(MulBF16(combined, bf16ConstBytes(plDim, gemma4PerLayerCombineScale)))
}

func TestPerLayerProjBatchedMatchesUnbatchedReference(t *testing.T) {
	requireNativeRuntime(t)
	const numLayers, pliDim, dModel = 2, 8, 16
	const eps = float32(1e-5)
	plDim := numLayers * pliDim
	projScale := float32(1 / math.Sqrt(float64(dModel)))
	hidden := toBF16Bytes(syntheticFloat32(dModel, 1))
	perLayer := toBF16Bytes(syntheticFloat32(plDim, 2))
	projW := toBF16Bytes(syntheticFloat32(plDim*dModel, 3))
	projNormW := toBF16Bytes(syntheticFloat32(pliDim, 4))

	got, err := perLayerProjBatched(copyView(projW), hidden, perLayer, projScale, projNormW, plDim, numLayers, pliDim, dModel, eps)
	if err != nil {
		t.Fatalf("perLayerProjBatched: %v", err)
	}
	want := perLayerProjUnbatchedRef(t, projW, hidden, perLayer, projScale, projNormW, numLayers, pliDim, dModel, eps)
	eqBytes(t, "perLayerProjBatched", got, want)
}

func TestPerLayerProjBatchedUsesScalarScaleBuffers(t *testing.T) {
	requireNativeRuntime(t)
	const numLayers, pliDim, dModel = 2, 8, 16
	const eps = float32(1e-5)
	plDim := numLayers * pliDim
	projScale := float32(1 / math.Sqrt(float64(dModel)))
	hidden := toBF16Bytes(syntheticFloat32(dModel, 21))
	perLayer := toBF16Bytes(syntheticFloat32(plDim, 22))
	projW := toBF16Bytes(syntheticFloat32(plDim*dModel, 23))
	projNormW := toBF16Bytes(syntheticFloat32(pliDim, 24))

	projKey := bf16ConstKey{n: plDim, v: projScale}
	combineKey := bf16ConstKey{n: plDim, v: gemma4PerLayerCombineScale}
	bf16ConstMu.Lock()
	delete(bf16ConstCache, projKey)
	delete(bf16ConstCache, combineKey)
	bf16ConstMu.Unlock()

	if _, err := perLayerProjBatched(copyView(projW), hidden, perLayer, projScale, projNormW, plDim, numLayers, pliDim, dModel, eps); err != nil {
		t.Fatalf("perLayerProjBatched: %v", err)
	}

	bf16ConstMu.Lock()
	_, projectedScaleCached := bf16ConstCache[projKey]
	_, combineScaleCached := bf16ConstCache[combineKey]
	bf16ConstMu.Unlock()
	if projectedScaleCached || combineScaleCached {
		t.Fatalf("perLayerProjBatched materialized plDim-wide scale buffers (projected=%v combine=%v), want scalar-bound BF16 scales", projectedScaleCached, combineScaleCached)
	}
}

func TestPerLayerProjBatchedInputGuards(t *testing.T) {
	const numLayers, pliDim, dModel = 2, 3, 4
	plDim := numLayers * pliDim
	hidden := make([]byte, dModel*bf16Size)
	perLayer := make([]byte, plDim*bf16Size)
	projNormW := make([]byte, pliDim*bf16Size)

	tests := []struct {
		name      string
		projView  bufView
		hidden    []byte
		perLayer  []byte
		projNormW []byte
		numLayers int
		pliDim    int
		dModel    int
	}{
		{"zero layers", bufView{}, hidden, perLayer, projNormW, 0, pliDim, dModel},
		{"bad hidden", bufView{}, hidden[:len(hidden)-1], perLayer, projNormW, numLayers, pliDim, dModel},
		{"bad per-layer", bufView{}, hidden, perLayer[:len(perLayer)-1], projNormW, numLayers, pliDim, dModel},
		{"bad norm", bufView{}, hidden, perLayer, projNormW[:len(projNormW)-1], numLayers, pliDim, dModel},
		{"nil resident view", bufView{}, hidden, perLayer, projNormW, numLayers, pliDim, dModel},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := perLayerProjBatched(tc.projView, tc.hidden, tc.perLayer, 1, tc.projNormW, tc.numLayers*tc.pliDim, tc.numLayers, tc.pliDim, tc.dModel, 1e-5)
			if err == nil {
				t.Fatal("perLayerProjBatched error = nil")
			}
		})
	}
}
