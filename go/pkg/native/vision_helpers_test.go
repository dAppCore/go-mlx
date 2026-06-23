// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

func TestVisionGridForPatchCount(t *testing.T) {
	tests := []struct {
		patches, pool int
		wantH, wantW  int
	}{
		{patches: 0, pool: 2, wantH: 0, wantW: 0},
		{patches: 12, pool: 1, wantH: 3, wantW: 4},
		{patches: 16, pool: 2, wantH: 4, wantW: 4},
		{patches: 18, pool: 2, wantH: 1, wantW: 18},
	}
	for _, tt := range tests {
		gotH, gotW := visionGridForPatchCount(tt.patches, tt.pool)
		if gotH != tt.wantH || gotW != tt.wantW {
			t.Fatalf("visionGridForPatchCount(%d, %d) = (%d, %d), want (%d, %d)", tt.patches, tt.pool, gotH, gotW, tt.wantH, tt.wantW)
		}
	}
}

func TestVisionPoolerBranches(t *testing.T) {
	hidden := toBF16Bytes([]float32{1, 2, 3, 4, 5, 6, 7, 8})
	got := bf16Floats(visionPooler(hidden, 2, 2, 2, 2, 2))
	want := []float32{8, 10}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("grid pool value %d = %v, want %v", i, got[i], want[i])
		}
	}

	group := bf16Floats(visionPooler(hidden, 1, 4, 2, 2, 1))
	wantGroup := []float32{4, 5}
	for i := range wantGroup {
		if group[i] != wantGroup[i] {
			t.Fatalf("group pool value %d = %v, want %v", i, group[i], wantGroup[i])
		}
	}

	pass := bf16Floats(visionPooler(toBF16Bytes([]float32{1, 2, 3, 4, 5, 6}), 3, 1, 2, 2, 1))
	wantPass := []float32{1, 2, 3, 4, 5, 6}
	for i := range wantPass {
		if pass[i] != wantPass[i] {
			t.Fatalf("pass pool value %d = %v, want %v", i, pass[i], wantPass[i])
		}
	}
}

func TestVisionStandardize(t *testing.T) {
	pooled := toBF16Bytes([]float32{2, 4, 6, 8})
	if got := visionStandardize(pooled, nil, nil, 2); &got[0] != &pooled[0] {
		t.Fatal("visionStandardize without weights should return the original slice")
	}
	got := bf16Floats(visionStandardize(pooled, toBF16Bytes([]float32{1, 2}), toBF16Bytes([]float32{3, 4}), 2))
	want := []float32{3, 8, 15, 24}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("standardized value %d = %v, want %v", i, got[i], want[i])
		}
	}
}

func TestVisionProjectorNoProjectionNormalisesRows(t *testing.T) {
	rows := toBF16Bytes([]float32{3, 4, 1, 2})
	got, err := visionProjector(rows, &VisionProjectorWeights{Eps: 0}, 2)
	if err != nil {
		t.Fatalf("visionProjector: %v", err)
	}
	values := bf16Floats(got)
	want := []float32{
		3 / 3.5355339,
		4 / 3.5355339,
		1 / 1.5811388,
		2 / 1.5811388,
	}
	for i := range want {
		if diff := values[i] - want[i]; diff < -0.01 || diff > 0.01 {
			t.Fatalf("projector value %d = %v, want about %v", i, values[i], want[i])
		}
	}
}

func TestVisionProjectorMLPBranch(t *testing.T) {
	requireNativeRuntime(t)
	rows := toBF16Bytes([]float32{3, 4})
	identity := toBF16Bytes([]float32{1, 0, 0, 1})
	got, err := visionProjector(rows, &VisionProjectorWeights{Linear1: identity, Linear2: identity, Eps: 0}, 2)
	if err != nil {
		t.Fatalf("visionProjector MLP branch: %v", err)
	}
	values := bf16Floats(got)
	n0, n1 := float32(3/3.5355339), float32(4/3.5355339)
	want := []float32{geluTanhScalar(n0), geluTanhScalar(n1)}
	for i := range want {
		if diff := values[i] - want[i]; diff < -0.02 || diff > 0.02 {
			t.Fatalf("MLP projector value %d = %v, want about %v", i, values[i], want[i])
		}
	}
}

func TestVisionValidationGuards(t *testing.T) {
	requireNativeRuntime(t)

	pixels := toBF16Bytes(syntheticFloat32(2, 21))
	weight := toBF16Bytes(syntheticFloat32(4, 23))
	if _, err := VisionPatchEmbed(pixels[:len(pixels)-1], weight, nil, 1, 2, 2); err == nil {
		t.Fatal("VisionPatchEmbed(short pixels) error = nil")
	}
	if _, err := VisionPatchEmbed(pixels, weight[:len(weight)-1], nil, 1, 2, 2); err == nil {
		t.Fatal("VisionPatchEmbed(short weight) error = nil")
	}
	if _, err := VisionPatchEmbed(pixels, weight, toBF16Bytes([]float32{1}), 1, 2, 2); err == nil {
		t.Fatal("VisionPatchEmbed(short position embedding) error = nil")
	}

	if _, err := matRowsF32([]float32{1}, []float32{1, 2}, 1, 2, 2); err == nil {
		t.Fatal("matRowsF32(size mismatch) error = nil")
	}
	if got, err := matRowsF32(syntheticFloat32(4, 25), nil, 0, 2, 2); err != nil || len(got) != 0 {
		t.Fatalf("matRowsF32(zero rows) = len %d, err %v; want empty nil-error result", len(got), err)
	}

	q := toBF16Bytes(syntheticFloat32(4, 27))
	kv := toBF16Bytes(syntheticFloat32(2, 29))
	if _, err := VisionSDPA(q, kv, kv, 1, 2, 0, 2, 1); err == nil {
		t.Fatal("VisionSDPA(zero KV heads) error = nil")
	}
	if _, err := VisionSDPA(q[:len(q)-1], kv, kv, 1, 2, 1, 2, 1); err == nil {
		t.Fatal("VisionSDPA(short q) error = nil")
	}
	if _, err := VisionSDPA(q, kv[:len(kv)-1], kv, 1, 2, 1, 2, 1); err == nil {
		t.Fatal("VisionSDPA(short k) error = nil")
	}
	if _, err := VisionSDPA(q, kv, kv[:len(kv)-1], 1, 2, 1, 2, 1); err == nil {
		t.Fatal("VisionSDPA(short v) error = nil")
	}

	in := []float32{1, 2, 3, 4}
	noRoPE := vision2DRoPEHeadMajor(in, 1, 1, 4, 1, 1, 0)
	for i, want := range in {
		if noRoPE[i] != want {
			t.Fatalf("no-RoPE value %d = %v, want %v", i, noRoPE[i], want)
		}
	}
}

func TestVisionKernelFailureGuards(t *testing.T) {
	requireNativeRuntime(t)

	const hidden, headDim, L = 4, 4, 1
	cfg := VisionConfig{
		Hidden: hidden, NumHeads: 1, NumKVHeads: 1, HeadDim: headDim,
		GridH: 1, GridW: 1, RMSNormEps: 1e-5,
	}
	weights := visionGuardLayerWeights(hidden, headDim, 6)
	x := toBF16Bytes(syntheticFloat32(L*hidden, 31))
	withWrongMainLibrary(t, func() {
		if _, err := VisionPatchEmbed(
			toBF16Bytes(syntheticFloat32(2, 33)),
			toBF16Bytes(syntheticFloat32(hidden*2, 35)),
			nil,
			L, 2, hidden,
		); err == nil {
			t.Fatal("VisionPatchEmbed(wrong library) error = nil")
		}
		resetNativePipelineCachesForCoverage()

		if _, err := VisionSDPA(
			toBF16Bytes(syntheticFloat32(L*headDim, 37)),
			toBF16Bytes(syntheticFloat32(L*headDim, 39)),
			toBF16Bytes(syntheticFloat32(L*headDim, 41)),
			L, 1, 1, headDim, 1,
		); err == nil {
			t.Fatal("VisionSDPA(wrong library) error = nil")
		}
		resetNativePipelineCachesForCoverage()

		if _, err := visionAttention(x, weights, cfg); err == nil {
			t.Fatal("visionAttention(wrong library) error = nil")
		}
		resetNativePipelineCachesForCoverage()

		if _, err := visionMLP(x, weights, L, hidden); err == nil {
			t.Fatal("visionMLP(wrong library) error = nil")
		}
		resetNativePipelineCachesForCoverage()

		if _, err := VisionEncoderLayer(x, weights, cfg); err == nil {
			t.Fatal("VisionEncoderLayer(wrong library) error = nil")
		}
	})
}

func visionGuardLayerWeights(hidden, headDim, ffDim int) *VisionLayerWeights {
	return &VisionLayerWeights{
		InputNorm:    toBF16Bytes(syntheticFloat32(hidden, 43)),
		PostAttnNorm: toBF16Bytes(syntheticFloat32(hidden, 45)),
		PreFFNorm:    toBF16Bytes(syntheticFloat32(hidden, 47)),
		PostFFNorm:   toBF16Bytes(syntheticFloat32(hidden, 49)),
		WQ:           toBF16Bytes(syntheticFloat32(headDim*hidden, 51)),
		WK:           toBF16Bytes(syntheticFloat32(headDim*hidden, 53)),
		WV:           toBF16Bytes(syntheticFloat32(headDim*hidden, 55)),
		WO:           toBF16Bytes(syntheticFloat32(hidden*headDim, 57)),
		QNorm:        toBF16Bytes(syntheticFloat32(headDim, 59)),
		KNorm:        toBF16Bytes(syntheticFloat32(headDim, 61)),
		WGate:        toBF16Bytes(syntheticFloat32(ffDim*hidden, 63)),
		WUp:          toBF16Bytes(syntheticFloat32(ffDim*hidden, 65)),
		WDown:        toBF16Bytes(syntheticFloat32(hidden*ffDim, 67)),
	}
}

func TestVisionTowerRejectsShortPositionEmbeddings(t *testing.T) {
	requireNativeRuntime(t)
	cfg := VisionConfig{Hidden: 2, PatchDim: 2, PoolKernel: 1}
	patches := toBF16Bytes([]float32{1, 2, 3, 4})
	w := &VisionWeights{
		PatchEmbedding:     toBF16Bytes([]float32{1, 0, 0, 1}),
		PositionEmbeddings: toBF16Bytes([]float32{1, 2}),
	}
	if _, err := VisionTower(patches, w, cfg); err == nil {
		t.Fatal("VisionTower(short position embeddings) error = nil")
	}
}

func TestVisionTowerRejectsNilWeights(t *testing.T) {
	requireNativeRuntime(t)
	cfg := VisionConfig{Hidden: 2, PatchDim: 2, PoolKernel: 1}
	patches := toBF16Bytes([]float32{1, 2})
	if _, err := VisionTower(patches, nil, cfg); err == nil {
		t.Fatal("VisionTower(nil weights) error = nil")
	}
}
