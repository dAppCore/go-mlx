// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package native

import (
	"os"
	"testing"
)

// TestNativeDecoder_PrimitivesMatchValueOps gates the compute seam impl: a
// Decoder-driven RMSNorm → Proj → Add (recorded into one command buffer, flushed
// once) must be byte-for-byte the proven value-level ops (RMSNormBF16 + MatVecBF16 +
// add). So the shared gemma4 orchestration, driving native through this Decoder,
// computes exactly what native's primitives always have — the seam is faithful.
func TestNativeDecoder_PrimitivesMatchValueOps(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, outDim = 8, 8 // outDim == dModel so the residual add is well-shaped
	const eps = float32(1e-6)
	mk := func(salt, n int) []byte {
		f := make([]float32, n)
		for i := range f {
			f[i] = float32((i*salt+7)%53-26) * 0.03
		}
		return toBF16Bytes(f)
	}
	x := mk(3, dModel)
	normW := mk(5, dModel)
	projW := mk(9, outDim*dModel)

	// value-level reference (the proven ops).
	normed, err := RMSNormBF16(x, normW, 1, dModel, eps)
	if err != nil {
		t.Fatalf("RMSNormBF16: %v", err)
	}
	proj, err := MatVecBF16(projW, normed, outDim, dModel)
	if err != nil {
		t.Fatalf("MatVecBF16: %v", err)
	}
	wantSum := make([]byte, dModel*bf16Size)
	for i := 0; i < dModel; i++ {
		v := bf16ToF32(proj[i*bf16Size], proj[i*bf16Size+1]) + bf16ToF32(x[i*bf16Size], x[i*bf16Size+1])
		h := f32ToBF16(v)
		wantSum[i*bf16Size], wantSum[i*bf16Size+1] = byte(h), byte(h>>8)
	}

	// Decoder-driven: record RMSNorm → Proj → Add into one batch, flush once, read.
	var gotProj, gotSum []byte
	withAutoreleasePool(func() {
		d := newNativeDecoder(dModel, dModel)
		bx, bnw, bw := d.Upload(x), d.Upload(normW), d.Upload(projW)
		bnormed, bout, bsum := d.Alloc(dModel*bf16Size), d.Alloc(outDim*bf16Size), d.Alloc(dModel*bf16Size)
		d.RMSNorm(bnormed, bx, bnw, 0, 0, 1, dModel, eps)
		d.Proj(bout, bnormed, bw, 0, outDim, dModel)
		d.Add(bsum, bout, bx, dModel)
		var rerr error
		if gotProj, rerr = d.Read(bout); rerr != nil {
			t.Fatalf("Read bout: %v", rerr)
		}
		if gotSum, rerr = d.Read(bsum); rerr != nil {
			t.Fatalf("Read bsum: %v", rerr)
		}
	})
	if string(gotProj) != string(proj) {
		t.Fatalf("Decoder RMSNorm→Proj != value RMSNormBF16→MatVecBF16:\n got %v\nwant %v", gotProj, proj)
	}
	if string(gotSum) != string(wantSum) {
		t.Fatalf("Decoder Add != value add:\n got %v\nwant %v", gotSum, wantSum)
	}
	t.Logf("native Decoder ≡ value-level ops (RMSNorm→Proj→Add recorded in one batch) — the seam is faithful")
}
