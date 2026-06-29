// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"os"
	"testing"
	"unsafe"
)

// buildMoEWeights makes a MoELayerWeights with deterministic pseudo-random bf16
// weights of the correct shapes — a fixture for the executor-wiring test.
func buildMoEWeights(numExperts, topK, dModel, dFF, expertDFF, salt int) *MoELayerWeights {
	gen := func(n, s int) []byte {
		f := make([]float32, n)
		for i := range f {
			f[i] = float32((i*s+13)%97-48) * 0.02
		}
		return toBF16Bytes(f)
	}
	scale := make([]float32, numExperts)
	for i := range scale {
		scale[i] = 0.5 + float32(i)*0.1
	}
	return &MoELayerWeights{
		NumExperts: numExperts, TopK: topK, ExpertDFF: expertDFF,
		PreFFNormW: gen(dModel, salt+1), PreFFNorm2W: gen(dModel, salt+2),
		PostFFNorm1W: gen(dModel, salt+3), PostFFNorm2W: gen(dModel, salt+4),
		PostFFNormW: gen(dModel, salt+5),
		WGate:       gen(dFF*dModel, salt+6), WUp: gen(dFF*dModel, salt+7), WDown: gen(dModel*dFF, salt+8),
		RouterNormWScaled: gen(dModel, salt+9), RouterW: gen(numExperts*dModel, salt+10),
		PerExpertScale: toBF16Bytes(scale),
		ExpGateW:       gen(numExperts*expertDFF*dModel, salt+11), ExpUpW: gen(numExperts*expertDFF*dModel, salt+12),
		ExpDownW: gen(numExperts*dModel*expertDFF, salt+13),
	}
}

// moeBlockRef is the oracle for MoEBlockBF16: it rebuilds BOTH branches from the
// parity-proven primitives (local MLP inline; expert branch via moeExpertsRef) and
// wires the five norms + dual-branch sum + residual exactly as
// pkg/metal/model/gemma4 decoder_layer.go's MoE branch. It calls the SAME MoERouter
// as the block, so the expert accumulation order — and thus the bf16 rounding —
// matches, allowing a byte-for-byte gate.
func moeBlockRef(t *testing.T, h []byte, w MoELayerWeights, dModel, dFF int, eps float32) []byte {
	t.Helper()
	numExperts, topK, expertDFF := w.NumExperts, w.TopK, w.ExpertDFF
	must := func(b []byte, err error) []byte {
		if err != nil {
			t.Fatalf("moeBlockRef op: %v", err)
		}
		return b
	}
	idx, weights, err := MoERouter(h, w.RouterNormWScaled, w.RouterW, w.PerExpertScale, numExperts, topK, dModel, eps)
	if err != nil {
		t.Fatalf("moeBlockRef router: %v", err)
	}
	// local dense MLP branch, rebuilt from primitives (no residual).
	h1In := must(RMSNormBF16(h, w.PreFFNormW, 1, dModel, eps))
	g := must(MatVecBF16(w.WGate, h1In, dFF, dModel))
	u := must(MatVecBF16(w.WUp, h1In, dFF, dModel))
	h1 := must(MatVecBF16(w.WDown, must(GeluGateMulBF16(g, u)), dModel, dFF))
	// expert branch on the separately-normed input, rebuilt via moeExpertsRef.
	h2In := must(RMSNormBF16(h, w.PreFFNorm2W, 1, dModel, eps))
	h2 := moeExpertsRef(t, h2In, idx, weights, w.ExpGateW, w.ExpUpW, w.ExpDownW, numExperts, topK, dModel, expertDFF)
	// independent norms, sum, post-norm, residual.
	h1n := must(RMSNormBF16(h1, w.PostFFNorm1W, 1, dModel, eps))
	h2n := must(RMSNormBF16(h2, w.PostFFNorm2W, 1, dModel, eps))
	ff := must(RMSNormBF16(must(AddBF16(h1n, h2n)), w.PostFFNormW, 1, dModel, eps))
	return must(AddBF16(h, ff))
}

// denseFFNRef is the NON-MoE feed-forward (decoder_layer.go's else branch): a single
// dense MLP, rms(h, PreFFNorm) → MLP → rms(·, PostFFNorm) → + h. Used only to prove
// MoEBlockBF16's expert branch genuinely contributes (the dual-branch output must
// differ from running the local MLP alone).
func denseFFNRef(t *testing.T, h []byte, w MoELayerWeights, dModel, dFF int, eps float32) []byte {
	t.Helper()
	must := func(b []byte, err error) []byte {
		if err != nil {
			t.Fatalf("denseFFNRef op: %v", err)
		}
		return b
	}
	ffIn := must(RMSNormBF16(h, w.PreFFNormW, 1, dModel, eps))
	ff := must(mlpTransformBF16(ffIn, w.WGate, w.WUp, w.WDown, dModel, dFF))
	return must(AddBF16(h, must(RMSNormBF16(ff, w.PostFFNormW, 1, dModel, eps))))
}

// TestMoEBlock gates the dual-branch MoE feed-forward composition. MoEBlockBF16 is
// byte-for-byte the independent reference that rebuilds both branches from primitives
// and wires the five norms + dual-branch sum + residual per the metal rule — proving
// the WIRING (each norm in the right place, both branches live, the single residual)
// since the sub-ops are individually gated elsewhere. A non-vacuous check confirms
// the expert branch actually contributes: the dual-branch output differs from the
// dense-MLP-only FFN. Local dFF and expertDFF deliberately differ (catch a dim mixup).
func TestMoEBlock(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const numExperts, topK, dModel, dFF, expertDFF = 8, 2, 256, 512, 384
	const eps = float32(1e-6)
	mk := func(n, salt int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = float32((i*salt+13)%97-48) * 0.02
		}
		return s
	}
	h := toBF16Bytes(mk(dModel, 29))
	w := MoELayerWeights{
		NumExperts:        numExperts,
		TopK:              topK,
		ExpertDFF:         expertDFF,
		PreFFNormW:        toBF16Bytes(mk(dModel, 3)),
		PreFFNorm2W:       toBF16Bytes(mk(dModel, 5)),
		PostFFNorm1W:      toBF16Bytes(mk(dModel, 7)),
		PostFFNorm2W:      toBF16Bytes(mk(dModel, 11)),
		PostFFNormW:       toBF16Bytes(mk(dModel, 13)),
		WGate:             toBF16Bytes(mk(dFF*dModel, 17)),
		WUp:               toBF16Bytes(mk(dFF*dModel, 19)),
		WDown:             toBF16Bytes(mk(dModel*dFF, 23)),
		RouterNormWScaled: toBF16Bytes(mk(dModel, 2)),
		RouterW:           toBF16Bytes(mk(numExperts*dModel, 43)),
		PerExpertScale:    toBF16Bytes([]float32{1.0, 0.5, 2.0, 0.25, 1.5, 0.75, 3.0, 0.1}),
		ExpGateW:          toBF16Bytes(mk(numExperts*expertDFF*dModel, 53)),
		ExpUpW:            toBF16Bytes(mk(numExperts*expertDFF*dModel, 71)),
		ExpDownW:          toBF16Bytes(mk(numExperts*dModel*expertDFF, 47)),
	}

	got, err := MoEBlockBF16(h, w, dModel, dFF, eps)
	if err != nil {
		t.Fatalf("MoEBlockBF16: %v", err)
	}
	want := moeBlockRef(t, h, w, dModel, dFF, eps)
	eqBytes(t, "MoEBlockBF16", got, want)

	// non-vacuous: the dual-branch output must differ from the dense-MLP-only FFN
	// (i.e. the expert branch is genuinely summed in, not silently dropped).
	dense := denseFFNRef(t, h, w, dModel, dFF, eps)
	same := len(dense) == len(got)
	for i := range got {
		if i < len(dense) && got[i] != dense[i] {
			same = false
			break
		}
	}
	if same {
		t.Fatal("MoEBlockBF16 output equals the dense-MLP-only FFN — the expert branch did not contribute")
	}
	t.Logf("MoEBlock (%d experts, top-%d, dFF %d / expertDFF %d): dual-branch ≡ composed reference and differs from dense-only FFN", numExperts, topK, dFF, expertDFF)
}

func TestMLPTransformBF16AllocationBudget(t *testing.T) {
	requireNativeRuntime(t)
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()

	const dModel, dFF = 64, 128
	x := toBF16Bytes(syntheticFloat32(dModel, 37))
	wGate := toBF16Bytes(syntheticFloat32(dFF*dModel, 17))
	wUp := toBF16Bytes(syntheticFloat32(dFF*dModel, 19))
	wDown := toBF16Bytes(syntheticFloat32(dModel*dFF, 23))
	if _, err := mlpTransformBF16(x, wGate, wUp, wDown, dModel, dFF); err != nil {
		t.Fatalf("mlpTransformBF16 warmup: %v", err)
	}

	var transformErr error
	allocs := testing.AllocsPerRun(5, func() {
		_, transformErr = mlpTransformBF16(x, wGate, wUp, wDown, dModel, dFF)
	})
	if transformErr != nil {
		t.Fatalf("mlpTransformBF16: %v", transformErr)
	}
	if allocs > 582 {
		t.Fatalf("mlpTransformBF16 allocations = %.0f, want <= 582", allocs)
	}
}

func TestMLPTransformBF16WritesDirectlyToReturnedOutput(t *testing.T) {
	requireNativeRuntime(t)

	const dModel, dFF = 64, 128
	x := toBF16Bytes(syntheticFloat32(dModel, 37))
	wGate := toBF16Bytes(syntheticFloat32(dFF*dModel, 17))
	wUp := toBF16Bytes(syntheticFloat32(dFF*dModel, 19))
	wDown := toBF16Bytes(syntheticFloat32(dModel*dFF, 23))

	scratch, err := getMLPTransformScratch(dModel, dFF)
	if err != nil {
		t.Fatalf("getMLPTransformScratch: %v", err)
	}
	scratchOut := unsafe.Slice((*byte)(scratch.mlp.down.Contents()), dModel*bf16Size)
	sentinel := bytes.Repeat([]byte{0x7d}, len(scratchOut))
	copy(scratchOut, sentinel)
	putMLPTransformScratch(scratch)

	got, err := mlpTransformBF16(x, wGate, wUp, wDown, dModel, dFF)
	if err != nil {
		t.Fatalf("mlpTransformBF16: %v", err)
	}
	want, err := mlpTransformBF16Into(make([]byte, dModel*bf16Size), x, wGate, wUp, wDown, dModel, dFF)
	if err != nil {
		t.Fatalf("mlpTransformBF16Into reference: %v", err)
	}
	eqBytes(t, "mlpTransformBF16 direct output", got, want)

	scratch, err = getMLPTransformScratch(dModel, dFF)
	if err != nil {
		t.Fatalf("getMLPTransformScratch after call: %v", err)
	}
	defer putMLPTransformScratch(scratch)
	scratchOut = unsafe.Slice((*byte)(scratch.mlp.down.Contents()), dModel*bf16Size)
	if !bytes.Equal(scratchOut, sentinel) {
		t.Fatal("mlpTransformBF16 wrote through pooled scratch output instead of returned output")
	}
}

func TestMoEBlockBF16AllocationBudget(t *testing.T) {
	requireNativeRuntime(t)
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()

	const numExperts, topK, dModel, dFF, expertDFF = 4, 2, 64, 128, 96
	const eps = float32(1e-5)
	h := toBF16Bytes(syntheticFloat32(dModel, 29))
	w := *buildMoEWeights(numExperts, topK, dModel, dFF, expertDFF, 3)
	if _, err := MoEBlockBF16(h, w, dModel, dFF, eps); err != nil {
		t.Fatalf("MoEBlockBF16 warmup: %v", err)
	}

	var blockErr error
	allocs := testing.AllocsPerRun(3, func() {
		_, blockErr = MoEBlockBF16(h, w, dModel, dFF, eps)
	})
	if blockErr != nil {
		t.Fatalf("MoEBlockBF16: %v", blockErr)
	}
	if allocs > 2950 {
		t.Fatalf("MoEBlockBF16 allocations = %.0f, want <= 2950", allocs)
	}
}

func TestMoEBlockBF16IntoWritesDirectlyToCallerOutput(t *testing.T) {
	requireNativeRuntime(t)
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()

	const numExperts, topK, dModel, dFF, expertDFF = 4, 2, 64, 128, 96
	const eps = float32(1e-5)
	h := toBF16Bytes(syntheticFloat32(dModel, 29))
	w := *buildMoEWeights(numExperts, topK, dModel, dFF, expertDFF, 3)
	want, err := MoEBlockBF16(h, w, dModel, dFF, eps)
	if err != nil {
		t.Fatalf("MoEBlockBF16: %v", err)
	}

	scratch, err := getMoEBlockBF16Scratch(dModel, dFF, expertDFF, topK)
	if err != nil {
		t.Fatalf("getMoEBlockBF16Scratch: %v", err)
	}
	sentinel := bytes.Repeat([]byte{0xa5}, len(scratch.out.bytes))
	copy(scratch.out.bytes, sentinel)
	putMoEBlockBF16Scratch(scratch)

	out := make([]byte, dModel*bf16Size)
	outPtr := unsafe.Pointer(&out[0])
	got, err := MoEBlockBF16Into(out, h, w, dModel, dFF, eps)
	if err != nil {
		t.Fatalf("MoEBlockBF16Into: %v", err)
	}
	if len(got) != dModel*bf16Size || unsafe.Pointer(&got[0]) != outPtr {
		t.Fatal("MoEBlockBF16Into did not reuse caller-owned output backing")
	}
	eqBytes(t, "MoEBlockBF16Into direct output", got, want)

	scratch, err = getMoEBlockBF16Scratch(dModel, dFF, expertDFF, topK)
	if err != nil {
		t.Fatalf("getMoEBlockBF16Scratch after call: %v", err)
	}
	defer putMoEBlockBF16Scratch(scratch)
	if !bytes.Equal(scratch.out.bytes, sentinel) {
		t.Fatal("MoEBlockBF16Into wrote through pooled block output instead of caller output")
	}
}

func TestMoEBlockBF16AfterRouterRejectsInvalidInputs(t *testing.T) {
	requireNativeRuntime(t)

	const numExperts, topK, dModel, dFF, expertDFF = 4, 2, 64, 128, 96
	const eps = float32(1e-5)
	h := toBF16Bytes(syntheticFloat32(dModel, 29))
	idx := []int32{0, 1}
	weights := toBF16Bytes([]float32{0.75, 0.25})
	w := *buildMoEWeights(numExperts, topK, dModel, dFF, expertDFF, 3)
	if _, err := moeBlockBF16AfterRouter(h[:len(h)-bf16Size], idx, weights, nil, w, dModel, dFF, eps); err == nil {
		t.Fatal("expected moeBlockBF16AfterRouter to reject short residual")
	}
	bad := w
	bad.ExpGateW = bad.ExpGateW[:len(bad.ExpGateW)-bf16Size]
	if _, err := moeBlockBF16AfterRouter(h, idx, weights, nil, bad, dModel, dFF, eps); err == nil {
		t.Fatal("expected moeBlockBF16AfterRouter to reject short expert gate weight")
	}
	if _, err := moeBlockBF16AfterRouter(nil, nil, nil, nil, MoELayerWeights{}, 0, 0, eps); err != nil {
		t.Fatalf("moeBlockBF16AfterRouter zero dimensions: %v", err)
	}
}

func TestMoEBlockBF16AfterRouterUsesProvidedHiddenBuffer(t *testing.T) {
	requireNativeRuntime(t)

	const numExperts, topK, dModel, dFF, expertDFF = 4, 2, 64, 128, 96
	const eps = float32(1e-5)
	hostH := toBF16Bytes(syntheticFloat32(dModel, 7))
	bufferH := toBF16Bytes(syntheticFloat32(dModel, 29))
	idx := []int32{0, 1}
	weights := toBF16Bytes([]float32{0.75, 0.25})
	w := *buildMoEWeights(numExperts, topK, dModel, dFF, expertDFF, 3)

	pinned, err := newPinnedNoCopyBytes(len(bufferH))
	if err != nil {
		t.Fatalf("newPinnedNoCopyBytes: %v", err)
	}
	defer pinned.Close()
	hBuf, err := pinned.copyBuffer(bufferH)
	if err != nil {
		t.Fatalf("copyBuffer: %v", err)
	}

	want, err := moeBlockBF16AfterRouter(bufferH, idx, weights, nil, w, dModel, dFF, eps)
	if err != nil {
		t.Fatalf("moeBlockBF16AfterRouter: %v", err)
	}
	got, err := moeBlockBF16AfterRouterWithBuffer(hostH, hBuf, idx, weights, nil, w, dModel, dFF, eps)
	if err != nil {
		t.Fatalf("moeBlockBF16AfterRouterWithBuffer: %v", err)
	}
	eqBytes(t, "MoEBlockBF16 provided hidden buffer", got, want)
}

func TestMoEBlockBF16ScratchClose(t *testing.T) {
	requireNativeRuntime(t)

	s, err := newMoEBlockBF16Scratch(64, 128, 96, 2)
	if err != nil {
		t.Fatalf("newMoEBlockBF16Scratch: %v", err)
	}
	if s.h == nil || s.h.buf == nil || s.weights == nil || s.weights.buf == nil || s.out == nil || s.out.buf == nil {
		t.Fatal("newMoEBlockBF16Scratch did not allocate pinned buffers")
	}
	s.Close()
	if s.h != nil || s.weights != nil || s.out != nil || s.dModel != 0 || s.dFF != 0 || s.expertDFF != 0 || s.topK != 0 {
		t.Fatal("Close did not clear pinned buffers and dimensions")
	}
	s.Close()
}

func TestMoEBlockPostCombineRejectsInvalidInputs(t *testing.T) {
	requireNativeRuntime(t)

	const dModel = 64
	h := toBF16Bytes(syntheticFloat32(dModel, 29))
	h1 := toBF16Bytes(syntheticFloat32(dModel, 31))
	h2 := toBF16Bytes(syntheticFloat32(dModel, 37))
	post1 := toBF16Bytes(syntheticFloat32(dModel, 41))
	post2 := toBF16Bytes(syntheticFloat32(dModel, 43))
	post := toBF16Bytes(syntheticFloat32(dModel, 47))
	if _, err := moeBlockPostCombineBF16(h[:len(h)-bf16Size], h1, h2, post1, bufView{}, post2, bufView{}, post, bufView{}, dModel, 1e-5); err == nil {
		t.Fatal("expected moeBlockPostCombineBF16 to reject short residual")
	}
	if _, err := moeBlockPostCombineBF16(h, h1, h2, post1[:len(post1)-bf16Size], bufView{}, post2, bufView{}, post, bufView{}, dModel, 1e-5); err == nil {
		t.Fatal("expected moeBlockPostCombineBF16 to reject short post norm")
	}
	zero, err := moeBlockPostCombineBF16(nil, nil, nil, nil, bufView{}, nil, bufView{}, nil, bufView{}, 0, 1e-5)
	if err != nil {
		t.Fatalf("moeBlockPostCombineBF16 zero dimensions: %v", err)
	}
	if len(zero) != 0 {
		t.Fatalf("moeBlockPostCombineBF16 zero dimensions len = %d, want 0", len(zero))
	}
}

func TestMoEBlockPostCombineScratchClose(t *testing.T) {
	requireNativeRuntime(t)

	s, err := newMoEBlockPostCombineScratch(64)
	if err != nil {
		t.Fatalf("newMoEBlockPostCombineScratch: %v", err)
	}
	if s.h == nil || s.h1 == nil || s.h2 == nil || s.out == nil {
		t.Fatal("newMoEBlockPostCombineScratch did not allocate pinned buffers")
	}
	s.Close()
	if s.h != nil || s.h1 != nil || s.h2 != nil || s.out != nil || s.dModel != 0 {
		t.Fatal("Close did not clear pinned buffers and dimensions")
	}
	s.Close()
}

func TestMoEBlockBF16CachesLocalDenseWeightsWithExperts(t *testing.T) {
	requireNativeRuntime(t)
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()

	const numExperts, topK, dModel, dFF, expertDFF = 4, 2, 64, 128, 96
	h := toBF16Bytes(syntheticFloat32(dModel, 29))
	w := moeLayerWeightsFixture(numExperts, topK, dModel, dFF, expertDFF, 3)
	idx, _, err := MoERouter(h, w.RouterNormWScaled, w.RouterW, w.PerExpertScale, numExperts, topK, dModel, 1e-5)
	if err != nil {
		t.Fatalf("MoERouter: %v", err)
	}
	resetResidentBufsForTest()

	if _, err := MoEBlockBF16(h, w, dModel, dFF, 1e-5); err != nil {
		t.Fatalf("MoEBlockBF16: %v", err)
	}

	key := func(b []byte) uintptr {
		return uintptr(unsafe.Pointer(&b[0]))
	}
	residentBufMu.Lock()
	got := len(residentBufs)
	required := map[uintptr]string{
		key(w.WGate):    "local gate",
		key(w.WUp):      "local up",
		key(w.WDown):    "local down",
		key(w.ExpGateW): "expert gate",
		key(w.ExpUpW):   "expert up",
		key(w.ExpDownW): "expert down",
	}
	missing := []string{}
	for k, name := range required {
		if _, ok := residentBufs[k]; !ok {
			missing = append(missing, name)
		}
	}
	expertGateSz, expertDownSz := expertDFF*dModel*bf16Size, dModel*expertDFF*bf16Size
	selectedSliceHits := 0
	for _, e32 := range idx {
		e := int(e32)
		if _, ok := residentBufs[key(w.ExpGateW[e*expertGateSz:(e+1)*expertGateSz])]; ok {
			selectedSliceHits++
		}
		if _, ok := residentBufs[key(w.ExpUpW[e*expertGateSz:(e+1)*expertGateSz])]; ok {
			selectedSliceHits++
		}
		if _, ok := residentBufs[key(w.ExpDownW[e*expertDownSz:(e+1)*expertDownSz])]; ok {
			selectedSliceHits++
		}
	}
	residentBufMu.Unlock()

	if len(missing) > 0 {
		t.Fatalf("MoEBlockBF16 missing resident weights %v (resident=%d)", missing, got)
	}
	if selectedSliceHits > 0 {
		t.Fatalf("MoEBlockBF16 cached %d selected expert slices; want whole expert tensors only", selectedSliceHits)
	}
	if got < len(required) {
		t.Fatalf("resident weights = %d, want at least %d local dense + whole expert tensors", got, len(required))
	}
}
