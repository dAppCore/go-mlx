// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package native

import (
	"os"
	"testing"
	"time"

	core "dappco.re/go"
)

// forwardLayer builds one layer's synthetic weights (salt varies them per layer).
func forwardLayer(dModel, nHeads, nKV, headDim, dFF, salt int) DecodeLayerWeights {
	qDim, kvDim := nHeads*headDim, nKV*headDim
	mk := func(n, s int) []byte {
		f := make([]float32, n)
		for i := range f {
			f[i] = float32((i*s+7)%101-50) * 0.02
		}
		return toBF16Bytes(f)
	}
	return DecodeLayerWeights{
		AttnNormW: mk(dModel, salt+13), WQ: mk(qDim*dModel, salt+53),
		WK: mk(kvDim*dModel, salt+71), WV: mk(kvDim*dModel, salt+83), WO: mk(dModel*qDim, salt+17),
		MLPNormW: mk(dModel, salt+19), WGate: mk(dFF*dModel, salt+61),
		WUp: mk(dFF*dModel, salt+29), WDown: mk(dModel*dFF, salt+47),
	}
}

// TestDecodeForward gates the multi-layer, multi-token forward against the
// parity-proven single step: DecodeForward must equal stepping DecodeStepKV
// token-by-token, layer-by-layer (each layer's own growing cache). This anchors
// the loop wiring — the residual stream flowing layer→layer, the per-layer cache
// growth across tokens, the per-token position — to the proven real step.
func TestDecodeForward(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF = 512, 8, 4, 64, 1024
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-5)
	const nLayers, T, maxLen = 3, 4, 8
	kvDim := nKV * headDim

	layers := make([]DecodeLayerWeights, nLayers)
	for l := range layers {
		layers[l] = forwardLayer(dModel, nHeads, nKV, headDim, dFF, (l+1)*100)
	}
	inputs := make([][]byte, T)
	for i := range inputs {
		f := make([]float32, dModel)
		for j := range f {
			f[j] = float32((j*(i+3)+5)%97-48) * 0.02
		}
		inputs[i] = toBF16Bytes(f)
	}

	// reference: step DecodeStepKV through the loop, each layer its own Go cache
	kC := make([][]byte, nLayers)
	vC := make([][]byte, nLayers)
	for l := range kC {
		kC[l] = make([]byte, maxLen*kvDim*bf16Size)
		vC[l] = make([]byte, maxLen*kvDim*bf16Size)
	}
	ref := make([][]byte, T)
	for tok := 0; tok < T; tok++ {
		x := inputs[tok]
		for l := 0; l < nLayers; l++ {
			w := layers[l]
			var err error
			x, err = DecodeStepKV(x, w.AttnNormW, w.WQ, w.WK, w.WV, w.WO, kC[l], vC[l], w.MLPNormW, w.WGate, w.WUp, w.WDown, dModel, nHeads, nKV, headDim, maxLen, dFF, tok, base, scale, eps)
			if err != nil {
				t.Fatalf("DecodeStepKV ref t=%d l=%d: %v", tok, l, err)
			}
		}
		ref[tok] = x
	}

	got, err := DecodeForward(inputs, layers, dModel, nHeads, nKV, headDim, maxLen, dFF, base, scale, eps)
	if err != nil {
		t.Fatalf("DecodeForward: %v", err)
	}
	if len(got) != T {
		t.Fatalf("DecodeForward returned %d outputs, want %d", len(got), T)
	}
	for tok := 0; tok < T; tok++ {
		eqBytes(t, "DecodeForward token", got[tok], ref[tok])
	}
	t.Logf("DecodeForward(%d layers × %d tokens, GQA %d/%d, growing cache): byte-identical to stepped DecodeStepKV", nLayers, T, nHeads, nKV)
}

// TestDecodeForwardICB gates the cache-grow ICB: replaying the recorded N-layer
// stack per token — bumping offBuf/nBuf and re-setting each layer's two cache-write
// offsets — must equal the proven re-encode DecodeForward byte-for-byte, over a
// cache that grows token by token. Run at 1 and 3 layers so the per-layer offset
// rebind and the cross-layer residual ping-pong are both exercised.
func TestDecodeForwardICB(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF = 512, 8, 4, 64, 1024
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-5)
	const T, maxLen = 5, 8

	for _, nLayers := range []int{1, 3} {
		layers := make([]DecodeLayerWeights, nLayers)
		for l := range layers {
			layers[l] = forwardLayer(dModel, nHeads, nKV, headDim, dFF, (l+1)*100)
		}
		inputs := make([][]byte, T)
		for i := range inputs {
			f := make([]float32, dModel)
			for j := range f {
				f[j] = float32((j*(i+3)+5)%97-48) * 0.02
			}
			inputs[i] = toBF16Bytes(f)
		}

		ref, err := DecodeForward(inputs, layers, dModel, nHeads, nKV, headDim, maxLen, dFF, base, scale, eps)
		if err != nil {
			t.Fatalf("DecodeForward (%d layers): %v", nLayers, err)
		}
		got, err := DecodeForwardICB(inputs, layers, dModel, nHeads, nKV, headDim, maxLen, dFF, base, scale, eps)
		if err != nil {
			t.Fatalf("DecodeForwardICB (%d layers): %v", nLayers, err)
		}
		if len(got) != T {
			t.Fatalf("DecodeForwardICB returned %d outputs, want %d", len(got), T)
		}
		for tok := 0; tok < T; tok++ {
			eqBytes(t, core.Sprintf("DecodeForwardICB L%d tok%d", nLayers, tok), got[tok], ref[tok])
		}
		t.Logf("DecodeForwardICB(%d layers × %d tokens, growing cache): byte-identical to re-encode DecodeForward — cache-grow ICB holds", nLayers, T)
	}
}

// TestDecodeForwardHostCost measures the real forward's per-token wall as the KV
// cache grows. The per-token host encode is a fixed op count regardless of window
// length (N layers × the same ops), so at these synthetic dims — where GPU work is
// tiny — the per-token cost stays ~flat as the cache fills, the structural reason
// the encode-bypass (single-submit per-token ICB) pays off: constant host work per
// token, flat memory pressure, no per-token sawtooth. Shared weights keep it
// AX-11-light; this is host-cost at synthetic dims, NOT real-model tok/s.
func TestDecodeForwardHostCost(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF = 512, 8, 4, 64, 1024
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-5)
	const nLayers = 24

	w := forwardLayer(dModel, nHeads, nKV, headDim, dFF, 100)
	layers := make([]DecodeLayerWeights, nLayers)
	for l := range layers {
		layers[l] = w // shared weights: host encode cost is bind-count, not which buffer
	}
	mkInputs := func(T int) [][]byte {
		in := make([][]byte, T)
		for i := range in {
			f := make([]float32, dModel)
			for j := range f {
				f[j] = float32((j*(i+3)+5)%97-48) * 0.02
			}
			in[i] = toBF16Bytes(f)
		}
		return in
	}
	// warm
	if _, err := DecodeForward(mkInputs(4), layers, dModel, nHeads, nKV, headDim, 4, dFF, base, scale, eps); err != nil {
		t.Fatalf("warm: %v", err)
	}
	for _, T := range []int{8, 16, 32} {
		inputs := mkInputs(T)
		t0 := time.Now()
		if _, err := DecodeForward(inputs, layers, dModel, nHeads, nKV, headDim, T, dFF, base, scale, eps); err != nil {
			t.Fatalf("DecodeForward T=%d: %v", T, err)
		}
		d := time.Since(t0)
		t.Logf("%2d-layer forward, %2d tokens (cache 1..%d): %.2f ms total, %6.1f µs/token",
			nLayers, T, T, float64(d.Microseconds())/1000, float64(d.Microseconds())/float64(T))
	}
}

// TestDecodeForwardICBEncodeBypass is the cache-grow rung's payoff: over the REAL
// growing-cache forward, re-encoding all 24*nLayers ops per token (DecodeForward)
// vs replaying the recorded stack and re-setting only offBuf/nBuf + 2*nLayers
// cache-write offsets (DecodeForwardICB). Both submit one commit+wait per token,
// so the delta is the per-token host encode the replay-with-rebind removes from a
// real decode loop — the encode-bypass made good on an actual growing KV cache.
// Host-cost at synthetic dims, NOT real-model tok/s.
func TestDecodeForwardICBEncodeBypass(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF = 512, 8, 4, 64, 1024
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-5)
	const nLayers = 24

	w := forwardLayer(dModel, nHeads, nKV, headDim, dFF, 100)
	layers := make([]DecodeLayerWeights, nLayers)
	for l := range layers {
		layers[l] = w
	}
	mkInputs := func(T int) [][]byte {
		in := make([][]byte, T)
		for i := range in {
			f := make([]float32, dModel)
			for j := range f {
				f[j] = float32((j*(i+3)+5)%97-48) * 0.02
			}
			in[i] = toBF16Bytes(f)
		}
		return in
	}
	// warm both paths
	_, _ = DecodeForward(mkInputs(4), layers, dModel, nHeads, nKV, headDim, 4, dFF, base, scale, eps)
	_, _ = DecodeForwardICB(mkInputs(4), layers, dModel, nHeads, nKV, headDim, 4, dFF, base, scale, eps)

	for _, T := range []int{8, 16, 32} {
		inputs := mkInputs(T)
		t0 := time.Now()
		if _, err := DecodeForward(inputs, layers, dModel, nHeads, nKV, headDim, T, dFF, base, scale, eps); err != nil {
			t.Fatalf("DecodeForward T=%d: %v", T, err)
		}
		reEnc := time.Since(t0)
		t1 := time.Now()
		if _, err := DecodeForwardICB(inputs, layers, dModel, nHeads, nKV, headDim, T, dFF, base, scale, eps); err != nil {
			t.Fatalf("DecodeForwardICB T=%d: %v", T, err)
		}
		icb := time.Since(t1)
		reUs := float64(reEnc.Microseconds()) / float64(T)
		icbUs := float64(icb.Microseconds()) / float64(T)
		t.Logf("%2d-layer forward, %2d tokens: re-encode %6.1f µs/tok, ICB-replay %6.1f µs/tok, host saved %6.1f µs/tok (%.2fx)",
			nLayers, T, reUs, icbUs, reUs-icbUs, reUs/icbUs)
	}
}

// TestForwardGPUvsWall splits the E2B-scale per-token wall into pure GPU
// execution (from the command-buffer timestamps) vs host/sync, so the fusion
// target is evidence not inference: if GPU << wall the cost is the ICB execution
// (840 serial barriers / replay / residency); if GPU ≈ wall it is real
// kernel work (gemv bandwidth + launches). Opt-in (NATIVE_REALSCALE).
func TestForwardGPUvsWall(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" || os.Getenv("NATIVE_REALSCALE") == "" {
		t.Skip("set MLX_METALLIB_PATH and NATIVE_REALSCALE")
	}
	const dModel, nHeads, nKV, headDim, dFF, nLayers = 1536, 8, 1, 256, 6144, 35
	const base, scale, eps = float32(1000000), float32(0.0625), float32(1e-6)
	w := forwardLayer(dModel, nHeads, nKV, headDim, dFF, 100)
	layers := make([]DecodeLayerWeights, nLayers)
	for l := range layers {
		layers[l] = w
	}
	mkInputs := func(T int) [][]byte {
		in := make([][]byte, T)
		for i := range in {
			f := make([]float32, dModel)
			for j := range f {
				f[j] = float32((j*(i+3)+5)%97-48) * 0.02
			}
			in[i] = toBF16Bytes(f)
		}
		return in
	}
	run := func(T int) (wallUs, gpuUs float64) {
		profileForward = true
		defer func() { profileForward = false }()
		profForwardGPUSec = 0
		t0 := time.Now()
		if _, err := DecodeForwardICB(mkInputs(T), layers, dModel, nHeads, nKV, headDim, T, dFF, base, scale, eps); err != nil {
			t.Fatalf("DecodeForwardICB T=%d: %v", T, err)
		}
		return float64(time.Since(t0).Microseconds()), profForwardGPUSec * 1e6
	}
	// Two-point separation: wall(T) = recording(one-time) + T·steady. The earlier
	// single-T wall/T conflated the two and read "host-bound"; subtracting isolates
	// the real steady-state per-token (and the GPU fraction of it). Warm first so
	// one-time PSO compilation lands in neither timed point (it would corrupt the
	// subtraction — it only happens on the first call).
	run(4)
	const T1, T2 = 16, 48
	w1, _ := run(T1)
	w2, g2 := run(T2)
	steady := (w2 - w1) / (T2 - T1)
	recording := w1 - T1*steady
	gpuPerTok := g2 / T2
	t.Logf("E2B-scale ICB forward (two-point T=%d,%d):", T1, T2)
	t.Logf("  STEADY-STATE per-token %7.1f µs — GPU-exec %7.1f µs (%.0f%%), host+sync %7.1f µs",
		steady, gpuPerTok, 100*gpuPerTok/steady, steady-gpuPerTok)
	t.Logf("  one-time ICB recording %7.0f µs (amortises over tokens; the single-T wall/T artifact)", recording)
	t.Logf("  → steady-state ≈ %.0f tok/s (bf16); GPU-bound, so 4-bit weights (qmv, ~1/4 the read) is the lever", 1e6/steady)
}

// TestDecodeForwardICBRealScale answers whether the encode-bypass survives at
// PRODUCTION scale: it runs the forward at gemma4-E2B's core decode dims (dModel
// 1536, 35 layers, headDim 256, MQA nKV=1, dFF 6144) where per-layer GPU work is
// real, not negligible — so the question "is decode still host-bound, do the
// savings still pay" gets a real number. Opt-in (NATIVE_REALSCALE set) since it is
// a heavier run. Parity is asserted at these dims first (byte-identical to the
// re-encode path), then the per-token A/B is timed.
//
// HONEST SCOPE: this is a host-cost PROXY at E2B's dimensions — a uniform dense
// layer, NOT exact E2B (its MoE blocks, sliding-window layers, KV-sharing, logit
// soft-cap are not modelled). It measures the host encode the ICB removes at real
// op-count/dims; it is not a real-model tok/s and produces no tokens (no embedding
// /lm_head/sampler). Shared weights keep the build light; the real distinct-weight
// working set is ~2.4 GB (reported), allocated once — flat per-token, no sawtooth.
func TestDecodeForwardICBRealScale(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" || os.Getenv("NATIVE_REALSCALE") == "" {
		t.Skip("set MLX_METALLIB_PATH and NATIVE_REALSCALE to run the E2B-scale measurement")
	}
	// gemma4-E2B core decode dims (text_config)
	const dModel, nHeads, nKV, headDim, dFF, nLayers = 1536, 8, 1, 256, 6144, 35
	const base, scale, eps = float32(1000000), float32(0.0625), float32(1e-6)
	const T, maxLen = 16, 16

	w := forwardLayer(dModel, nHeads, nKV, headDim, dFF, 100)
	layers := make([]DecodeLayerWeights, nLayers)
	for l := range layers {
		layers[l] = w
	}
	perLayerBytes := (nHeads*headDim*dModel + 2*nKV*headDim*dModel + dModel*nHeads*headDim + 2*dFF*dModel + dModel*dFF) * bf16Size
	inputs := make([][]byte, T)
	for i := range inputs {
		f := make([]float32, dModel)
		for j := range f {
			f[j] = float32((j*(i+3)+5)%97-48) * 0.02
		}
		inputs[i] = toBF16Bytes(f)
	}

	// parity at real scale, then timing
	ref, err := DecodeForward(inputs, layers, dModel, nHeads, nKV, headDim, maxLen, dFF, base, scale, eps)
	if err != nil {
		t.Fatalf("DecodeForward: %v", err)
	}
	got, err := DecodeForwardICB(inputs, layers, dModel, nHeads, nKV, headDim, maxLen, dFF, base, scale, eps)
	if err != nil {
		t.Fatalf("DecodeForwardICB: %v", err)
	}
	for tok := 0; tok < T; tok++ {
		eqBytes(t, core.Sprintf("E2B-scale tok%d", tok), got[tok], ref[tok])
	}

	t0 := time.Now()
	if _, err := DecodeForward(inputs, layers, dModel, nHeads, nKV, headDim, maxLen, dFF, base, scale, eps); err != nil {
		t.Fatalf("DecodeForward timed: %v", err)
	}
	reEnc := time.Since(t0)
	t1 := time.Now()
	if _, err := DecodeForwardICB(inputs, layers, dModel, nHeads, nKV, headDim, maxLen, dFF, base, scale, eps); err != nil {
		t.Fatalf("DecodeForwardICB timed: %v", err)
	}
	icb := time.Since(t1)
	reUs := float64(reEnc.Microseconds()) / float64(T)
	icbUs := float64(icb.Microseconds()) / float64(T)
	t.Logf("E2B-scale (dModel %d, %d layers, headDim %d, MQA, dFF %d), %d tokens — parity OK:", dModel, nLayers, headDim, dFF, T)
	t.Logf("  re-encode %7.1f µs/tok, ICB-replay %7.1f µs/tok, host saved %7.1f µs/tok (%.2fx)",
		reUs, icbUs, reUs-icbUs, reUs/icbUs)
	t.Logf("  distinct-weight working set ≈ %.2f GB (%.1f MB/layer × %d), allocated once — flat per-token",
		float64(perLayerBytes)*float64(nLayers)/1e9, float64(perLayerBytes)/1e6, nLayers)
}
