// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"os"
	"testing"
	"time"

	"dappco.re/go/mlx/pkg/model"
	g4 "dappco.re/go/mlx/pkg/model/gemma4"
)

// pleQuantModel assembles a small e2b-shaped PLE quant model (4-bit main+PLE embedding, bf16 PLE
// projection — the shape the GPU next-inputs seam handles).
func pleQuantModel(t testing.TB, numLayers, dFF, vocab int) (*QuantModel, model.Arch) {
	const dModel, nHeads, nKV, headDim = 128, 2, 1, 64
	const pliDim, gs, bits = 64, 64, 4
	cfg := g4.Config{
		HiddenSize: dModel, NumHiddenLayers: numLayers, IntermediateSize: dFF,
		NumAttentionHeads: nHeads, NumKeyValueHeads: nKV, HeadDim: headDim, VocabSize: vocab, RMSNormEps: 1e-6,
		HiddenSizePerLayerInput: pliDim, VocabSizePerLayerInput: vocab,
		Quantization: &model.QuantConfig{GroupSize: gs, Bits: bits},
	}
	arch, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	ts := quantGemma4Tensors(t, arch, gs, bits)
	addPLETensors(t, ts, arch, gs, bits)
	lm, err := model.Assemble(ts, arch, model.StandardWeightNames())
	if err != nil {
		t.Fatalf("Assemble: %v", err)
	}
	g, err := loadedToQuant(lm, gs, bits)
	if err != nil {
		t.Fatalf("loadedToQuant: %v", err)
	}
	if !g.HasPLE() {
		t.Fatal("fixture should have the per-layer-input tower")
	}
	return g, arch
}

// TestChainedGPUDecodeMatchesHost gates the chained-GPU decode: with the GPU next-inputs seam ON (each
// step produces the next emb+pli on-GPU, one command buffer/token) the token sequence must equal the host
// embed/PLE chained path. A bug in the on-GPU emb/pli, the no-input stepBody, or the cache/pos bookkeeping
// diverges the tokens. Also pins that the GPU path is actually wired (not silently falling back).
func TestChainedGPUDecodeMatchesHost(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	g, arch := pleQuantModel(t, 2, 256, 32)
	const maxLen, N = 16, 8
	prompt := []int32{1, 5, 3, 2}

	sessGPU, err := NewArchQuantSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("session GPU: %v", err)
	}
	if sessGPU.encNextInputsGPU == nil {
		t.Fatal("expected the GPU next-inputs seam wired (e2b-shaped PLE session)")
	}
	chainedGPUInputsDisabled = false
	gpuGen, err := sessGPU.Generate(prompt, N, -1)
	if err != nil {
		t.Fatalf("Generate (GPU): %v", err)
	}

	chainedGPUInputsDisabled = true
	defer func() { chainedGPUInputsDisabled = false }()
	sessHost, err := NewArchQuantSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("session host: %v", err)
	}
	hostGen, err := sessHost.Generate(prompt, N, -1)
	if err != nil {
		t.Fatalf("Generate (host): %v", err)
	}

	if len(gpuGen) != len(hostGen) || len(gpuGen) != N {
		t.Fatalf("token count: GPU %d, host %d, want %d", len(gpuGen), len(hostGen), N)
	}
	for i := range gpuGen {
		if gpuGen[i] != hostGen[i] {
			t.Fatalf("token %d: chained-GPU %d != host %d (GPU=%v host=%v)", i, gpuGen[i], hostGen[i], gpuGen, hostGen)
		}
	}
	t.Logf("chained-GPU decode matches host embed/PLE path: %v", gpuGen)
}

// TestChainedGPUDecodeHeadroom measures the per-token GPU-execution span vs wall across a chained-GPU
// decode — the host/sync gap a submit-ahead pipeline could overlap. Reported at 16 AND 32 layers: the
// fixed per-token sync is a smaller fraction at depth, so this is the evidence for whether the 2-ICB
// submit-ahead (piece b) is worth its build cost. Diagnostic (logs), not a pass/fail gate.
func TestChainedGPUDecodeHeadroom(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	prompt := []int32{1, 5, 3, 7, 2, 9}
	const maxLen, N = 128, 48
	for _, numLayers := range []int{16, 32} {
		g, arch := pleQuantModel(t, numLayers, 6144, 8192)
		sess, err := NewArchQuantSession(g, arch, maxLen)
		if err != nil {
			t.Fatalf("%dL session: %v", numLayers, err)
		}
		if sess.encNextInputsGPU == nil {
			t.Fatalf("%dL: chained-GPU path not wired", numLayers)
		}
		if err := sess.PrefillTokens(prompt); err != nil {
			t.Fatalf("%dL prefill: %v", numLayers, err)
		}
		// warmup (untimed) then a measured run.
		if _, err := sess.GenerateFromCache(4, -1); err != nil {
			t.Fatalf("%dL warmup: %v", numLayers, err)
		}
		pieceTimingOn = true
		chainedGPUSpanNs = 0
		t0 := time.Now()
		if _, err := sess.GenerateFromCache(N, -1); err != nil {
			pieceTimingOn = false
			t.Fatalf("%dL generate: %v", numLayers, err)
		}
		wall := time.Since(t0)
		pieceTimingOn = false
		_ = sess.Close()
		gpu := time.Duration(chainedGPUSpanNs)
		headroom := float64(wall-gpu) / float64(wall) * 100
		t.Logf("%2dL: wall %.2fms  gpu-span %.2fms  per-tok wall %.3fms gpu %.3fms  host/sync headroom %.1f%% (submit-ahead ceiling)",
			numLayers, float64(wall.Microseconds())/1000, float64(gpu.Microseconds())/1000,
			float64(wall.Microseconds())/1000/float64(N), float64(gpu.Microseconds())/1000/float64(N), headroom)
	}
}

func benchChainedDecodePLE(b *testing.B, gpuInputs bool) {
	if os.Getenv(MetallibPathEnv) == "" {
		b.Skip("metallib not set")
	}
	g, arch := pleQuantModel(b, 16, 6144, 8192)
	const maxLen, N = 96, 32
	prompt := []int32{1, 5, 3, 7, 2, 9}
	chainedGPUInputsDisabled = !gpuInputs
	defer func() { chainedGPUInputsDisabled = false }()
	b.SetBytes(int64(N))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		sess, err := NewArchQuantSession(g, arch, maxLen)
		if err != nil {
			b.Fatal(err)
		}
		if err := sess.PrefillTokens(prompt); err != nil {
			b.Fatal(err)
		}
		b.StartTimer()
		if _, err := sess.GenerateFromCache(N, -1); err != nil {
			b.Fatal(err)
		}
		b.StopTimer()
		_ = sess.Close()
		b.StartTimer()
	}
}

// 16-layer e2b-shaped PLE decode: host embed/PLE chained (2 buffers/token) vs chained-GPU (1).
func benchChainedDecodePLEHost(b *testing.B) { benchChainedDecodePLE(b, false) }
func benchChainedDecodePLEGpu(b *testing.B)  { benchChainedDecodePLE(b, true) }

func BenchmarkChainedDecodePLEHost(b *testing.B) { benchChainedDecodePLEHost(b) }
func BenchmarkChainedDecodePLEGpu(b *testing.B)  { benchChainedDecodePLEGpu(b) }
