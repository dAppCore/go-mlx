// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"os"
	"testing"
	"time"
)

// TestRealE2BChainedGPUParityAndSpeed validates the chained-GPU + submit-ahead decode on the ACTUAL
// gemma-4 e2b-4bit checkpoint (not a synthetic fixture): the GPU next-inputs seam must wire, and the
// host / chained-GPU / pipelined paths must produce token-IDENTICAL output on real weights, while the
// GPU paths report their real decode tok/s. This is the real-model gate the synthetic suite can't give —
// the thing that says "the wins translate to the served model". Gated behind LEM_REAL_E2B (loads ~2.7GB);
// loads the weights ONCE and builds three sessions sharing them (independent KV caches).
func TestRealE2BChainedGPUParityAndSpeed(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	if os.Getenv("LEM_REAL_E2B") == "" {
		t.Skip("set LEM_REAL_E2B=1 to run the real e2b-4bit validation (loads ~2.7GB)")
	}
	dir := resolveE2B4bitDir(t)
	const maxLen, warmup, N = 320, 8, 64

	// Load the checkpoint ONCE; build fresh sessions sharing the weight shards.
	lm, dm, err := loadRegistered(dir)
	if err != nil {
		t.Fatalf("loadRegistered: %v", err)
	}
	defer func() { _ = dm.Close() }()
	if !quantised(lm) {
		t.Fatalf("expected a quantised e2b checkpoint")
	}
	sb, err := buildShardBuffers(dm)
	if err != nil {
		t.Fatalf("buildShardBuffers: %v", err)
	}
	defer func() { _ = sb.Close() }()
	qm, err := loadedToQuant(lm, lm.Embed.GroupSize, lm.Embed.Bits)
	if err != nil {
		t.Fatalf("loadedToQuant: %v", err)
	}
	newSess := func() *ArchSession {
		s, serr := newArchQuantSessionShards(qm, lm.Arch, maxLen, sb)
		if serr != nil {
			t.Fatalf("newArchQuantSessionShards: %v", serr)
		}
		return s
	}

	probe := newSess()
	if probe.encNextInputsGPU == nil {
		t.Fatal("real e2b-4bit: GPU next-inputs seam NOT wired (chained-GPU path inactive)")
	}
	if probe.recordPeerICB == nil {
		t.Fatal("real e2b-4bit: peer-ICB recorder NOT set (pipeline path inactive)")
	}
	prompt := []int32{2, 1000, 2500, 4000, 8000, 16000}

	run := func(name string, host, pipe bool) ([]int32, float64, float64) {
		chainedGPUInputsDisabled = host
		pipelinedGPUDecodeEnabled = pipe
		sess := newSess()
		if err := sess.PrefillTokens(prompt); err != nil {
			t.Fatalf("%s prefill: %v", name, err)
		}
		if _, err := sess.GenerateFromCache(warmup, -1); err != nil {
			t.Fatalf("%s warmup: %v", name, err)
		}
		pieceTimingOn = true
		chainedGPUSpanNs = 0
		t0 := time.Now()
		timed, err := sess.GenerateFromCache(N, -1)
		wall := time.Since(t0)
		pieceTimingOn = false
		if err != nil {
			t.Fatalf("%s generate: %v", name, err)
		}
		gpuFrac := float64(chainedGPUSpanNs) / float64(wall.Nanoseconds()) * 100
		return timed, float64(N) / wall.Seconds(), gpuFrac
	}

	hostTok, hostTps, _ := run("host", true, false)
	chainTok, chainTps, chainGPU := run("chained-GPU", false, false)
	pipeTok, pipeTps, pipeGPU := run("pipelined", false, true)
	chainedGPUInputsDisabled = false
	pipelinedGPUDecodeEnabled = false

	eq := func(a, b []int32) bool {
		if len(a) != len(b) {
			return false
		}
		for i := range a {
			if a[i] != b[i] {
				return false
			}
		}
		return true
	}
	if !eq(chainTok, hostTok) {
		t.Fatalf("chained-GPU tokens diverge from host on real e2b:\n host=%v\n gpu =%v", hostTok, chainTok)
	}
	if !eq(pipeTok, hostTok) {
		t.Fatalf("pipelined tokens diverge from host on real e2b:\n host=%v\n pipe=%v", hostTok, pipeTok)
	}
	t.Logf("real e2b-4bit decode tok/s (tg%d): host %.1f  chained-GPU %.1f (%.2fx, gpu-busy %.0f%%)  pipelined %.1f (%.2fx, gpu-busy %.0f%%) — tokens identical",
		N, hostTps, chainTps, chainTps/hostTps, chainGPU, pipeTps, pipeTps/hostTps, pipeGPU)

	// Per-piece GPU split: force the serial path (separate PLE / layer-stack / head command buffers, each
	// its own commit+wait so its wall ≈ its GPU time) and attribute per-token GPU time. Locates which
	// kernel dominates — the lever to chase to beat the cgo engine.
	stepGreedyChainDisabled = true
	defer func() { stepGreedyChainDisabled = false }()
	sb2 := newSess()
	if err := sb2.PrefillTokens(prompt); err != nil {
		t.Fatalf("breakdown prefill: %v", err)
	}
	if _, err := sb2.GenerateFromCache(warmup, -1); err != nil {
		t.Fatalf("breakdown warmup: %v", err)
	}
	pieceTimingOn = true
	pieceNs = [3]int64{}
	if _, err := sb2.GenerateFromCache(N, -1); err != nil {
		pieceTimingOn = false
		t.Fatalf("breakdown generate: %v", err)
	}
	pieceTimingOn = false
	stepGreedyChainDisabled = false
	per := func(ns int64) float64 { return float64(ns) / 1e6 / float64(N) }
	t.Logf("per-token GPU split (serial, ms): PLE %.3f  layer-stack %.3f  head %.3f  (sum %.3f)",
		per(pieceNs[0]), per(pieceNs[1]), per(pieceNs[2]), per(pieceNs[0]+pieceNs[1]+pieceNs[2]))

	// Barrier-cost ceiling (TIMING-ONLY; output races): record the ICB with NO barriers and measure the
	// pipelined per-token GPU span. The gap to the barriered span is what the coarse SetBarriers cost —
	// the headroom a finer recorded-barrier schedule could reclaim in the layer stack.
	allBarriersOffForTest = true
	pipelinedGPUDecodeEnabled = true
	defer func() { allBarriersOffForTest = false; pipelinedGPUDecodeEnabled = false }()
	sbar := newSess()
	if err := sbar.PrefillTokens(prompt); err != nil {
		t.Fatalf("nobarrier prefill: %v", err)
	}
	if _, err := sbar.GenerateFromCache(warmup, -1); err != nil {
		t.Fatalf("nobarrier warmup: %v", err)
	}
	pieceTimingOn = true
	chainedGPUSpanNs = 0
	tnb := time.Now()
	if _, err := sbar.GenerateFromCache(N, -1); err != nil {
		pieceTimingOn = false
		t.Fatalf("nobarrier generate: %v", err)
	}
	wallNb := time.Since(tnb)
	pieceTimingOn = false
	allBarriersOffForTest = false
	pipelinedGPUDecodeEnabled = false
	nbGpuPerTok := float64(chainedGPUSpanNs) / 1e6 / float64(N)
	barGpuPerTok := per(pieceNs[1]) // barriered layer-stack per token (reference)
	t.Logf("barrier ceiling: pipelined no-barrier per-token GPU %.3fms (wall %.1f tok/s) vs barriered layer-stack %.3fms — barrier cost headroom",
		nbGpuPerTok, float64(N)/wallNb.Seconds(), barGpuPerTok)

	// Fine-grained replay: barrier-FREE ICB + a resource-scoped encoder memory barrier at each true dep
	// (instead of the coarse all-prior SetBarrier full drain). Should pipeline the tiny decode kernels and
	// reclaim the barrier headroom while staying token-correct. Measure GPU span + tok/s + parity vs host.
	fineGrainedReplay = true
	pipelinedGPUDecodeEnabled = true
	defer func() { fineGrainedReplay = false; pipelinedGPUDecodeEnabled = false }()
	sfg := newSess()
	if err := sfg.PrefillTokens(prompt); err != nil {
		t.Fatalf("fine-grained prefill: %v", err)
	}
	if _, err := sfg.GenerateFromCache(warmup, -1); err != nil {
		t.Fatalf("fine-grained warmup: %v", err)
	}
	pieceTimingOn = true
	chainedGPUSpanNs = 0
	tfg := time.Now()
	fgTok, err := sfg.GenerateFromCache(N, -1)
	wallFg := time.Since(tfg)
	pieceTimingOn = false
	fineGrainedReplay = false
	pipelinedGPUDecodeEnabled = false
	if err != nil {
		t.Fatalf("fine-grained generate: %v", err)
	}
	fgGpuPerTok := float64(chainedGPUSpanNs) / 1e6 / float64(N)
	t.Logf("fine-grained pipelined: %.1f tok/s  %.3fms/token GPU  tokens-match-host=%v",
		float64(N)/wallFg.Seconds(), fgGpuPerTok, eq(fgTok, hostTok))
}

func resolveE2B4bitDir(t *testing.T) string {
	home := os.Getenv("HOME")
	base := home + "/.cache/huggingface/hub/models--mlx-community--gemma-4-e2b-it-4bit/snapshots"
	entries, err := os.ReadDir(base)
	if err != nil {
		t.Skipf("e2b-4bit snapshot dir not found (%v)", err)
	}
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		dir := base + "/" + e.Name()
		if _, serr := os.Stat(dir + "/config.json"); serr == nil {
			return dir
		}
	}
	t.Skip("no e2b-4bit snapshot with config.json")
	return ""
}
