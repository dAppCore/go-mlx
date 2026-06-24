// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"os"
	"testing"
	"time"

	"dappco.re/go/mlx/pkg/model"
	"github.com/tmc/apple/foundation"
	"github.com/tmc/apple/metal"
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

	// FFN-only barrier ceiling: drop just the gate/gelu/down barriers (racy, timing-only) — how much GPU
	// a fused FFN megakernel could reclaim. The delta vs the full-barriered pipeline scopes piece-(A).
	ffnBarriersOffForTest = true
	pipelinedGPUDecodeEnabled = true
	defer func() { ffnBarriersOffForTest = false; pipelinedGPUDecodeEnabled = false }()
	sffn := newSess()
	if err := sffn.PrefillTokens(prompt); err != nil {
		t.Fatalf("ffn-probe prefill: %v", err)
	}
	if _, err := sffn.GenerateFromCache(warmup, -1); err != nil {
		t.Fatalf("ffn-probe warmup: %v", err)
	}
	pieceTimingOn = true
	chainedGPUSpanNs = 0
	tffn := time.Now()
	if _, err := sffn.GenerateFromCache(N, -1); err != nil {
		pieceTimingOn = false
		t.Fatalf("ffn-probe generate: %v", err)
	}
	wallFfn := time.Since(tffn)
	pieceTimingOn = false
	ffnBarriersOffForTest = false
	pipelinedGPUDecodeEnabled = false
	ffnGpuPerTok := float64(chainedGPUSpanNs) / 1e6 / float64(N)
	fullPipeGpuPerTok := (pipeGPU / 100.0) * 1000.0 / pipeTps // full-barriered pipelined GPU ms/token
	t.Logf("FFN-fusion ceiling: drop gate/gelu/down barriers -> per-token GPU %.3fms (%.1f tok/s) vs full %.3fms — fused-FFN reclaim %.3fms/token (~%.0f tok/s if realised)",
		ffnGpuPerTok, float64(N)/wallFfn.Seconds(), fullPipeGpuPerTok, fullPipeGpuPerTok-ffnGpuPerTok,
		1000.0/((1000.0/pipeTps)-(fullPipeGpuPerTok-ffnGpuPerTok)))

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

// TestRealE2BWithinLayerOpCost breaks the per-token GPU cost down to the INDIVIDUAL ICB op: each decode op
// is executed as its own command buffer and timed by GPUEndTime-GPUStartTime. A kernel's GPU span is
// value-independent (it depends on dispatch sizes, not the stale buffer contents the isolated op happens to
// read), so the timing is correct even though the op runs over whatever the warmup left behind. Two outputs,
// both NON-racy: (1) Σ per-op span = the true serial compute floor — barriered(5.757ms) − Σ is the ACTUAL
// reclaimable barrier-drain cost (the racy no-barrier 1.834ms over-counts because it also overlaps deps);
// (2) the per-op histogram shows WHERE the cost concentrates. This is the discriminator the advisor flagged:
// if it lives in a few fat gemvs (q/o/gate/up/down) the cost is near a compute floor and projection-fusion
// is low-value; if it spreads across many skinny dispatches, dispatch-count reduction pays.
func TestRealE2BWithinLayerOpCost(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	if os.Getenv("LEM_REAL_E2B") == "" {
		t.Skip("set LEM_REAL_E2B=1 to run the real e2b-4bit op-cost breakdown (loads ~2.7GB)")
	}
	dir := resolveE2B4bitDir(t)
	const maxLen, warmup = 320, 8
	lm, dm, err := loadRegistered(dir)
	if err != nil {
		t.Fatalf("loadRegistered: %v", err)
	}
	defer func() { _ = dm.Close() }()
	sb, err := buildShardBuffers(dm)
	if err != nil {
		t.Fatalf("buildShardBuffers: %v", err)
	}
	defer func() { _ = sb.Close() }()
	qm, err := loadedToQuant(lm, lm.Embed.GroupSize, lm.Embed.Bits)
	if err != nil {
		t.Fatalf("loadedToQuant: %v", err)
	}
	sess, err := newArchQuantSessionShards(qm, lm.Arch, maxLen, sb)
	if err != nil {
		t.Fatalf("newArchQuantSessionShards: %v", err)
	}
	prompt := []int32{2, 1000, 2500, 4000, 8000, 16000}
	if err := sess.PrefillTokens(prompt); err != nil {
		t.Fatalf("prefill: %v", err)
	}
	if _, err := sess.GenerateFromCache(warmup, -1); err != nil {
		t.Fatalf("warmup: %v", err)
	}
	r := sess.state.icb
	if r == nil {
		t.Fatal("no recorded ICB replay (encode-bypass inactive)")
	}

	// time ONE ICB op (range [op,op+1)) as its own command buffer; min over iters = the cleanest GPU span.
	timeOp := func(op uint, iters int) float64 {
		minNs := math.MaxFloat64
		for i := 0; i < iters; i++ {
			var ns float64
			withAutoreleasePool(func() {
				cb := queue.CommandBuffer()
				enc := cb.ComputeCommandEncoder()
				enc.UseResourcesCountUsage(r.residentRes, uint(len(r.residentRes)), metal.MTLResourceUsageRead|metal.MTLResourceUsageWrite)
				enc.ExecuteCommandsInBufferWithRange(r.icb, foundation.NSRange{Location: op, Length: 1})
				enc.EndEncoding()
				cb.Commit()
				cb.WaitUntilCompleted()
				ns = float64(cb.GPUEndTime()-cb.GPUStartTime()) * 1e9
			})
			if ns < minNs {
				minNs = ns
			}
		}
		return minNs / 1e3 // µs
	}

	// (1) whole-stack compute floor: Σ per-op min span over every layer op.
	total := r.opsPerLayer * uint(r.nLayers)
	var sumUs float64
	for op := r.rng.Location; op < r.rng.Location+total; op++ {
		sumUs += timeOp(op, 20)
	}
	t.Logf("Σ per-op GPU span over %d layer ops = %.3f ms (TRUE serial compute, no host sync, no overlap) "+
		"— barriered layer-stack 5.757ms ⇒ reclaimable barrier-drain ≈ %.3f ms; racy no-barrier 1.834ms over-counts (it overlaps deps)",
		total, sumUs/1e3, 5.757-sumUs/1e3)

	// (2) per-op histogram for the first owns-cache GLOBAL layer — annotate the structural ops; the fat
	// gemvs (q/o/gate/up/down) stand out by magnitude.
	li := 0
	for ; li < r.nLayers; li++ {
		if r.specs[li].OwnsCache() && r.specs[li].Attention != model.SlidingAttention {
			break
		}
	}
	if li == r.nLayers {
		li = 0 // fall back to layer 0 if no global owns-cache layer
	}
	base := r.rng.Location + uint(li)*r.opsPerLayer
	t.Logf("--- per-op GPU µs, global owns-cache layer %d (ops %d..%d) ---", li, base, base+r.opsPerLayer-1)
	var layerSum float64
	for k := uint(0); k < r.opsPerLayer; k++ {
		op := base + k
		label := ""
		switch int(op) {
		case r.sdpaIdx[li]:
			label = " <- SDPA"
		case r.kRopeIdx[li]:
			label = " <- K rope->cache"
		case r.vIdx[li]:
			label = " <- V proj->cache"
		case r.vNormIdx[li]:
			label = " <- V norm"
		}
		us := timeOp(op, 50)
		layerSum += us
		t.Logf("  op %2d (idx %3d): %7.2f µs%s", k, op, us, label)
	}
	t.Logf("layer %d Σ = %.2f µs (× %d layers ≈ %.3f ms)", li, layerSum, r.nLayers, layerSum*float64(r.nLayers)/1e3)
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
