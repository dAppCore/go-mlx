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

	run := func(name string, host, pipe bool) ([]int32, float64) {
		chainedGPUInputsDisabled = host
		pipelinedGPUDecodeEnabled = pipe
		sess := newSess()
		if err := sess.PrefillTokens(prompt); err != nil {
			t.Fatalf("%s prefill: %v", name, err)
		}
		if _, err := sess.GenerateFromCache(warmup, -1); err != nil {
			t.Fatalf("%s warmup: %v", name, err)
		}
		t0 := time.Now()
		timed, err := sess.GenerateFromCache(N, -1)
		wall := time.Since(t0)
		if err != nil {
			t.Fatalf("%s generate: %v", name, err)
		}
		return timed, float64(N) / wall.Seconds()
	}

	hostTok, hostTps := run("host", true, false)
	chainTok, chainTps := run("chained-GPU", false, false)
	pipeTok, pipeTps := run("pipelined", false, true)
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
	t.Logf("real e2b-4bit decode tok/s (tg%d): host %.1f  chained-GPU %.1f (%.2fx)  pipelined %.1f (%.2fx) — tokens identical",
		N, hostTps, chainTps, chainTps/hostTps, pipeTps, pipeTps/hostTps)
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
