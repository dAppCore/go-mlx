// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package gemma4

import (
	"context"
	"math"
	"os"
	"testing"

	"dappco.re/go/mlx/internal/metaltest"
	"dappco.re/go/mlx/pkg/metal"
	"dappco.re/go/mlx/pkg/native"
	"dappco.re/go/mlx/pkg/safetensors"
	"dappco.re/go/mlx/pkg/tokenizer"
)

// TestAssistantDraftParityNativeVsMetal is the cross-engine MTP parity INSTRUMENT:
// the same bf16 target + bf16 drafter, the same prompt ids, prefetched through BOTH
// engines, then one draft block from the identical boundary state. metal is the
// measured-healthy reference (62-83% draft acceptance on real pairs); native drafts
// at 0% on the same checkpoints, so this test localises WHERE native diverges:
//
//	stage 1 — the SEED: the retained post-final-norm boundary hidden each engine
//	          feeds its first draft step. Large divergence here = the prefill/
//	          boundary-retention path is the bug (the draft stack never had a chance).
//	stage 2 — the DRAFT BLOCK: the k greedy tokens each drafter proposes from that
//	          seed. Divergence here with an agreeing seed = the drafter forward /
//	          target-KV plumbing is the bug.
//
// It lives in the metal package (needs metal's unexported prompt preparation) and
// drives native through its exported parity seams (PrepareAssistantPrompt,
// BoundaryNormedHidden, DraftBlockFromSession). Native is CGO-free, so the import
// adds no build burden; native additionally needs MLX_METALLIB_PATH, so the test
// skips without it. Until native's defect is fixed this test FAILS on stage 2 by
// design — it is the failing-test-first reproducer, and the regression gate after.
func TestAssistantDraftParityNativeVsMetal(t *testing.T) {
	if !metaltest.RunMetalTests {
		t.Skip("build with -tags metal_runtime to run the cross-engine parity instrument")
	}
	if !metal.MetalAvailable() {
		t.Skip("Metal runtime unavailable")
	}
	if os.Getenv(native.MetallibPathEnv) == "" {
		t.Skipf("set %s for the native engine side", native.MetallibPathEnv)
	}
	targetPath := metaltest.HFModelPath(t, "mlx-community/gemma-4-E2B-it-bf16")
	assistantPath := metaltest.HFModelPath(t, "mlx-community/gemma-4-E2B-it-assistant-bf16")

	const draftTokens = 4
	tok, err := tokenizer.LoadTokenizer(targetPath + "/tokenizer.json")
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}
	ids := tok.Encode(gemma4Turn("Name the planets of the solar system in order."))
	if len(ids) < 4 {
		t.Fatalf("prompt tokenised to %d ids, want a real prompt", len(ids))
	}
	lastToken := ids[len(ids)-1]

	// ---- metal side: prefill + boundary hidden + one draft block ----
	m, err := metal.LoadAndInit(targetPath, metal.LoadConfig{})
	if err != nil {
		t.Fatalf("metal.LoadAndInit: %v", err)
	}
	defer m.Close()
	pair, err := AttachGemma4Assistant(m, assistantPath)
	if err != nil {
		t.Fatalf("AttachGemma4Assistant: %v", err)
	}
	defer pair.Close()

	var metalHidden []float32
	var metalTokens []int32
	if deviceErr := m.WithDevice(func() {
		prepared, perr := prepareGemma4AssistantPrompt(context.Background(), m, pair, ids, metal.GenerateConfig{MaxTokens: 16})
		if perr != nil {
			err = perr
			return
		}
		defer func() { metal.FreeCaches(prepared.Caches) }()
		defer metal.Free(prepared.Logits, prepared.Hidden)

		if eerr := metal.Eval(prepared.Hidden); eerr != nil {
			err = eerr
			return
		}
		metalHidden = append([]float32(nil), prepared.Hidden.Floats()...)

		block, berr := pair.DraftBlock(lastToken, prepared.Hidden, prepared.Caches, draftTokens)
		if berr != nil {
			err = berr
			return
		}
		defer block.Close()
		metalTokens = append([]int32(nil), block.Tokens...)
	}); deviceErr != nil {
		t.Fatalf("metal WithDevice: %v", deviceErr)
	}
	if err != nil {
		t.Fatalf("metal side: %v", err)
	}

	// ---- native side: same ids through the CGO-free engine ----
	// maxLen must EXCEED the E2B sliding window (512): native's paged sliding caches only
	// allocate ring pages when slidingWindow < maxLen (decode_forward_arch.go initDevicePagedKV),
	// and its bf16 prefill hard-errors on a ringless sliding cache.
	sess, err := native.LoadDir(targetPath, 640)
	if err != nil {
		t.Fatalf("native.LoadDir: %v", err)
	}
	defer sess.Close()
	npair, err := native.LoadAssistantPairDirs(targetPath, assistantPath)
	if err != nil {
		t.Fatalf("native.LoadAssistantPairDirs: %v", err)
	}
	defer npair.Close()

	if err := sess.PrepareAssistantPrompt(ids); err != nil {
		t.Fatalf("native PrepareAssistantPrompt: %v", err)
	}
	nativeHiddenBF16, err := sess.BoundaryNormedHidden()
	if err != nil {
		t.Fatalf("native BoundaryNormedHidden: %v", err)
	}
	nativeHidden := bf16BytesToFloat32(nativeHiddenBF16)

	nblock, err := npair.DraftBlockFromSession(sess, lastToken, draftTokens)
	if err != nil {
		t.Fatalf("native DraftBlockFromSession: %v", err)
	}
	nativeTokens := append([]int32(nil), nblock.Tokens...)

	// ---- stage 0: which CONVENTION does each seed carry? ----
	// rms magnitude fingerprints pre-norm residual (large, outlier dims O(100)) vs
	// post-final-norm (O(1-10)); norming metal's vector with the checkpoint's own
	// norm weights and diffing against native's tells us whether the two engines sit
	// exactly one final-norm apart.
	dm, err := safetensors.LoadDirMmap(targetPath)
	if err != nil {
		t.Fatalf("stage 0: load target tensors: %v", err)
	}
	defer dm.Close()
	var normW []float32
	for name, tensor := range dm.Tensors {
		// the multimodal wrapper prefixes tensor names (language_model.…); match the
		// final-norm suffix instead of an exact name.
		if len(name) >= len("model.norm.weight") && name[len(name)-len("model.norm.weight"):] == "model.norm.weight" {
			normW = bf16BytesToFloat32(tensor.Data)
			t.Logf("stage 0 final norm tensor: %s (len %d)", name, len(normW))
			break
		}
	}
	if len(normW) != len(metalHidden) {
		t.Fatalf("stage 0: no final norm tensor matching hidden size %d found", len(metalHidden))
	}
	t.Logf("stage 0 rms: metal=%.4f native=%.4f normW[660]=%.4f", rmsOf(metalHidden), rmsOf(nativeHidden), normW[660])
	normedMetal := hostRMSNorm(metalHidden, normW)
	var mnMax float64
	for i := range normedMetal {
		if d := math.Abs(float64(normedMetal[i]) - float64(nativeHidden[i])); d > mnMax {
			mnMax = d
		}
	}
	t.Logf("stage 0 maxAbs(RMSNorm(metalHidden) - nativeHidden) = %.6f  → ~0 means metal feeds the PRE-norm residual and native feeds POST-norm (one norm apart)", mnMax)

	// ---- stage 1: the seed hidden ----
	if len(metalHidden) != len(nativeHidden) {
		t.Fatalf("stage 1: boundary hidden lengths differ: metal %d vs native %d", len(metalHidden), len(nativeHidden))
	}
	var maxAbs, sumAbs float64
	maxIdx := -1
	for i := range metalHidden {
		d := math.Abs(float64(metalHidden[i]) - float64(nativeHidden[i]))
		sumAbs += d
		if d > maxAbs {
			maxAbs, maxIdx = d, i
		}
	}
	meanAbs := sumAbs / float64(len(metalHidden))
	t.Logf("stage 1 SEED HIDDEN: len=%d maxAbs=%.6f (dim %d: metal=%.6f native=%.6f) meanAbs=%.6f",
		len(metalHidden), maxAbs, maxIdx, metalHidden[maxIdx], nativeHidden[maxIdx], meanAbs)

	// bf16 across two independent kernel stacks: small drift is expected, gross
	// divergence means the boundary/prefill path itself is the defect.
	const seedTolerance = 0.25
	seedAgrees := maxAbs <= seedTolerance
	if !seedAgrees {
		t.Errorf("stage 1 FAIL: boundary hiddens diverge (maxAbs %.4f > %.2f) — the native prefill/boundary retention is the defect; the draft stack never had a chance", maxAbs, seedTolerance)
	}

	// ---- stage 2: the draft block ----
	t.Logf("stage 2 DRAFT BLOCK: metal=%v native=%v", metalTokens, metalTokensOrNative(nativeTokens))
	t.Logf("stage 2 decoded: metal=%q native=%q", tok.Decode(metalTokens), tok.Decode(nativeTokens))
	mismatch := 0
	for i := 0; i < len(metalTokens) && i < len(nativeTokens); i++ {
		if metalTokens[i] != nativeTokens[i] {
			mismatch++
		}
	}
	if len(metalTokens) != len(nativeTokens) || mismatch > 0 {
		if seedAgrees {
			t.Errorf("stage 2 FAIL: seeds agree but draft tokens diverge (%d/%d mismatched) — the defect is INSIDE native's draft forward or its target-KV plumbing", mismatch, len(metalTokens))
		} else {
			t.Errorf("stage 2: draft tokens diverge (%d/%d) — expected, downstream of the stage 1 seed divergence", mismatch, len(metalTokens))
		}
	}

	// ---- stage 3: the VERIFY — does native's target accept its own healthy drafts? ----
	// The drafts above are the drafter's greedy proposals from a metal-agreeing state;
	// under greedy verification the target's argmax should accept most of them (metal
	// accepts 62-83% live). A near-zero acceptance HERE isolates the defect to native's
	// verify row/position mapping rather than anything draft-side.
	vr, err := npair.VerifyDraftBlockFromSession(sess, nativeTokens)
	if err != nil {
		t.Fatalf("stage 3: native VerifyDraftBlockFromSession: %v", err)
	}
	t.Logf("stage 3 VERIFY: drafted=%v targetSays=%v accepted=%d rejected=%d replacement=%d allAccepted=%v",
		vr.DraftedTokens, vr.TargetTokens, vr.AcceptedCount, vr.RejectedCount, vr.ReplacementToken, vr.AllAccepted)
	t.Logf("stage 3 decoded: drafted=%q targetSays=%q", tok.Decode(nativeTokens), tok.Decode(vr.TargetTokens))
	if vr.AcceptedCount == 0 && seedAgrees && mismatch == 0 {
		t.Errorf("stage 3 FAIL: the target rejects ALL healthy drafts — native's verify row/position mapping is the defect")
	}
}

// metalTokensOrNative exists only to keep the log line symmetrical when native
// returned fewer tokens than requested.
func metalTokensOrNative(tokens []int32) []int32 { return tokens }

// bf16BytesToFloat32 widens a little-endian bf16 byte slab to float32 for diffing.
func bf16BytesToFloat32(b []byte) []float32 {
	out := make([]float32, len(b)/2)
	for i := range out {
		bits := uint32(b[2*i]) | uint32(b[2*i+1])<<8
		out[i] = math.Float32frombits(bits << 16)
	}
	return out
}

// rmsOf is the root-mean-square magnitude fingerprint of a hidden vector.
func rmsOf(x []float32) float64 {
	var sum float64
	for _, v := range x {
		sum += float64(v) * float64(v)
	}
	return math.Sqrt(sum / float64(len(x)))
}

// hostRMSNorm applies plain RMSNorm (x/rms · w) with the checkpoint's stored weights —
// the mlx-community gemma4 exports ship the (1+w) fold already baked, so plain multiply
// is the checkpoint-faithful application.
func hostRMSNorm(x []float32, w []float32) []float32 {
	const eps = 1e-6
	var sum float64
	for _, v := range x {
		sum += float64(v) * float64(v)
	}
	inv := 1.0 / math.Sqrt(sum/float64(len(x))+eps)
	out := make([]float32, len(x))
	for i := range x {
		out[i] = float32(float64(x[i]) * inv * float64(w[i]))
	}
	return out
}
