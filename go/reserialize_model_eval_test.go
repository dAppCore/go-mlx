// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"bytes"
	"context"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	infspine "dappco.re/go/inference/spine"
	session "dappco.re/go/inference/state/session"
	"dappco.re/go/mlx/internal/metaltest"
	"dappco.re/go/mlx/pkg/metal"
)

// TestSessionReserialize_CaptureRestoreCapture_Eval is the HOT LEAD instrument
// for this lane: blob -> restore -> reserialize must equal the original blob.
//
// The native engine (pkg/native) shipped exactly this CLASS of bug, fixed in
// commit 442ac99c: RestoreState wrote the session's dormant (paged) store
// while SerializeState read the live (ICB) store, so restore silently
// zeroed live history and a save -> restore -> save cycle exported an empty
// conversation with no error anywhere in the chain.
//
// Neither of pkg/metal's own existing "round trip" tests would catch the
// metal-side equivalent:
//   - TestSessionSnapshot_CaptureRestoreRoundTrip_Good
//     (pkg/metal/session_snapshot_drive_cover_test.go) restores into a fresh
//     session and asserts only err == nil — it never re-captures to compare
//     against the snapshot it restored.
//   - TestSession_SnapshotRestoreFork_Eval
//     (pkg/metal/session_snapshot_model_eval_test.go) restores into a fresh
//     session and resumes generation, but likewise never re-captures to
//     compare against the original snapshot before perturbing state.
//
// This test closes that gap from the session package's public surface
// (Session.CaptureKV / Session.RestoreKV), against the real metal engine (no
// fakes — internal/sessionfake.Handle decouples CaptureKV from RestoreKV by
// construction, so a fake-backed test is structurally incapable of catching
// this class of bug). It lived beside the session package before the lift to
// dappco.re/go/inference/state/session; it re-homes to the go-mlx root because
// it is a METAL-engine integration test — the wrapped handle now crosses
// metalSessionAdapter, so the kvconv snapshot bridge is inside the loop being
// proven. Two session shapes reachable through the public LoadConfig surface
// are exercised: plain and paged KV cache. Pipelined (FixedKVCache
// pending-commit) and compiled/ICB-replay shapes are internal pkg/metal
// mechanisms gated behind experimental flags with no LoadConfig surface —
// not reachable from this lane; see the audit report.
//
//	GO_MLX_BENCH_MODEL=google/gemma-4-e2b-it go test \
//	  -tags 'metal_runtime model_eval' -run TestSessionReserialize -v ./go/
func TestSessionReserialize_CaptureRestoreCapture_Eval(t *testing.T) {
	if !metaltest.RunModelEvalTests {
		t.Skip("model-eval test; build with -tags model_eval")
	}
	repo := core.Getenv("GO_MLX_BENCH_MODEL")
	if repo == "" {
		repo = "google/gemma-4-e2b-it"
	}
	dir := metaltest.HFModelPath(t, repo)

	cases := []struct {
		name string
		cfg  metal.LoadConfig
	}{
		{name: "plain", cfg: metal.LoadConfig{ContextLen: 512}},
		{name: "paged", cfg: metal.LoadConfig{ContextLen: 512, KVCacheMode: string(metal.KVCacheModePaged)}},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			model, err := metal.LoadAndInit(dir, tc.cfg)
			if err != nil {
				t.Fatalf("LoadAndInit(%s): %v", tc.name, err)
			}
			defer model.Close()
			ctx := context.Background()

			// Build a small but non-trivial live conversation: a token
			// prefill followed by a few generated tokens, so the snapshot
			// carries both prefilled and generated history — the exact
			// shape the native bug's "continuations decoded against empty
			// history" symptom depended on.
			source := session.New(newMetalSessionAdapter(model.NewSession()), infspine.ModelInfo{}, nil)
			defer source.Close()
			if err := source.PrefillTokens(ctx, []int32{2, 100, 200, 300, 400}); err != nil {
				t.Fatalf("[%s] PrefillTokens: %v", tc.name, err)
			}
			if _, err := source.Generate(inference.WithMaxTokens(4)); err != nil {
				t.Fatalf("[%s] Generate: %v", tc.name, err)
			}

			snap1, err := source.CaptureKV()
			if err != nil {
				t.Fatalf("[%s] CaptureKV (pre-restore): %v", tc.name, err)
			}
			if snap1 == nil || len(snap1.Layers) == 0 || len(snap1.Tokens) == 0 {
				t.Fatalf("[%s] pre-restore snapshot carried no state: %+v", tc.name, snap1)
			}
			if len(snap1.Generated) == 0 {
				t.Fatalf("[%s] pre-restore snapshot carried no generated tokens — Generate did not advance state", tc.name)
			}
			blobA, err := snap1.MarshalBinary()
			if err != nil {
				t.Fatalf("[%s] MarshalBinary (pre-restore): %v", tc.name, err)
			}

			// Restore into a FRESH session (not the same one) — the fresh
			// session starts with zero state, so the second capture can only
			// carry the restored content, not leftover state from source.
			target := session.New(newMetalSessionAdapter(model.NewSession()), infspine.ModelInfo{}, nil)
			defer target.Close()
			if err := target.RestoreKV(snap1); err != nil {
				t.Fatalf("[%s] RestoreKV: %v", tc.name, err)
			}

			snap2, err := target.CaptureKV()
			if err != nil {
				t.Fatalf("[%s] CaptureKV (post-restore): %v", tc.name, err)
			}
			blobB, err := snap2.MarshalBinary()
			if err != nil {
				t.Fatalf("[%s] MarshalBinary (post-restore): %v", tc.name, err)
			}

			if !bytes.Equal(blobA, blobB) {
				firstDiff := -1
				n := min(len(blobA), len(blobB))
				for i := 0; i < n; i++ {
					if blobA[i] != blobB[i] {
						firstDiff = i
						break
					}
				}
				t.Fatalf("[%s] reserialize mismatch: capture -> restore -> capture changed the blob "+
					"(len %d vs %d, first differing byte at offset %d) — RestoreKV is not writing the "+
					"same store CaptureKV reads; this is the native-bug CLASS of asymmetry in the metal engine",
					tc.name, len(blobA), len(blobB), firstDiff)
			}
		})
	}
}
