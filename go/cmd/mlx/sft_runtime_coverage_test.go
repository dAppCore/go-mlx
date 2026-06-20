// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package main

import (
	"context"
	"strings"
	"testing"

	core "dappco.re/go"
)

// sft over a TINY synthetic loads the model for real and advances through the
// SFT config build into m.TrainSFT — the load-success + config + train-call
// path. The degenerate 1-layer synthetic + tiny vocab produces no trainable
// batches, so training itself fails (exit 1); a real end-to-end SFT is an
// AX-11 multi-GB run this suite avoids. -metrics-lp/-capture default off so the
// run stays self-contained.
func TestRunSFT_SyntheticLoadAndTrainAttempt_Bad(t *testing.T) {
	requireSyntheticRuntime(t)
	dir := t.TempDir()
	model := writeSyntheticGemma3Model(t, dir)
	data := core.JoinPath(dir, "train.jsonl")
	if r := core.WriteFile(data, []byte(
		`{"messages":[{"role":"user","content":"hello"},{"role":"assistant","content":"world"}]}`+"\n"+
			`{"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"there"}]}`+"\n"), 0o644); !r.OK {
		t.Fatalf("write data: %v", r.Value)
	}
	ckpt := core.JoinPath(dir, "ckpt")
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{
		"sft", "--model", model, "--data", data,
		"--epochs", "1", "--batch", "1", "--grad-accum", "1",
		"--eval-every", "0", "--metrics-lp", "off", "--capture", "off",
		"--checkpoint-every", "0", "--rank", "2", "--max-seq", "32",
		"--context", "64",
		"--save", core.JoinPath(ckpt, "adapter.safetensors"),
	}, stdout, stderr)
	if code != 1 {
		t.Fatalf("exit = %d, want 1 (synthetic produces no trainable batches); stderr=%q", code, stderr.String())
	}
	// Confirms the runner reached the real load (the "loading" notice) and the
	// training call (the "training:" error), i.e. the whole config-build tail ran.
	if !strings.Contains(stderr.String(), "loading") || !strings.Contains(stderr.String(), "training:") {
		t.Fatalf("stderr = %q, want both the load notice and the training-call error", stderr.String())
	}
}
