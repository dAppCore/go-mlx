// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package main

import (
	"context"
	"strings"
	"testing"

	core "dappco.re/go"
)

// sft over a TINY synthetic with a --checkpoint-dir but no -metrics-lp off
// constructs the metrics line-protocol sink + the default save path before
// reaching TrainSFT (which fails on the degenerate synthetic). Covers the
// metrics-sink + save-path-default + lp-default branches the off-path test
// skips. The CLI's plain dataset.Config produces no trainable batches, so the
// terminal state is the training error (exit 1) — a real end-to-end SFT is an
// AX-11 multi-GB run this suite avoids.
func TestRunSFT_SyntheticMetricsSink_Bad(t *testing.T) {
	requireSyntheticRuntime(t)
	dir := t.TempDir()
	model := writeSyntheticGemma3Model(t, dir)
	data := core.JoinPath(dir, "train.jsonl")
	if r := core.WriteFile(data, []byte(
		`{"messages":[{"role":"user","content":"hello"},{"role":"assistant","content":"world"}]}`+"\n"), 0o644); !r.OK {
		t.Fatalf("write data: %v", r.Value)
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{
		"sft", "--model", model, "--data", data,
		"--epochs", "1", "--batch", "1", "--grad-accum", "1",
		"--eval-every", "0", "--checkpoint-every", "0", "--rank", "2", "--max-seq", "16",
		"--checkpoint-dir", core.JoinPath(dir, "ckpt"), // lp + save default under here
		"--run-id", "sft-cov-run",
	}, stdout, stderr)
	if code != 1 {
		t.Fatalf("exit = %d, want 1 (synthetic no trainable batches); stderr=%q", code, stderr.String())
	}
	if !strings.Contains(stderr.String(), "loading") || !strings.Contains(stderr.String(), "training:") {
		t.Fatalf("stderr = %q, want the load notice + training error after sink construction", stderr.String())
	}
}
