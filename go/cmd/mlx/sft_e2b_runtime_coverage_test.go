// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package main

import (
	"context"
	"strings"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/mlx/internal/metaltest"
)

// sft over the real e2b-4bit model runs LoRA training to completion and reaches
// runSFTCommand's success tail (the steps/last-loss line + the adapter line).
// The synthetic gemma3 fixture cannot reach this: its degenerate BPE tokenizer
// (empty merges, special-token-only vocab) encodes every input to ZERO tokens,
// so every sample is rejected as too short (vLen<2) and TrainSFT errors with
// "no trainable batches". Only a real tokenizer produces trainable spans.
//
// Verified the cgo metal engine does LoRA backward on a QUANTIZED (4-bit) base —
// frozen quantized weights + a tiny rank-2 adapter, so the gradient/optimizer
// footprint stays small. The run is intentionally minimal (4 tiny samples, 1
// epoch, batch 1, grad-accum 1, eval/metrics/capture all off): the goal is
// coverage of the result-printing tail, not meaningful fine-tuning, so a
// degenerate loss is still a PASS — only a hard error fails. A weightless
// checkout skips green via HFModelPath's t.Skip.
//
// Eval (--valid / --eval-every) is deliberately NOT exercised here: the val-loss
// and checkpoint-score branches need eval-generation passes DURING training,
// which is real extra cost for a handful of statements. The partial tail (steps
// + adapter) is the bounded win.
func TestRunSFT_E2BLoRACompletes_Good(t *testing.T) {
	requireSyntheticRuntime(t)
	model := metaltest.HFModelPath(t, "mlx-community/gemma-4-e2b-it-4bit")

	dir := t.TempDir()
	data := core.JoinPath(dir, "train.jsonl")
	var lines strings.Builder
	for i := 0; i < 4; i++ {
		lines.WriteString(`{"messages":[{"role":"user","content":"Say hi."},{"role":"assistant","content":"Hi there, friend."}]}`)
		lines.WriteString("\n")
	}
	if r := core.WriteFile(data, []byte(lines.String()), 0o644); !r.OK {
		t.Fatalf("seed training data: %v", r.Value)
	}

	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{
		"sft", "--model", model, "--data", data,
		"--epochs", "1", "--batch", "1", "--grad-accum", "1",
		"--eval-every", "0", "--metrics-lp", "off", "--capture", "off",
		"--checkpoint-every", "0", "--rank", "2", "--max-seq", "64",
		"--save", core.JoinPath(dir, "ckpt", "adapter.safetensors"),
	}, stdout, stderr)
	if code != 0 {
		t.Fatalf("exit = %d, want 0 (LoRA SFT completes on e2b-4bit); stderr=%q", code, stderr.String())
	}
	out := stdout.String()
	// The success tail: the steps/last-loss summary line + the saved-adapter line.
	if !strings.Contains(out, "steps") || !strings.Contains(out, "last-loss") {
		t.Fatalf("stdout = %q, want the steps/last-loss summary line", out)
	}
	if !strings.Contains(out, "adapter") {
		t.Fatalf("stdout = %q, want the saved-adapter line", out)
	}
}

// sft over e2b-4bit with --metrics-lp pointed at a file (NOT --influx-url, which
// would hit the network) wires the v0-schema line-protocol metrics sink and
// covers the sink branches runSFTCommand's bare run leaves cold: the sink setup
// + default run-id derivation (sft.go ~180/188), the ProbeSink wire, and the
// flush / "metrics N lines → <path>" tail (sft.go ~248-261). Train-loss lines
// alone make Lines()>0, so no eval/cascade pass is needed. Same minimal 4-bit
// LoRA run as above; a weightless checkout skips green.
func TestRunSFT_E2BMetricsSink_Good(t *testing.T) {
	requireSyntheticRuntime(t)
	model := metaltest.HFModelPath(t, "mlx-community/gemma-4-e2b-it-4bit")

	dir := t.TempDir()
	data := core.JoinPath(dir, "train.jsonl")
	var lines strings.Builder
	for i := 0; i < 4; i++ {
		lines.WriteString(`{"messages":[{"role":"user","content":"Say hi."},{"role":"assistant","content":"Hi there, friend."}]}`)
		lines.WriteString("\n")
	}
	if r := core.WriteFile(data, []byte(lines.String()), 0o644); !r.OK {
		t.Fatalf("seed training data: %v", r.Value)
	}
	lpPath := core.JoinPath(dir, "metrics.lp")

	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{
		"sft", "--model", model, "--data", data,
		"--epochs", "1", "--batch", "1", "--grad-accum", "1",
		"--eval-every", "0", "--capture", "off",
		"--metrics-lp", lpPath, // file sink — no --influx-url (no network)
		"--checkpoint-every", "0", "--rank", "2", "--max-seq", "64",
		"--save", core.JoinPath(dir, "ckpt", "adapter.safetensors"),
	}, stdout, stderr)
	if code != 0 {
		t.Fatalf("exit = %d, want 0 (LoRA SFT with metrics sink); stderr=%q", code, stderr.String())
	}
	out := stdout.String()
	if !strings.Contains(out, "metrics") {
		t.Fatalf("stdout = %q, want the 'metrics N lines' summary", out)
	}
	if !core.Stat(lpPath).OK {
		t.Fatalf("metrics line-protocol file %q was not written", lpPath)
	}
}
