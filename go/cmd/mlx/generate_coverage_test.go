// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package main

import (
	"context"
	"strings"
	"testing"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/mlx"
	"dappco.re/go/mlx/pkg/metal"
)

// generate with a bad flag fails to parse (exit 2).
func TestRunGenerate_BadFlag_Bad(t *testing.T) {
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{"generate", "-not-a-flag", "x"}, stdout, stderr)
	if code != 2 {
		t.Fatalf("exit = %d, want 2 (bad flag)", code)
	}
}

// generate --native with --trace enters the native trace loader; a bad model
// path should fail as a load error, not as a CLI incompatibility.
func TestRunGenerate_NativeTraceBadModel_Bad(t *testing.T) {
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{"generate", "-native", "-trace", core.JoinPath(t.TempDir(), "nope")}, stdout, stderr)
	if code != 1 {
		t.Fatalf("exit = %d, want 1 (native trace load failure)", code)
	}
	if !strings.Contains(stderr.String(), "native trace") || !strings.Contains(stderr.String(), "load:") {
		t.Fatalf("stderr = %q, want native trace load error", stderr.String())
	}
}

// generate --native with --state now enters the native state path; a bad model
// path should fail as a load error, not as a CLI incompatibility.
func TestRunGenerate_NativeStateBadModel_Bad(t *testing.T) {
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{"generate", "-native", "-state", "chat", core.JoinPath(t.TempDir(), "nope")}, stdout, stderr)
	if code != 1 {
		t.Fatalf("exit = %d, want 1 (native state load failure)", code)
	}
	if !strings.Contains(stderr.String(), "native state") || !strings.Contains(stderr.String(), "load:") {
		t.Fatalf("stderr = %q, want native state load error", stderr.String())
	}
}

// generate over a TINY synthetic with -context + -kv-cache + -kv-storage flags
// applies all three load options and decodes (the load-option branch coverage).
func TestRunGenerate_LoadOptionFlags_Good(t *testing.T) {
	requireSyntheticRuntime(t)
	model := writeSyntheticGemma3Model(t, t.TempDir())
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{
		"generate", "-max-tokens", "6", "-temp", "0", "-draft", "",
		"-context", "64", "-kv-cache", "paged", "-kv-storage", "fp16",
		"-prompt", "hello", model,
	}, stdout, stderr)
	if code != 0 {
		t.Fatalf("exit = %d, want 0; stderr=%q", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "tok/s") {
		t.Fatalf("stdout = %q, want the decode-rate report", stdout.String())
	}
}

// generate --native over a TINY synthetic loads through the no-cgo contract
// path (printing the native notice) and then reaches the warm step. The native
// contract requires bf16 weight tables; the fp32 synthetic trips the warm guard
// (exit 1), which exercises both the native-load branch and the warm-error
// return — a real bf16 model is an AX-11 multi-GB load this suite avoids.
func TestRunGenerate_NativeBackendWarmGuard_Bad(t *testing.T) {
	requireSyntheticRuntime(t)
	model := writeSyntheticGemma3Model(t, t.TempDir())
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{
		"generate", "-native", "-max-tokens", "6", "-temp", "0", "-prompt", "hello", model,
	}, stdout, stderr)
	if code != 1 {
		t.Fatalf("exit = %d, want 1 (native warm guard on fp32 weights); stderr=%q", code, stderr.String())
	}
	if !strings.Contains(stderr.String(), "no-cgo native token-loop contract") {
		t.Fatalf("stderr = %q, want the native-backend load notice", stderr.String())
	}
}

// generate --native against a nonexistent model reaches the contract loader's
// error path (exit 1).
func TestRunGenerate_NativeBadModel_Bad(t *testing.T) {
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{
		"generate", "-native", "-max-tokens", "4", core.JoinPath(t.TempDir(), "nope"),
	}, stdout, stderr)
	if code != 1 {
		t.Fatalf("exit = %d, want 1 (native load failure)", code)
	}
	if !strings.Contains(stderr.String(), "load:") {
		t.Fatalf("stderr = %q, want a load error", stderr.String())
	}
}

// generate -trace over a TINY synthetic with -pipeline=false forces the serial
// decode loop through the trace lanes (the non-pipelined gate branch).
func TestRunGenerate_TraceSerial_Good(t *testing.T) {
	requireSyntheticRuntime(t)
	model := writeSyntheticGemma3Model(t, t.TempDir())
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{
		"generate", "-trace", "-pipeline=false", "-max-tokens", "5", "-draft", "", "-prompt", "hello", model,
	}, stdout, stderr)
	if code != 0 {
		t.Fatalf("exit = %d, want 0; stderr=%q", code, stderr.String())
	}
}

// fakeMTPTextModel embeds the inference.TextModel interface (nil — its other
// methods are never called by printGenerateMTPMetrics) and adds the MTPMetrics()
// provider so the metrics-present branch is exercised without a real
// speculative model.
type fakeMTPTextModel struct {
	inference.TextModel
	metrics *metal.MTPMetrics
}

func (f *fakeMTPTextModel) MTPMetrics() *metal.MTPMetrics { return f.metrics }

// printGenerateMTPMetrics prints the acceptance line when the model exposes
// non-nil MTP metrics (the speculative-lane instrument read).
func TestPrintGenerateMTPMetrics_WithMetrics_Good(t *testing.T) {
	buf := core.NewBuffer()
	printGenerateMTPMetrics(buf, &fakeMTPTextModel{metrics: &metal.MTPMetrics{
		AcceptanceRate: 0.8, AcceptedTokens: 16, ProposedTokens: 20, TargetVerifyCalls: 4,
	}})
	out := buf.String()
	if !strings.Contains(out, "mtp:") || !strings.Contains(out, "acceptance") {
		t.Fatalf("stdout = %q, want the MTP acceptance line", out)
	}
}

func TestPrintTokenPhaseBudget_NoGPUWait_Good(t *testing.T) {
	buf := core.NewBuffer()
	printTokenPhaseBudget(buf, "native", mlx.Metrics{TokenPhases: []mlx.TokenPhaseTrace{
		{Step: 0, TotalDuration: time.Millisecond, ForwardDuration: time.Millisecond},
		{Step: 1, TotalDuration: 2 * time.Millisecond, ForwardDuration: 2 * time.Millisecond},
		{Step: 2, FinalToken: true, TotalDuration: time.Millisecond, ForwardDuration: time.Millisecond},
	}})
	out := buf.String()
	if strings.Contains(out, "Inf") {
		t.Fatalf("stdout = %q, want finite/no-GPU ceiling text", out)
	}
	if !strings.Contains(out, "tok/s ceiling if zeroed: n/a") {
		t.Fatalf("stdout = %q, want no-GPU ceiling marker", out)
	}
}

// printGenerateMTPMetrics with a provider that returns nil metrics stays silent.
func TestPrintGenerateMTPMetrics_NilMetrics_Good(t *testing.T) {
	buf := core.NewBuffer()
	printGenerateMTPMetrics(buf, &fakeMTPTextModel{metrics: nil})
	if buf.String() != "" {
		t.Fatalf("stdout = %q, want empty for a nil-metrics provider", buf.String())
	}
}
