// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"bytes"
	"context"
	"strings"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	mlx "dappco.re/go/mlx"
)

// tune with a bad flag fails to parse (exit 2).
func TestRunTune_BadFlag_Bad(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := runTuneCommand(context.Background(), []string{"-not-a-flag"}, &stdout, &stderr); code != 2 {
		t.Fatalf("exit = %d, want 2 (bad flag)", code)
	}
}

// tune with an unexpected positional argument is a usage error (exit 2).
func TestRunTune_UnexpectedPositional_Bad(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runTuneCommand(context.Background(), []string{"stray-arg"}, &stdout, &stderr)
	if code != 2 {
		t.Fatalf("exit = %d, want 2 (unexpected positional)", code)
	}
	if !strings.Contains(stderr.String(), "unexpected positional") {
		t.Fatalf("stderr = %q, want the positional notice", stderr.String())
	}
}

// tune -json emits JSONL result events + a selected event when an MTP block
// wins, persisting the profile.
func TestRunTune_JSONMTPWins_Good(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	model := writeTuneFixtureModel(t)
	profileDir := t.TempDir()
	swapTuneRunner(t, func(_ context.Context, req tuneMeasurementRequest) ([]inference.TuningResult, error) {
		return fakeTuneResults(req.Workload, 30.0, map[int]float64{5: 55.0}), nil
	})
	var stdout, stderr bytes.Buffer
	code := runTuneCommand(context.Background(), []string{"--model", model, "--profile-dir", profileDir, "--json"}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("exit = %d; stderr=%q", code, stderr.String())
	}
	out := stdout.String()
	if !strings.Contains(out, `"persisted":"true"`) || !strings.Contains(out, `"profile_path"`) {
		t.Fatalf("JSONL = %q, want a persisted selected event", out)
	}
}

// tune -json with AR winning emits the not-persisted selected event.
func TestRunTune_JSONARWins_Good(t *testing.T) {
	model := writeTuneFixtureModel(t)
	swapTuneRunner(t, func(_ context.Context, req tuneMeasurementRequest) ([]inference.TuningResult, error) {
		return fakeTuneResults(req.Workload, 99.0, map[int]float64{5: 40.0}), nil
	})
	var stdout, stderr bytes.Buffer
	code := runTuneCommand(context.Background(), []string{"--model", model, "--profile-dir", t.TempDir(), "--json"}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("exit = %d; stderr=%q", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), `"persisted":"false"`) {
		t.Fatalf("JSONL = %q, want the AR-wins not-persisted event", stdout.String())
	}
}

// tune where every candidate errored persists nothing and exits 1.
func TestRunTune_AllCandidatesFailed_Bad(t *testing.T) {
	model := writeTuneFixtureModel(t)
	swapTuneRunner(t, func(_ context.Context, req tuneMeasurementRequest) ([]inference.TuningResult, error) {
		return []inference.TuningResult{
			{Candidate: inference.TuningCandidate{ID: "ar", Workload: req.Workload}, Error: "ar boom"},
			{Candidate: inference.TuningCandidate{ID: "mtp-block-5", Workload: req.Workload}, Error: "verify OOM"},
		}, nil
	})
	var stdout, stderr bytes.Buffer
	code := runTuneCommand(context.Background(), []string{"--model", model, "--profile-dir", t.TempDir()}, &stdout, &stderr)
	if code != 1 {
		t.Fatalf("exit = %d, want 1 (all candidates failed)", code)
	}
	if !strings.Contains(stderr.String(), "every candidate failed") {
		t.Fatalf("stderr = %q, want the all-failed notice", stderr.String())
	}
}

// tune persists a profile even when the machine hash is unavailable — the
// discovery seam errors, so the warning fires but the MTP winner is still
// written (without a machine hash).
func TestRunTune_MachineHashUnavailable_Good(t *testing.T) {
	origDiscover := runDiscoverLocalRuntime
	origDevice := runGetDeviceInfo
	t.Cleanup(func() { runDiscoverLocalRuntime = origDiscover; runGetDeviceInfo = origDevice })
	runGetDeviceInfo = func() mlx.DeviceInfo { return mlx.DeviceInfo{} }
	runDiscoverLocalRuntime = func(_ context.Context, _ mlx.LocalDiscoveryConfig) (inference.MachineDiscoveryReport, error) {
		return inference.MachineDiscoveryReport{}, core.NewError("no device")
	}
	model := writeTuneFixtureModel(t)
	profileDir := t.TempDir()
	swapTuneRunner(t, func(_ context.Context, req tuneMeasurementRequest) ([]inference.TuningResult, error) {
		return fakeTuneResults(req.Workload, 30.0, map[int]float64{5: 55.0}), nil
	})
	var stdout, stderr bytes.Buffer
	code := runTuneCommand(context.Background(), []string{"--model", model, "--profile-dir", profileDir}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("exit = %d; stderr=%q", code, stderr.String())
	}
	if !strings.Contains(stderr.String(), "machine hash unavailable") {
		t.Fatalf("stderr = %q, want the hash-unavailable warning", stderr.String())
	}
	if !strings.Contains(stdout.String(), "profile") {
		t.Fatalf("stdout = %q, want the profile-written summary", stdout.String())
	}
}

// parseTuneDraftBlocks skips empty entries between commas, rejects an
// all-empty/blank list, and rejects a non-numeric entry.
func TestParseTuneDraftBlocks_Coverage_Good(t *testing.T) {
	blocks, err := parseTuneDraftBlocks("4, ,5,,6")
	if err != nil || len(blocks) != 3 {
		t.Fatalf("parse(gappy) = (%v, %v), want 3 blocks", blocks, err)
	}
	if _, err := parseTuneDraftBlocks("  , ,"); err == nil {
		t.Fatal("parse(blank) error = nil, want no-blocks error")
	}
	if _, err := parseTuneDraftBlocks("4,oops"); err == nil {
		t.Fatal("parse(non-numeric) error = nil, want invalid-block error")
	}
	if _, err := parseTuneDraftBlocks("1"); err == nil {
		t.Fatal("parse(out-of-range) error = nil, want range error")
	}
}
