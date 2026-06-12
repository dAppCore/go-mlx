// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"dappco.re/go/inference"
)

// The tune verb's plumbing is tested with a FAKE measurement runner — no
// model is ever loaded. The live runner is the operator's supervised step.

func writeTuneFixtureModel(t *testing.T) string {
	t.Helper()
	model := t.TempDir()
	if err := os.WriteFile(filepath.Join(model, "config.json"), []byte(`{"model_type":"gemma4_text"}`), 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}
	assistant := filepath.Join(model, "assistant")
	if err := os.MkdirAll(assistant, 0o755); err != nil {
		t.Fatalf("mkdir assistant: %v", err)
	}
	if err := os.WriteFile(filepath.Join(assistant, "config.json"), []byte(`{"model_type":"gemma4_assistant"}`), 0o644); err != nil {
		t.Fatalf("write assistant config: %v", err)
	}
	if err := os.WriteFile(filepath.Join(assistant, "model.safetensors"), []byte("stub"), 0o644); err != nil {
		t.Fatalf("write weights stub: %v", err)
	}
	return model
}

func fakeTuneResults(workload inference.TuningWorkload, ar float64, blocks map[int]float64) []inference.TuningResult {
	results := []inference.TuningResult{
		tuneResultFromMetricsForTest(tuneARCandidateID, nil, ar, workload),
	}
	for block, rate := range blocks {
		results = append(results, tuneResultFromMetricsForTest(
			"mtp-block-"+itoa(block),
			map[string]string{tuneDraftBlockLabel: itoa(block)},
			rate, workload))
	}
	return results
}

func itoa(n int) string {
	return string(rune('0' + n))
}

func tuneResultFromMetricsForTest(id string, labels map[string]string, decodeRate float64, workload inference.TuningWorkload) inference.TuningResult {
	measurements := inference.TuningMeasurements{DecodeTokensPerSec: decodeRate}
	return inference.TuningResult{
		Candidate:    inference.TuningCandidate{ID: id, Workload: workload, Labels: labels},
		Measurements: measurements,
		Score:        inference.ScoreTuningMeasurements(workload, measurements),
	}
}

// swapTuneRunner installs a fake measurement runner for one test.
func swapTuneRunner(t *testing.T, fake func(context.Context, tuneMeasurementRequest) ([]inference.TuningResult, error)) {
	t.Helper()
	previous := runTuneMeasurements
	runTuneMeasurements = fake
	t.Cleanup(func() { runTuneMeasurements = previous })
}

// An MTP block that beats AR is selected and persisted; the profile round-trips
// through the serve-side reader with the winning block intact.
func TestTune_MTPWinsPersistsProfile_Good(t *testing.T) {
	model := writeTuneFixtureModel(t)
	profileDir := t.TempDir()
	var gotReq tuneMeasurementRequest
	swapTuneRunner(t, func(_ context.Context, req tuneMeasurementRequest) ([]inference.TuningResult, error) {
		gotReq = req
		return fakeTuneResults(req.Workload, 32.8, map[int]float64{4: 48.0, 5: 52.0, 6: 50.1}), nil
	})

	var stdout, stderr bytes.Buffer
	code := runTuneCommand(context.Background(), []string{"--model", model, "--profile-dir", profileDir, "--max-tokens", "64"}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("tune exit = %d, stderr: %s", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "mtp-block-5") {
		t.Fatalf("summary missing the block-5 winner: %s", stdout.String())
	}
	if gotReq.DraftPath == "" || len(gotReq.Blocks) != 3 {
		t.Fatalf("runner request = %+v, want detected drafter + 3 blocks", gotReq)
	}

	// Round-trip: the serve-side reader finds the persisted block.
	block, path := loadTunedDraftBlock(profileDir, model, "")
	if block != 5 || path == "" {
		t.Fatalf("loadTunedDraftBlock = %d/%q, want the persisted block 5", block, path)
	}
}

// AR winning means NO persistence — serve must never engage a losing block.
func TestTune_ARWinsPersistsNothing_Bad(t *testing.T) {
	model := writeTuneFixtureModel(t)
	profileDir := t.TempDir()
	swapTuneRunner(t, func(_ context.Context, req tuneMeasurementRequest) ([]inference.TuningResult, error) {
		return fakeTuneResults(req.Workload, 60.0, map[int]float64{4: 48.0, 5: 52.0}), nil
	})

	var stdout, stderr bytes.Buffer
	code := runTuneCommand(context.Background(), []string{"--model", model, "--profile-dir", profileDir}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("tune exit = %d, stderr: %s", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "AR") || !strings.Contains(stdout.String(), "no profile written") {
		t.Fatalf("summary must say AR won and nothing persisted: %s", stdout.String())
	}
	entries, err := os.ReadDir(profileDir)
	if err != nil {
		t.Fatalf("read profile dir: %v", err)
	}
	if len(entries) != 0 {
		t.Fatalf("profile dir has %d entries, want none when AR wins", len(entries))
	}
}

// Flag validation and structural failures exit non-zero without touching the
// runner or the disk.
func TestTune_BadInputs_Ugly(t *testing.T) {
	swapTuneRunner(t, func(_ context.Context, _ tuneMeasurementRequest) ([]inference.TuningResult, error) {
		t.Fatal("runner must not run on invalid input")
		return nil, nil
	})

	var stdout, stderr bytes.Buffer
	if code := runTuneCommand(context.Background(), []string{}, &stdout, &stderr); code != 2 {
		t.Fatalf("missing --model exit = %d, want 2", code)
	}
	if code := runTuneCommand(context.Background(), []string{"--model", "/m", "--depths", "1,99"}, &stdout, &stderr); code != 2 {
		t.Fatalf("out-of-range depths exit = %d, want 2", code)
	}
	if code := runTuneCommand(context.Background(), []string{"--model", "/m", "--workload", "nonsense"}, &stdout, &stderr); code != 2 {
		t.Fatalf("bad workload exit = %d, want 2", code)
	}
	// A model with no drafter has nothing to tune.
	bare := t.TempDir()
	if err := os.WriteFile(filepath.Join(bare, "config.json"), []byte(`{"model_type":"gemma4_text"}`), 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}
	if code := runTuneCommand(context.Background(), []string{"--model", bare}, &stdout, &stderr); code != 1 {
		t.Fatalf("drafterless exit = %d, want 1 with the nothing-to-tune message", code)
	}
	if !strings.Contains(stderr.String(), "nothing to tune") {
		t.Fatalf("stderr missing the nothing-to-tune message: %s", stderr.String())
	}
}

// Runner errors surface as failures (exit 1); per-candidate errors are
// reported but do not block selection among the survivors.
func TestTune_RunnerErrors_Ugly(t *testing.T) {
	model := writeTuneFixtureModel(t)
	swapTuneRunner(t, func(_ context.Context, _ tuneMeasurementRequest) ([]inference.TuningResult, error) {
		return nil, contextErr{}
	})
	var stdout, stderr bytes.Buffer
	if code := runTuneCommand(context.Background(), []string{"--model", model}, &stdout, &stderr); code != 1 {
		t.Fatalf("runner error exit = %d, want 1", code)
	}

	// One candidate errors, the rest still select.
	profileDir := t.TempDir()
	swapTuneRunner(t, func(_ context.Context, req tuneMeasurementRequest) ([]inference.TuningResult, error) {
		results := fakeTuneResults(req.Workload, 30.0, map[int]float64{5: 52.0})
		results = append(results, inference.TuningResult{
			Candidate: inference.TuningCandidate{ID: "mtp-block-6", Workload: req.Workload},
			Error:     "verify forward OOM",
		})
		return results, nil
	})
	stdout.Reset()
	stderr.Reset()
	if code := runTuneCommand(context.Background(), []string{"--model", model, "--profile-dir", profileDir}, &stdout, &stderr); code != 0 {
		t.Fatalf("partial-failure exit = %d, stderr: %s", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "mtp-block-5") {
		t.Fatalf("summary missing the surviving winner: %s", stdout.String())
	}
}

type contextErr struct{}

func (contextErr) Error() string { return "bench runner exploded" }

// The serve-side auto-profile reader: newest matching profile wins, machine
// hashes must agree when both sides carry one, corrupt files are skipped, and
// --no-auto-profile (handled by the caller passing flagBlock>0 or the flag)
// never reaches the reader.
func TestLoadTunedDraftBlock_Matching_Good(t *testing.T) {
	dir := t.TempDir()
	write := func(name string, profile inference.TuningProfile) {
		t.Helper()
		if err := writeTuningProfile(filepath.Join(dir, name), profile); err != nil {
			t.Fatalf("write profile: %v", err)
		}
	}
	write("old.json", inference.TuningProfile{
		Key:           inference.TuningProfileKey{Model: inference.ModelIdentity{Path: "/m"}},
		Candidate:     inference.TuningCandidate{ID: "mtp-block-4", Labels: map[string]string{tuneDraftBlockLabel: "4"}},
		CreatedAtUnix: 100,
	})
	write("new.json", inference.TuningProfile{
		Key:           inference.TuningProfileKey{Model: inference.ModelIdentity{Path: "/m"}},
		Candidate:     inference.TuningCandidate{ID: "mtp-block-5", Labels: map[string]string{tuneDraftBlockLabel: "5"}},
		CreatedAtUnix: 200,
	})
	write("other-model.json", inference.TuningProfile{
		Key:           inference.TuningProfileKey{Model: inference.ModelIdentity{Path: "/other"}},
		Candidate:     inference.TuningCandidate{ID: "mtp-block-6", Labels: map[string]string{tuneDraftBlockLabel: "6"}},
		CreatedAtUnix: 300,
	})
	write("other-machine.json", inference.TuningProfile{
		Key:           inference.TuningProfileKey{MachineHash: "machine-b", Model: inference.ModelIdentity{Path: "/m"}},
		Candidate:     inference.TuningCandidate{ID: "mtp-block-7", Labels: map[string]string{tuneDraftBlockLabel: "7"}},
		CreatedAtUnix: 400,
	})
	if err := os.WriteFile(filepath.Join(dir, "corrupt.json"), []byte("{nope"), 0o644); err != nil {
		t.Fatalf("write corrupt: %v", err)
	}

	block, path := loadTunedDraftBlock(dir, "/m", "machine-a")
	if block != 5 || !strings.HasSuffix(path, "new.json") {
		t.Fatalf("loadTunedDraftBlock = %d/%q, want newest matching block 5", block, path)
	}
	if block, _ := loadTunedDraftBlock(dir, "/nope", "machine-a"); block != 0 {
		t.Fatalf("unmatched model block = %d, want 0", block)
	}
}
