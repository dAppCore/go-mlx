// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

// TestReadJSONBody_RejectsOversizedBody — admin body reads must refuse
// >64KB to prevent memory-exhaustion DoS via adversarial large POST.
func TestReadJSONBody_RejectsOversizedBody(t *testing.T) {
	body := bytes.Repeat([]byte("x"), 128*1024)
	req := httptest.NewRequest(http.MethodPost, "/v1/admin/test", bytes.NewReader(body))
	var target map[string]any
	if err := readJSONBody(req, &target); err == nil {
		t.Fatal("expected error for 128KB body, got nil")
	}
}

// TestReadJSONBody_AcceptsSmallBody — legitimate admin payloads must pass.
func TestReadJSONBody_AcceptsSmallBody(t *testing.T) {
	body := []byte(`{"model":"lemer-lite","max_candidates":4}`)
	req := httptest.NewRequest(http.MethodPost, "/v1/admin/test", bytes.NewReader(body))
	var target map[string]any
	if err := readJSONBody(req, &target); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if target["model"] != "lemer-lite" {
		t.Errorf("expected model=lemer-lite, got %v", target["model"])
	}
}

// TestClampAutoTuneRequest_ClampsHugeValues — adversarial inputs must
// be clamped to the resource caps before reaching the worker.
func TestClampAutoTuneRequest_ClampsHugeValues(t *testing.T) {
	req := adminAutoTuneRequest{
		Model:         "/some/path",
		MaxCandidates: 1_000_000,
		MaxTokens:     1_000_000,
		Runs:          1_000_000,
	}
	clamped := clampAutoTuneRequest(req)
	if clamped.MaxCandidates != maxAutoTuneCandidates {
		t.Errorf("MaxCandidates: got %d want %d", clamped.MaxCandidates, maxAutoTuneCandidates)
	}
	if clamped.MaxTokens != maxAutoTuneTokens {
		t.Errorf("MaxTokens: got %d want %d", clamped.MaxTokens, maxAutoTuneTokens)
	}
	if clamped.Runs != maxAutoTuneRuns {
		t.Errorf("Runs: got %d want %d", clamped.Runs, maxAutoTuneRuns)
	}
}

// TestClampAutoTuneRequest_PreservesSmallValues — values within the
// caps must round-trip unchanged so legitimate callers keep their
// chosen budget.
func TestClampAutoTuneRequest_PreservesSmallValues(t *testing.T) {
	req := adminAutoTuneRequest{MaxCandidates: 4, MaxTokens: 256, Runs: 3}
	clamped := clampAutoTuneRequest(req)
	if clamped.MaxCandidates != 4 || clamped.MaxTokens != 256 || clamped.Runs != 3 {
		t.Errorf("clamp altered small values: got %+v", clamped)
	}
}

// TestAdminJobRegistry_Semaphore_RefusesSecond — second concurrent
// auto-tune kickoff must fail-fast, not block. Tuning is GPU-bound
// and single-instance; refusing the second is the right answer.
func TestAdminJobRegistry_Semaphore_RefusesSecond(t *testing.T) {
	reg := newAdminJobRegistry(context.Background(), io.Discard, "")
	if !reg.tryAcquireSlot() {
		t.Fatal("first acquire should succeed on a fresh registry")
	}
	if reg.tryAcquireSlot() {
		t.Fatal("second acquire should fail — slot is held")
	}
	reg.releaseSlot()
	if !reg.tryAcquireSlot() {
		t.Fatal("acquire after release should succeed")
	}
	reg.releaseSlot()
}

// TestAdminJobRegistry_Prune_EvictsOldFinished — done/failed jobs
// older than maxJobAge must be evicted. Keeps the registry bounded
// across long-running serve processes.
func TestAdminJobRegistry_Prune_EvictsOldFinished(t *testing.T) {
	reg := newAdminJobRegistry(context.Background(), io.Discard, "")
	reg.jobs["old-done"] = &adminAutoTuneJob{
		ID: "old-done", Status: "done", StartedAt: time.Now().Add(-2 * maxJobAge),
	}
	reg.jobs["old-failed"] = &adminAutoTuneJob{
		ID: "old-failed", Status: "failed", StartedAt: time.Now().Add(-2 * maxJobAge),
	}
	reg.jobs["recent-done"] = &adminAutoTuneJob{
		ID: "recent-done", Status: "done", StartedAt: time.Now(),
	}

	reg.mu.Lock()
	reg.prune()
	reg.mu.Unlock()

	if _, exists := reg.jobs["old-done"]; exists {
		t.Error("old done job not evicted")
	}
	if _, exists := reg.jobs["old-failed"]; exists {
		t.Error("old failed job not evicted")
	}
	if _, exists := reg.jobs["recent-done"]; !exists {
		t.Error("recent done job evicted (should be kept)")
	}
}

// TestAdminJobRegistry_PersistRoundtrip — a job written to the
// registry's persistPath must reload into a fresh registry pointed
// at the same path. Survives serve restarts.
func TestAdminJobRegistry_PersistRoundtrip(t *testing.T) {
	tmp := t.TempDir() + "/admin-jobs.json"

	reg := newAdminJobRegistry(context.Background(), io.Discard, tmp)
	reg.commitLocked(func() {
		reg.jobs["job-1"] = &adminAutoTuneJob{
			ID: "job-1", Status: "done", Model: "/some/model",
			StartedAt: time.Now().UTC().Add(-5 * time.Minute),
		}
	})

	// Fresh registry — must load the persisted job.
	reg2 := newAdminJobRegistry(context.Background(), io.Discard, tmp)
	job, ok := reg2.jobs["job-1"]
	if !ok {
		t.Fatal("expected job-1 to load from persistence, got nothing")
	}
	if job.Status != "done" || job.Model != "/some/model" {
		t.Errorf("loaded job state wrong: %+v", job)
	}
}

// TestAdminJobRegistry_RestoreMarksInFlightAsFailed — jobs that
// were "pending" or "running" at write time must restore as "failed"
// with a clear restart message (the goroutine that would have
// completed them no longer exists post-restart).
func TestAdminJobRegistry_RestoreMarksInFlightAsFailed(t *testing.T) {
	tmp := t.TempDir() + "/admin-jobs.json"

	// Seed a registry with an in-flight job + persist it.
	seed := newAdminJobRegistry(context.Background(), io.Discard, tmp)
	seed.commitLocked(func() {
		seed.jobs["in-flight"] = &adminAutoTuneJob{
			ID: "in-flight", Status: "running", Model: "/m",
			StartedAt: time.Now().UTC().Add(-10 * time.Minute),
		}
	})

	// Restore — the in-flight job must come back as failed.
	restored := newAdminJobRegistry(context.Background(), io.Discard, tmp)
	job, ok := restored.jobs["in-flight"]
	if !ok {
		t.Fatal("in-flight job not restored")
	}
	if job.Status != "failed" {
		t.Errorf("expected status=failed, got %q", job.Status)
	}
	if job.Error == "" {
		t.Error("expected non-empty Error explaining the restart")
	}
	if job.FinishedAt == nil {
		t.Error("expected FinishedAt to be set on restored-failed job")
	}
}

// TestAdminJobRegistry_PersistEmpty — when persistPath is empty
// (test mode), all helpers stay no-op without error.
func TestAdminJobRegistry_PersistEmpty(t *testing.T) {
	reg := newAdminJobRegistry(context.Background(), io.Discard, "")
	if err := reg.persistLocked(); err != nil {
		t.Errorf("expected nil error on empty persistPath, got %v", err)
	}
	// commitLocked must not panic.
	reg.commitLocked(func() {
		reg.jobs["x"] = &adminAutoTuneJob{ID: "x", Status: "done", StartedAt: time.Now()}
	})
}

// TestAdminJobRegistry_Prune_KeepsInFlight — pending/running jobs
// must never be evicted regardless of age. They're load-bearing
// references for in-flight goroutines.
func TestAdminJobRegistry_Prune_KeepsInFlight(t *testing.T) {
	reg := newAdminJobRegistry(context.Background(), io.Discard, "")
	reg.jobs["old-running"] = &adminAutoTuneJob{
		ID: "old-running", Status: "running", StartedAt: time.Now().Add(-2 * maxJobAge),
	}
	reg.jobs["old-pending"] = &adminAutoTuneJob{
		ID: "old-pending", Status: "pending", StartedAt: time.Now().Add(-2 * maxJobAge),
	}

	reg.mu.Lock()
	reg.prune()
	reg.mu.Unlock()

	if _, exists := reg.jobs["old-running"]; !exists {
		t.Error("running job evicted — must never happen")
	}
	if _, exists := reg.jobs["old-pending"]; !exists {
		t.Error("pending job evicted — must never happen")
	}
}
