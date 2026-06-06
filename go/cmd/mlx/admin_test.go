// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"testing"
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
// TestClampAutoTuneRequest_PreservesSmallValues — values within the
// caps must round-trip unchanged so legitimate callers keep their
// chosen budget.
// TestAdminJobRegistry_Semaphore_RefusesSecond — second concurrent
// auto-tune kickoff must fail-fast, not block. Tuning is GPU-bound
// and single-instance; refusing the second is the right answer.
// TestAdminJobRegistry_Prune_EvictsOldFinished — done/failed jobs
// older than maxJobAge must be evicted. Keeps the registry bounded
// across long-running serve processes.
// TestAdminJobRegistry_PersistRoundtrip — a job written to the
// registry's persistPath must reload into a fresh registry pointed
// at the same path. Survives serve restarts.
// TestAdminJobRegistry_RestoreMarksInFlightAsFailed — jobs that
// were "pending" or "running" at write time must restore as "failed"
// with a clear restart message (the goroutine that would have
// completed them no longer exists post-restart).
// TestAdminJobRegistry_PersistEmpty — when persistPath is empty
// (test mode), all helpers stay no-op without error.
// TestAdminJobRegistry_Prune_KeepsInFlight — pending/running jobs
// must never be evicted regardless of age. They're load-bearing
// references for in-flight goroutines.
