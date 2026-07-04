// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
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

// TestAdmin_ReadJSONBody_Ugly — a body within the size cap but not valid
// JSON surfaces the unmarshal error (distinct from the oversize-cap
// rejection covered above).
func TestAdmin_ReadJSONBody_Ugly(t *testing.T) {
	req := httptest.NewRequest(http.MethodPost, "/v1/admin/test", strings.NewReader("{definitely not json"))
	var target map[string]any
	if err := readJSONBody(req, &target); err == nil {
		t.Fatal("expected unmarshal error for malformed JSON, got nil")
	}
}

// TestAdmin_WriteJSON_Good — writeJSON sets the JSON content-type, the
// requested status, and a parseable body.
func TestAdmin_WriteJSON_Good(t *testing.T) {
	rr := httptest.NewRecorder()
	writeJSON(rr, http.StatusCreated, map[string]string{"ok": "yes"})
	if rr.Code != http.StatusCreated {
		t.Fatalf("status = %d, want 201", rr.Code)
	}
	if ct := rr.Header().Get("content-type"); ct != "application/json" {
		t.Fatalf("content-type = %q, want application/json", ct)
	}
	var decoded map[string]string
	if err := json.Unmarshal(rr.Body.Bytes(), &decoded); err != nil {
		t.Fatalf("body not JSON: %v", err)
	}
	if decoded["ok"] != "yes" {
		t.Fatalf("body = %#v, want ok=yes", decoded)
	}
}

// TestAdmin_WriteJSON_Ugly — an unmarshalable value (a channel can't be
// JSON-encoded) falls back to 500 + a hand-rolled error envelope rather
// than writing a half-serialised body.
func TestAdmin_WriteJSON_Ugly(t *testing.T) {
	rr := httptest.NewRecorder()
	writeJSON(rr, http.StatusOK, map[string]any{"bad": make(chan int)})
	if rr.Code != http.StatusInternalServerError {
		t.Fatalf("status = %d, want 500 on marshal failure", rr.Code)
	}
	if !strings.Contains(rr.Body.String(), "marshal failed") {
		t.Fatalf("body = %q, want marshal-failed envelope", rr.Body.String())
	}
}

// TestAdmin_NotImplementedHandler_Good — the placeholder answers 501 and
// names both the endpoint and what's blocking it, so a caller hitting an
// unwired route gets an actionable message rather than a 404.
func TestAdmin_NotImplementedHandler_Good(t *testing.T) {
	h := adminNotImplementedHandler("serve/reload", "no resolver wired")
	req := httptest.NewRequest(http.MethodPost, "/v1/admin/serve/reload", nil)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)
	if rr.Code != http.StatusNotImplemented {
		t.Fatalf("status = %d, want 501", rr.Code)
	}
	var body map[string]string
	if err := json.Unmarshal(rr.Body.Bytes(), &body); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if body["endpoint"] != "serve/reload" || body["blocker"] != "no resolver wired" {
		t.Fatalf("body = %#v, want endpoint + blocker", body)
	}
}

// TestAdmin_NowJobID_Good — the auto-tune job id carries its prefix and a
// purely numeric nanosecond tail. (Uniqueness across calls is "extremely
// improbable", not guaranteed — two calls in the same nanosecond would
// collide — so we assert the deterministic format, not inequality.)
func TestAdmin_NowJobID_Good(t *testing.T) {
	id := nowJobID()
	const prefix = "autotune-"
	if !strings.HasPrefix(id, prefix) {
		t.Fatalf("nowJobID = %q, want %s prefix", id, prefix)
	}
	tail := strings.TrimPrefix(id, prefix)
	if tail == "" {
		t.Fatalf("nowJobID = %q, want a non-empty nanosecond tail", id)
	}
	for _, r := range tail {
		if r < '0' || r > '9' {
			t.Fatalf("nowJobID tail = %q, want all digits", tail)
		}
	}
}

// TestAdmin_MachineHandler_Bad — a non-GET method is rejected 405 before
// any machine-discovery work runs (the mutation guard is the handler's
// own logic; the discovery happy-path is environment-dependent and not
// asserted here).
func TestAdmin_MachineHandler_Bad(t *testing.T) {
	for _, method := range []string{http.MethodPost, http.MethodPut, http.MethodDelete} {
		t.Run(method, func(t *testing.T) {
			req := httptest.NewRequest(method, "/v1/admin/machine", nil)
			rr := httptest.NewRecorder()
			adminMachineHandler(rr, req)
			if rr.Code != http.StatusMethodNotAllowed {
				t.Fatalf("method %s: status = %d, want 405", method, rr.Code)
			}
		})
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
