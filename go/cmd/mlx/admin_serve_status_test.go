// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"dappco.re/go/inference"
)

// TestBuildAdminServeStatusConfig_FromCandidate — every TuningCandidate
// field copies into the corresponding adminServeStatusConfig field.
// Documents the 1:1 mapping so future TuningCandidate additions are
// caught by a follow-up test failure here.
func TestBuildAdminServeStatusConfig_FromCandidate(t *testing.T) {
	c := inference.TuningCandidate{
		ContextLength:        8192,
		ParallelSlots:        4,
		PromptCache:          true,
		PromptCacheMinTokens: 64,
		CachePolicy:          "rotating",
		CacheMode:            "fp16",
		BatchSize:            32,
		PrefillChunkSize:     1024,
		ExpectedQuantization: 4,
		MemoryLimitBytes:     64 << 30,
		CacheLimitBytes:      8 << 30,
		WiredLimitBytes:      32 << 30,
		Adapter:              inference.AdapterIdentity{Path: "/tmp/adapter"},
	}
	got := buildAdminServeStatusConfig(c, 0)
	if got.ContextLength != 8192 || got.ParallelSlots != 4 || !got.PromptCache {
		t.Errorf("scalar field copy failed: %+v", got)
	}
	if got.CacheMode != "fp16" || got.CachePolicy != "rotating" {
		t.Errorf("string field copy failed: %+v", got)
	}
	if got.MemoryLimitBytes != 64<<30 || got.CacheLimitBytes != 8<<30 || got.WiredLimitBytes != 32<<30 {
		t.Errorf("memory caps copy failed: %+v", got)
	}
	if got.AdapterPath != "/tmp/adapter" {
		t.Errorf("adapter path copy failed: %q", got.AdapterPath)
	}
}

// TestBuildAdminServeStatusConfig_ContextOverride — explicit --context
// flag must win over the profile's ContextLength so operators can
// shrink memory footprint without re-tuning.
func TestBuildAdminServeStatusConfig_ContextOverride(t *testing.T) {
	c := inference.TuningCandidate{ContextLength: 8192}
	got := buildAdminServeStatusConfig(c, 2048)
	if got.ContextLength != 2048 {
		t.Errorf("override failed: got %d want 2048", got.ContextLength)
	}
}

// TestBuildAdminServeStatusConfig_NoOverride_ZeroLeaves — contextOverride=0
// must leave the candidate's value untouched (zero is the "no override"
// sentinel, not a request to set context to 0).
func TestBuildAdminServeStatusConfig_NoOverride_ZeroLeaves(t *testing.T) {
	c := inference.TuningCandidate{ContextLength: 8192}
	got := buildAdminServeStatusConfig(c, 0)
	if got.ContextLength != 8192 {
		t.Errorf("zero-override clobbered candidate: got %d want 8192", got.ContextLength)
	}
}

// TestAdminServeStatusHandler_GETReturnsJSON — GET returns the
// snapshot as JSON. Caller (GUI / agent / curl) parses the shape
// without recomputation; runtime + config fields are present.
func TestAdminServeStatusHandler_GETReturnsJSON(t *testing.T) {
	snap := adminServeStatus{
		ModelPath:    "/some/model",
		ProfilePath:  "/some/profile.json",
		Runtime:      adminRuntimeMetal,
		LoadedAtUnix: 1700000000,
		Config: adminServeStatusConfig{
			ContextLength: 8192,
			CacheMode:     "fp16",
			PromptCache:   true,
		},
	}
	req := httptest.NewRequest(http.MethodGet, "/v1/admin/serve/status", nil)
	rr := httptest.NewRecorder()
	adminServeStatusHandler(snap).ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rr.Code)
	}
	if got := rr.Header().Get("content-type"); got != "application/json" {
		t.Errorf("Content-Type: got %q, want application/json", got)
	}

	var decoded adminServeStatus
	if err := json.Unmarshal(rr.Body.Bytes(), &decoded); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if decoded.ModelPath != snap.ModelPath {
		t.Errorf("ModelPath: got %q want %q", decoded.ModelPath, snap.ModelPath)
	}
	if decoded.Runtime != "metal" {
		t.Errorf("Runtime: got %q want metal", decoded.Runtime)
	}
	if decoded.Config.CacheMode != "fp16" {
		t.Errorf("Config.CacheMode: got %q want fp16", decoded.Config.CacheMode)
	}
}

// TestAdminServeStatusHandler_NonGETRejected — POST / PUT / DELETE
// must be refused with 405 (the endpoint is a snapshot, never mutated
// via this route).
func TestAdminServeStatusHandler_NonGETRejected(t *testing.T) {
	h := adminServeStatusHandler(adminServeStatus{})
	for _, method := range []string{http.MethodPost, http.MethodPut, http.MethodDelete, http.MethodPatch} {
		t.Run(method, func(t *testing.T) {
			req := httptest.NewRequest(method, "/v1/admin/serve/status", nil)
			rr := httptest.NewRecorder()
			h.ServeHTTP(rr, req)
			if rr.Code != http.StatusMethodNotAllowed {
				t.Errorf("method %s: got %d, want 405", method, rr.Code)
			}
		})
	}
}
