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
