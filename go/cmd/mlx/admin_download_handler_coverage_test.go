// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	core "dappco.re/go"
)

// adminDownloadHandler GET ?job=<known> returns 200 with the job JSON — the
// poll-success branch (admin_download.go ~251). A job is seeded directly into
// the registry (same package) so no download runs.
func TestAdminDownloadHandler_GetKnownJob_Good(t *testing.T) {
	reg := newAdminDownloadRegistry(context.Background(), io.Discard)
	reg.mu.Lock()
	reg.jobs["download-seeded"] = &adminDownloadJob{
		ID:        "download-seeded",
		Status:    "running",
		Repo:      "mlx-community/gemma-4-e2b-it-4bit",
		Revision:  "main",
		StartedAt: time.Now().UTC(),
		BytesDone: 42,
	}
	reg.mu.Unlock()
	handler := adminDownloadHandler(reg, &fakeHFTreeAPI{})

	get := httptest.NewRequest(http.MethodGet, "/v1/admin/models/download?job=download-seeded", nil)
	rec := httptest.NewRecorder()
	handler(rec, get)
	if rec.Code != http.StatusOK {
		t.Fatalf("known-job status = %d, want 200; body=%s", rec.Code, rec.Body.String())
	}
	body := rec.Body.String()
	if !strings.Contains(body, `"id":"download-seeded"`) || !strings.Contains(body, `"status":"running"`) {
		t.Fatalf("job poll body = %s, want the seeded job fields", body)
	}
}

// adminDownloadHandler POST with a malformed allowlist file returns 500 — the
// allowlist-parse-error branch (admin_download.go ~275). HOME is redirected to
// a temp dir holding a corrupt allowed-models.json so loadAllowedModels errors.
func TestAdminDownloadHandler_AllowlistParseError_Bad(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	allowDir := core.PathJoin(home, "Lethean", "data")
	if r := core.MkdirAll(allowDir, 0o755); !r.OK {
		t.Fatalf("mkdir allow dir: %v", r.Value)
	}
	if r := core.WriteFile(core.PathJoin(allowDir, "allowed-models.json"), []byte("{not json"), 0o644); !r.OK {
		t.Fatalf("seed allowlist: %v", r.Value)
	}

	reg := newAdminDownloadRegistry(context.Background(), io.Discard)
	handler := adminDownloadHandler(reg, &fakeHFTreeAPI{})

	post := httptest.NewRequest(http.MethodPost, "/v1/admin/models/download",
		strings.NewReader(`{"repo":"mlx-community/gemma-4-e2b-it-4bit","revision":"main"}`))
	rec := httptest.NewRecorder()
	handler(rec, post)
	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("malformed-allowlist status = %d, want 500; body=%s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), "allowlist parse") {
		t.Fatalf("body = %s, want the allowlist-parse error", rec.Body.String())
	}
}
