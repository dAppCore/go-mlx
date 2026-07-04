// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	core "dappco.re/go"
)

// TestGenerateAdminToken_Format — fresh tokens must carry the
// lthn-mlx_ prefix and be 52 chars total (9 prefix + 43 base64url
// chars for 32 bytes of entropy).
func TestGenerateAdminToken_Format(t *testing.T) {
	tok, err := generateAdminToken()
	if err != nil {
		t.Fatalf("generate: %v", err)
	}
	if !core.HasPrefix(tok, "lthn-mlx_") {
		t.Errorf("missing lthn-mlx_ prefix: %q", tok)
	}
	if len(tok) != 52 {
		t.Errorf("unexpected length: got %d want 52 (token %q)", len(tok), tok)
	}
}

// TestGenerateAdminToken_Unique — two generates must produce
// different tokens (otherwise crypto/rand is broken).
func TestGenerateAdminToken_Unique(t *testing.T) {
	a, err := generateAdminToken()
	if err != nil {
		t.Fatalf("generate a: %v", err)
	}
	b, err := generateAdminToken()
	if err != nil {
		t.Fatalf("generate b: %v", err)
	}
	if a == b {
		t.Errorf("two generates produced identical tokens — entropy broken: %q", a)
	}
}

// TestEnsureAdminToken_GeneratesIfAbsent — first call on a fresh
// path generates + writes a token + reports generated=true.
func TestEnsureAdminToken_GeneratesIfAbsent(t *testing.T) {
	tmp := t.TempDir()
	path := core.PathJoin(tmp, "admin.token")

	tok, generated, err := ensureAdminToken(path)
	if err != nil {
		t.Fatalf("ensure: %v", err)
	}
	if !generated {
		t.Error("expected generated=true for fresh path")
	}
	if !core.HasPrefix(tok, "lthn-mlx_") {
		t.Errorf("unexpected token shape: %q", tok)
	}
}

// TestEnsureAdminToken_RoundTrips — second call on an existing path
// returns the same token + generated=false.
func TestEnsureAdminToken_RoundTrips(t *testing.T) {
	tmp := t.TempDir()
	path := core.PathJoin(tmp, "admin.token")

	first, _, err := ensureAdminToken(path)
	if err != nil {
		t.Fatalf("ensure 1: %v", err)
	}
	second, generated, err := ensureAdminToken(path)
	if err != nil {
		t.Fatalf("ensure 2: %v", err)
	}
	if generated {
		t.Error("expected generated=false on re-read of existing path")
	}
	if first != second {
		t.Errorf("token changed across reads: %q vs %q", first, second)
	}
}

// TestRequireBearerOnAdmin_DeniesNoAuth — admin path without Bearer
// header must 401, never reach the wrapped handler.
func TestRequireBearerOnAdmin_DeniesNoAuth(t *testing.T) {
	var innerCalled bool
	inner := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		innerCalled = true
	})
	h := requireBearerOnAdmin(inner, "lthn-mlx_test", io.Discard)

	req := httptest.NewRequest(http.MethodGet, "/v1/admin/machine", nil)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusUnauthorized {
		t.Errorf("expected 401, got %d", rr.Code)
	}
	if innerCalled {
		t.Error("inner handler reached without auth — middleware bypass")
	}
	if got := rr.Header().Get("www-authenticate"); got != `Bearer realm="lthn-mlx-admin"` {
		t.Errorf("WWW-Authenticate: got %q", got)
	}
}

// TestRequireBearerOnAdmin_DeniesWrongToken — wrong Bearer token
// must 401, never reach the wrapped handler.
func TestRequireBearerOnAdmin_DeniesWrongToken(t *testing.T) {
	var innerCalled bool
	inner := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		innerCalled = true
	})
	h := requireBearerOnAdmin(inner, "lthn-mlx_correct", io.Discard)

	req := httptest.NewRequest(http.MethodGet, "/v1/admin/machine", nil)
	req.Header.Set("Authorization", "Bearer lthn-mlx_wrong")
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusUnauthorized {
		t.Errorf("expected 401, got %d", rr.Code)
	}
	if innerCalled {
		t.Error("inner handler reached with wrong token — middleware bypass")
	}
}

// TestRequireBearerOnAdmin_AcceptsCorrectToken — correct Bearer
// token must pass through to the wrapped handler.
func TestRequireBearerOnAdmin_AcceptsCorrectToken(t *testing.T) {
	var innerCalled bool
	inner := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		innerCalled = true
		w.WriteHeader(http.StatusOK)
	})
	h := requireBearerOnAdmin(inner, "lthn-mlx_test", io.Discard)

	req := httptest.NewRequest(http.MethodGet, "/v1/admin/machine", nil)
	req.Header.Set("Authorization", "Bearer lthn-mlx_test")
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)

	if !innerCalled {
		t.Error("inner handler not reached with correct token")
	}
	if rr.Code != http.StatusOK {
		t.Errorf("expected 200, got %d (body: %s)", rr.Code, rr.Body.String())
	}
}

// TestRequireBearerOnAdmin_AllowsInferencePath — non-admin paths
// (chat completions, etc.) must pass through without auth.
func TestRequireBearerOnAdmin_AllowsInferencePath(t *testing.T) {
	for _, path := range []string{
		"/v1/chat/completions",
		"/v1/completions",
		"/v1/messages",
		"/api/chat",
		"/v1/models",
		"/v1/health",
		"/",
	} {
		t.Run(path, func(t *testing.T) {
			var innerCalled bool
			inner := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				innerCalled = true
				w.WriteHeader(http.StatusOK)
			})
			h := requireBearerOnAdmin(inner, "lthn-mlx_test", io.Discard)

			req := httptest.NewRequest(http.MethodPost, path, nil)
			rr := httptest.NewRecorder()
			h.ServeHTTP(rr, req)

			if !innerCalled {
				t.Errorf("inference path %q blocked by admin auth", path)
			}
		})
	}
}

// TestRequireBearerOnAdmin_AdminNoSlash — /v1/admin (no trailing
// slash) is NOT covered by the prefix /v1/admin/ — passes through.
// In production composition, the ServeMux 301s it to /v1/admin/
// which the second request then auth-checks. Either way, no bypass.
func TestRequireBearerOnAdmin_AdminNoSlash(t *testing.T) {
	var innerCalled bool
	inner := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		innerCalled = true
		w.WriteHeader(http.StatusNotFound) // inner can 404 — point is auth wasn't required
	})
	h := requireBearerOnAdmin(inner, "lthn-mlx_test", io.Discard)

	req := httptest.NewRequest(http.MethodGet, "/v1/admin", nil)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)

	if !innerCalled {
		t.Error("inner handler not reached for /v1/admin (no slash) — middleware over-broad")
	}
}
