// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	mlx "dappco.re/go/mlx"
)

// mockMachineHash wires the discovery seam to report a fixed machine hash, so
// the reload handler's confirm-machine gate is deterministic.
func mockMachineHash(t *testing.T, hash string) {
	t.Helper()
	origDiscover := runDiscoverLocalRuntime
	origDevice := runGetDeviceInfo
	t.Cleanup(func() { runDiscoverLocalRuntime = origDiscover; runGetDeviceInfo = origDevice })
	runGetDeviceInfo = func() mlx.DeviceInfo { return mlx.DeviceInfo{Architecture: "apple9"} }
	runDiscoverLocalRuntime = func(_ context.Context, _ mlx.LocalDiscoveryConfig) (inference.MachineDiscoveryReport, error) {
		return inference.MachineDiscoveryReport{Labels: map[string]string{"machine_hash": hash}}, nil
	}
}

func postReload(t *testing.T, h http.HandlerFunc, body string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, "/v1/admin/serve/reload", strings.NewReader(body))
	req.RemoteAddr = "10.0.0.1:1234"
	w := httptest.NewRecorder()
	h(w, req)
	return w
}

// The real reload handler rejects an invalid JSON body (400).
func TestRealAdminReload_InvalidBody_Bad(t *testing.T) {
	h := adminReloadHandler(newHotSwapResolver("/boot", "", 0, nil), io.Discard)
	if w := postReload(t, h, "{not json"); w.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400 (invalid body)", w.Code)
	}
}

// The real reload handler denies (400) when neither model nor model_path is set.
func TestRealAdminReload_MissingTarget_Bad(t *testing.T) {
	h := adminReloadHandler(newHotSwapResolver("/boot", "", 0, nil), io.Discard)
	w := postReload(t, h, `{"confirm_machine":"abc"}`)
	if w.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400 (missing target)", w.Code)
	}
	if !strings.Contains(w.Body.String(), "model or model_path required") {
		t.Fatalf("body = %q, want the missing-target reason", w.Body.String())
	}
}

// The real reload handler denies (400) when confirm_machine is absent.
func TestRealAdminReload_MissingConfirm_Bad(t *testing.T) {
	h := adminReloadHandler(newHotSwapResolver("/boot", "", 0, nil), io.Discard)
	w := postReload(t, h, `{"model":"some-model"}`)
	if w.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400 (missing confirm)", w.Code)
	}
	if !strings.Contains(w.Body.String(), "confirm_machine required") {
		t.Fatalf("body = %q, want the confirm-required reason", w.Body.String())
	}
}

// The real reload handler returns 500 (adminReloadFail) when the machine hash
// is unavailable — the discovery seam errors.
func TestRealAdminReload_MachineHashUnavailable_Bad(t *testing.T) {
	origDiscover := runDiscoverLocalRuntime
	origDevice := runGetDeviceInfo
	t.Cleanup(func() { runDiscoverLocalRuntime = origDiscover; runGetDeviceInfo = origDevice })
	runGetDeviceInfo = func() mlx.DeviceInfo { return mlx.DeviceInfo{} }
	runDiscoverLocalRuntime = func(_ context.Context, _ mlx.LocalDiscoveryConfig) (inference.MachineDiscoveryReport, error) {
		return inference.MachineDiscoveryReport{}, core.NewError("no device")
	}
	h := adminReloadHandler(newHotSwapResolver("/boot", "", 0, nil), io.Discard)
	w := postReload(t, h, `{"model":"m","confirm_machine":"abc"}`)
	if w.Code != http.StatusInternalServerError {
		t.Fatalf("status = %d, want 500 (machine hash unavailable)", w.Code)
	}
	if !strings.Contains(w.Body.String(), "machine hash unavailable") {
		t.Fatalf("body = %q, want the hash-unavailable reason", w.Body.String())
	}
}

// The real reload handler denies (400) when confirm_machine does not match the
// live machine hash.
func TestRealAdminReload_ConfirmMismatch_Bad(t *testing.T) {
	mockMachineHash(t, "real-hash")
	h := adminReloadHandler(newHotSwapResolver("/boot", "", 0, nil), io.Discard)
	w := postReload(t, h, `{"model":"m","confirm_machine":"wrong-hash"}`)
	if w.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400 (confirm mismatch)", w.Code)
	}
	if !strings.Contains(w.Body.String(), "mismatch") {
		t.Fatalf("body = %q, want the mismatch reason", w.Body.String())
	}
}

// With a matching confirm, an unresolvable basename target is denied (400) at
// the name-resolution gate (no such model dir / no sidecar).
func TestRealAdminReload_UnresolvableName_Bad(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	mockMachineHash(t, "real-hash")
	h := adminReloadHandler(newHotSwapResolver("/boot", "", 0, nil), io.Discard)
	w := postReload(t, h, `{"model":"no-such-model","confirm_machine":"real-hash"}`)
	if w.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400 (unresolvable name); body=%q", w.Code, w.Body.String())
	}
}

// With a matching confirm and a model_path that escapes the standard model dir,
// the bind gate denies (400).
func TestRealAdminReload_PathEscape_Bad(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	mockMachineHash(t, "real-hash")
	h := adminReloadHandler(newHotSwapResolver("/boot", "", 0, nil), io.Discard)
	w := postReload(t, h, `{"model_path":"/etc/passwd","confirm_machine":"real-hash"}`)
	if w.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400 (path escape); body=%q", w.Code, w.Body.String())
	}
}

// With everything valid down to the load, a model dir that exists with a
// sidecar but is not a loadable model trips the Replace load-failed path
// (adminReloadFail 500). The model lives under the standard model dir.
func TestRealAdminReload_LoadFailed_Bad(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	mockMachineHash(t, "real-hash")
	// Create a model dir under ~/Lethean/data/models/<name> with the required
	// .sha256 sidecar but no real weights, so resolution passes and Replace's
	// load fails.
	modelDir := core.PathJoin(standardModelDir(), "broken-model")
	if r := core.MkdirAll(modelDir, 0o755); !r.OK {
		t.Fatalf("mkdir model: %v", r.Value)
	}
	if r := core.WriteFile(core.PathJoin(modelDir, ".sha256"), []byte("deadbeef  model.safetensors\n"), 0o644); !r.OK {
		t.Fatalf("write sidecar: %v", r.Value)
	}
	h := adminReloadHandler(newHotSwapResolver("/boot", "", 0, nil), io.Discard)
	w := postReload(t, h, `{"model":"broken-model","confirm_machine":"real-hash"}`)
	if w.Code != http.StatusInternalServerError {
		t.Fatalf("status = %d, want 500 (load failed); body=%q", w.Code, w.Body.String())
	}
	if !strings.Contains(w.Body.String(), "load failed") {
		t.Fatalf("body = %q, want the load-failed reason", w.Body.String())
	}
}
