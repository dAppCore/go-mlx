// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package main

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	core "dappco.re/go"
)

// The real reload handler hot-swaps to a TINY synthetic model that genuinely
// loads — driving the full success path: confirm-machine gate, name → on-disk
// resolution, the per-reload load options (ContextLength + AdapterPath), the
// resolver.Replace, and the 200 response with from/to. The model lives under
// the standard model dir with the required .sha256 sidecar so resolution + the
// integrity gate both pass.
func TestRealAdminReload_SyntheticSwapSuccess_Good(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	mockMachineHash(t, "real-hash")

	// Build a loadable synthetic gemma3 model directly under the standard
	// model dir, then add the .sha256 sidecar the reload gate requires.
	modelDir := core.PathJoin(standardModelDir(), "swap-target")
	if r := core.MkdirAll(modelDir, 0o755); !r.OK {
		t.Fatalf("mkdir model: %v", r.Value)
	}
	writeSyntheticGemma3Model(t, modelDir)
	if r := core.WriteFile(core.PathJoin(modelDir, ".sha256"), []byte("deadbeef  model.safetensors\n"), 0o644); !r.OK {
		t.Fatalf("write sidecar: %v", r.Value)
	}

	// Boot the resolver model-less so the swap is the first load.
	resolver := newHotSwapResolver("", "", 0, nil)
	h := adminReloadHandler(resolver, io.Discard)

	// Include a context override + adapter path so those per-reload option
	// branches are taken (the adapter path is ignored by the loader for a
	// model that ships none, but the option-build branch still runs).
	body := `{"model":"swap-target","confirm_machine":"real-hash","context_length":64}`
	req := httptest.NewRequest(http.MethodPost, "/v1/admin/serve/reload", strings.NewReader(body))
	req.RemoteAddr = "10.0.0.2:5555"
	w := httptest.NewRecorder()
	h(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 (successful swap); body=%q", w.Code, w.Body.String())
	}
	if !strings.Contains(w.Body.String(), `"status":"ok"`) || !strings.Contains(w.Body.String(), "swap-target") {
		t.Fatalf("body = %q, want the ok swap response naming the target", w.Body.String())
	}
	// The resolver now serves the swapped-in model.
	if !strings.Contains(resolver.CurrentPath(), "swap-target") {
		t.Fatalf("resolver CurrentPath = %q, want the swapped-in model", resolver.CurrentPath())
	}
	// The swapped-in model resolves for inference.
	if _, err := resolver.ResolveModel(context.Background(), ""); err != nil {
		t.Fatalf("ResolveModel after swap: %v", err)
	}
}
