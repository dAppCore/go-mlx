// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"dappco.re/go/inference"
	mlx "dappco.re/go/mlx"
)

// fakeResolver — test seam for the reload handler. We don't load
// real metal models in tests.
type fakeResolver struct {
	current       string
	replaceCalls  int
	replaceErr    error
	replaceNewPath string
}

func (f *fakeResolver) CurrentPath() string { return f.current }
func (f *fakeResolver) Replace(newPath string, _ []mlx.LoadOption) (*loadedModel, string, error) {
	f.replaceCalls++
	if f.replaceErr != nil {
		return nil, "", f.replaceErr
	}
	prev := &loadedModel{modelPath: f.current}
	f.current = newPath
	if f.replaceNewPath != "" {
		f.current = f.replaceNewPath
	}
	return prev, f.current, nil
}

// reloadHandlerForTest mirrors adminReloadHandler but takes the
// adminReloadServer interface so we can wire fakeResolver. Kept
// here rather than exporting the production handler's parameter
// list because the production wire-up always carries a concrete
// *hotSwapResolver — the test seam is only for isolated runs.
func reloadHandlerForTest(srv adminReloadServer, stderr io.Writer) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		var req adminReloadRequest
		if err := readJSONBody(r, &req); err != nil {
			http.Error(w, "invalid body: "+err.Error(), http.StatusBadRequest)
			return
		}
		from := srv.CurrentPath()
		modelName := strings.TrimSpace(req.Model)
		if modelName == "" {
			adminReloadDeny(w, stderr, from, modelName, "model required")
			return
		}
		if req.Confirmation == "" {
			adminReloadDeny(w, stderr, from, modelName, "confirmation required (machine_hash from /v1/admin/machine)")
			return
		}
		expected, err := currentMachineProfileHash(r.Context())
		if err != nil {
			adminReloadFail(w, stderr, from, modelName, "machine hash unavailable: "+err.Error(), http.StatusInternalServerError)
			return
		}
		if req.Confirmation != expected {
			adminReloadDeny(w, stderr, from, modelName, "confirmation mismatch")
			return
		}
		toPath, err := resolveModelNameToPath(modelName)
		if err != nil {
			adminReloadDeny(w, stderr, from, modelName, err.Error())
			return
		}
		_, newPath, err := srv.Replace(toPath, nil)
		if err != nil {
			adminReloadFail(w, stderr, from, modelName, "load failed: "+err.Error(), http.StatusInternalServerError)
			return
		}
		writeJSON(w, http.StatusOK, adminReloadResponse{
			Status: "ok", From: from, To: newPath,
		})
	}
}

// withModelsDir creates a temp ~/Lethean/data/models layout, points
// the HOME env at the temp root, and returns a cleanup. Tests use
// this to populate fake models so resolveModelNameToPath can find
// them.
func withModelsDir(t *testing.T, modelNames ...string) (root string, cleanup func()) {
	t.Helper()
	tmp := t.TempDir()
	prevHome := os.Getenv("HOME")
	_ = os.Setenv("HOME", tmp)
	root = filepath.Join(tmp, "Lethean", "data", "models")
	for _, name := range modelNames {
		dir := filepath.Join(root, name)
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatalf("mkdir %s: %v", dir, err)
		}
		// Write a minimal .sha256 so resolveModelNameToPath accepts.
		manifest := filepath.Join(dir, shaManifestFilename)
		if err := os.WriteFile(manifest, []byte("deadbeef  weights.bin\n"), 0o600); err != nil {
			t.Fatalf("write manifest: %v", err)
		}
	}
	return root, func() { _ = os.Setenv("HOME", prevHome) }
}

// TestResolveModelNameToPath_RejectsTraversal — `..` / `/` / leading
// `.` in the model name must be rejected before any filesystem
// lookup. Path-injection class per §4.F-7.1.
// TestPathWithinDir guards Mantis #1780 (F-7 N-2): containment uses
// filepath.Rel semantics, not a raw byte prefix, so a sibling dir that
// merely shares a prefix is correctly rejected while a real child passes.
func TestPathWithinDir_Good(t *testing.T) {
	cases := []struct {
		root, target string
		want         bool
	}{
		{"/m/models", "/m/models", true},
		{"/m/models", "/m/models/gemma", true},
		{"/m/models", "/m/models/a/b/c", true},
		{"/m/models", "/m/models-evil", false},   // sibling sharing prefix
		{"/m/models", "/m/models-evil/x", false}, // sibling subtree
		{"/m/models", "/etc/passwd", false},      // outside tree
		{"/m/models", "/m", false},               // parent
	}
	for _, c := range cases {
		if got := pathWithinDir(c.root, c.target); got != c.want {
			t.Errorf("pathWithinDir(%q, %q) = %v, want %v", c.root, c.target, got, c.want)
		}
	}
}

func TestResolveModelNameToPath_RejectsTraversal(t *testing.T) {
	_, cleanup := withModelsDir(t)
	defer cleanup()

	cases := []string{
		"../etc/passwd",
		"foo/bar",
		".hidden",
		"..",
	}
	for _, name := range cases {
		_, err := resolveModelNameToPath(name)
		if err == nil {
			t.Errorf("expected error for %q, got nil", name)
		}
	}
}

// TestResolveModelNameToPath_RequiresManifest — a model dir without
// a .sha256 sidecar must be refused per §4.F-7.2 (no hot-swap to
// unverified-integrity models).
func TestResolveModelNameToPath_RequiresManifest(t *testing.T) {
	root, cleanup := withModelsDir(t)
	defer cleanup()

	// Build a dir with NO sha256 manifest.
	dir := filepath.Join(root, "bare-model")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	_, err := resolveModelNameToPath("bare-model")
	if err == nil {
		t.Fatal("expected error for model without .sha256 sidecar, got nil")
	}
	if !strings.Contains(err.Error(), shaManifestFilename) {
		t.Errorf("error should name the missing sidecar: %v", err)
	}
}

// TestResolveModelNameToPath_AcceptsValid — a properly-formed model
// (basename + .sha256) returns the resolved path. The resolved path
// goes through PathEvalSymlinks, so we compare via filepath.EvalSymlinks
// in the test too (macOS /var → /private/var would otherwise diverge).
func TestResolveModelNameToPath_AcceptsValid(t *testing.T) {
	root, cleanup := withModelsDir(t, "good-model")
	defer cleanup()
	rootResolved, err := filepath.EvalSymlinks(root)
	if err != nil {
		t.Fatalf("EvalSymlinks root: %v", err)
	}

	path, err := resolveModelNameToPath("good-model")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.HasPrefix(path, rootResolved) {
		t.Errorf("resolved path %q does not stay under root %q", path, rootResolved)
	}
}

// TestReadModelManifest_ParsesShasumFormat — manifest entries in
// the standard shasum -a 256 format must round-trip cleanly.
func TestReadModelManifest_ParsesShasumFormat(t *testing.T) {
	tmp := t.TempDir()
	dir := filepath.Join(tmp, "m")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	body := "" +
		"# comment line\n" +
		"\n" +
		"abc123  weights.bin\n" +
		"deadbeef  config.json\n"
	if err := os.WriteFile(filepath.Join(dir, shaManifestFilename), []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	m, err := readModelManifest(dir)
	if err != nil {
		t.Fatalf("unexpected: %v", err)
	}
	if got, want := m["weights.bin"], "abc123"; got != want {
		t.Errorf("weights.bin: got %q want %q", got, want)
	}
	if got, want := m["config.json"], "deadbeef"; got != want {
		t.Errorf("config.json: got %q want %q", got, want)
	}
	if len(m) != 2 {
		t.Errorf("expected 2 entries, got %d", len(m))
	}
}

// TestWriteAndReadModelManifest_Roundtrip — write+read must
// preserve every entry.
func TestWriteAndReadModelManifest_Roundtrip(t *testing.T) {
	tmp := t.TempDir()
	digests := map[string]string{
		"weights.bin": "a1b2c3",
		"config.json": "d4e5f6",
		"tok.json":    "fedcba",
	}
	if err := writeModelManifest(tmp, digests); err != nil {
		t.Fatalf("write: %v", err)
	}
	got, err := readModelManifest(tmp)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if len(got) != len(digests) {
		t.Errorf("got %d entries, want %d", len(got), len(digests))
	}
	for k, v := range digests {
		if got[k] != v {
			t.Errorf("%s: got %q want %q", k, got[k], v)
		}
	}
}

// TestAdminReload_MissingConfirmation — request without
// confirmation must 400 + audit. The handler must NOT reach the
// resolver.Replace call.
func TestAdminReload_MissingConfirmation(t *testing.T) {
	_, cleanup := withModelsDir(t, "good-model")
	defer cleanup()

	srv := &fakeResolver{current: "/initial/path"}
	h := reloadHandlerForTest(srv, io.Discard)

	body := `{"model":"good-model"}`
	req := httptest.NewRequest(http.MethodPost, "/v1/admin/serve/reload", strings.NewReader(body))
	w := httptest.NewRecorder()
	h(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("got status %d want 400", w.Code)
	}
	if srv.replaceCalls != 0 {
		t.Errorf("Replace was called %d times — expected 0 on missing-confirmation path", srv.replaceCalls)
	}
}

// TestAdminReload_ConfirmationMismatch — wrong confirmation MUST
// refuse without calling Replace.
func TestAdminReload_ConfirmationMismatch(t *testing.T) {
	_, cleanup := withModelsDir(t, "good-model")
	defer cleanup()

	srv := &fakeResolver{current: "/initial/path"}
	h := reloadHandlerForTest(srv, io.Discard)

	body := `{"model":"good-model","confirmation":"wrong-hash"}`
	req := httptest.NewRequest(http.MethodPost, "/v1/admin/serve/reload", strings.NewReader(body))
	w := httptest.NewRecorder()
	h(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("got status %d want 400", w.Code)
	}
	if srv.replaceCalls != 0 {
		t.Errorf("Replace called %d times on bad confirmation, want 0", srv.replaceCalls)
	}
}

// TestAdminReload_MethodGuard — non-POST methods refuse with 405.
func TestAdminReload_MethodGuard(t *testing.T) {
	srv := &fakeResolver{}
	h := reloadHandlerForTest(srv, io.Discard)
	req := httptest.NewRequest(http.MethodGet, "/v1/admin/serve/reload", nil)
	w := httptest.NewRecorder()
	h(w, req)
	if w.Code != http.StatusMethodNotAllowed {
		t.Errorf("GET got %d want 405", w.Code)
	}
}

// TestAdminReload_NameWithSlash — a model name with `/` MUST be
// refused before the manifest check (path-traversal class). Tested
// via direct call to resolveModelNameToPath rather than through the
// handler since the handler depends on a live machine hash that's
// flaky in CI; the gate logic is what we care about.
func TestAdminReload_NameWithSlash(t *testing.T) {
	_, cleanup := withModelsDir(t)
	defer cleanup()

	if _, err := resolveModelNameToPath("good/../etc"); err == nil {
		t.Fatal("expected refusal for name containing /, got nil")
	}
}

// TestHotSwapResolver_CurrentPathBeforeLoad — CurrentPath returns
// the boot path before any ResolveModel call.
func TestHotSwapResolver_CurrentPathBeforeLoad(t *testing.T) {
	r := newHotSwapResolver("/boot/path", nil)
	if r.CurrentPath() != "/boot/path" {
		t.Errorf("got %q want /boot/path", r.CurrentPath())
	}
}

// TestHotSwapResolver_ImplementsResolverInterface — the openai mux
// expects ResolveModel(ctx, name) → (TextModel, error). The bridge
// via openaiResolver() must satisfy that interface; this test pins
// the contract at compile time.
func TestHotSwapResolver_ImplementsResolverInterface(t *testing.T) {
	r := newHotSwapResolver("/p", nil)
	resolver := r.openaiResolver()
	if resolver == nil {
		t.Fatal("openaiResolver returned nil")
	}
	// We can't actually call ResolveModel without a real model; the
	// type check at compile time is the load-bearing assertion.
	var _ interface {
		ResolveModel(ctx context.Context, name string) (inference.TextModel, error)
	} = resolver
}

// TestAdminReloadResponse_JSONShape — the response JSON must carry
// the four documented fields with exact key names so external
// consumers can decode reliably.
func TestAdminReloadResponse_JSONShape(t *testing.T) {
	resp := adminReloadResponse{
		Status: "ok", From: "/a", To: "/b", LoadedAt: 12345,
	}
	b, err := json.Marshal(resp)
	if err != nil {
		t.Fatal(err)
	}
	got := string(b)
	for _, want := range []string{`"status":"ok"`, `"from_model_path":"/a"`, `"to_model_path":"/b"`, `"loaded_at_unix":12345`} {
		if !strings.Contains(got, want) {
			t.Errorf("response JSON missing %q in %q", want, got)
		}
	}
}
