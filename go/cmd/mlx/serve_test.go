// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
	mlx "dappco.re/go/mlx"
)

// Without a drafter there is no speculative lane to explain, so the notice is
// empty and serve prints nothing extra — the reactive default changes NOTHING
// for drafterless models.
func TestSpeculativeServeNotice_NoDraftIsSilent_Good(t *testing.T) {
	if got := speculativeServeNotice(mlx.DraftDetection{}, 0); got != "" {
		t.Fatalf("speculativeServeNotice(none) = %q, want empty (no drafter → no notice)", got)
	}
}

// With a drafter engaged the operator MUST see the ACTIVE pair: which drafter,
// which ladder rung chose it, and the draft block the verify forwards run.
func TestSpeculativeServeNotice_ActivePairReported_Good(t *testing.T) {
	det := mlx.DraftDetection{
		Source:    mlx.DraftSourceAssistantDir,
		DraftPath: "/models/gemma-4-31b/assistant",
		Note:      "auto-detected assistant/ beside the weights",
	}
	got := speculativeServeNotice(det, 0)
	lower := strings.ToLower(got)
	for _, want := range []string{"active", "/models/gemma-4-31b/assistant", "block 5", "auto-detected"} {
		if !strings.Contains(lower, strings.ToLower(want)) {
			t.Fatalf("notice %q missing %q — the boot line must report the ACTIVE pair + block", got, want)
		}
	}
	if got := speculativeServeNotice(det, 6); !strings.Contains(got, "block 6") {
		t.Fatalf("notice %q missing explicit block 6", got)
	}
}

// resolveServeDraft flag semantics: 'auto' runs the ladder, ” disables, an
// explicit path forces — and -draft-detect=false stands the ladder down while
// still honouring an explicit path.
func TestResolveServeDraft_FlagSemantics_Good(t *testing.T) {
	model := t.TempDir()
	if err := os.WriteFile(filepath.Join(model, "config.json"), []byte(`{"model_type":"gemma4_text"}`), 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}
	assistant := filepath.Join(model, "assistant")
	if err := os.MkdirAll(assistant, 0o755); err != nil {
		t.Fatalf("mkdir assistant: %v", err)
	}
	if err := os.WriteFile(filepath.Join(assistant, "config.json"), []byte(`{"model_type":"gemma4_assistant"}`), 0o644); err != nil {
		t.Fatalf("write assistant config: %v", err)
	}
	if err := os.WriteFile(filepath.Join(assistant, "model.safetensors"), []byte("stub"), 0o644); err != nil {
		t.Fatalf("write assistant weights stub: %v", err)
	}

	if det := resolveServeDraft(model, "auto", true); det.Source != mlx.DraftSourceAssistantDir {
		t.Fatalf("auto detection = %+v, want assistant-dir", det)
	}
	if det := resolveServeDraft(model, "", true); det.Active() {
		t.Fatalf("--draft '' detection = %+v, want disabled", det)
	}
	if det := resolveServeDraft(model, "auto", false); det.Active() {
		t.Fatalf("-draft-detect=false detection = %+v, want disabled", det)
	}
	if det := resolveServeDraft(model, "/forced/path", false); det.Source != mlx.DraftSourceFlag || det.DraftPath != "/forced/path" {
		t.Fatalf("explicit detection = %+v, want the flag honoured even with detect off", det)
	}
}

// Native MTP currently accepts explicit native-loadable draft models; assistant
// auto-detection waits for a native assistant-only loader so --native does not
// fail just because an assistant/ directory sits beside the target.
func TestResolveNativeServeDraft_ExplicitOnly_Good(t *testing.T) {
	model := t.TempDir()
	if det := resolveNativeServeDraft(model, "auto"); det.Active() {
		t.Fatalf("native auto detection = %+v, want inactive until native assistant loader exists", det)
	}
	if det := resolveNativeServeDraft(model, ""); det.Active() {
		t.Fatalf("native empty draft detection = %+v, want inactive", det)
	}
	if det := resolveNativeServeDraft(model, "/draft/native"); det.Source != mlx.DraftSourceFlag || det.DraftPath != "/draft/native" {
		t.Fatalf("native explicit detection = %+v, want explicit flag path", det)
	}
}

// reloadLoadOpts overlays per-reload options on top of the auto-tuned boot
// options, last-wins, so a reload that carries only ContextLength keeps the
// boot baseline (Mantis #1785 F-7 N-7).
func TestHotSwapResolver_ReloadLoadOptsOverlay_Good(t *testing.T) {
	base := []mlx.LoadOption{mlx.WithContextLength(4096), mlx.WithKVCacheMode("paged")}
	r := newHotSwapResolver("/boot", "", 0, base)
	merged := r.reloadLoadOpts([]mlx.LoadOption{mlx.WithContextLength(8192)})
	if len(merged) != len(base)+1 {
		t.Fatalf("merged opts = %d, want base(%d)+overlay(1) — overlay appends, never replaces", len(merged), len(base))
	}
	// A nil overlay returns exactly the boot options.
	if got := r.reloadLoadOpts(nil); len(got) != len(base) {
		t.Fatalf("nil-overlay opts = %d, want the %d boot options", len(got), len(base))
	}
}

// setOnLoad / setDraftDetect are the pre-first-load wiring hooks; reloadDetection
// stands down when draftDetect was never armed.
func TestHotSwapResolver_WiringHooks_Good(t *testing.T) {
	r := newHotSwapResolver("/boot", "", 0, nil)
	// Detection not armed → reloadDetection returns an inactive decision.
	if det := r.reloadDetection("/some/target"); det.Active() {
		t.Fatalf("reloadDetection before arming = %+v, want inactive", det)
	}
	// Arm detection (disabled) and register an onLoad hook — neither should
	// panic, and both are accepted before the first load.
	r.setOnLoad(func(inference.TextModel) {})
	r.setDraftDetect(mlx.DraftDetectOptions{Disabled: true})
	if r.onLoad == nil {
		t.Fatal("setOnLoad did not register the hook")
	}
	// Armed-but-disabled still stands down for a non-Gemma-4 target.
	if det := r.reloadDetection("/some/target"); det.Active() {
		t.Fatalf("reloadDetection armed-disabled = %+v, want inactive", det)
	}
}

// resolveServeDraftBlock: with no drafter detected (or an explicit block) the
// profile lookup is skipped and the flag block is returned with no note.
func TestResolveServeDraftBlock_NoDrafterSkipsProfile_Good(t *testing.T) {
	// Inactive detection → returns the flag block (0) with no note.
	block, note := resolveServeDraftBlock(context.Background(), mlx.DraftDetection{}, "/m", 0, false, t.TempDir())
	if block != 0 || note != "" {
		t.Fatalf("inactive detection = (%d, %q), want (0, \"\")", block, note)
	}
	// Explicit flag block short-circuits even when detection is active.
	active := mlx.DraftDetection{Source: mlx.DraftSourceFlag, DraftPath: "/d"}
	block, note = resolveServeDraftBlock(context.Background(), active, "/m", 5, false, t.TempDir())
	if block != 5 || note != "" {
		t.Fatalf("explicit flag = (%d, %q), want (5, \"\")", block, note)
	}
	// -no-auto-profile stands the lookup down too.
	block, _ = resolveServeDraftBlock(context.Background(), active, "/m", 0, true, t.TempDir())
	if block != 0 {
		t.Fatalf("no-auto-profile block = %d, want 0 (lookup skipped)", block)
	}
}

// serve --print-admin-token reveals (and generates on first use) the admin
// token, then exits 0 WITHOUT binding a listener — HOME is redirected so the
// real ~/Lethean token is never touched.
func TestRunServe_PrintAdminToken_Good(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{"serve", "--print-admin-token"}, stdout, stderr)
	if code != 0 {
		t.Fatalf("exit = %d, want 0; stderr=%q", code, stderr.String())
	}
	if !strings.Contains(stderr.String(), "admin token") {
		t.Fatalf("stderr = %q, want the admin token line", stderr.String())
	}
}

// serve --rotate-admin-token regenerates + prints the token and exits 0,
// binding no listener.
func TestRunServe_RotateAdminToken_Good(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{"serve", "--rotate-admin-token"}, stdout, stderr)
	if code != 0 {
		t.Fatalf("exit = %d, want 0; stderr=%q", code, stderr.String())
	}
	if !strings.Contains(stderr.String(), "rotated") {
		t.Fatalf("stderr = %q, want the rotated-token line", stderr.String())
	}
}

// serve with an unknown -kv-cache mode is a usage error (exit 2) before any
// listener binds.
func TestRunServe_UnknownKVCacheMode_Bad(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{"serve", "-kv-cache", "not-a-mode"}, stdout, stderr)
	if code != 2 {
		t.Fatalf("exit = %d, want 2 (unknown kv-cache mode)", code)
	}
	if !strings.Contains(stderr.String(), "unknown -kv-cache mode") {
		t.Fatalf("stderr = %q, want the unknown-mode notice", stderr.String())
	}
}

// serve model-less boots the listener + admin surface and shuts down cleanly
// when the context is cancelled (the graceful-shutdown return-0 path). HOME is
// redirected; the listen address is loopback :0 so the kernel assigns a free
// port and nothing collides with a real serve on :36911.
func TestRunServe_ModellessGracefulShutdown_Good(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	ctx, cancel := context.WithCancel(context.Background())
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	done := make(chan int, 1)
	go func() {
		done <- runCommand(ctx, []string{
			"serve", "-addr", "127.0.0.1:0", "-shutdown-timeout", "2s", "-state-conversations=false",
		}, stdout, stderr)
	}()
	// Give the listener a beat to bind, then cancel → graceful shutdown.
	time.Sleep(150 * time.Millisecond)
	cancel()
	select {
	case code := <-done:
		if code != 0 {
			t.Fatalf("model-less serve exit = %d, want 0 on graceful shutdown; stderr=%q", code, stderr.String())
		}
	case <-time.After(10 * time.Second):
		t.Fatal("serve did not shut down within 10s of context cancel")
	}
	if !strings.Contains(stderr.String(), "model-less") {
		t.Fatalf("stderr = %q, want the model-less boot notice", stderr.String())
	}
}
