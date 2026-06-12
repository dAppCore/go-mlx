// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

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

// resolveServeDraft flag semantics: 'auto' runs the ladder, '' disables, an
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
