// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"os"
	"path/filepath"
	"testing"
)

// Detection is path-shape only — these tests build the conventions in temp
// dirs and never open weights.

func writeDetectModelDir(t *testing.T, dir, modelType string) {
	t.Helper()
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatalf("mkdir %s: %v", dir, err)
	}
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{"model_type": "`+modelType+`"}`), 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors"), []byte("stub"), 0o644); err != nil {
		t.Fatalf("write safetensors stub: %v", err)
	}
}

func TestDraftDetect_AssistantDir_Good(t *testing.T) {
	model := t.TempDir()
	writeDetectModelDir(t, model, "gemma4_text")
	writeDetectModelDir(t, filepath.Join(model, "assistant"), "gemma4_assistant")

	det := DetectGemma4DraftPath(model, "", DraftDetectOptions{})
	if !det.Active() || det.Source != DraftSourceAssistantDir {
		t.Fatalf("detection = %+v, want assistant-dir", det)
	}
	if det.DraftPath != filepath.Join(model, "assistant") {
		t.Fatalf("draft path = %s, want %s", det.DraftPath, filepath.Join(model, "assistant"))
	}
}

func TestDraftDetect_PairBundleSibling_Good(t *testing.T) {
	bundle := t.TempDir()
	target := filepath.Join(bundle, "target")
	writeDetectModelDir(t, target, "gemma4_text")
	writeDetectModelDir(t, filepath.Join(bundle, "assistant"), "gemma4_assistant")
	// An mtplx_pair.json at the bundle root is IGNORED — directory shape decides.
	if err := os.WriteFile(filepath.Join(bundle, "mtplx_pair.json"), []byte(`{"layout":{}}`), 0o644); err != nil {
		t.Fatalf("write pair manifest: %v", err)
	}

	det := DetectGemma4DraftPath(target, "", DraftDetectOptions{})
	if !det.Active() || det.Source != DraftSourceSiblingAssistant {
		t.Fatalf("detection = %+v, want the target/ + assistant/ bundle", det)
	}
	if det.DraftPath != filepath.Join(bundle, "assistant") {
		t.Fatalf("draft path = %s, want bundle assistant", det.DraftPath)
	}
}

func TestDraftDetect_MTPDirAndSiblingGGUF_Good(t *testing.T) {
	model := t.TempDir()
	writeDetectModelDir(t, model, "gemma4_text")
	if err := os.MkdirAll(filepath.Join(model, "MTP"), 0o755); err != nil {
		t.Fatalf("mkdir MTP: %v", err)
	}
	gguf := filepath.Join(model, "MTP", "gemma-4-31B-it-Q8_0-MTP.gguf")
	if err := os.WriteFile(gguf, []byte("stub"), 0o644); err != nil {
		t.Fatalf("write gguf stub: %v", err)
	}
	det := DetectGemma4DraftPath(model, "", DraftDetectOptions{})
	if !det.Active() || det.Source != DraftSourceMTPDir || det.DraftPath != gguf {
		t.Fatalf("detection = %+v, want the MTP/ gguf %s", det, gguf)
	}

	// Sibling convention: mtp-*.gguf beside the weights (no MTP/ dir).
	model2 := t.TempDir()
	writeDetectModelDir(t, model2, "gemma4_text")
	sibling := filepath.Join(model2, "mtp-gemma-4-31B.gguf")
	if err := os.WriteFile(sibling, []byte("stub"), 0o644); err != nil {
		t.Fatalf("write sibling gguf: %v", err)
	}
	det = DetectGemma4DraftPath(model2, "", DraftDetectOptions{})
	if !det.Active() || det.Source != DraftSourceMTPSibling || det.DraftPath != sibling {
		t.Fatalf("detection = %+v, want the sibling gguf %s", det, sibling)
	}
}

func TestDraftDetect_Precedence_Good(t *testing.T) {
	// All conventions present at once: the explicit flag wins over everything;
	// without a flag, assistant/ outranks MTP/.
	model := t.TempDir()
	writeDetectModelDir(t, model, "gemma4_text")
	writeDetectModelDir(t, filepath.Join(model, "assistant"), "gemma4_assistant")
	if err := os.MkdirAll(filepath.Join(model, "MTP"), 0o755); err != nil {
		t.Fatalf("mkdir MTP: %v", err)
	}
	if err := os.WriteFile(filepath.Join(model, "MTP", "x-MTP.gguf"), []byte("stub"), 0o644); err != nil {
		t.Fatalf("write gguf: %v", err)
	}

	det := DetectGemma4DraftPath(model, "/explicit/drafter", DraftDetectOptions{})
	if det.Source != DraftSourceFlag || det.DraftPath != "/explicit/drafter" {
		t.Fatalf("detection = %+v, want the explicit flag to win", det)
	}
	det = DetectGemma4DraftPath(model, "", DraftDetectOptions{})
	if det.Source != DraftSourceAssistantDir {
		t.Fatalf("detection = %+v, want assistant/ to outrank MTP/", det)
	}
}

func TestDraftDetect_NonGemma4Unaffected_Bad(t *testing.T) {
	model := t.TempDir()
	writeDetectModelDir(t, model, "qwen3")
	writeDetectModelDir(t, filepath.Join(model, "assistant"), "gemma4_assistant")

	if det := DetectGemma4DraftPath(model, "", DraftDetectOptions{}); det.Active() {
		t.Fatalf("detection = %+v, want non-gemma4 targets untouched", det)
	}
	// The explicit flag still wins regardless of family — the operator said so.
	if det := DetectGemma4DraftPath(model, "/forced", DraftDetectOptions{}); det.Source != DraftSourceFlag {
		t.Fatalf("detection = %+v, want the explicit flag honoured", det)
	}
}

func TestDraftDetect_Disabled_Bad(t *testing.T) {
	model := t.TempDir()
	writeDetectModelDir(t, model, "gemma4_text")
	writeDetectModelDir(t, filepath.Join(model, "assistant"), "gemma4_assistant")

	if det := DetectGemma4DraftPath(model, "", DraftDetectOptions{Disabled: true}); det.Active() {
		t.Fatalf("detection = %+v, want the typed disable to stand the ladder down", det)
	}
}

func TestDraftDetect_DegenerateShapes_Ugly(t *testing.T) {
	// assistant/ without weights is not a drafter.
	model := t.TempDir()
	writeDetectModelDir(t, model, "gemma4_text")
	if err := os.MkdirAll(filepath.Join(model, "assistant"), 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(model, "assistant", "config.json"), []byte(`{"model_type":"gemma4_assistant"}`), 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}
	if det := DetectGemma4DraftPath(model, "", DraftDetectOptions{}); det.Active() {
		t.Fatalf("detection = %+v, want weightless assistant/ ignored", det)
	}

	// Two GGUFs in MTP/ is ambiguous — stand down rather than guess.
	if err := os.MkdirAll(filepath.Join(model, "MTP"), 0o755); err != nil {
		t.Fatalf("mkdir MTP: %v", err)
	}
	for _, name := range []string{"a-MTP.gguf", "b-MTP.gguf"} {
		if err := os.WriteFile(filepath.Join(model, "MTP", name), []byte("stub"), 0o644); err != nil {
			t.Fatalf("write gguf: %v", err)
		}
	}
	if det := DetectGemma4DraftPath(model, "", DraftDetectOptions{}); det.Active() {
		t.Fatalf("detection = %+v, want ambiguous MTP/ contents to stand down", det)
	}

	// Missing model entirely / blank paths.
	if det := DetectGemma4DraftPath("", "", DraftDetectOptions{}); det.Active() {
		t.Fatalf("detection = %+v, want blank model path inert", det)
	}
	if det := DetectGemma4DraftPath(filepath.Join(model, "nope"), "", DraftDetectOptions{}); det.Active() {
		t.Fatalf("detection = %+v, want missing model dir inert", det)
	}
}
