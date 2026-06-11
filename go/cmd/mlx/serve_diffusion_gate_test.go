// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"testing"

	core "dappco.re/go"
)

func TestServeArchitectureGate_DiffusionRefused_Good(t *testing.T) {
	dir := t.TempDir()
	cfg := `{"model_type":"diffusion_gemma","architectures":["DiffusionGemmaForBlockDiffusion"]}`
	if res := core.WriteFile(core.PathJoin(dir, "config.json"), []byte(cfg), 0o644); !res.OK {
		t.Fatalf("write config: %+v", res)
	}
	if err := serveArchitectureGate(dir); err == nil {
		t.Fatal("diffusion_gemma was not refused")
	}
}

func TestServeArchitectureGate_ARServed_Good(t *testing.T) {
	dir := t.TempDir()
	cfg := `{"model_type":"gemma4","architectures":["Gemma4ForConditionalGeneration"]}`
	if res := core.WriteFile(core.PathJoin(dir, "config.json"), []byte(cfg), 0o644); !res.OK {
		t.Fatalf("write config: %+v", res)
	}
	if err := serveArchitectureGate(dir); err != nil {
		t.Fatalf("gemma4 refused: %v", err)
	}
	// Unreadable config must not block serving — the loader's own probe
	// owns that failure with a better message.
	if err := serveArchitectureGate(t.TempDir()); err != nil {
		t.Fatalf("missing config refused: %v", err)
	}
}
