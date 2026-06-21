// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package native

import (
	"testing"

	"dappco.re/go/mlx/pkg/model"
	_ "dappco.re/go/mlx/pkg/model/gemma4"  // register the gemma4 loaders for the dispatch check
	_ "dappco.re/go/mlx/pkg/model/mistral" // register the mistral loaders
)

// TestLoadDirReactiveDispatch pins the reactive registry dispatch the generic loader runs: probeArch
// resolves a config's architecture the metal way (profile.ResolveArchitecture over model_type + the
// nested text_config + architectures, with a raw-model_type fallback), and model.LookupLoader finds a
// registered loader for the arches the backend serves. An unknown model_type resolves to no loader —
// a clean error, not a panic — so the backend stays model-agnostic: it knows the registry, not gemma4.
func TestLoadDirReactiveDispatch(t *testing.T) {
	hasLoader := func(cfg string) bool {
		resolved, raw, err := probeArch([]byte(cfg))
		if err != nil {
			t.Fatalf("probeArch(%s): %v", cfg, err)
		}
		return model.LookupLoader(resolved) != nil || model.LookupLoader(raw) != nil
	}
	if !hasLoader(`{"model_type":"gemma4"}`) {
		t.Fatal("gemma4 config should dispatch to a registered loader")
	}
	if !hasLoader(`{"model_type":"gemma4_unified","text_config":{"model_type":"gemma4_text"}}`) {
		t.Fatal("gemma4 multimodal wrapper should dispatch to a registered loader")
	}
	if !hasLoader(`{"model_type":"mistral3","architectures":["Mistral3ForConditionalGeneration"],"text_config":{"model_type":"ministral3"}}`) {
		t.Fatal("mistral3 config should dispatch to a registered loader")
	}
	if hasLoader(`{"model_type":"nonesuch_arch"}`) {
		t.Fatal("an unregistered model_type must resolve to no loader")
	}
}
