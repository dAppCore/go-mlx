// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

// PARKED — these two tests were relocated into the gemma4 test package by the
// RFC.model-sdk Task-4 architecture-test move, but they exercise metal-internal
// symbols that stayed in package metal and are not (and should not be) exported:
//
//   - TestGemma4_LoadModel_Dispatch_Good          -> metal's unexported loadModel
//   - TestGemma4_ValidLayerQuantization_…_Good     -> metal's unexported validGemma4LayerQuantization (decode.go)
//
// They cannot compile from package gemma4 without exporting metal internals
// solely for a test (forbidden). Their correct home is metal's own test suite
// (where loadModel/validGemma4LayerQuantization live). Parked here verbatim
// pending that relocation; this directory is Go-ignored so it does not block the
// gemma4 test build. See the session report.
package gemma4

import (
	"testing"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
)

func TestGemma4_ValidLayerQuantization_AllowsAffineProductAndBenchBits_Good(t *testing.T) {
	if !validGemma4LayerQuantization(64, 5) {
		t.Fatal("validGemma4LayerQuantization(64, 5) = false, want q5 bench tier accepted")
	}
	if !validGemma4LayerQuantization(64, 6) {
		t.Fatal("validGemma4LayerQuantization(64, 6) = false, want q6 product default accepted")
	}
}

func TestGemma4_LoadModel_Dispatch_Good(t *testing.T) {
	dir := t.TempDir()
	_ = coreio.Local.Write(core.JoinPath(dir, "config.json"), `{
		"model_type": "gemma4_text",
		"hidden_size": 8,
		"num_hidden_layers": 1,
		"intermediate_size": 16,
		"num_attention_heads": 1,
		"num_key_value_heads": 1,
		"head_dim": 4,
		"hidden_size_per_layer_input": 0
	}`)

	_, err := loadModel(dir)
	if err == nil {
		t.Fatal("expected tokenizer error, proving dispatch reached Gemma4 loader")
	}
	if !core.Contains(err.Error(), "tokenizer") && !core.Contains(err.Error(), "gemma4") {
		t.Fatalf("expected gemma4 loader error, got: %v", err)
	}
}
