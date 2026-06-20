// SPDX-Licence-Identifier: EUPL-1.2

// The last reachable arms after the main branch sweep: the nested
// text_config / quantization-block malformed-key and missing-colon
// returns in the config walker, the MiniMax M2 config-read failure (only
// reachable by calling the inspector directly — a successful Inspect has
// already read the config), the JANG not-present-but-probed no-op, and
// the AutoRound native tensor-map validation failure. Each is exercised
// at the tightest reachable layer.

package model

import (
	"testing"

	core "dappco.re/go"
	mp "dappco.re/go/mlx/pack"
)

// TestModelConfigProbe_NestedKeyAndColonErrors drives the malformed-key
// (ParseJSONStringRaw error) and missing-colon arms inside the nested
// text_config (config_probe_unmarshal.go:283-289) and quantization-block
// (config_probe_unmarshal.go:489-495) walkers. The earlier TextConfig /
// QuantBlock branch tests hit the `data[i] != '"'` and value-error arms,
// but not these two key-side returns.
func TestModelConfigProbe_NestedKeyAndColonErrors(t *testing.T) {
	cases := []struct {
		name string
		json string
	}{
		// text_config: a key with an invalid \x escape.
		{"text_bad_key_escape", `{"text_config":{"mod\xel":"x"}}`},
		// text_config: a valid key with no following colon.
		{"text_missing_colon", `{"text_config":{"model_type" "x"}}`},
		// text_config: EOF right after a valid key (no colon, no more data).
		{"text_eof_after_key", `{"text_config":{"model_type"`},
		// quantization: a key with an invalid \x escape.
		{"quant_bad_key_escape", `{"quantization":{"bi\xts":4}}`},
		// quantization: a valid key with no following colon.
		{"quant_missing_colon", `{"quantization":{"bits" 4}}`},
		// quantization: EOF right after a valid key.
		{"quant_eof_after_key", `{"quantization_config":{"bits"`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var probe modelConfigProbe
			if err := probe.UnmarshalJSON([]byte(tc.json)); err == nil {
				t.Fatalf("expected error for %q", tc.json)
			}
		})
	}
}

// TestInspectMiniMaxM2_ConfigReadError drives the config-read failure arm
// (pack.go:465-468) directly: through Inspect the config is always
// readable by the time the MiniMax inspector runs, so the only way to
// observe the read error is to point a minimax_m2 pack's ConfigPath at a
// directory and call the inspector.
func TestInspectMiniMaxM2_ConfigReadError(t *testing.T) {
	dir := t.TempDir()
	cfgDir := core.PathJoin(dir, "config.json")
	if result := core.MkdirAll(cfgDir, 0o755); !result.OK {
		t.Fatalf("MkdirAll config.json dir: %v", result.Value)
	}
	pack := mp.ModelPack{Architecture: "minimax_m2", ConfigPath: cfgDir, Root: dir}
	inspectModelPackMiniMaxM2(&pack)
	if !pack.HasIssue(mp.ModelPackIssueInvalidConfig) {
		t.Fatalf("issues = %+v, want invalid-config from a directory ConfigPath", pack.Issues)
	}
	if pack.MiniMaxM2 != nil {
		t.Fatalf("MiniMaxM2 = %+v, want nil when the config read fails", pack.MiniMaxM2)
	}
}

// TestInspectMiniMaxM2_NotMiniMaxNoOp pins the early-return guard
// (pack.go:461-463): a non-minimax architecture, or an empty ConfigPath,
// is a no-op.
func TestInspectMiniMaxM2_NotMiniMaxNoOp(t *testing.T) {
	pack := mp.ModelPack{Architecture: "qwen3", ConfigPath: "/whatever"}
	inspectModelPackMiniMaxM2(&pack)
	if pack.MiniMaxM2 != nil || len(pack.Issues) != 0 {
		t.Fatalf("non-minimax arch should be a no-op, got %+v", pack)
	}
	pack2 := mp.ModelPack{Architecture: "minimax_m2", ConfigPath: ""}
	inspectModelPackMiniMaxM2(&pack2)
	if pack2.MiniMaxM2 != nil || len(pack2.Issues) != 0 {
		t.Fatalf("empty ConfigPath should be a no-op, got %+v", pack2)
	}
}

// TestInspectModelPackJANG_NotPresentNoOp drives the info==nil return
// (pack_quantinspect.go:23-25): an UNPOPULATED dir index makes has()
// return true, so the inspector calls jang.ReadConfig against a root with
// no jang_config.json — which returns (nil, nil) on IsNotExist — and the
// inspector returns without touching the pack.
func TestInspectModelPackJANG_NotPresentNoOp(t *testing.T) {
	dir := t.TempDir() // no jang_config.json
	pack := mp.ModelPack{}
	inspectModelPackJANG(&pack, dir, &modelPackDirIndex{}) // populated=false → has()=true
	if pack.JANG != nil || len(pack.Issues) != 0 {
		t.Fatalf("absent jang_config.json should be a no-op, got JANG=%+v issues=%+v", pack.JANG, pack.Issues)
	}
}

// TestInspectModelPack_AutoRoundTensorMapValidationFails drives the
// ValidateSafetensorsTensorMap failure arm (pack_quantinspect.go:73-76):
// a native AutoRound pack that DECLARES a tensor map but ships safetensors
// missing those packed tensors fails validation with the unsupported
// AutoRound issue.
func TestInspectModelPack_AutoRoundTensorMapValidationFails(t *testing.T) {
	dir := t.TempDir()
	writeModelPackFile(t, core.PathJoin(dir, "config.json"), `{
		"model_type": "gemma4_text",
		"vocab_size": 32000,
		"hidden_size": 8,
		"num_hidden_layers": 1,
		"max_position_embeddings": 2048
	}`)
	writeModelPackFile(t, core.PathJoin(dir, "auto_round_config.json"), `{
		"bits": 4,
		"group_size": 32,
		"sym": true,
		"data_type": "int",
		"quant_method": "auto-round",
		"packing_format": "auto_round",
		"tensors": [
			{
				"name": "model.layers.0.self_attn.q_proj.weight",
				"packed": "model.layers.0.self_attn.q_proj.weight.packed",
				"scales": "model.layers.0.self_attn.q_proj.weight.scales",
				"zero_points": "model.layers.0.self_attn.q_proj.weight.zeros",
				"shape": [4, 8],
				"bits": 4,
				"group_size": 32,
				"sym": true
			}
		]
	}`)
	writeModelPackFile(t, core.PathJoin(dir, "tokenizer.json"), modelPackTokenizerJSON)
	// A stub safetensors that does NOT contain the declared packed tensors
	// — so ValidateSafetensorsTensorMap fails.
	writeModelPackFile(t, core.PathJoin(dir, "model-00001-of-00001.safetensors"), "stub")

	pack, err := Inspect(dir)
	if err != nil {
		t.Fatalf("Inspect error = %v", err)
	}
	if !pack.HasIssue(mp.ModelPackIssueUnsupportedAutoRound) {
		t.Fatalf("issues = %+v, want unsupported-AutoRound from tensor-map validation failure", pack.Issues)
	}
}
