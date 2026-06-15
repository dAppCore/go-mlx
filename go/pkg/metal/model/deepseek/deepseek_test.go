// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package deepseek

import (
	"testing"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
	"dappco.re/go/mlx/pkg/metal"
)

func TestDeepSeek_LoadStagedModelValidatesMLA_Good(t *testing.T) {
	dir := t.TempDir()
	config := `{
		"architectures": ["DeepseekV3ForCausalLM"],
		"model_type": "deepseek_v3",
		"hidden_size": 1024,
		"num_hidden_layers": 2,
		"num_attention_heads": 8,
		"num_key_value_heads": 2,
		"vocab_size": 32000,
		"n_routed_experts": 64,
		"q_lora_rank": 1536,
		"kv_lora_rank": 512,
		"qk_nope_head_dim": 128,
		"qk_rope_head_dim": 64,
		"v_head_dim": 128
	}`
	if err := coreio.Local.Write(core.JoinPath(dir, "config.json"), config); err != nil {
		t.Fatalf("write config: %v", err)
	}
	writeMinimalTokenizer(t, dir)

	model, err := loadStagedModel(dir, []byte(config))
	if err != nil {
		t.Fatalf("loadStagedModel(deepseek) error = %v", err)
	}
	if model.ModelType() != "deepseek" || model.NumLayers() != 2 {
		t.Fatalf("model metadata = %s/%d, want deepseek/2", model.ModelType(), model.NumLayers())
	}
	if model.Tokenizer() == nil {
		t.Fatal("Tokenizer() = nil, want staged loader to expose tokenizer metadata")
	}
	if model.mla.KVLoRARank != 512 || model.mla.QKHeadDim != 192 || model.mla.VHeadDim != 128 {
		t.Fatalf("DeepSeek MLA plan = %+v, want kv rank 512 qk head 192 v head 128", model.mla)
	}
	info := metal.ModelInfo{Architecture: model.ModelType(), NumLayers: model.NumLayers()}
	model.FillModelInfo(&info)
	if info.VocabSize != 32000 || info.HiddenSize != 1024 {
		t.Fatalf("FillModelInfo = %+v, want vocab=32000 hidden=1024", info)
	}
}

func TestDeepSeek_LoadStagedModelValidatesMLA_Bad(t *testing.T) {
	base := `{
		"architectures": ["DeepseekV3ForCausalLM"],
		"model_type": "deepseek_v3",
		"hidden_size": 1024,
		"num_hidden_layers": 2,
		"num_attention_heads": 8,
		"num_key_value_heads": 2,
		"vocab_size": 32000,
		"n_routed_experts": 64,
		%s
	}`
	cases := []struct {
		name string
		mla  string
		want string
	}{
		{
			name: "missing-kv-lora",
			mla:  `"qk_nope_head_dim": 128, "qk_rope_head_dim": 64, "v_head_dim": 128`,
			want: "kv_lora_rank",
		},
		{
			name: "missing-rope-split",
			mla:  `"kv_lora_rank": 512, "qk_nope_head_dim": 128, "v_head_dim": 128`,
			want: "qk_nope_head_dim and qk_rope_head_dim",
		},
		{
			name: "bad-qk-sum",
			mla:  `"kv_lora_rank": 512, "qk_nope_head_dim": 128, "qk_rope_head_dim": 64, "qk_head_dim": 256, "v_head_dim": 128`,
			want: "qk_head_dim",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			config := core.Sprintf(base, tc.mla)
			_, err := loadStagedModel(t.TempDir(), []byte(config))
			if err == nil || !core.Contains(err.Error(), tc.want) {
				t.Fatalf("loadStagedModel(deepseek invalid MLA) error = %v, want %q", err, tc.want)
			}
		})
	}
}

func requireMetalRuntime(t testing.TB) {
	t.Helper()
	if !metal.MetalAvailable() {
		t.Skip("Metal runtime unavailable")
	}
}

// TestDeepSeek_RegistryDispatch_Good drives the init()-registered loader closure
// through the real loader registry (metal.LoadAndInit → loadModel →
// lookupModelLoader → closure), the path the runtime takes for a checkpoint
// directory. The staged loader reads config + tokenizer only — no weights, no
// GPU allocation — so a synthetic t.TempDir() pack exercises the closure's
// success branch without loading a model.
func TestDeepSeek_RegistryDispatch_Good(t *testing.T) {
	requireMetalRuntime(t)

	dir := t.TempDir()
	config := `{
		"architectures": ["DeepseekV3ForCausalLM"],
		"model_type": "deepseek_v3",
		"hidden_size": 1024,
		"num_hidden_layers": 2,
		"num_attention_heads": 8,
		"num_key_value_heads": 2,
		"vocab_size": 32000,
		"n_routed_experts": 64,
		"q_lora_rank": 1536,
		"kv_lora_rank": 512,
		"qk_nope_head_dim": 128,
		"qk_rope_head_dim": 64,
		"v_head_dim": 128
	}`
	if err := coreio.Local.Write(core.JoinPath(dir, "config.json"), config); err != nil {
		t.Fatalf("write config: %v", err)
	}
	writeMinimalTokenizer(t, dir)

	model, err := metal.LoadAndInit(dir)
	if err != nil {
		t.Fatalf("LoadAndInit(deepseek) error = %v", err)
	}
	if model.ModelType() != "deepseek" {
		t.Fatalf("ModelType() = %q, want deepseek", model.ModelType())
	}
	if info := model.Info(); info.VocabSize != 32000 || info.HiddenSize != 1024 {
		t.Fatalf("Info() = %+v, want vocab=32000 hidden=1024", info)
	}
}

// TestDeepSeek_RegistryDispatch_Bad drives the same registry path with a config
// that probes as deepseek but fails staged validation (no MLA fields),
// exercising the init() closure's core.E wrap branch — the error path callers
// see for a malformed DeepSeek checkpoint.
func TestDeepSeek_RegistryDispatch_Bad(t *testing.T) {
	requireMetalRuntime(t)

	dir := t.TempDir()
	config := `{
		"architectures": ["DeepseekV3ForCausalLM"],
		"model_type": "deepseek_v3",
		"hidden_size": 1024,
		"num_hidden_layers": 2,
		"num_attention_heads": 8,
		"num_key_value_heads": 2,
		"vocab_size": 32000,
		"n_routed_experts": 64
	}`
	if err := coreio.Local.Write(core.JoinPath(dir, "config.json"), config); err != nil {
		t.Fatalf("write config: %v", err)
	}
	writeMinimalTokenizer(t, dir)

	if _, err := metal.LoadAndInit(dir); err == nil {
		t.Fatal("LoadAndInit(invalid deepseek) error = nil, want validation diagnostic")
	}
}

// TestDeepSeek_ParseStagedConfigFallback_Good covers parseStagedConfig's
// detection branches: a bare config with no model_type and no architectures
// falls back to "deepseek", and the explicit field is normalised to the canonical
// id rather than echoed verbatim.
func TestDeepSeek_ParseStagedConfigFallback_Good(t *testing.T) {
	cfg, err := parseStagedConfig([]byte(`{"hidden_size": 1024}`))
	if err != nil {
		t.Fatalf("parseStagedConfig fallback error = %v", err)
	}
	if cfg.ModelType != "deepseek" {
		t.Fatalf("fallback ModelType = %q, want deepseek", cfg.ModelType)
	}
}

// TestDeepSeek_ParseStagedConfigReject_Bad covers the reject branch: a config
// whose detected type resolves to something other than deepseek is refused, so
// the deepseek loader never silently adopts a foreign checkpoint.
func TestDeepSeek_ParseStagedConfigReject_Bad(t *testing.T) {
	if _, err := parseStagedConfig([]byte(`{"model_type": "llama", "hidden_size": 1024}`)); err == nil ||
		!core.Contains(err.Error(), "deepseek config") {
		t.Fatalf("parseStagedConfig(llama) error = %v, want deepseek-config diagnostic", err)
	}
}

// TestDeepSeek_ParseStagedConfigMalformed_Ugly covers the JSON-unmarshal failure
// branch — truncated bytes surface a parse error, never a zero-value config.
func TestDeepSeek_ParseStagedConfigMalformed_Ugly(t *testing.T) {
	if _, err := parseStagedConfig([]byte(`{"hidden_size": `)); err == nil {
		t.Fatal("parseStagedConfig(truncated) error = nil, want JSON parse error")
	}
}

// TestDeepSeek_Validate_Bad walks the staged validation branches independently of
// loading: each malformed config must trip its own diagnostic so the loader
// rejects a partial checkpoint with a specific message.
func TestDeepSeek_Validate_Bad(t *testing.T) {
	full := func() stagedConfig {
		return stagedConfig{
			HiddenSize:        1024,
			NumHiddenLayers:   2,
			VocabSize:         32000,
			NumAttentionHeads: 8,
			NumKeyValueHeads:  2,
			NRoutedExperts:    64,
			KVLoRARank:        512,
			QKNoPEHeadDim:     128,
			QKRoPEHeadDim:     64,
			VHeadDim:          128,
		}
	}
	cases := []struct {
		name   string
		mutate func(*stagedConfig)
		want   string
	}{
		{name: "missing-hidden-size", mutate: func(c *stagedConfig) { c.HiddenSize = 0 }, want: "hidden size, layer count, and vocab size"},
		{name: "missing-layers", mutate: func(c *stagedConfig) { c.NumHiddenLayers = 0 }, want: "hidden size, layer count, and vocab size"},
		{name: "missing-vocab", mutate: func(c *stagedConfig) { c.VocabSize = 0 }, want: "hidden size, layer count, and vocab size"},
		{name: "missing-attn-heads", mutate: func(c *stagedConfig) { c.NumAttentionHeads = 0 }, want: "attention and key/value head counts"},
		{name: "missing-kv-heads", mutate: func(c *stagedConfig) { c.NumKeyValueHeads = 0 }, want: "attention and key/value head counts"},
		{name: "missing-experts", mutate: func(c *stagedConfig) { c.NRoutedExperts = 0 }, want: "expert count"},
		{name: "missing-mla", mutate: func(c *stagedConfig) { c.KVLoRARank = 0 }, want: "kv_lora_rank"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg := full()
			tc.mutate(&cfg)
			err := cfg.validate()
			if err == nil || !core.Contains(err.Error(), tc.want) {
				t.Fatalf("validate(%s) error = %v, want %q", tc.name, err, tc.want)
			}
		})
	}
}

// TestDeepSeek_MLAPlanInfersHeadDim_Good covers the deepSeekMLAPlan branch that
// infers qk_head_dim from the no-rope/rope split when the config omits the
// combined field — the common case for checkpoints that list only the parts.
func TestDeepSeek_MLAPlanInfersHeadDim_Good(t *testing.T) {
	cfg := stagedConfig{
		KVLoRARank:    512,
		QKNoPEHeadDim: 128,
		QKRoPEHeadDim: 64,
		VHeadDim:      128,
	}
	plan, err := cfg.deepSeekMLAPlan()
	if err != nil {
		t.Fatalf("deepSeekMLAPlan inferred-head-dim error = %v", err)
	}
	if plan.QKHeadDim != 192 {
		t.Fatalf("inferred QKHeadDim = %d, want 192 (128+64)", plan.QKHeadDim)
	}
}

// TestDeepSeek_MLAPlanMissingVDim_Bad covers the v_head_dim validation branch:
// the no-rope/rope split is present but the value head dim is absent, so the plan
// is rejected before any kernel would consume it.
func TestDeepSeek_MLAPlanMissingVDim_Bad(t *testing.T) {
	cfg := stagedConfig{
		KVLoRARank:    512,
		QKNoPEHeadDim: 128,
		QKRoPEHeadDim: 64,
	}
	if _, err := cfg.deepSeekMLAPlan(); err == nil || !core.Contains(err.Error(), "qk_head_dim and v_head_dim") {
		t.Fatalf("deepSeekMLAPlan(missing v_head_dim) error = %v, want qk_head_dim/v_head_dim diagnostic", err)
	}
}

// TestDeepSeek_FirstArchitectureName covers the architecture→registry-id mapping:
// any DeepSeek* architecture resolves to "deepseek"; an unrelated architecture or
// an empty list returns "" so the caller falls back to model_type.
func TestDeepSeek_FirstArchitectureName(t *testing.T) {
	cases := []struct {
		name  string
		archs []string
		want  string
	}{
		{name: "deepseek-v3", archs: []string{"DeepseekV3ForCausalLM"}, want: "deepseek"},
		{name: "deepseek-v2", archs: []string{"DeepseekV2ForCausalLM"}, want: "deepseek"},
		{name: "non-deepseek", archs: []string{"LlamaForCausalLM"}, want: ""},
		{name: "empty", archs: nil, want: ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := firstDeepSeekArchitectureName(tc.archs); got != tc.want {
				t.Fatalf("firstDeepSeekArchitectureName(%v) = %q, want %q", tc.archs, got, tc.want)
			}
		})
	}
}

// TestDeepSeek_DecodeUnavailableError names the operation and the missing
// capability so a caller that reaches for native decode on the staged loader gets
// a diagnostic instead of a silent nil from Forward.
func TestDeepSeek_DecodeUnavailableError(t *testing.T) {
	model := &StagedModel{}
	err := model.DecodeUnavailableError("generate")
	if err == nil {
		t.Fatal("DecodeUnavailableError = nil, want diagnostic")
	}
	msg := err.Error()
	if !core.Contains(msg, "generate") || !core.Contains(msg, "sparse-expert decode kernels") {
		t.Fatalf("DecodeUnavailableError message = %q, want operation + missing-kernel note", msg)
	}
}

func writeMinimalTokenizer(t *testing.T, dir string) {
	t.Helper()
	tokenizer := `{
		"model": {"type": "BPE", "vocab": {"hello": 0, "<unk>": 1}, "merges": []},
		"pre_tokenizer": {"type": "ByteLevel"},
		"decoder": {"type": "ByteLevel"}
	}`
	if err := coreio.Local.Write(core.JoinPath(dir, "tokenizer.json"), tokenizer); err != nil {
		t.Fatalf("write tokenizer: %v", err)
	}
}
