// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	"testing"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
	"dappco.re/go/mlx/pkg/metal"
)

// gemma4DenseTextConfigJSON is the synthetic two-layer dense config the forward
// tests load — identical in shape to the one TestGemma4_LoadAndForwardDenseModel
// uses, so the gemma4TinyWeights fixture matches its dimensions exactly.
const gemma4DenseTextConfigJSON = `{
	"model_type": "gemma4_text",
	"hidden_size": 8,
	"num_hidden_layers": 2,
	"intermediate_size": 16,
	"num_attention_heads": 1,
	"num_key_value_heads": 1,
	"head_dim": 4,
	"global_head_dim": 8,
	"vocab_size": 10,
	"max_position_embeddings": 16,
	"rms_norm_eps": 1e-6,
	"sliding_window": 4,
	"sliding_window_pattern": 2,
	"num_kv_shared_layers": 0,
	"hidden_size_per_layer_input": 0,
	"use_double_wide_mlp": false,
	"layer_types": ["sliding_attention", "full_attention"]
}`

// loadGemma4DenseTestModel writes the synthetic dense config + tokenizer +
// gemma4TinyWeights into a temp dir and loads it, registering cleanup. It is the
// shared force-Eval harness the forward.go tests reuse for real decode paths.
func loadGemma4DenseTestModel(t *testing.T) *Gemma4Model {
	t.Helper()
	requireMetalRuntime(t)

	dir := t.TempDir()
	if err := coreio.Local.Write(core.JoinPath(dir, "config.json"), gemma4DenseTextConfigJSON); err != nil {
		t.Fatalf("write config.json: %v", err)
	}
	writeMinimalTokenizer(t, dir)
	if err := metal.SaveSafetensors(core.JoinPath(dir, "model.safetensors"), gemma4TinyWeights()); err != nil {
		t.Fatalf("SaveSafetensors: %v", err)
	}

	model, err := LoadGemma4(dir)
	if err != nil {
		t.Fatalf("LoadGemma4: %v", err)
	}
	t.Cleanup(func() { closeGemma4(model) })
	return model
}

// TestForward_ForwardLastTokenLogits_Good runs prefill while projecting only the
// final sequence position, so the logits collapse to [1,1,vocab] rather than the
// full [1,L,vocab] — the memory-bounded prefill path attached MTP assistants seed
// from.
func TestForward_ForwardLastTokenLogits_Good(t *testing.T) {
	model := loadGemma4DenseTestModel(t)

	tokens := metal.FromValues([]int32{2, 3, 4}, 1, 3)
	caches := model.NewCache()
	logits := model.ForwardLastTokenLogits(tokens, nil, caches)
	if err := metal.Eval(logits); err != nil {
		t.Fatalf("Eval logits: %v", err)
	}
	defer func() {
		metal.Free(tokens, logits)
		metal.FreeCaches(caches)
	}()

	shape := logits.Shape()
	if len(shape) != 3 || shape[0] != 1 || shape[1] != 1 || shape[2] != 10 {
		t.Fatalf("last-token logits shape = %v, want [1 1 10]", shape)
	}
}

// TestForward_ForwardGreedyToken_Good runs the greedy decode path and asserts the
// returned token is a single in-vocabulary id. Greedy selection is monotonic
// under final logit softcapping, so this path skips materialising softcapped
// logits — the assertion is on the chosen token, not a logits tensor.
func TestForward_ForwardGreedyToken_Good(t *testing.T) {
	model := loadGemma4DenseTestModel(t)

	tokens := metal.FromValues([]int32{2, 3, 4}, 1, 3)
	caches := model.NewCache()
	tok := model.ForwardGreedyToken(tokens, nil, caches)
	if err := metal.Eval(tok); err != nil {
		t.Fatalf("Eval token: %v", err)
	}
	defer func() {
		metal.Free(tokens, tok)
		metal.FreeCaches(caches)
	}()

	ids := tok.DataInt32()
	if len(ids) != 1 {
		t.Fatalf("greedy token len = %d, want 1", len(ids))
	}
	if ids[0] < 0 || ids[0] >= int32(model.Cfg.VocabSize) {
		t.Fatalf("greedy token = %d, want 0 <= id < vocab %d", ids[0], model.Cfg.VocabSize)
	}
}

// TestForward_ForwardGreedyTokenWithSuppression_Good masks two candidate ids
// before argmax and asserts neither is ever selected — the suppression list must
// be honoured even when those ids would otherwise win. Suppressing the whole
// vocabulary except one id forces a deterministic survivor.
func TestForward_ForwardGreedyTokenWithSuppression_Good(t *testing.T) {
	model := loadGemma4DenseTestModel(t)

	// Suppress every id except 7 — the only survivor argmax can pick.
	var suppress []int32
	const survivor int32 = 7
	for id := int32(0); id < model.Cfg.VocabSize; id++ {
		if id != survivor {
			suppress = append(suppress, id)
		}
	}

	tokens := metal.FromValues([]int32{2, 3, 4}, 1, 3)
	caches := model.NewCache()
	tok := model.ForwardGreedyTokenWithSuppression(tokens, nil, caches, suppress)
	if err := metal.Eval(tok); err != nil {
		t.Fatalf("Eval token: %v", err)
	}
	defer func() {
		metal.Free(tokens, tok)
		metal.FreeCaches(caches)
	}()

	ids := tok.DataInt32()
	if len(ids) != 1 {
		t.Fatalf("suppressed greedy token len = %d, want 1", len(ids))
	}
	if ids[0] != survivor {
		t.Fatalf("suppressed greedy token = %d, want the lone survivor %d", ids[0], survivor)
	}
}

// TestForward_ForwardGreedyTokenWithSuppression_Ugly passes an empty suppression
// list — the suppressor must degrade to a plain greedy argmax and still return a
// single valid token (the len(suppressTokens)==0 branch).
func TestForward_ForwardGreedyTokenWithSuppression_Ugly(t *testing.T) {
	model := loadGemma4DenseTestModel(t)

	tokens := metal.FromValues([]int32{2, 3, 4}, 1, 3)
	caches := model.NewCache()
	tok := model.ForwardGreedyTokenWithSuppression(tokens, nil, caches, nil)
	if err := metal.Eval(tok); err != nil {
		t.Fatalf("Eval token: %v", err)
	}
	defer func() {
		metal.Free(tokens, tok)
		metal.FreeCaches(caches)
	}()

	ids := tok.DataInt32()
	if len(ids) != 1 || ids[0] < 0 || ids[0] >= int32(model.Cfg.VocabSize) {
		t.Fatalf("empty-suppression greedy token = %v, want one valid id", ids)
	}
}
