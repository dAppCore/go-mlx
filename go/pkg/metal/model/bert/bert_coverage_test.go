// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package bert

import (
	"testing"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
	"dappco.re/go/mlx/pkg/metal"
)

// TestBERT_ForwardInert_Good pins the staged loader's three inert primitives:
// the bidirectional encoder has no autoregressive forward pass, no masked
// forward pass, and no LoRA surface yet, so Forward, ForwardMasked, and ApplyLoRA
// all return nil until the scorer kernels land. Callers depend on the nil being
// explicit (not a panic) so the runtime can probe the staged loader's shape.
func TestBERT_ForwardInert_Good(t *testing.T) {
	model := &bertStagedModel{modelType: "bert"}

	if out := model.Forward(nil, nil); out != nil {
		t.Fatalf("Forward() = %#v, want nil for staged encoder", out)
	}
	if out := model.ForwardMasked(nil, nil, nil); out != nil {
		t.Fatalf("ForwardMasked() = %#v, want nil for staged encoder", out)
	}
	if adapter := model.ApplyLoRA(metal.LoRAConfig{}); adapter != nil {
		t.Fatalf("ApplyLoRA() = %#v, want nil for staged loader without LoRA surface", adapter)
	}
}

// TestBERT_LoadStagedModelMalformedConfig_Ugly drives loadBERTStagedModel with
// truncated JSON so the parseBERTStagedConfig failure surfaces as the loader's
// early error return — a malformed config.json never reaches tokenizer load.
func TestBERT_LoadStagedModelMalformedConfig_Ugly(t *testing.T) {
	_, err := loadBERTStagedModel(t.TempDir(), []byte(`{"hidden_size":`), "bert")
	if err == nil {
		t.Fatal("loadBERTStagedModel(truncated config) error = nil, want JSON parse error")
	}
}

// TestBERT_LoadStagedModelMissingTokenizer_Bad gives loadBERTStagedModel a valid
// config but an empty checkpoint directory — no tokenizer.json — so the loader
// fails at LoadTokenizer and wraps the diagnostic with the bert.load context. The
// config parses and validates clean; the failure is the missing tokenizer alone.
func TestBERT_LoadStagedModelMissingTokenizer_Bad(t *testing.T) {
	config := `{
		"architectures": ["BertModel"],
		"model_type": "bert",
		"hidden_size": 384,
		"num_hidden_layers": 6,
		"num_attention_heads": 12,
		"intermediate_size": 1536,
		"vocab_size": 30522,
		"max_position_embeddings": 512
	}`
	_, err := loadBERTStagedModel(t.TempDir(), []byte(config), "bert")
	if err == nil || !core.Contains(err.Error(), "load tokenizer") {
		t.Fatalf("loadBERTStagedModel(no tokenizer) error = %v, want load tokenizer diagnostic", err)
	}
}

// TestBERT_RegistryDispatchRerank_Bad drives the registry path for the
// bert_rerank id with a config that probes as bert_rerank (model_type) but fails
// staged validation (no hidden size / layers / vocab), exercising the init()
// bert_rerank closure's core.E wrap branch — the error a caller sees for a
// malformed sequence-classification checkpoint. Companion to
// TestBERT_RegistryDispatch_Bad, which exercises the bert closure's wrap.
func TestBERT_RegistryDispatchRerank_Bad(t *testing.T) {
	requireMetalRuntime(t)

	dir := t.TempDir()
	config := `{"architectures": ["BertForSequenceClassification"], "model_type": "bert_rerank"}`
	if err := coreio.Local.Write(core.JoinPath(dir, "config.json"), config); err != nil {
		t.Fatalf("write config: %v", err)
	}
	writeMinimalTokenizer(t, dir)

	_, err := metal.LoadAndInit(dir)
	if err == nil {
		t.Fatal("LoadAndInit(invalid bert_rerank) error = nil, want validation diagnostic")
	}
	if !core.Contains(err.Error(), "bert_rerank native load") {
		t.Fatalf("LoadAndInit error = %v, want bert_rerank native load wrap", err)
	}
}

// TestBERTPoolCLS_Bad walks the CLS pooler's guard branch: anything that is not a
// rank-3 hidden state with a positive sequence length returns (nil, false) so the
// Score path can reject a malformed encoder output instead of indexing into it.
func TestBERTPoolCLS_Bad(t *testing.T) {
	requireMetalRuntime(t)

	cases := []struct {
		name   string
		hidden *metal.Array
	}{
		{name: "nil", hidden: nil},
		{name: "rank-2", hidden: metal.FromValues([]float32{1, 2, 3, 4}, 2, 2)},
		{name: "rank-1", hidden: metal.FromValues([]float32{1, 2, 3}, 3)},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if tc.hidden != nil {
				defer metal.Free(tc.hidden)
			}
			pooled, ok := bertPoolCLS(tc.hidden)
			if ok || pooled != nil {
				metal.Free(pooled)
				t.Fatalf("bertPoolCLS(%s) = (%v, %v), want (nil, false)", tc.name, pooled, ok)
			}
		})
	}
}

// TestBERTPoolMean_InvalidHidden_Bad covers the mean pooler's hidden-state guard:
// a non-rank-3 (or empty-sequence) hidden state returns (nil, false) before the
// mask is even consulted — the guard fires on hidden shape alone.
func TestBERTPoolMean_InvalidHidden_Bad(t *testing.T) {
	requireMetalRuntime(t)

	hidden := metal.FromValues([]float32{1, 2, 3, 4}, 2, 2)
	defer metal.Free(hidden)

	if pooled, ok := bertPoolMean(hidden, nil); ok || pooled != nil {
		metal.Free(pooled)
		t.Fatalf("bertPoolMean(rank-2 hidden) = (%v, %v), want (nil, false)", pooled, ok)
	}
}

// TestBERTPoolMean_NilMask_Good covers the unmasked mean branch: with a nil
// attention mask the pooler averages every position uniformly via metal.Mean
// rather than masking — the [2,3,2] hidden state collapses to its per-feature
// column means.
func TestBERTPoolMean_NilMask_Good(t *testing.T) {
	requireMetalRuntime(t)

	hidden := metal.FromValues([]float32{
		1, 2,
		3, 4,
		5, 6,
		10, 20,
		30, 40,
		50, 60,
	}, 2, 3, 2)
	defer metal.Free(hidden)

	pooled, ok := bertPoolMean(hidden, nil)
	if !ok {
		t.Fatal("bertPoolMean(nil mask) ok = false, want true")
	}
	defer metal.Free(pooled)
	metal.Materialize(pooled)

	if gotShape := pooled.Shape(); len(gotShape) != 2 || gotShape[0] != 2 || gotShape[1] != 2 {
		t.Fatalf("shape = %v, want [2 2]", gotShape)
	}
	assertFloat32SliceClose(t, pooled.Floats(), []float32{3, 4, 30, 40}, 1e-5)
}

// TestBERTRerankHead_Score_Bad walks the Score guard branches that all return
// (nil, false): a nil classifier (no head), an unknown pool mode (default case),
// and a pooler that itself rejects the hidden state (the !ok branch after a valid
// mode). Each is the rerank API refusing to score rather than producing garbage.
func TestBERTRerankHead_Score_Bad(t *testing.T) {
	requireMetalRuntime(t)

	hidden := metal.FromValues([]float32{
		2, 3,
		4, 5,
	}, 1, 2, 2)
	weight := metal.FromValues([]float32{
		1, 2,
		-1, 1,
	}, 2, 2)
	bias := metal.FromValues([]float32{0.5, -0.5}, 2)
	defer metal.Free(hidden, weight, bias)

	// nil classifier — no head wired, Score cannot project.
	t.Run("nil-classifier", func(t *testing.T) {
		head := bertRerankHead{PoolMode: bertPoolingCLS}
		if logits, ok := head.Score(hidden, nil); ok || logits != nil {
			metal.Free(logits)
			t.Fatalf("Score(nil classifier) = (%v, %v), want (nil, false)", logits, ok)
		}
	})

	// unknown pool mode — neither cls nor mean, so the switch default fires.
	t.Run("unknown-pool-mode", func(t *testing.T) {
		head := bertRerankHead{
			Classifier: metal.NewLinear(weight, bias),
			PoolMode:   bertPoolingMode("max"),
		}
		if logits, ok := head.Score(hidden, nil); ok || logits != nil {
			metal.Free(logits)
			t.Fatalf("Score(unknown mode) = (%v, %v), want (nil, false)", logits, ok)
		}
	})

	// valid mode but the pooler rejects the hidden state — the !ok branch after
	// a recognised mode. A rank-2 hidden state trips bertPoolCLS's guard.
	t.Run("pooler-rejects-hidden", func(t *testing.T) {
		badHidden := metal.FromValues([]float32{1, 2, 3, 4}, 2, 2)
		defer metal.Free(badHidden)
		head := bertRerankHead{
			Classifier: metal.NewLinear(weight, bias),
			PoolMode:   bertPoolingCLS,
		}
		if logits, ok := head.Score(badHidden, nil); ok || logits != nil {
			metal.Free(logits)
			t.Fatalf("Score(rank-2 hidden) = (%v, %v), want (nil, false)", logits, ok)
		}
	})
}

// TestBERTRerankHead_ScoreEmptyMode_Good covers the default-mode branch: an
// unset PoolMode falls back to CLS pooling, so the head scores the first token of
// the sequence exactly as an explicit cls request would.
func TestBERTRerankHead_ScoreEmptyMode_Good(t *testing.T) {
	requireMetalRuntime(t)

	hidden := metal.FromValues([]float32{
		2, 3,
		4, 5,
	}, 1, 2, 2)
	weight := metal.FromValues([]float32{
		1, 2,
		-1, 1,
	}, 2, 2)
	bias := metal.FromValues([]float32{0.5, -0.5}, 2)
	head := bertRerankHead{Classifier: metal.NewLinear(weight, bias)} // PoolMode unset → CLS
	defer metal.Free(hidden, weight, bias)

	logits, ok := head.Score(hidden, nil)
	if !ok {
		t.Fatal("Score(empty mode) ok = false, want true")
	}
	defer metal.Free(logits)
	metal.Materialize(logits)

	if gotShape := logits.Shape(); len(gotShape) != 2 || gotShape[0] != 1 || gotShape[1] != 2 {
		t.Fatalf("shape = %v, want [1 2]", gotShape)
	}
	// CLS pooling selects [2,3]; logits = W·[2,3] + b = [2*1+3*2+0.5, 2*-1+3*1-0.5] = [8.5, 0.5].
	assertFloat32SliceClose(t, logits.Floats(), []float32{8.5, 0.5}, 1e-5)
}

// TestBERTRerankHead_ScoreMeanMode_Good covers the mean-pooling arm of the Score
// switch: an explicit mean PoolMode routes through bertPoolMean, averaging the
// sequence before the classifier projects it. With a nil mask the [1,2,2] hidden
// state averages to its per-feature column mean, then the head scores that.
func TestBERTRerankHead_ScoreMeanMode_Good(t *testing.T) {
	requireMetalRuntime(t)

	hidden := metal.FromValues([]float32{
		2, 3,
		4, 5,
	}, 1, 2, 2)
	weight := metal.FromValues([]float32{
		1, 2,
		-1, 1,
	}, 2, 2)
	bias := metal.FromValues([]float32{0.5, -0.5}, 2)
	head := bertRerankHead{
		Classifier: metal.NewLinear(weight, bias),
		PoolMode:   bertPoolingMean,
	}
	defer metal.Free(hidden, weight, bias)

	logits, ok := head.Score(hidden, nil)
	if !ok {
		t.Fatal("Score(mean mode) ok = false, want true")
	}
	defer metal.Free(logits)
	metal.Materialize(logits)

	if gotShape := logits.Shape(); len(gotShape) != 2 || gotShape[0] != 1 || gotShape[1] != 2 {
		t.Fatalf("shape = %v, want [1 2]", gotShape)
	}
	// Mean pooling averages [2,3] and [4,5] → [3,4]; logits = W·[3,4] + b =
	// [3*1+4*2+0.5, 3*-1+4*1-0.5] = [11.5, 0.5].
	assertFloat32SliceClose(t, logits.Floats(), []float32{11.5, 0.5}, 1e-5)
}
