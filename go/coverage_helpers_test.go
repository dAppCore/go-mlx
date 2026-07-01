// SPDX-Licence-Identifier: EUPL-1.2

// Coverage for the genuinely-reachable pure-Go helper surface that does not
// need a real Metal model: medium path math + the LoadModelFromMedium
// convenience wrapper, the eval no-forward tensor-encode guard, thinking-token
// filtering, the SFT/identity/label helpers, and the trivial allocator
// delegations. The post-guard real-forward bodies stay real-model-bound and
// are accounted for in the package coverage ceiling.

package mlx

import (
	"context"
	"strings"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/parser"
	coreio "dappco.re/go/io"
	mp "dappco.re/go/mlx/pack"
	"dappco.re/go/mlx/pkg/metal"
)

// emptyPieceTokenizer is a TokenizerImpl whose IDToken always returns "",
// forcing FilterThinkingTokens down its Decode-fallback branch. Decode echoes
// the id as text so the assertion can verify the visible output.
type emptyPieceTokenizer struct{}

func (emptyPieceTokenizer) Encode(text string) []int32 { return nil }
func (emptyPieceTokenizer) Decode(ids []int32) string {
	var b strings.Builder
	for _, id := range ids {
		b.WriteByte(byte('a' + (id % 26)))
	}
	return b.String()
}
func (t emptyPieceTokenizer) DecodeOne(id int32) string  { return t.Decode([]int32{id}) }
func (emptyPieceTokenizer) TokenID(string) (int32, bool) { return 0, false }
func (emptyPieceTokenizer) IDToken(int32) string         { return "" }
func (emptyPieceTokenizer) BOS() int32                   { return 1 }
func (emptyPieceTokenizer) EOS() int32                   { return 2 }
func (emptyPieceTokenizer) HasBOSToken() bool            { return false }

func TestFilterThinkingTokens_DecodeFallbackAndGuard(t *testing.T) {
	// Nil tokenizer guard.
	if _, err := FilterThinkingTokens(nil, []int32{1}, parser.Config{}, ModelInfo{}); err == nil {
		t.Fatalf("nil tokenizer should error")
	}

	// Valid tokenizer whose IDToken returns "" → exercises the Decode
	// fallback inside the streaming loop.
	tok := NewTokenizer(emptyPieceTokenizer{})
	out, err := FilterThinkingTokens(tok, []int32{0, 1, 2}, parser.Config{}, ModelInfo{Architecture: "gemma4_text"})
	if err != nil {
		t.Fatalf("FilterThinkingTokens errored: %v", err)
	}
	if out.Text == "" {
		t.Fatalf("expected decoded fallback text, got empty")
	}
}

func TestLoadTokenizer_MissingFileError(t *testing.T) {
	if _, err := LoadTokenizer("/nonexistent/tokenizer.json"); err == nil {
		t.Fatalf("LoadTokenizer on a missing path should error")
	}
}

func TestMediumRelativePath_AllArms(t *testing.T) {
	// target == root → empty.
	if got := mediumRelativePath("root", "root"); got != "" {
		t.Fatalf("target==root should be empty, got %q", got)
	}
	// empty target → empty.
	if got := mediumRelativePath("root", ""); got != "" {
		t.Fatalf("empty target should be empty, got %q", got)
	}
	// root == "" → trim leading slash off target.
	if got := mediumRelativePath("", "/models/demo/x.json"); got != "models/demo/x.json" {
		t.Fatalf("empty root = %q, want models/demo/x.json", got)
	}
	// Hot prefix path.
	if got := mediumRelativePath("models/demo", "models/demo/config.json"); got != "config.json" {
		t.Fatalf("prefix path = %q, want config.json", got)
	}
	// Cold path: target is NOT a clean prefix of root → falls through to
	// PathRel. "models/demo" → "models/other/x" relative is "../other/x".
	if got := mediumRelativePath("models/demo", "models/other/x"); got != "../other/x" {
		t.Fatalf("cold relative = %q, want ../other/x", got)
	}
}

func TestMediumModelRoot_AndCleanPath(t *testing.T) {
	// Weight-file suffix → parent dir.
	if got := mediumModelRoot("models/demo/model.safetensors"); got != "models/demo" {
		t.Fatalf("safetensors root = %q, want models/demo", got)
	}
	// Bare weight file with no dir → "" (the dir == "." remap).
	if got := mediumModelRoot("model.gguf"); got != "" {
		t.Fatalf("bare gguf root = %q, want empty", got)
	}
	// Non-weight path → cleaned passthrough.
	if got := mediumModelRoot("models/demo"); got != "models/demo" {
		t.Fatalf("dir root = %q, want models/demo", got)
	}
	// cleanMediumPath "." → "".
	if got := cleanMediumPath("."); got != "" {
		t.Fatalf("cleanMediumPath(.) = %q, want empty", got)
	}
	if got := cleanMediumPath("  a/b  "); got != "a/b" {
		t.Fatalf("cleanMediumPath trim = %q, want a/b", got)
	}
}

func TestStagePathFromMedium_Guards(t *testing.T) {
	// Nil medium guard.
	if _, _, err := stagePathFromMedium(nil, "x"); err == nil {
		t.Fatalf("nil medium should error")
	}
	// Path not present in medium.
	medium := coreio.NewMemoryMedium()
	if _, _, err := stagePathFromMedium(medium, "models/absent"); err == nil {
		t.Fatalf("absent path should error")
	}
}

// TestLoadModelFromMedium_Wrapper drives the convenience wrapper end-to-end
// with an in-memory medium + a faked native loader so the whole staging tree
// (copyMediumTree → walkMedium → copyMediumFile → mediumRelativePath) runs
// without a GPU. Mirrors the existing LoadModel(WithMedium) test but through
// the LoadModelFromMedium entry that was previously uncovered.
func TestLoadModelFromMedium_Wrapper(t *testing.T) {
	medium := coreio.NewMemoryMedium()
	if err := medium.Write("models/demo/config.json", `{"model_type":"gemma3"}`); err != nil {
		t.Fatalf("write config: %v", err)
	}
	if err := medium.Write("models/demo/tokenizer.json", `{"model":{"type":"BPE","vocab":{},"merges":[]}}`); err != nil {
		t.Fatalf("write tokenizer: %v", err)
	}
	if err := medium.Write("models/demo/model.gguf", "stub"); err != nil {
		t.Fatalf("write weights: %v", err)
	}

	originalLoadNativeModel := loadNativeModel
	t.Cleanup(func() { loadNativeModel = originalLoadNativeModel })

	var stagedPath string
	loadNativeModel = func(modelPath string, _ metal.LoadConfig) (NativeModel, error) {
		stagedPath = modelPath
		if result := core.Stat(core.PathJoin(modelPath, "config.json")); !result.OK {
			t.Fatalf("staged config missing")
		}
		return &fakeNativeModel{}, nil
	}

	model, err := LoadModelFromMedium(medium, "models/demo", WithContextLength(1024))
	if err != nil {
		t.Fatalf("LoadModelFromMedium() error = %v", err)
	}
	if stagedPath == "" {
		t.Fatalf("expected staged path to be passed to native loader")
	}
	if err := model.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
}

// TestEvaluateDatasetBatch_EncodesThenNoForward drives evaluateDatasetBatch
// with a valid batch and the fakeNativeModel (which is NOT a
// nativeEvalInternalModel), so the synthetic tensor-encode block executes and
// the run stops at the no-forward assertion — covering the encode legs without
// a real Metal forward.
func TestEvaluateDatasetBatch_EncodesThenNoForward(t *testing.T) {
	model := &Model{model: &fakeNativeModel{}}
	batch := SFTBatch{
		Batch:   Batch{Tokens: [][]int{{1, 2, 3}, {4, 5, 6}}},
		Targets: [][]int{{1, 2, 3}, {4, 5, 6}},
	}
	_, err := model.evaluateDatasetBatch(context.Background(), batch)
	if err == nil {
		t.Fatalf("expected no-forward error from a non-eval-capable model")
	}
}

// TestNewModelEvalRunner_NilModelClosures invokes the runner closures with a
// nil *Model so each model==nil guard arm inside Info/LoadAdapter/BuildBatches
// executes (the deeper forward bodies need a real model).
func TestNewModelEvalRunner_NilModelClosures(t *testing.T) {
	runner := NewModelEvalRunner(nil)
	ctx := context.Background()

	if info := runner.Info(ctx); info.Architecture != "" {
		t.Fatalf("nil-model Info should be empty, got %+v", info)
	}
	if _, err := runner.LoadAdapter(ctx, "/path"); err == nil {
		t.Fatalf("nil-model LoadAdapter should error")
	}
	if _, err := runner.BuildBatches(ctx, nil, nil); err == nil {
		t.Fatalf("nil-model BuildBatches should error")
	}

	// Cancelled context arms.
	cancelled, cancel := context.WithCancel(ctx)
	cancel()
	if _, err := runner.LoadAdapter(cancelled, "/path"); err == nil {
		t.Fatalf("cancelled LoadAdapter should error")
	}
}

func TestSFTEffectiveBatchSize_Delegates(t *testing.T) {
	// Pure delegation to train.SFTEffectiveBatchSize — just confirm the
	// accumulation multiplier flows through.
	got := SFTEffectiveBatchSize(SFTConfig{BatchSize: 4, GradientAccumulationSteps: 3})
	if got != 12 {
		t.Fatalf("SFTEffectiveBatchSize = %d, want 12", got)
	}
}

func TestDiscoveredModelIdentity(t *testing.T) {
	id := discoveredModelIdentity(inference.DiscoveredModel{
		Path:       "/m/demo",
		ModelType:  "gemma4_text",
		QuantBits:  4,
		QuantGroup: 32,
		QuantType:  "q4",
	})
	if id.Path != "/m/demo" || id.Architecture != "gemma4_text" || id.QuantBits != 4 {
		t.Fatalf("discoveredModelIdentity wrong: %+v", id)
	}
}

func TestModelPackIdentity_FillsMissingFields(t *testing.T) {
	// Empty fallback → all fields come from the pack.
	pack := mp.ModelPack{
		Path:          "/m/pack",
		Architecture:  "gemma4_text",
		QuantBits:     4,
		QuantGroup:    32,
		QuantType:     "q4",
		ContextLength: 8192,
		NumLayers:     34,
		HiddenSize:    2560,
		VocabSize:     256000,
	}
	id := modelPackIdentity(pack, inference.ModelIdentity{})
	if id.Path != "/m/pack" || id.Architecture != "gemma4_text" || id.NumLayers != 34 || id.VocabSize != 256000 {
		t.Fatalf("modelPackIdentity empty-fallback wrong: %+v", id)
	}

	// Non-empty fallback wins; pack only fills the gaps.
	fallback := inference.ModelIdentity{Path: "/keep", Architecture: "keep-arch"}
	merged := modelPackIdentity(pack, fallback)
	if merged.Path != "/keep" || merged.Architecture != "keep-arch" {
		t.Fatalf("modelPackIdentity should not overwrite set fields: %+v", merged)
	}
	if merged.NumLayers != 34 || merged.HiddenSize != 2560 {
		t.Fatalf("modelPackIdentity should fill unset fields from pack: %+v", merged)
	}
}

func TestSplitExecutorStepLabels(t *testing.T) {
	if got := splitExecutorLayerStepLabel("layer ", 7, 3); got != "layer 7 step 3" {
		t.Fatalf("splitExecutorLayerStepLabel = %q", got)
	}
	if got := splitExecutorStepLabel("decode ", 5); got != "decode 5" {
		t.Fatalf("splitExecutorStepLabel = %q", got)
	}
}

func TestMLXAllocatorDelegations(t *testing.T) {
	// GC and SeedRandom are thin delegations to the metal runtime — drive
	// them to cover the one-line bodies. SeedRandom returns an error only on
	// a runtime fault; a fixed seed succeeds.
	GC()
	if err := SeedRandom(42); err != nil {
		t.Fatalf("SeedRandom(42) error = %v", err)
	}
}

// castClassify unpacks the core.Result returned by the migrated
// inference.TextModel.Classify into the (results, error) pair the tests were
// written against — results is the []inference.ClassifyResult payload (nil on
// failure, via the comma-ok assertion) and error is resultError's unwrapped
// failure (nil when OK). Keeps the call sites' control flow byte-identical
// after the Err/Close/Classify/BatchGenerate → core.Result migration.
func castClassify(r core.Result) ([]inference.ClassifyResult, error) {
	results, _ := r.Value.([]inference.ClassifyResult)
	return results, resultError(r)
}

// castBatch is castClassify for BatchGenerate — unpacks the core.Result into
// the ([]inference.BatchResult, error) pair the tests expect.
func castBatch(r core.Result) ([]inference.BatchResult, error) {
	results, _ := r.Value.([]inference.BatchResult)
	return results, resultError(r)
}

// castTextModel unpacks the core.Result returned by the migrated
// inference.Backend.LoadModel into the (inference.TextModel, error) pair the
// tests expect — the model payload (nil on failure) and resultError's error.
func castTextModel(r core.Result) (inference.TextModel, error) {
	model, _ := r.Value.(inference.TextModel)
	return model, resultError(r)
}
