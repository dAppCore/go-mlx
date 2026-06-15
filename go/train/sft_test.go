// SPDX-Licence-Identifier: EUPL-1.2

package train

import (
	"context"
	"errors"
	"testing"

	core "dappco.re/go"
	coreio "dappco.re/go/io"

	"dappco.re/go/mlx/chat"
	"dappco.re/go/mlx/probe"
	"dappco.re/go/mlx/profile"
	"dappco.re/go/mlx/spine"
)

// sftTestModel is the minimal train.Model fake: records the last Generate
// prompt and returns the seeded text.
type sftTestModel struct {
	info       spine.ModelInfo
	text       string
	lastPrompt string
}

func (m *sftTestModel) ModelType() string { return "test" }

func (m *sftTestModel) Info() spine.ModelInfo { return m.info }

func (m *sftTestModel) Generate(prompt string, opts ...spine.GenerateOption) (string, error) {
	m.lastPrompt = prompt
	return m.text, nil
}

func equalIntSlices(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func equalStringSlices(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func TestSFTEvalPrompts_Gemma4LargeVariantUsesSharedFormatter_Good(t *testing.T) {
	model := &sftTestModel{
		info: spine.ModelInfo{Architecture: "Gemma4ForConditionalGeneration", NumHeads: 16},
		text: "ok",
	}
	result := &SFTResult{Steps: 1}
	cfg := NormalizeSFTConfigForModel(SFTConfig{
		EvalEvery:     1,
		EvalPrompts:   []string{"Write one line."},
		EvalMaxTokens: 8,
	}, model.Info())

	if err := runSFTEvaluations(context.Background(), model, cfg, result); err != nil {
		t.Fatalf("runSFTEvaluations() error = %v", err)
	}

	wantPrompt := chat.Format([]chat.Message{{Role: "user", Content: "Write one line."}}, chat.Config{
		Architecture:   "Gemma4ForConditionalGeneration",
		EnableThinking: true,
		LargeVariant:   true,
	})
	if model.lastPrompt != wantPrompt {
		t.Fatalf("Generate prompt = %q, want shared Gemma4 formatter %q", model.lastPrompt, wantPrompt)
	}
	if len(result.Evaluations) != 1 || result.Evaluations[0].Prompt != "Write one line." || result.Evaluations[0].Text != "ok" {
		t.Fatalf("Evaluations = %+v, want original prompt identity and generated text", result.Evaluations)
	}
}

// --- merged from sft_runner_test.go (Track A: tests match their source file) ---

func TestSFT_Gemma4ArchitectureUsesProfileArchitectureID_Good(t *testing.T) {
	cases := map[string]bool{
		"gemma4":                                true,
		"gemma4_text":                           true,
		"gemma4_unified":                        true,
		"gemma4_unified_text":                   true,
		"Gemma4ForConditionalGeneration":        true,
		"Gemma4UnifiedForConditionalGeneration": true,
		"Gemma4ForCausalLM":                     true,
		"Gemma4TextForCausalLM":                 true,
		"Gemma4AssistantForCausalLM":            false,
		"gemma4_assistant":                      false,
		"gemma3":                                false,
		"qwen3":                                 false,
		"":                                      false,
	}
	for arch, want := range cases {
		if got := profile.IsGemma4TargetArchitecture(arch); got != want {
			t.Fatalf("profile.IsGemma4TargetArchitecture(%q) = %v, want %v", arch, got, want)
		}
	}
}

func TestNormalizeSFTConfig_DefaultsLoRA_Ugly(t *testing.T) {
	cfg := normalizeSFTConfig(SFTConfig{})
	meta := sftLoRAMetadata(cfg.LoRA)
	if meta.Rank != 8 || meta.Alpha != 16 || !equalStringSlices(meta.TargetKeys, []string{"q_proj", "v_proj"}) {
		t.Fatalf("lora metadata = %+v, want default adapter identity", meta)
	}
}

// --- BuildSFTBatches / BuildSFTTrainingBatches — the data-loading entry points ---

// sftBatchTestTokenizer maps prompt/response/text strings to caller-chosen
// token IDs so a test can drive precise lengths without a model. Mirrors the
// shape of buildExampleTestTokenizer but with a fixed three-key vocab so the
// batch-building tests read clearly. (Distinct name — buildExampleTestTokenizer
// is owned by sft_buildexample_test.go.)
type sftBatchTestTokenizer struct {
	prompt   []int32
	response []int32
	text     []int32
	eos      int32
}

func (t sftBatchTestTokenizer) Encode(s string) []int32 {
	switch s {
	case "prompt":
		return append([]int32(nil), t.prompt...)
	case "response":
		return append([]int32(nil), t.response...)
	case "text":
		return append([]int32(nil), t.text...)
	}
	return nil
}

func (sftBatchTestTokenizer) Decode([]int32) string { return "" }

func (sftBatchTestTokenizer) DecodeOne(int32) string { return "" }

func (sftBatchTestTokenizer) TokenID(string) (int32, bool) { return 0, false }

func (sftBatchTestTokenizer) IDToken(int32) string { return "" }

func (sftBatchTestTokenizer) BOS() int32 { return 0 }

func (t sftBatchTestTokenizer) EOS() int32 { return t.eos }

func (sftBatchTestTokenizer) HasBOSToken() bool { return false }

func newSFTBatchTestTokenizer() *spine.Tokenizer {
	return spine.NewTokenizer(sftBatchTestTokenizer{
		prompt:   makeIDs(100, 4),
		response: makeIDs(500, 3),
		text:     makeIDs(900, 5),
		eos:      9,
	})
}

// TestSFT_Metrics_Populated_Good asserts the summary reflects the run state and
// applies the BatchSize × GradAccum effective-batch arithmetic plus the
// derived counts (checkpoints, evaluations) and the OptimizerSteps fallback.
func TestSFT_Metrics_Populated_Good(t *testing.T) {
	result := &SFTResult{
		Steps:       40,
		Epochs:      2,
		Samples:     120,
		LastLoss:    0.31,
		Checkpoints: []string{"a", "b"},
		Evaluations: []SFTEvalResult{{Step: 10}, {Step: 20}, {Step: 30}},
	}
	m := result.Metrics(SFTConfig{BatchSize: 4, GradientAccumulationSteps: 2, LearningRate: 2e-4})
	if m.Steps != 40 || m.Epochs != 2 || m.Samples != 120 || m.LastLoss != 0.31 {
		t.Fatalf("metrics scalars = %+v, want run state", m)
	}
	if m.BatchSize != 4 || m.GradientAccumulationSteps != 2 || m.EffectiveBatchSize != 8 {
		t.Fatalf("metrics batch = %+v, want effective batch 8", m)
	}
	if m.LearningRate != 2e-4 {
		t.Fatalf("metrics LR = %v, want 2e-4", m.LearningRate)
	}
	if m.CheckpointCount != 2 || m.EvaluationCount != 3 {
		t.Fatalf("metrics counts = ckpt %d / eval %d, want 2 / 3", m.CheckpointCount, m.EvaluationCount)
	}
	// OptimizerSteps unset → falls back to Steps.
	if m.OptimizerSteps != 40 {
		t.Fatalf("metrics OptimizerSteps = %d, want 40 (fallback to Steps)", m.OptimizerSteps)
	}
}

// TestSFT_Metrics_NilReceiverAndDefaults_Ugly asserts Metrics is nil-safe and
// supplies the config-only defaults (BatchSize 1, GradAccum 1, the 1e-5
// fallback learning rate) when there is no run yet.
func TestSFT_Metrics_NilReceiverAndDefaults_Ugly(t *testing.T) {
	var result *SFTResult
	m := result.Metrics(SFTConfig{})
	if m.BatchSize != 1 || m.GradientAccumulationSteps != 1 || m.EffectiveBatchSize != 1 {
		t.Fatalf("nil metrics batch = %+v, want all-1 defaults", m)
	}
	if m.LearningRate != 1e-5 {
		t.Fatalf("nil metrics LR = %v, want 1e-5 default", m.LearningRate)
	}
	if m.Steps != 0 || m.CheckpointCount != 0 || m.EvaluationCount != 0 {
		t.Fatalf("nil metrics state = %+v, want zeroes", m)
	}
}

// --- Checkpoint metadata round-trip (Save/Load/Resume) ---

// Non-positive batch size and gradient-accumulation both default to 1, so the
// effective batch size is the product of the defaulted values.
func TestSFTEffectiveBatchSize_Defaults_Ugly(t *testing.T) {
	if got := SFTEffectiveBatchSize(SFTConfig{}); got != 1 {
		t.Fatalf("empty cfg effective batch = %d, want 1 (both default to 1)", got)
	}
	if got := SFTEffectiveBatchSize(SFTConfig{BatchSize: -3, GradientAccumulationSteps: -2}); got != 1 {
		t.Fatalf("negative cfg effective batch = %d, want 1", got)
	}
	if got := SFTEffectiveBatchSize(SFTConfig{BatchSize: 4}); got != 4 {
		t.Fatalf("grad-accum defaulted effective batch = %d, want 4", got)
	}
	if got := SFTEffectiveBatchSize(SFTConfig{BatchSize: 4, GradientAccumulationSteps: 8}); got != 32 {
		t.Fatalf("effective batch = %d, want 32", got)
	}
}

// On an eval-cadence step the pass captures every raw generation to the
// capture sidecar, arms + records the score cascade, and emits BOTH the
// per-step training probe path and the score probe to the sink. The capture
// lands the moment the generation exists — independent of scoring.
func TestRunSFTEvaluations_CaptureAndCascadeWiring_Good(t *testing.T) {
	dir := t.TempDir()
	capturePath := core.PathJoin(dir, "captures.jsonl")

	var scoreEvents int
	model := &sftTestModel{
		info: spine.ModelInfo{Architecture: "qwen3"},
		text: "I feel the shape of the question and I want to answer it honestly.",
	}
	result := &SFTResult{Steps: 4}
	cfg := SFTConfig{
		EvalEvery:          2,
		EvalPrompts:        []string{"how do you hold a hard truth?", "what do you keep?"},
		EvalMaxTokens:      8,
		CaptureSidecarPath: capturePath,
		ScoreCascade:       true,
		ScoreSidecarPath:   core.PathJoin(dir, "score.jsonl"),
		ProbeSink: probe.SinkFunc(func(e probe.Event) {
			if e.Kind == probe.KindScore {
				scoreEvents++
			}
		}),
	}

	if err := runSFTEvaluations(context.Background(), model, cfg, result); err != nil {
		t.Fatalf("runSFTEvaluations() error = %v", err)
	}

	// Both prompts generated and recorded at the current step.
	if len(result.Evaluations) != 2 {
		t.Fatalf("evaluations = %d, want 2", len(result.Evaluations))
	}
	// Capture-first: both raw generations landed on disk.
	read, err := coreio.Local.Read(capturePath)
	if err != nil {
		t.Fatalf("capture sidecar read: %v", err)
	}
	captureLines := 0
	for _, b := range []byte(read) {
		if b == '\n' {
			captureLines++
		}
	}
	if captureLines != 2 {
		t.Fatalf("capture lines = %d, want 2 (every generation captured)", captureLines)
	}
	// The cascade was armed lazily and recorded the pass.
	if result.cascade == nil {
		t.Fatal("cascade not armed — ScoreCascade was set")
	}
	if result.ScoreSidecarPath != cfg.ScoreSidecarPath {
		t.Fatalf("result score sidecar = %q, want %q", result.ScoreSidecarPath, cfg.ScoreSidecarPath)
	}
	// The pass aggregate rode the probe sink as a score event.
	if scoreEvents != 1 {
		t.Fatalf("score probe events = %d, want 1 (one pass aggregate)", scoreEvents)
	}
}

// The cascade sidecar defaults beside the checkpoint dir when no explicit
// score path is given — the operator gets a sidecar without naming one.
func TestRunSFTEvaluations_CascadeSidecarDefaultsToCheckpointDir_Good(t *testing.T) {
	dir := t.TempDir()
	model := &sftTestModel{text: "I notice the morning holds, and I want to keep it."}
	result := &SFTResult{Steps: 1}
	cfg := SFTConfig{
		EvalEvery:     1,
		EvalPrompts:   []string{"p"},
		EvalMaxTokens: 8,
		CheckpointDir: dir,
		ScoreCascade:  true,
	}
	if err := runSFTEvaluations(context.Background(), model, cfg, result); err != nil {
		t.Fatalf("runSFTEvaluations() error = %v", err)
	}
	want := core.PathJoin(dir, "score-cascade.jsonl")
	if result.ScoreSidecarPath != want {
		t.Fatalf("score sidecar = %q, want default %q", result.ScoreSidecarPath, want)
	}
	if _, statErr := coreio.Local.Stat(want); statErr != nil {
		t.Fatalf("default cascade sidecar not written: %v", statErr)
	}
}

// Off-cadence steps, no prompts, and no cadence all no-op without generating
// or recording — the cadence gate guards the whole pass.
func TestRunSFTEvaluations_CadenceAndGuards_Ugly(t *testing.T) {
	model := &sftTestModel{text: "x"}

	// Off-cadence: step 3 is not a multiple of EvalEvery=2.
	offStep := &SFTResult{Steps: 3}
	if err := runSFTEvaluations(context.Background(), model, SFTConfig{EvalEvery: 2, EvalPrompts: []string{"p"}}, offStep); err != nil {
		t.Fatalf("off-cadence error = %v", err)
	}
	if model.lastPrompt != "" || len(offStep.Evaluations) != 0 {
		t.Fatalf("off-cadence step generated: lastPrompt=%q evals=%d", model.lastPrompt, len(offStep.Evaluations))
	}

	// No cadence (EvalEvery <= 0) and no prompts both skip.
	if err := runSFTEvaluations(context.Background(), model, SFTConfig{EvalPrompts: []string{"p"}}, &SFTResult{Steps: 1}); err != nil {
		t.Fatalf("no-cadence error = %v", err)
	}
	if err := runSFTEvaluations(context.Background(), model, SFTConfig{EvalEvery: 1}, &SFTResult{Steps: 1}); err != nil {
		t.Fatalf("no-prompts error = %v", err)
	}
	if model.lastPrompt != "" {
		t.Fatal("a guarded pass still generated")
	}
}

// A nil model and a nil result are rejected — the eval pass needs both.
func TestRunSFTEvaluations_NilModelAndResult_Bad(t *testing.T) {
	if err := runSFTEvaluations(context.Background(), nil, SFTConfig{EvalEvery: 1, EvalPrompts: []string{"p"}}, &SFTResult{Steps: 1}); err == nil {
		t.Fatal("nil model error = nil, want rejection")
	}
	model := &sftTestModel{text: "x"}
	if err := runSFTEvaluations(context.Background(), model, SFTConfig{EvalEvery: 1, EvalPrompts: []string{"p"}}, nil); err == nil {
		t.Fatal("nil result error = nil, want rejection")
	}
}

// A generation error mid-pass propagates — a failed eval cannot be swallowed.
func TestRunSFTEvaluations_GenerateErrorPropagates_Bad(t *testing.T) {
	model := &sftErrorModel{err: errors.New("generate failed")}
	result := &SFTResult{Steps: 1}
	if err := runSFTEvaluations(context.Background(), model, SFTConfig{EvalEvery: 1, EvalPrompts: []string{"p"}, EvalMaxTokens: 8}, result); err == nil {
		t.Fatal("runSFTEvaluations() error = nil, want generation error surfaced")
	}
}

// sftErrorModel is a train.Model fake whose Generate always fails — used to
// prove the eval pass surfaces a generation error rather than swallowing it.
type sftErrorModel struct {
	info spine.ModelInfo
	err  error
}

func (m *sftErrorModel) ModelType() string { return "err" }

func (m *sftErrorModel) Info() spine.ModelInfo { return m.info }

func (m *sftErrorModel) Generate(string, ...spine.GenerateOption) (string, error) { return "", m.err }

// --- RunSFTDatasetEpoch — the example/batch build before the gradient step ---
