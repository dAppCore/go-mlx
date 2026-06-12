// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/mlx/chat"
	"dappco.re/go/mlx/dataset"
	"dappco.re/go/mlx/pkg/metal"
	"dappco.re/go/mlx/probe"
	"dappco.re/go/mlx/profile"
	"dappco.re/go/mlx/train"
)

type fakeSFTTokenizer struct {
	encoded map[string][]int32
	eos     int32
}

func (t fakeSFTTokenizer) Encode(text string) []int32 {
	if tokens, ok := t.encoded[text]; ok {
		return append([]int32(nil), tokens...)
	}
	out := make([]int32, 0, len(text))
	for _, r := range text {
		out = append(out, int32(r))
	}
	return out
}

func (t fakeSFTTokenizer) Decode(tokens []int32) string {
	builder := core.NewBuilder()
	for _, token := range tokens {
		builder.WriteString(core.Sprintf("%d", token))
	}
	return builder.String()
}

func (t fakeSFTTokenizer) TokenID(text string) (int32, bool) {
	tokens := t.Encode(text)
	if len(tokens) != 1 {
		return 0, false
	}
	return tokens[0], true
}

func (t fakeSFTTokenizer) IDToken(id int32) string { return core.Sprintf("%d", id) }
func (t fakeSFTTokenizer) DecodeOne(id int32) string {
	return t.Decode([]int32{id})
}
func (t fakeSFTTokenizer) BOS() int32        { return 0 }
func (t fakeSFTTokenizer) EOS() int32        { return t.eos }
func (t fakeSFTTokenizer) HasBOSToken() bool { return false }

func TestSFTSliceDataset_Reset_Good(t *testing.T) {
	dataset := dataset.NewSliceDataset([]dataset.Sample{
		{Prompt: "a", Response: "b"},
	})

	first, ok, err := dataset.Next()
	if err != nil {
		t.Fatalf("Next() error = %v", err)
	}
	if !ok || first.Prompt != "a" {
		t.Fatalf("first Next() = %+v ok=%v", first, ok)
	}
	if _, ok, err := dataset.Next(); err != nil || ok {
		t.Fatalf("exhausted Next() ok=%v err=%v, want ok=false err=nil", ok, err)
	}
	if err := dataset.Reset(); err != nil {
		t.Fatalf("Reset() error = %v", err)
	}
	again, ok, err := dataset.Next()
	if err != nil {
		t.Fatalf("Next() after Reset error = %v", err)
	}
	if !ok || again.Response != "b" {
		t.Fatalf("Next() after Reset = %+v ok=%v", again, ok)
	}
}

func TestBuildSFTBatches_MasksPromptAndAppendsEOS_Good(t *testing.T) {
	tokenizer := NewTokenizer(fakeSFTTokenizer{
		encoded: map[string][]int32{
			"prompt":   {10, 11},
			"response": {20, 21},
		},
		eos: 2,
	})
	dataset := dataset.NewSliceDataset([]dataset.Sample{{Prompt: "prompt", Response: "response"}})

	batches, err := BuildSFTBatches(tokenizer, dataset, SFTConfig{BatchSize: 1})
	if err != nil {
		t.Fatalf("BuildSFTBatches() error = %v", err)
	}
	if len(batches) != 1 {
		t.Fatalf("batches len = %d, want 1", len(batches))
	}
	got := batches[0]
	wantInputs := []int{10, 11, 20, 21}
	wantTargets := []int{11, 20, 21, 2}
	wantMask := []float32{0, 1, 1, 1}
	if !equalIntSlices(got.Batch.Tokens[0], wantInputs) {
		t.Fatalf("inputs = %v, want %v", got.Batch.Tokens[0], wantInputs)
	}
	if !equalIntSlices(got.Targets[0], wantTargets) {
		t.Fatalf("targets = %v, want %v", got.Targets[0], wantTargets)
	}
	if !equalFloat32Slices(got.Batch.LossMask[0], wantMask) {
		t.Fatalf("loss mask = %v, want %v", got.Batch.LossMask[0], wantMask)
	}
}

func TestBuildSFTBatches_TextSampleTrainsWholeSequence_Good(t *testing.T) {
	tokenizer := NewTokenizer(fakeSFTTokenizer{
		encoded: map[string][]int32{"full": {5, 6, 7}},
		eos:     9,
	})
	dataset := dataset.NewSliceDataset([]dataset.Sample{{Text: "full"}})

	batches, err := BuildSFTBatches(tokenizer, dataset, SFTConfig{BatchSize: 1, NoEOS: true})
	if err != nil {
		t.Fatalf("BuildSFTBatches() error = %v", err)
	}
	if len(batches) != 1 {
		t.Fatalf("batches len = %d, want 1", len(batches))
	}
	if !equalIntSlices(batches[0].Batch.Tokens[0], []int{5, 6}) {
		t.Fatalf("inputs = %v, want [5 6]", batches[0].Batch.Tokens[0])
	}
	if !equalIntSlices(batches[0].Targets[0], []int{6, 7}) {
		t.Fatalf("targets = %v, want [6 7]", batches[0].Targets[0])
	}
	if !equalFloat32Slices(batches[0].Batch.LossMask[0], []float32{1, 1}) {
		t.Fatalf("loss mask = %v, want [1 1]", batches[0].Batch.LossMask[0])
	}
}

func TestBuildSFTBatches_NilTokenizer_Bad(t *testing.T) {
	_, err := BuildSFTBatches(nil, dataset.NewSliceDataset([]dataset.Sample{{Text: "x"}}), SFTConfig{})
	if err == nil {
		t.Fatal("expected nil tokenizer error")
	}
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

func equalFloat32Slices(a, b []float32) bool {
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

func TestModelTrainSFT_NilModel_Bad(t *testing.T) {
	var model *Model
	_, err := model.TrainSFT(context.Background(), dataset.NewSliceDataset([]dataset.Sample{{Text: "x"}}), SFTConfig{})
	if err == nil {
		t.Fatal("expected nil model error")
	}
}

func TestModelTrainSFT_ValidationBranches_Bad(t *testing.T) {
	model := &Model{model: &fakeNativeModel{}}
	if _, err := model.TrainSFT(context.Background(), nil, SFTConfig{}); err == nil {
		t.Fatal("expected nil dataset error")
	}
	if _, err := model.TrainSFT(context.Background(), dataset.NewSliceDataset([]dataset.Sample{{Text: "x"}}), SFTConfig{}); err == nil {
		t.Fatal("expected nil tokenizer error")
	}

	model.tok = NewTokenizer(&metal.Tokenizer{})
	if _, err := model.TrainSFT(context.Background(), dataset.NewSliceDataset([]dataset.Sample{{Text: "x"}}), SFTConfig{}); err == nil {
		t.Fatal("expected nil LoRA adapter error")
	}
}

func TestDatasetConfigForModel_Gemma4OfficialArchitectureUsesSharedFormatter_Good(t *testing.T) {
	cfg := DatasetConfigForModel(ModelInfo{Architecture: "Gemma4ForConditionalGeneration", NumHeads: 16})
	if got := chat.TemplateName(cfg.ChatTemplate); got != "gemma4" {
		t.Fatalf("TemplateName = %q, want gemma4 for official Gemma4 architecture", got)
	}
	if !cfg.ChatTemplate.LargeVariant {
		t.Fatal("LargeVariant = false, want true for 16-head Gemma4 model")
	}
	got := chat.Format([]chat.Message{{Role: "user", Content: "Write one line."}}, cfg.ChatTemplate)
	if !core.Contains(got, "<|turn>user\nWrite one line.<turn|>") {
		t.Fatalf("formatted prompt = %q, want shared Gemma4 turn syntax", got)
	}
	if !core.Contains(got, "<|think|>") {
		t.Fatalf("formatted prompt = %q, want thinking-enabled Gemma4 rendering (registry default)", got)
	}

	for _, info := range []ModelInfo{
		{Architecture: "Gemma4AssistantForCausalLM", NumHeads: 16},
		{Architecture: "qwen3", NumHeads: 16},
		{Architecture: "Gemma4ForCausalLM", NumHeads: 8},
	} {
		cfg := DatasetConfigForModel(info)
		if cfg.ChatTemplate.LargeVariant {
			t.Fatalf("DatasetConfigForModel(%+v).LargeVariant = true, want false outside large Gemma4 targets", info)
		}
	}
}

func TestBuildSFTTrainingBatches_UsesAccumulationAsEffectiveBatch_Good(t *testing.T) {
	tokenizer := NewTokenizer(fakeSFTTokenizer{
		encoded: map[string][]int32{
			"p1": {1},
			"r1": {2},
			"p2": {3},
			"r2": {4},
		},
		eos: 9,
	})
	dataset := dataset.NewJSONL([]dataset.Sample{
		{Prompt: "p1", Response: "r1"},
		{Prompt: "p2", Response: "r2"},
	})

	batches, err := BuildSFTTrainingBatches(tokenizer, dataset, SFTConfig{
		BatchSize:                 1,
		GradientAccumulationSteps: 2,
	})
	if err != nil {
		t.Fatalf("BuildSFTTrainingBatches() error = %v", err)
	}
	if len(batches) != 1 {
		t.Fatalf("batches len = %d, want one effective optimizer batch", len(batches))
	}
	if len(batches[0].Batch.Tokens) != 2 {
		t.Fatalf("batch sequences = %d, want 2 micro-batches", len(batches[0].Batch.Tokens))
	}
	if !equalFloat32Slices(batches[0].Batch.LossMask[0], []float32{1, 1}) ||
		!equalFloat32Slices(batches[0].Batch.LossMask[1], []float32{1, 1}) {
		t.Fatalf("loss masks = %v, want response-only masks preserved", batches[0].Batch.LossMask)
	}
}

func TestBuildSFTTrainingBatches_NilDataset_Bad(t *testing.T) {
	tokenizer := NewTokenizer(fakeSFTTokenizer{eos: 9})
	_, err := BuildSFTTrainingBatches(tokenizer, nil, SFTConfig{})
	if err == nil {
		t.Fatal("expected nil dataset error")
	}
}

func TestBuildSFTTrainingBatches_PackedDataset_Ugly(t *testing.T) {
	tokenizer := NewTokenizer(fakeSFTTokenizer{
		encoded: map[string][]int32{
			"p1": {1},
			"r1": {2},
			"p2": {3},
			"r2": {4},
		},
		eos: 9,
	})
	dataset := dataset.NewSliceDataset([]dataset.Sample{
		{Prompt: "p1", Response: "r1"},
		{Prompt: "p2", Response: "r2"},
	})

	batches, err := BuildSFTTrainingBatches(tokenizer, dataset, SFTConfig{
		BatchSize:       1,
		MaxSeqLen:       8,
		SequencePacking: true,
	})
	if err != nil {
		t.Fatalf("BuildSFTTrainingBatches() error = %v", err)
	}
	if len(batches) != 1 || len(batches[0].Batch.Tokens) != 1 {
		t.Fatalf("batches = %+v, want one packed sequence", batches)
	}
	if !equalIntSlices(batches[0].Batch.Tokens[0], []int{1, 2, 3, 4}) {
		t.Fatalf("packed inputs = %v, want [1 2 3 4]", batches[0].Batch.Tokens[0])
	}
}

func TestSFTCheckpointMetadata_RoundTrip_Good(t *testing.T) {
	dir := t.TempDir()
	meta := SFTCheckpointMetadata{
		Version:                   SFTCheckpointMetadataVersion,
		Path:                      dir,
		AdapterPath:               core.PathJoin(dir, "adapter.safetensors"),
		Step:                      7,
		OptimizerStep:             7,
		Epoch:                     2,
		Samples:                   13,
		Loss:                      0.125,
		LearningRate:              2e-4,
		BatchSize:                 2,
		GradientAccumulationSteps: 4,
		SequencePacking:           true,
		EvalTemperature:           0.4,
		Model:                     "qwen3",
		LoRA: SFTLoRAMetadata{
			Rank:                 16,
			Alpha:                32,
			TargetKeys:           []string{"q_proj", "v_proj"},
			AllowExtendedTargets: true,
		},
	}

	if err := SaveSFTCheckpointMetadata(dir, meta); err != nil {
		t.Fatalf("SaveSFTCheckpointMetadata() error = %v", err)
	}
	got, err := LoadSFTCheckpointMetadata(dir)
	if err != nil {
		t.Fatalf("LoadSFTCheckpointMetadata() error = %v", err)
	}
	if got.Step != 7 || got.Epoch != 2 || got.GradientAccumulationSteps != 4 || got.EvalTemperature != 0.4 || got.LoRA.Rank != 16 || !got.LoRA.AllowExtendedTargets {
		t.Fatalf("metadata = %+v, want round-tripped training state", got)
	}
}

func TestLoadSFTCheckpointMetadata_Missing_Bad(t *testing.T) {
	_, err := LoadSFTCheckpointMetadata(core.PathJoin(t.TempDir(), "missing"))
	if err == nil {
		t.Fatal("expected missing metadata error")
	}
}

func TestLoadSFTResumeMetadata_LoadsAdjacentMetadata_Ugly(t *testing.T) {
	dir := t.TempDir()
	meta := SFTCheckpointMetadata{
		Version:                   SFTCheckpointMetadataVersion,
		Path:                      dir,
		Step:                      11,
		OptimizerStep:             11,
		Epoch:                     3,
		Samples:                   21,
		Loss:                      0.5,
		GradientAccumulationSteps: 2,
	}
	if err := SaveSFTCheckpointMetadata(dir, meta); err != nil {
		t.Fatalf("SaveSFTCheckpointMetadata() error = %v", err)
	}
	result := &SFTResult{}
	if err := ApplySFTResumeMetadata(result, SFTConfig{ResumePath: dir}); err != nil {
		t.Fatalf("ApplySFTResumeMetadata() error = %v", err)
	}
	if result.ResumedFrom == nil || result.ResumedFrom.Step != 11 || result.ResumePath != dir {
		t.Fatalf("resume result = %+v, want metadata attached", result)
	}
}

func TestSFTResult_Metrics_Good(t *testing.T) {
	result := &SFTResult{
		Steps:       4,
		Epochs:      2,
		Samples:     9,
		LastLoss:    0.75,
		Checkpoints: []string{"a", "b"},
		Evaluations: []SFTEvalResult{{Step: 2}, {Step: 4}},
	}

	metrics := result.Metrics(SFTConfig{
		BatchSize:                 2,
		GradientAccumulationSteps: 3,
		LearningRate:              2e-4,
	})
	if metrics.OptimizerSteps != 4 || metrics.EffectiveBatchSize != 6 || metrics.CheckpointCount != 2 || metrics.EvaluationCount != 2 {
		t.Fatalf("metrics = %+v, want SFT counters", metrics)
	}
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

func TestSFTAdapter_SanitisesProbeSink_Good(t *testing.T) {
	native := &fakeNativeModel{loraAdapter: &metal.LoRAAdapter{}}
	adapter, err := (&Model{model: native}).sftAdapter(SFTConfig{LoRA: LoRAConfig{ProbeSink: probe.NewRecorder(), Lambda: 0.25}})
	if err != nil {
		t.Fatalf("sftAdapter() error = %v", err)
	}
	if adapter == nil || native.lastLoRAConfig.ProbeSink != nil || native.lastLoRAConfig.Lambda != 0.25 {
		t.Fatalf("adapter=%+v native config=%+v, want adapter with sanitised probe config", adapter, native.lastLoRAConfig)
	}
}

func TestSFTAdapter_Gemma4UsesSharedLoRATargetPolicy_Good(t *testing.T) {
	native := &fakeNativeModel{
		info:        metal.ModelInfo{Architecture: "gemma4_text"},
		loraAdapter: &metal.LoRAAdapter{},
	}
	model := &Model{model: native}
	adapter, err := model.sftAdapter(train.NormalizeSFTConfigForModel(SFTConfig{}, model.Info()))
	if err != nil {
		t.Fatalf("sftAdapter() error = %v", err)
	}
	if adapter == nil {
		t.Fatal("sftAdapter() adapter = nil")
	}
	wantTargets := profile.DefaultLoRATargets("gemma4")
	if !equalStringSlices(native.lastLoRAConfig.TargetKeys, wantTargets) {
		t.Fatalf("TargetKeys = %v, want shared Gemma 4 defaults %v", native.lastLoRAConfig.TargetKeys, wantTargets)
	}
	if !equalStringSlices(native.lastLoRAConfig.TargetLayers, wantTargets) {
		t.Fatalf("TargetLayers = %v, want shared Gemma 4 defaults %v", native.lastLoRAConfig.TargetLayers, wantTargets)
	}
}
