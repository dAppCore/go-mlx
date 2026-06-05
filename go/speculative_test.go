// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"testing"

	core "dappco.re/go"
	mp "dappco.re/go/mlx/pack"
	"dappco.re/go/mlx/pkg/metal"
	"dappco.re/go/mlx/pkg/metal/model/gemma4"
)

func TestSpeculative_Model_GenerateSpeculative_Good(t *testing.T) {
	target := &Model{model: &fakeNativeModel{tokens: []metal.Token{
		{ID: 1, Text: "A"},
		{ID: 2, Text: "B"},
	}}}
	draftNative := &fakeNativeModel{tokens: []metal.Token{
		{ID: 1, Text: "A"},
		{ID: 3, Text: "C"},
	}}
	draft := &Model{model: draftNative}

	result, err := target.GenerateSpeculative(context.Background(), draft, "prompt", SpeculativeDecodeConfig{
		MaxTokens:   2,
		DraftTokens: 2,
	})
	if err != nil {
		t.Fatalf("GenerateSpeculative() error = %v", err)
	}
	if result.Text != "AB" {
		t.Fatalf("Text = %q, want target greedy text AB", result.Text)
	}
	if result.Metrics.AcceptedTokens != 1 || result.Metrics.RejectedTokens != 1 {
		t.Fatalf("Metrics = %+v, want one accepted and one rejected", result.Metrics)
	}
	if result.Metrics.TargetCalls != 1 || result.Metrics.DraftCalls != 1 {
		t.Fatalf("calls = %+v, want one target and one draft call", result.Metrics)
	}
	if draftNative.lastGenerateConfig.MaxTokens != 2 {
		t.Fatalf("draft MaxTokens = %d, want 2", draftNative.lastGenerateConfig.MaxTokens)
	}
}

func TestSpeculative_Model_GenerateSpeculative_Bad(t *testing.T) {
	target := &Model{model: &fakeNativeModel{}}
	if _, err := target.GenerateSpeculative(context.Background(), nil, "prompt", SpeculativeDecodeConfig{}); err == nil {
		t.Fatal("GenerateSpeculative(nil draft) error = nil, want guard")
	}
	if _, err := (*Model)(nil).GenerateSpeculative(context.Background(), target, "prompt", SpeculativeDecodeConfig{}); err == nil {
		t.Fatal("GenerateSpeculative(nil target) error = nil, want guard")
	}
}

func TestSpeculative_Model_GenerateSpeculative_Ugly(t *testing.T) {
	target := &Model{model: &fakeNativeModel{}}
	draft := &Model{model: &fakeNativeModel{}}
	if _, err := target.GenerateSpeculative(nil, draft, "prompt", SpeculativeDecodeConfig{MaxTokens: -1}); err == nil {
		t.Fatal("GenerateSpeculative(negative max) error = nil, want validation")
	}
	if _, err := target.GenerateSpeculative(nil, draft, "prompt", SpeculativeDecodeConfig{DraftTokens: -1}); err == nil {
		t.Fatal("GenerateSpeculative(negative draft) error = nil, want validation")
	}
}

func TestSpeculative_LoadSpeculativePair_Good(t *testing.T) {
	oldLoad := loadNativeModel
	defer func() { loadNativeModel = oldLoad }()

	tokenizer, err := metal.LoadTokenizer(writeRootTokenizer(t))
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}
	loadNativeModel = func(path string, cfg metal.LoadConfig) (nativeModel, error) {
		return &fakeNativeModel{
			info:      metal.ModelInfo{Architecture: path, VocabSize: 256, QuantBits: 4, QuantGroup: 64, NumLayers: 1},
			tokenizer: tokenizer,
			tokens:    []metal.Token{{ID: 1, Text: "A"}},
		}, nil
	}

	pair, err := LoadSpeculativePair("/models/target", "/models/target-assistant", SpeculativePairConfig{
		TargetOptions:  []LoadOption{WithAutoMemoryPlan(false)},
		DraftOptions:   []LoadOption{WithAutoMemoryPlan(false)},
		TokenizerProbe: []string{"hello"},
	})
	if err != nil {
		t.Fatalf("LoadSpeculativePair() error = %v", err)
	}
	defer pair.Close()
	if pair.Target == nil || pair.Draft == nil {
		t.Fatalf("pair = %+v, want both models", pair)
	}
	if len(pair.Report.TokenizerProbe) != 1 || pair.Report.Target.VocabSize != 256 || pair.Report.Draft.VocabSize != 256 {
		t.Fatalf("Report = %+v, want compatibility details", pair.Report)
	}
	result, err := pair.Generate(context.Background(), "prompt", SpeculativeDecodeConfig{MaxTokens: 1, DraftTokens: 1})
	if err != nil {
		t.Fatalf("pair.Generate() error = %v", err)
	}
	if result.Metrics.AcceptedTokens != 1 {
		t.Fatalf("Metrics = %+v, want accepted target/draft token", result.Metrics)
	}
}

func TestSpeculative_LoadSpeculativePair_Gemma4Assistant_Good(t *testing.T) {
	oldLoad := loadNativeModel
	oldInspect := inspectSpeculativeDraftModelPack
	oldAttach := attachGemma4AssistantDraft
	defer func() {
		loadNativeModel = oldLoad
		inspectSpeculativeDraftModelPack = oldInspect
		attachGemma4AssistantDraft = oldAttach
	}()

	tokenizer, err := metal.LoadTokenizer(writeRootTokenizer(t))
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}
	targetNative := &fakeNativeModel{
		info:      metal.ModelInfo{Architecture: "gemma4_text", VocabSize: 256, HiddenSize: 8, QuantBits: 4, QuantGroup: 64, NumLayers: 2},
		tokenizer: tokenizer,
		gemma4AssistantResult: gemma4.Gemma4AssistantGenerateResult{
			Tokens:         []metal.Token{{ID: 1, Text: "A"}},
			Text:           "A",
			TargetTokens:   1,
			DraftTokens:    2,
			AcceptedTokens: 1,
			RejectedTokens: 1,
			TargetCalls:    2,
			DraftCalls:     1,
		},
	}
	loadNativeModel = func(path string, cfg metal.LoadConfig) (nativeModel, error) {
		return targetNative, nil
	}
	inspectSpeculativeDraftModelPack = func(path string, opts ...mp.ModelPackOption) (mp.ModelPack, error) {
		return mp.ModelPack{Architecture: "gemma4_assistant"}, nil
	}
	attachGemma4AssistantDraft = func(target nativeModel, draftPath string) (*gemma4.Gemma4AssistantPair, error) {
		if target != targetNative {
			t.Fatalf("assistant target = %T, want targetNative", target)
		}
		tokenOrdering := metal.FromValues([]int64{0, 1, 2, 3}, 4)
		return &gemma4.Gemma4AssistantPair{
			Assistant: &gemma4.Gemma4AssistantModel{
				Tok:                      tokenizer,
				Cfg:                      &gemma4.Gemma4TextConfig{TransformerConfig: metal.TransformerConfig{VocabSize: 256, HiddenSize: 4, MaxPositionEmbeddings: 4096}},
				BackboneHiddenSize:       8,
				UseOrderedEmbeddings:     true,
				NumCentroids:             2048,
				CentroidIntermediateTopK: 32,
				Layers:                   make([]*gemma4.Gemma4AssistantLayer, 4),
				TokenOrdering:            tokenOrdering,
			},
		}, nil
	}

	pair, err := LoadSpeculativePair("/models/target", "/models/target-assistant", SpeculativePairConfig{
		TargetOptions:  []LoadOption{WithAutoMemoryPlan(false)},
		DraftOptions:   []LoadOption{WithAutoMemoryPlan(false)},
		TokenizerProbe: []string{"hello"},
	})
	if err != nil {
		t.Fatalf("LoadSpeculativePair() error = %v", err)
	}
	defer pair.Close()
	if pair.Target == nil || pair.Draft != nil || pair.Gemma4Assistant == nil {
		t.Fatalf("pair target=%v draft=%v assistant=%v, want target plus native assistant", pair.Target, pair.Draft, pair.Gemma4Assistant)
	}
	if pair.Report.Draft.Architecture != "gemma4_assistant" || pair.Report.Draft.NumLayers != 4 {
		t.Fatalf("Report.Draft = %+v, want gemma4_assistant metadata", pair.Report.Draft)
	}
	if pair.Report.AssistantLayout == nil ||
		pair.Report.AssistantLayout.Architecture != "gemma4_assistant" ||
		!pair.Report.AssistantLayout.OrderedEmbeddings ||
		pair.Report.AssistantLayout.Centroids != 2048 ||
		pair.Report.AssistantLayout.CentroidIntermediateTopK != 32 ||
		!pair.Report.AssistantLayout.FourLayerDrafter ||
		pair.Report.AssistantLayout.TokenOrderingDType != "int64" ||
		len(pair.Report.AssistantLayout.TokenOrderingShape) != 1 ||
		pair.Report.AssistantLayout.TokenOrderingShape[0] != 4 {
		t.Fatalf("Report.AssistantLayout = %+v, want ordered four-layer assistant layout", pair.Report.AssistantLayout)
	}
	result, err := pair.Generate(context.Background(), "prompt", SpeculativeDecodeConfig{MaxTokens: 1, DraftTokens: 2})
	if err != nil {
		t.Fatalf("pair.Generate() error = %v", err)
	}
	if result.Text != "A" || result.Metrics.AcceptedTokens != 1 || result.Metrics.RejectedTokens != 1 {
		t.Fatalf("pair.Generate() = %+v, want native Gemma 4 assistant decode result", result)
	}
	if result.Mode != SpeculativeDecodeModeMTP {
		t.Fatalf("pair.Generate() mode = %q, want %q", result.Mode, SpeculativeDecodeModeMTP)
	}
	if targetNative.gemma4AssistantPair != pair.Gemma4Assistant {
		t.Fatal("GenerateGemma4Assistant did not receive attached assistant pair")
	}
	if targetNative.lastGemma4AssistantPrompt != "prompt" || targetNative.lastGemma4AssistantDraftTokens != 2 {
		t.Fatalf("GenerateGemma4Assistant args prompt=%q draft=%d", targetNative.lastGemma4AssistantPrompt, targetNative.lastGemma4AssistantDraftTokens)
	}
}

func TestSpeculative_Gemma4AssistantUsesProductionDraftDefault_Good(t *testing.T) {
	oldLoad := loadNativeModel
	oldInspect := inspectSpeculativeDraftModelPack
	oldAttach := attachGemma4AssistantDraft
	defer func() {
		loadNativeModel = oldLoad
		inspectSpeculativeDraftModelPack = oldInspect
		attachGemma4AssistantDraft = oldAttach
	}()

	tokenizer, err := metal.LoadTokenizer(writeRootTokenizer(t))
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}
	targetNative := &fakeNativeModel{
		info:      metal.ModelInfo{Architecture: "gemma4_text", VocabSize: 256, HiddenSize: 8, QuantBits: 6, QuantGroup: 64, NumLayers: 2},
		tokenizer: tokenizer,
		gemma4AssistantResult: gemma4.Gemma4AssistantGenerateResult{
			Tokens:         []metal.Token{{ID: 1, Text: "A"}},
			Text:           "A",
			TargetTokens:   1,
			DraftTokens:    MTPDefaultDraftTokens,
			AcceptedTokens: 1,
			TargetCalls:    1,
			DraftCalls:     1,
		},
	}
	loadNativeModel = func(path string, cfg metal.LoadConfig) (nativeModel, error) {
		return targetNative, nil
	}
	inspectSpeculativeDraftModelPack = func(path string, opts ...mp.ModelPackOption) (mp.ModelPack, error) {
		return mp.ModelPack{Architecture: "gemma4_assistant"}, nil
	}
	attachGemma4AssistantDraft = func(target nativeModel, draftPath string) (*gemma4.Gemma4AssistantPair, error) {
		return &gemma4.Gemma4AssistantPair{
			Assistant: &gemma4.Gemma4AssistantModel{
				Tok:                      tokenizer,
				Cfg:                      &gemma4.Gemma4TextConfig{TransformerConfig: metal.TransformerConfig{VocabSize: 256, HiddenSize: 4, MaxPositionEmbeddings: 4096}},
				BackboneHiddenSize:       8,
				UseOrderedEmbeddings:     true,
				NumCentroids:             2048,
				CentroidIntermediateTopK: 32,
				Layers:                   make([]*gemma4.Gemma4AssistantLayer, 4),
			},
		}, nil
	}

	pair, err := LoadSpeculativePair("/models/target", "/models/target-assistant", SpeculativePairConfig{
		TargetOptions:  []LoadOption{WithAutoMemoryPlan(false)},
		DraftOptions:   []LoadOption{WithAutoMemoryPlan(false)},
		TokenizerProbe: []string{"hello"},
	})
	if err != nil {
		t.Fatalf("LoadSpeculativePair() error = %v", err)
	}
	defer pair.Close()

	_, err = pair.Generate(context.Background(), "prompt", SpeculativeDecodeConfig{MaxTokens: 4})
	if err != nil {
		t.Fatalf("pair.Generate() error = %v", err)
	}
	if targetNative.lastGemma4AssistantDraftTokens != MTPDefaultDraftTokens {
		t.Fatalf("default assistant draft tokens = %d, want production default %d", targetNative.lastGemma4AssistantDraftTokens, MTPDefaultDraftTokens)
	}
}

func TestSpeculative_LoadLocalGemma4AssistantPair_Good(t *testing.T) {
	if !metal.MetalAvailable() {
		t.Skip("Metal runtime unavailable; skipping local speculative pair smoke")
	}
	targetPath := core.Trim(core.Env("GO_MLX_GEMMA4_TARGET_MODEL"))
	assistantPath := core.Trim(core.Env("GO_MLX_GEMMA4_ASSISTANT_MODEL"))
	if targetPath == "" || assistantPath == "" {
		t.Skip("set GO_MLX_GEMMA4_TARGET_MODEL and GO_MLX_GEMMA4_ASSISTANT_MODEL to run the local speculative pair smoke")
	}
	pair, err := LoadSpeculativePair(targetPath, assistantPath, SpeculativePairConfig{
		TargetOptions:  []LoadOption{WithAutoMemoryPlan(false)},
		DraftOptions:   []LoadOption{WithAutoMemoryPlan(false)},
		TokenizerProbe: []string{"hello"},
	})
	if err != nil {
		t.Fatalf("LoadSpeculativePair(%s, %s): %v", targetPath, assistantPath, err)
	}
	defer pair.Close()
	if pair.Target == nil || pair.Draft != nil || pair.Gemma4Assistant == nil {
		t.Fatalf("pair target=%v draft=%v assistant=%v, want target plus Gemma 4 assistant", pair.Target, pair.Draft, pair.Gemma4Assistant)
	}
	if pair.Report.Draft.Architecture != "gemma4_assistant" {
		t.Fatalf("Report.Draft = %+v, want gemma4_assistant", pair.Report.Draft)
	}
}

func TestSpeculative_LoadSpeculativePair_Bad(t *testing.T) {
	oldLoad := loadNativeModel
	defer func() { loadNativeModel = oldLoad }()

	tokenizer, err := metal.LoadTokenizer(writeRootTokenizer(t))
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}
	targetNative := &fakeNativeModel{
		info:      metal.ModelInfo{Architecture: "gemma4_text", VocabSize: 10, QuantBits: 4, QuantGroup: 64, NumLayers: 1},
		tokenizer: tokenizer,
	}
	draftNative := &fakeNativeModel{
		info:      metal.ModelInfo{Architecture: "gemma4_assistant", VocabSize: 11, QuantBits: 4, QuantGroup: 64, NumLayers: 1},
		tokenizer: tokenizer,
	}
	loadNativeModel = func(path string, _ metal.LoadConfig) (nativeModel, error) {
		if core.Contains(path, "assistant") {
			return draftNative, nil
		}
		return targetNative, nil
	}

	_, err = LoadSpeculativePair("/models/target", "/models/target-assistant", SpeculativePairConfig{
		TargetOptions: []LoadOption{WithAutoMemoryPlan(false)},
		DraftOptions:  []LoadOption{WithAutoMemoryPlan(false)},
	})
	if err == nil {
		t.Fatal("LoadSpeculativePair(vocab mismatch) error = nil, want validation")
	}
	if targetNative.closeCalls == 0 || draftNative.closeCalls == 0 {
		t.Fatalf("closeCalls = target:%d draft:%d, want both closed after validation error", targetNative.closeCalls, draftNative.closeCalls)
	}
}

func TestSpeculative_LoadSpeculativePair_Ugly(t *testing.T) {
	oldLoad := loadNativeModel
	defer func() { loadNativeModel = oldLoad }()

	loadNativeModel = func(path string, _ metal.LoadConfig) (nativeModel, error) {
		tokenizer := &metal.Tokenizer{}
		if core.Contains(path, "assistant") {
			tokenizer = nil
		}
		return &fakeNativeModel{
			info:      metal.ModelInfo{Architecture: path, VocabSize: 10, QuantBits: 4, QuantGroup: 64, NumLayers: 1},
			tokenizer: tokenizer,
		}, nil
	}

	if _, err := LoadSpeculativePair("", "/models/draft", SpeculativePairConfig{}); err == nil {
		t.Fatal("LoadSpeculativePair(empty target) error = nil, want path validation")
	}
	_, err := LoadSpeculativePair("/models/target", "/models/target-assistant", SpeculativePairConfig{
		TargetOptions: []LoadOption{WithAutoMemoryPlan(false)},
		DraftOptions:  []LoadOption{WithAutoMemoryPlan(false)},
	})
	if err == nil {
		t.Fatal("LoadSpeculativePair(nil draft tokenizer) error = nil, want validation")
	}
}
