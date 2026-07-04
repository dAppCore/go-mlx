// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"

	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/metal"
)

func ExampleModel_GenerateSpeculative() {
	target := &Model{model: &fakeNativeModel{tokens: []metal.Token{
		{ID: 1, Text: "A"},
		{ID: 2, Text: "B"},
	}}}
	draft := &Model{model: &fakeNativeModel{tokens: []metal.Token{
		{ID: 1, Text: "A"},
		{ID: 3, Text: "C"},
	}}}

	result, err := target.GenerateSpeculative(context.Background(), draft, "prompt", SpeculativeDecodeConfig{
		MaxTokens:   2,
		DraftTokens: 2,
	})

	core.Println(err == nil, result.Text, result.Metrics.AcceptedTokens, result.Metrics.RejectedTokens)
	// Output: true AB 1 1
}

func ExampleLoadSpeculativePair() {
	tokenizer, cleanup, ok := exampleSpeculativeTokenizer()
	if !ok {
		return
	}
	defer cleanup()
	oldLoad := loadNativeModel
	defer func() { loadNativeModel = oldLoad }()
	loadNativeModel = func(path string, _ metal.LoadConfig) (NativeModel, error) {
		return &fakeNativeModel{
			info:      metal.ModelInfo{Architecture: path, VocabSize: 256, QuantBits: 4, QuantGroup: 64, NumLayers: 1},
			tokenizer: tokenizer,
			tokens:    []metal.Token{{ID: 1, Text: "A"}},
		}, nil
	}

	pair, err := LoadSpeculativePair("/models/target", "/models/draft", SpeculativePairConfig{
		TargetOptions:  []LoadOption{WithAutoMemoryPlan(false)},
		DraftOptions:   []LoadOption{WithAutoMemoryPlan(false)},
		TokenizerProbe: []string{"hello"},
	})
	if pair != nil {
		defer pair.Close()
	}

	core.Println(err == nil, pair.Report.Target.VocabSize, len(pair.Report.TokenizerProbe))
	// Output: true 256 1
}

func ExampleSpeculativePair_Generate() {
	pair := &SpeculativePair{
		Target: &Model{model: &fakeNativeModel{tokens: []metal.Token{
			{ID: 1, Text: "A"},
			{ID: 2, Text: "B"},
		}}},
		Draft: &Model{model: &fakeNativeModel{tokens: []metal.Token{
			{ID: 1, Text: "A"},
			{ID: 3, Text: "C"},
		}}},
	}

	result, err := pair.Generate(context.Background(), "prompt", SpeculativeDecodeConfig{MaxTokens: 2, DraftTokens: 2})

	core.Println(err == nil, result.Text, result.Metrics.AcceptedTokens, result.Metrics.RejectedTokens)
	// Output: true AB 1 1
}

func ExampleSpeculativePair_Close() {
	targetNative := &fakeNativeModel{}
	draftNative := &fakeNativeModel{}
	pair := &SpeculativePair{
		Target: &Model{model: targetNative},
		Draft:  &Model{model: draftNative},
	}

	err := pair.Close()

	core.Println(err == nil, targetNative.closeCalls, draftNative.closeCalls)
	// Output: true 1 1
}

func exampleSpeculativeTokenizer() (*metal.Tokenizer, func(), bool) {
	dirResult := core.MkdirTemp("", "go-mlx-speculative-example-*")
	if !dirResult.OK {
		return nil, func() {}, false
	}
	dir := dirResult.Value.(string)
	path := core.PathJoin(dir, "tokenizer.json")
	if result := core.WriteFile(path, []byte(rootTokenizerJSON), 0o644); !result.OK {
		core.RemoveAll(dir)
		return nil, func() {}, false
	}
	tokenizer, err := metal.LoadTokenizer(path)
	if err != nil {
		core.RemoveAll(dir)
		return nil, func() {}, false
	}
	return tokenizer, func() { core.RemoveAll(dir) }, true
}
