// SPDX-Licence-Identifier: EUPL-1.2

package daemon

import (
	"context"

	core "dappco.re/go"
	"dappco.re/go/inference"
	mlx "dappco.re/go/mlx"
)

// Runnable example that invokes the native runner constructor and reports the
// resolved default model name (no model is loaded — the backend is stubbed).
func ExampleNewNativeGenerateRunner() {
	runner := NewNativeGenerateRunner(NativeGenerateConfig{
		ModelPaths: map[string]string{"default": "/models/main"},
	})
	defer func() { _ = runner.Close() }()
	core.Println(runner.defaultModel)
	// Output: default
}

// exampleNativeModel is a deterministic stand-in for a loaded go-mlx model so
// the runnable examples never touch real weights.
type exampleNativeModel struct{ closed bool }

func (m *exampleNativeModel) GenerateStream(_ context.Context, _ string, _ ...mlx.GenerateOption) <-chan mlx.Token {
	ch := make(chan mlx.Token, 1)
	ch <- mlx.Token{Text: "hi"}
	close(ch)
	return ch
}

func (m *exampleNativeModel) ChatStream(_ context.Context, _ []inference.Message, _ ...mlx.GenerateOption) <-chan mlx.Token {
	ch := make(chan mlx.Token)
	close(ch)
	return ch
}
func (m *exampleNativeModel) WarmPromptCache(string) error { return nil }
func (m *exampleNativeModel) Metrics() mlx.Metrics         { return mlx.Metrics{} }
func (m *exampleNativeModel) Err() error                   { return nil }
func (m *exampleNativeModel) Close() error                 { m.closed = true; return nil }

// ExampleNativeGenerateRunner_Generate runs a prompt through a cached native
// model (stubbed here) and prints the generated text.
func ExampleNativeGenerateRunner_Generate() {
	runner := NewNativeGenerateRunner(NativeGenerateConfig{
		ModelPaths: map[string]string{"default": "/models/main"},
	})
	runner.loadModel = func(string, ...mlx.LoadOption) (nativeGenerateModel, error) {
		return &exampleNativeModel{}, nil
	}
	defer func() { _ = runner.Close() }()

	result, err := runner.Generate(context.Background(), GenerateRequest{Prompt: "hello"})
	if err != nil {
		core.Println(err)
		return
	}
	core.Println(result.Text)
	// Output: hi
}

// ExampleNativeGenerateRunner_Close loads a model, then releases it and reports
// that the runner closed cleanly.
func ExampleNativeGenerateRunner_Close() {
	model := &exampleNativeModel{}
	runner := NewNativeGenerateRunner(NativeGenerateConfig{
		ModelPaths: map[string]string{"default": "/models/main"},
	})
	runner.loadModel = func(string, ...mlx.LoadOption) (nativeGenerateModel, error) {
		return model, nil
	}
	if _, err := runner.Generate(context.Background(), GenerateRequest{Prompt: "hello"}); err != nil {
		core.Println(err)
		return
	}
	core.Println(runner.Close())
	core.Println(model.closed)
	// Output:
	// <nil>
	// true
}
