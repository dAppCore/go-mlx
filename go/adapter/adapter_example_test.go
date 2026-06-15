// SPDX-Licence-Identifier: EUPL-1.2

// Runnable examples for the adapter surface — one per public symbol, as the
// v0.9.0 audit TEST-STANDARD requires. Each example drives the adapter through
// the in-package stubTextModel (defined in adapter_test.go) so the output is
// deterministic without loading a real inference engine, mirroring exactly how
// a LEM consumer wires adapter.New(model, "mlx") over a loaded model.

package adapter_test

import (
	"context"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/mlx/adapter"
)

func ExampleNew() {
	model := &stubTextModel{}
	a := adapter.New(model, "mlx")
	core.Println(a.Name(), a.Available())
	// Output: mlx true
}

func ExampleAdapter_Name() {
	a := adapter.New(&stubTextModel{}, "mlx")
	core.Println(a.Name())
	// Output: mlx
}

func ExampleAdapter_Available() {
	a := adapter.New(&stubTextModel{}, "mlx")
	core.Println(a.Available())
	// Available reports false once the model is released.
	_ = a.Close()
	core.Println(a.Available())
	// Output:
	// true
	// false
}

func ExampleAdapter_Model() {
	model := &stubTextModel{}
	a := adapter.New(model, "mlx")
	core.Println(a.Model() == model)
	// Output: true
}

func ExampleAdapter_Close() {
	a := adapter.New(&stubTextModel{}, "mlx")
	core.Println(a.Close())
	// Close is idempotent — a second call is a clean no-op.
	core.Println(a.Close())
	// Output:
	// <nil>
	// <nil>
}

func ExampleAdapter_Generate() {
	model := &stubTextModel{tokens: []inference.Token{{Text: "Hello"}, {Text: " world"}}}
	a := adapter.New(model, "mlx")
	result, _ := a.Generate(context.Background(), "greet", adapter.GenOpts{MaxTokens: 16})
	core.Println(result.Text)
	// Output: Hello world
}

func ExampleAdapter_GenerateStream() {
	model := &stubTextModel{tokens: []inference.Token{{Text: "a"}, {Text: "b"}, {Text: "c"}}}
	a := adapter.New(model, "mlx")
	var streamed string
	_ = a.GenerateStream(context.Background(), "stream", adapter.GenOpts{}, func(token string) error {
		streamed += token
		return nil
	})
	core.Println(streamed)
	// Output: abc
}

func ExampleAdapter_Chat() {
	model := &stubTextModel{chatTokens: []inference.Token{{Text: "chat"}, {Text: " reply"}}}
	a := adapter.New(model, "mlx")
	result, _ := a.Chat(context.Background(), []inference.Message{{Role: "user", Content: "hi"}}, adapter.GenOpts{})
	core.Println(result.Text)
	// Output: chat reply
}

func ExampleAdapter_ChatStream() {
	model := &stubTextModel{chatTokens: []inference.Token{{Text: "x"}, {Text: "y"}, {Text: "z"}}}
	a := adapter.New(model, "mlx")
	var streamed string
	_ = a.ChatStream(context.Background(), []inference.Message{{Role: "user", Content: "hi"}}, adapter.GenOpts{}, func(token string) error {
		streamed += token
		return nil
	})
	core.Println(streamed)
	// Output: xyz
}

func ExampleAdapter_InspectAttention() {
	model := &stubTextModel{attention: &inference.AttentionSnapshot{NumLayers: 2, Architecture: "gemma3"}}
	a := adapter.New(model, "mlx")
	snap, _ := a.InspectAttention(context.Background(), "prompt")
	core.Println(snap.Architecture, snap.NumLayers)
	// Output: gemma3 2
}
