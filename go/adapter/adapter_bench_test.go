// SPDX-Licence-Identifier: EUPL-1.2

// adapter_bench_test.go measures the adapter's per-call allocations on the
// buffered Generate + Chat paths — the surface LEM consumers drive (AX-11). No
// model load: the wrapped inference.TextModel is the in-package stub that yields
// a fixed token slice, so the benchmark isolates the adapter's OWN per-call work
// (the range-over-func accumulation into the value-typed core.Builder, the
// genOptsToInference slice build, the &metrics fetch) from the real engine,
// which lives behind the inference.TextModel interface in pkg/metal.
//
// GenOpts carries both MaxTokens and Temp > 0 so genOptsToInference takes the
// worst-case two-option branch every iteration. The stub yields several tokens
// to exercise the Builder grow path the way a real decode loop does.
//
// External test package — exercises the exported surface exactly as the
// backend.go caller (adapter.New(model, "mlx")) does.

package adapter_test

import (
	"context"
	"testing"

	"dappco.re/go/inference"
	"dappco.re/go/mlx/adapter"
)

// benchTokens is the synthetic stream every buffered benchmark accumulates.
// Eight short tokens approximate a tiny decode without loading a model.
var benchTokens = []inference.Token{
	{Text: "The "}, {Text: "quick "}, {Text: "brown "}, {Text: "fox "},
	{Text: "jumps "}, {Text: "over "}, {Text: "the "}, {Text: "dog."},
}

// benchOpts forces the worst-case two-option genOptsToInference branch.
var benchOpts = adapter.GenOpts{MaxTokens: 64, Temp: 0.2}

func BenchmarkAdapterGenerate(b *testing.B) {
	model := &stubTextModel{tokens: benchTokens}
	a := adapter.New(model, "mlx")
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		result, err := a.Generate(ctx, "prompt", benchOpts)
		if err != nil {
			b.Fatalf("Generate() error = %v", err)
		}
		if result.Text == "" {
			b.Fatal("Generate() returned empty text")
		}
	}
}

func BenchmarkAdapterChat(b *testing.B) {
	model := &stubTextModel{chatTokens: benchTokens}
	a := adapter.New(model, "mlx")
	ctx := context.Background()
	messages := []inference.Message{{Role: "user", Content: "hi"}}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		result, err := a.Chat(ctx, messages, benchOpts)
		if err != nil {
			b.Fatalf("Chat() error = %v", err)
		}
		if result.Text == "" {
			b.Fatal("Chat() returned empty text")
		}
	}
}
