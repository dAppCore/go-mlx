// SPDX-Licence-Identifier: EUPL-1.2

package daemon

import (
	"bytes"
	"context"
	"testing"

	core "dappco.re/go"
	mlx "dappco.re/go/mlx"
)

func BenchmarkGenerateRequestFromRequest(b *testing.B) {
	req := Request{
		Prompt:      "ping",
		Model:       "main",
		Messages:    []Message{{Role: "system", Content: "you are helpful"}, {Role: "user", Content: "hello"}, {Role: "assistant", Content: "hi"}},
		MaxTokens:   64,
		Temperature: 0.7,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = generateRequestFromRequest(req)
	}
}

func BenchmarkCopyStringMap(b *testing.B) {
	in := map[string]string{
		"default":  "/models/qwen",
		"backup":   "/models/llama",
		"thinking": "/models/gemma",
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = copyStringMap(in)
	}
}

func BenchmarkNormalizeAction(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = normalizeAction("  GENERATE  ")
	}
}

func BenchmarkRegistryDispatch_Stub(b *testing.B) {
	r := NewRegistry("violet", "test")
	ctx := context.Background()
	req := Request{Action: "info"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = r.Dispatch(ctx, req)
	}
}

func BenchmarkNewNativeGenerateRunner(b *testing.B) {
	cfg := NativeGenerateConfig{
		ModelPaths: map[string]string{
			"default": "/m/qwen",
			"backup":  "/m/llama",
		},
		DefaultModelName: "default",
		DefaultMaxTokens: 256,
		LoadOptions:      []mlx.LoadOption{},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = NewNativeGenerateRunner(cfg)
	}
}

func BenchmarkToMLXMessages(b *testing.B) {
	msgs := []Message{
		{Role: "system", Content: "you are helpful"},
		{Role: "user", Content: "hello"},
		{Role: "assistant", Content: "hi"},
		{Role: "user", Content: "explain"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = toMLXMessages(msgs)
	}
}

// BenchmarkFrameTrimAndParse measures the per-frame normalize-and-parse
// pair that runs inside handleConn for every request.
func BenchmarkFrameTrimAndParse(b *testing.B) {
	raw := []byte(`  {"action":"generate","prompt":"ping","model":"main","max_tokens":64,"temperature":0.7}  `)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		trimmed := bytes.TrimSpace(raw)
		if len(trimmed) == 0 {
			continue
		}
		line := core.AsString(trimmed)
		var req Request
		if result := core.JSONUnmarshalString(line, &req); !result.OK {
			b.Fatal(result.Value)
		}
	}
}

func BenchmarkRegistryActions(b *testing.B) {
	r := NewRegistry("violet", "test")
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = r.Actions()
	}
}
