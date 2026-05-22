// SPDX-Licence-Identifier: EUPL-1.2

package daemon

import (
	"bytes"
	"context"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
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

// BenchmarkFrameTrimAndParse_Hoisted mirrors the handleConn shape
// where req is declared outside the loop and re-zeroed per frame.
func BenchmarkFrameTrimAndParse_Hoisted(b *testing.B) {
	raw := []byte(`  {"action":"generate","prompt":"ping","model":"main","max_tokens":64,"temperature":0.7}  `)
	var req Request
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		trimmed := bytes.TrimSpace(raw)
		if len(trimmed) == 0 {
			continue
		}
		line := core.AsString(trimmed)
		req = Request{}
		if result := core.JSONUnmarshalString(line, &req); !result.OK {
			b.Fatal(result.Value)
		}
	}
}

// BenchmarkNativeRunner_ModelForCached drives the modelFor read path
// concurrently to exercise the RWMutex read fast-path on a populated
// runner cache. The model is pre-loaded once.
func BenchmarkNativeRunner_ModelForCached(b *testing.B) {
	runner := &NativeGenerateRunner{
		modelPaths: map[string]string{"main": "/m/main"},
		models:     map[string]nativeGenerateModel{"main": &noopGenerateModel{}},
	}
	b.ReportAllocs()
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			_, _ = runner.modelFor("main", "/m/main")
		}
	})
}

type noopGenerateModel struct{}

func (n *noopGenerateModel) GenerateStream(_ context.Context, _ string, _ ...mlx.GenerateOption) <-chan mlx.Token {
	ch := make(chan mlx.Token)
	close(ch)
	return ch
}

func (n *noopGenerateModel) ChatStream(_ context.Context, _ []inference.Message, _ ...mlx.GenerateOption) <-chan mlx.Token {
	ch := make(chan mlx.Token)
	close(ch)
	return ch
}

func (n *noopGenerateModel) WarmPromptCache(string) error { return nil }
func (n *noopGenerateModel) Metrics() mlx.Metrics         { return mlx.Metrics{} }
func (n *noopGenerateModel) Err() error                   { return nil }
func (n *noopGenerateModel) Close() error                 { return nil }

func BenchmarkRegistryActions(b *testing.B) {
	r := NewRegistry("violet", "test")
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = r.Actions()
	}
}

// discardWriter implements core.Writer with zero-cost Write.
type discardWriter struct{}

func (discardWriter) Write(p []byte) (int, error) { return len(p), nil }

// BenchmarkWriteJSONLine_TypicalResp measures the per-response
// marshal-and-emit path used by handleConn.
func BenchmarkWriteJSONLine_TypicalResp(b *testing.B) {
	resp := Response{
		"status": "ok",
		"action": "generate",
		"text":   "the quick brown fox jumps over the lazy dog",
		"model":  "main",
	}
	w := discardWriter{}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := writeJSONLine(w, resp); err != nil {
			b.Fatal(err)
		}
	}
}
