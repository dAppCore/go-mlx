// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the OpenAI-compatibility admin handlers — health,
// wake/sleep, cache-entries. Per AX-11 — these run on the same
// process as the wire handlers and end up in liveness probes /
// monitoring loops that hit the endpoint at a high steady rate
// (orchestrators ping /v1/health every few seconds). The label
// parser also fires per cache-entries request and scales with the
// number of query-string filters supplied by the caller.
//
// Run:    go test -bench='BenchmarkAdmin' -benchtime=100ms -benchmem -run='^$' ./go/openai

package openai

import (
	"context"
	"iter"
	"net/http"
	"net/http/httptest"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	openaicompat "dappco.re/go/inference/openai"
)

// Sinks defeat compiler DCE.
var (
	adminBenchSinkLabels map[string]string
	adminBenchSinkString string
	adminBenchSinkCode   int
)

// --- adminCacheEntryLabels — pure query-string fan-out ---

func BenchmarkAdmin_CacheEntryLabels_NoFilters(b *testing.B) {
	req := httptest.NewRequest(http.MethodGet, DefaultAdminCacheEntriesPath+"?model=qwen3", nil)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		adminBenchSinkLabels = adminCacheEntryLabels(req)
	}
}

func BenchmarkAdmin_CacheEntryLabels_FewFilters(b *testing.B) {
	req := httptest.NewRequest(http.MethodGet,
		DefaultAdminCacheEntriesPath+"?model=qwen3&tenant=local&adapter=probe-lora",
		nil)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		adminBenchSinkLabels = adminCacheEntryLabels(req)
	}
}

func BenchmarkAdmin_CacheEntryLabels_ManyFilters(b *testing.B) {
	// Eight labels — realistic upper bound for orchestrator-driven
	// fan-out queries (tenant + adapter + region + workload + role
	// + version + cohort + env).
	req := httptest.NewRequest(http.MethodGet,
		DefaultAdminCacheEntriesPath+"?model=qwen3&tenant=local&adapter=probe-lora&region=eu&workload=chat&role=primary&version=1&cohort=a&env=prod",
		nil)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		adminBenchSinkLabels = adminCacheEntryLabels(req)
	}
}

// --- adminHealthHandler.ServeHTTP — default body assembly path ---

func BenchmarkAdmin_HealthHandler_Default(b *testing.B) {
	resolver := openaicompat.NewStaticResolver(map[string]inference.TextModel{"qwen": &adminBenchMockModel{}})
	handler := &adminHealthHandler{resolver: resolver}
	req := httptest.NewRequest(http.MethodGet, DefaultHealthPath, nil)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rec := httptest.NewRecorder()
		handler.ServeHTTP(rec, req)
		adminBenchSinkCode = rec.Code
	}
}

// Custom callback path — same handler but exercises the user-supplied
// Health closure + the post-fill defaulting branches.
func BenchmarkAdmin_HealthHandler_Custom(b *testing.B) {
	resolver := openaicompat.NewStaticResolver(map[string]inference.TextModel{"qwen": &adminBenchMockModel{}})
	cfg := AdminConfig{
		Health: func(context.Context) (Health, error) {
			return Health{Status: "ok", Runtime: "go-mlx", Models: []string{"qwen3"}}, nil
		},
	}
	handler := &adminHealthHandler{resolver: resolver, cfg: cfg}
	req := httptest.NewRequest(http.MethodGet, DefaultHealthPath, nil)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rec := httptest.NewRecorder()
		handler.ServeHTTP(rec, req)
		adminBenchSinkCode = rec.Code
	}
}

// --- adminActionHandler.ServeHTTP — wake/sleep callback dispatch ---

func BenchmarkAdmin_ActionHandler_Wake(b *testing.B) {
	handler := &adminActionHandler{action: "wake", callback: func(context.Context) error { return nil }}
	req := httptest.NewRequest(http.MethodPost, DefaultAdminWakePath, nil)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rec := httptest.NewRecorder()
		handler.ServeHTTP(rec, req)
		adminBenchSinkCode = rec.Code
	}
}

// --- adminCacheEntriesHandler.ServeHTTP — full happy path with lister ---

func BenchmarkAdmin_CacheEntriesHandler_TypicalEntries(b *testing.B) {
	entries := []inference.CacheBlockRef{
		{ID: "blk-a", Kind: "prefix", TokenCount: 256, Labels: map[string]string{"tenant": "local"}},
		{ID: "blk-b", Kind: "prefix", TokenCount: 256, Labels: map[string]string{"tenant": "local"}},
		{ID: "blk-c", Kind: "prefix", TokenCount: 128, Labels: map[string]string{"tenant": "local"}},
		{ID: "blk-d", Kind: "prefix", TokenCount: 64, Labels: map[string]string{"tenant": "local"}},
	}
	model := &adminBenchMockModel{entries: entries}
	resolver := openaicompat.NewStaticResolver(map[string]inference.TextModel{"qwen": model})
	handler := &adminCacheEntriesHandler{resolver: resolver}
	req := httptest.NewRequest(http.MethodGet, DefaultAdminCacheEntriesPath+"?model=qwen&tenant=local", nil)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rec := httptest.NewRecorder()
		handler.ServeHTTP(rec, req)
		adminBenchSinkCode = rec.Code
	}
}

// --- Health body marshal — what every health probe writes back ---

func BenchmarkAdmin_HealthBodyMarshal(b *testing.B) {
	health := Health{
		Status:  "ok",
		Runtime: "go-mlx",
		Models:  []string{"qwen3", "gemma4-2b", "llama3-8b"},
		Time:    1716297600,
		Labels:  map[string]string{"region": "eu-west", "tenant": "local"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		adminBenchSinkString = core.JSONMarshalString(health)
	}
}

func BenchmarkAdmin_ActionResponseMarshal(b *testing.B) {
	resp := ActionResponse{
		Action: "wake",
		Status: "ok",
		Labels: map[string]string{"runtime": "go-mlx"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		adminBenchSinkString = core.JSONMarshalString(resp)
	}
}

// adminBenchMockModel is a minimal TextModel + CacheEntryLister +
// CacheService that satisfies the resolver + entries-handler path
// without dragging the GPU-backed metal model into the bench.
type adminBenchMockModel struct {
	entries []inference.CacheBlockRef
}

func (m *adminBenchMockModel) Generate(context.Context, string, ...inference.GenerateOption) iter.Seq[inference.Token] {
	return func(func(inference.Token) bool) {}
}

func (m *adminBenchMockModel) Chat(context.Context, []inference.Message, ...inference.GenerateOption) iter.Seq[inference.Token] {
	return func(func(inference.Token) bool) {}
}

func (m *adminBenchMockModel) Classify(context.Context, []string, ...inference.GenerateOption) ([]inference.ClassifyResult, error) {
	return nil, nil
}

func (m *adminBenchMockModel) BatchGenerate(context.Context, []string, ...inference.GenerateOption) ([]inference.BatchResult, error) {
	return nil, nil
}

func (m *adminBenchMockModel) ModelType() string { return "mock" }
func (m *adminBenchMockModel) Info() inference.ModelInfo {
	return inference.ModelInfo{Architecture: "qwen3"}
}
func (m *adminBenchMockModel) Metrics() inference.GenerateMetrics { return inference.GenerateMetrics{} }
func (m *adminBenchMockModel) Err() error                         { return nil }
func (m *adminBenchMockModel) Close() error                       { return nil }

func (m *adminBenchMockModel) CacheEntries(context.Context, map[string]string) ([]inference.CacheBlockRef, error) {
	return append([]inference.CacheBlockRef(nil), m.entries...), nil
}

func (m *adminBenchMockModel) CacheStats(context.Context) (inference.CacheStats, error) {
	return inference.CacheStats{Blocks: len(m.entries), CacheMode: "block-q8"}, nil
}
