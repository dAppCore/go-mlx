// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for split_remote_ffn.go — the HTTP request-build hot path
// driving an FFN forward across the network. Per AX-11 — ForwardFFN
// fires once per omitted FFN layer per generated token; a 32-layer
// split with 4 omitted layers generating 100 tokens issues 400 calls
// per Generate. Header-set + payload-marshal allocations all show up
// in this hot loop.
//
// Run:    go test -bench='BenchmarkRemoteSplitFFN' -benchmem -run='^$' ./go

package mlx

import (
	"context"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

// Sinks defeat compiler DCE.
var (
	remoteSplitFFNBenchSinkResult SplitFFNResult
	remoteSplitFFNBenchSinkErr    error
)

// --- ForwardFFN end-to-end via in-process HTTP test server ---

func BenchmarkRemoteSplitFFN_ForwardFFN_NoExtraHeaders(b *testing.B) {
	srv := core.NewHTTPTestServer(core.HandlerFunc(func(w core.ResponseWriter, r *core.Request) {
		w.Header().Set("Content-Type", "application/json")
		core.WriteString(w, core.JSONMarshalString(RemoteSplitFFNResponse{Hidden: []float32{1, 2, 3}}))
	}))
	defer srv.Close()
	executor, err := NewRemoteSplitFFNExecutor(RemoteSplitFFNConfig{
		Endpoint: inference.SplitEndpoint{
			ID:   "ffn-bench",
			Role: inference.SplitEndpointRoleFFN,
			URL:  srv.URL,
		},
	})
	if err != nil {
		b.Fatalf("NewRemoteSplitFFNExecutor: %v", err)
	}
	hidden := []float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		remoteSplitFFNBenchSinkResult, remoteSplitFFNBenchSinkErr = executor.ForwardFFN(ctx, SplitFFNRequest{
			Layer:  i % 32,
			Hidden: hidden,
		})
	}
}

func BenchmarkRemoteSplitFFN_ForwardFFN_WithHeadersAndLabels(b *testing.B) {
	srv := core.NewHTTPTestServer(core.HandlerFunc(func(w core.ResponseWriter, r *core.Request) {
		w.Header().Set("Content-Type", "application/json")
		core.WriteString(w, core.JSONMarshalString(RemoteSplitFFNResponse{Hidden: []float32{1, 2, 3}}))
	}))
	defer srv.Close()
	executor, err := NewRemoteSplitFFNExecutor(RemoteSplitFFNConfig{
		Endpoint: inference.SplitEndpoint{
			ID:   "ffn-bench",
			Role: inference.SplitEndpointRoleFFN,
			URL:  srv.URL,
			Labels: map[string]string{
				"shard":   "0",
				"region":  "eu-west-1",
				"version": "v1",
			},
		},
		Headers: map[string]string{
			"Authorization": "Bearer secret-token",
			"X-Trace-Id":    "trace-abc-123",
			"X-Tenant-Id":   "tenant-42",
		},
	})
	if err != nil {
		b.Fatalf("NewRemoteSplitFFNExecutor: %v", err)
	}
	hidden := []float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		remoteSplitFFNBenchSinkResult, remoteSplitFFNBenchSinkErr = executor.ForwardFFN(ctx, SplitFFNRequest{
			Layer:  i % 32,
			Hidden: hidden,
		})
	}
}

// --- Constructor — fires once per split-inference plan ---

func BenchmarkRemoteSplitFFN_NewExecutor_NoExtraHeaders(b *testing.B) {
	cfg := RemoteSplitFFNConfig{
		Endpoint: inference.SplitEndpoint{
			ID:   "ffn-bench",
			Role: inference.SplitEndpointRoleFFN,
			URL:  "http://localhost:8080/ffn",
		},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		exec, err := NewRemoteSplitFFNExecutor(cfg)
		if err != nil {
			b.Fatalf("NewRemoteSplitFFNExecutor: %v", err)
		}
		_ = exec.userHeader // touch field
	}
}

func BenchmarkRemoteSplitFFN_NewExecutor_WithHeadersAndLabels(b *testing.B) {
	cfg := RemoteSplitFFNConfig{
		Endpoint: inference.SplitEndpoint{
			ID:   "ffn-bench",
			Role: inference.SplitEndpointRoleFFN,
			URL:  "http://localhost:8080/ffn",
			Labels: map[string]string{
				"shard":  "0",
				"region": "eu-west-1",
			},
		},
		Headers: map[string]string{
			"Authorization": "Bearer secret-token",
			"X-Trace-Id":    "trace-abc-123",
		},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		exec, err := NewRemoteSplitFFNExecutor(cfg)
		if err != nil {
			b.Fatalf("NewRemoteSplitFFNExecutor: %v", err)
		}
		_ = exec.userHeader // touch field
	}
}
