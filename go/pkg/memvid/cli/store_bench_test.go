// SPDX-Licence-Identifier: EUPL-1.2

package cli

import (
	"context"
	"errors"
	"testing"

	"dappco.re/go/mlx/pkg/memvid"
)

func BenchmarkCommandError_Error(b *testing.B) {
	cmdErr := &CommandError{
		Args:   []string{"view", "/tmp/trace.mv2", "--frame-id", "1234", "--json"},
		Stdout: "  some stdout  ",
		Stderr: "  some stderr describing the failure  ",
		Err:    errors.New("exit status 1"),
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = cmdErr.Error()
	}
}

func BenchmarkCommandLooksNotFound(b *testing.B) {
	cmdErr := &CommandError{
		Stdout: "permission denied opening /tmp/trace.mv2",
		Stderr: "frame 42 was not found in segment",
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = commandLooksNotFound(cmdErr)
	}
}

func BenchmarkPut_ArgBuild(b *testing.B) {
	ctx := context.Background()
	runner := func(_ context.Context, _ []byte, _ string, _ ...string) ([]byte, string, string, error) {
		return []byte(`{"memory":{"frame_count":1},"reports":[]}`), "", "", nil
	}
	store, err := Open("/tmp/trace.mv2", WithBinary("/bin/memvid"), withRunner(runner))
	if err != nil {
		b.Fatalf("Open() error = %v", err)
	}
	opts := memvid.PutOptions{
		URI:   "mlx://chunk/1234",
		Title: "trace entry",
		Kind:  "log",
		Track: "session",
		Tags:  map[string]string{"a": "1", "b": "2", "c": "3"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := store.Put(ctx, "payload", opts); err != nil {
			b.Fatalf("Put() error = %v", err)
		}
	}
}

// BenchmarkPut_NoOpts: minimal Put — no URI/title/kind/track/tags/labels. The
// 5 fixed flags + raw-write toggle path. Lowest-overhead Put baseline.
func BenchmarkPut_NoOpts(b *testing.B) {
	ctx := context.Background()
	runner := func(_ context.Context, _ []byte, _ string, _ ...string) ([]byte, string, string, error) {
		return []byte(`{"memory":{"frame_count":1},"reports":[]}`), "", "", nil
	}
	store, err := Open("/tmp/trace.mv2", WithBinary("/bin/memvid"), withRunner(runner))
	if err != nil {
		b.Fatalf("Open() error = %v", err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := store.Put(ctx, "payload", memvid.PutOptions{}); err != nil {
			b.Fatalf("Put() error = %v", err)
		}
	}
}

// BenchmarkPut_ManyTags: 8-tag stress — exercises the keys-slice + sort +
// string concat hot path. Worst-case alloc footprint inside Put.
func BenchmarkPut_ManyTags(b *testing.B) {
	ctx := context.Background()
	runner := func(_ context.Context, _ []byte, _ string, _ ...string) ([]byte, string, string, error) {
		return []byte(`{"memory":{"frame_count":1},"reports":[]}`), "", "", nil
	}
	store, err := Open("/tmp/trace.mv2", WithBinary("/bin/memvid"), withRunner(runner))
	if err != nil {
		b.Fatalf("Open() error = %v", err)
	}
	opts := memvid.PutOptions{
		Tags: map[string]string{
			"a": "1", "b": "2", "c": "3", "d": "4",
			"e": "5", "f": "6", "g": "7", "h": "8",
		},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := store.Put(ctx, "payload", opts); err != nil {
			b.Fatalf("Put() error = %v", err)
		}
	}
}

// BenchmarkPut_Labels: label-only fast path — single append loop, no map sort.
func BenchmarkPut_Labels(b *testing.B) {
	ctx := context.Background()
	runner := func(_ context.Context, _ []byte, _ string, _ ...string) ([]byte, string, string, error) {
		return []byte(`{"memory":{"frame_count":1},"reports":[]}`), "", "", nil
	}
	store, err := Open("/tmp/trace.mv2", WithBinary("/bin/memvid"), withRunner(runner))
	if err != nil {
		b.Fatalf("Open() error = %v", err)
	}
	opts := memvid.PutOptions{
		Labels: []string{"alpha", "beta", "gamma", "delta"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := store.Put(ctx, "payload", opts); err != nil {
			b.Fatalf("Put() error = %v", err)
		}
	}
}

// BenchmarkResolve_ChunkID: Get/Resolve by chunk_id — the random-access
// path that Search calls N times per query and that Snider's State load
// path hits per chunk_id lookup. This is THE golden-path JSON-parse-and-
// build-ref hot loop.
func BenchmarkResolve_ChunkID(b *testing.B) {
	ctx := context.Background()
	runner := func(_ context.Context, _ []byte, _ string, _ ...string) ([]byte, string, string, error) {
		return []byte(`{"frame":{"id":1234,"uri":"mlx://chunk/1234","title":"trace","search_text":"fallback","payload_length":4096,"metadata":{"caption":"caption"}},"content":"payload bytes for chunk 1234"}`), "", "", nil
	}
	store, err := Open("/tmp/trace.mv2", WithBinary("/bin/memvid"), withRunner(runner))
	if err != nil {
		b.Fatalf("Open() error = %v", err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := store.Resolve(ctx, 1234); err != nil {
			b.Fatalf("Resolve() error = %v", err)
		}
	}
}

// BenchmarkResolve_Get: Get is Resolve without the ChunkRef construction.
// Snider's "load text by chunk_id" minimal path.
func BenchmarkResolve_Get(b *testing.B) {
	ctx := context.Background()
	runner := func(_ context.Context, _ []byte, _ string, _ ...string) ([]byte, string, string, error) {
		return []byte(`{"frame":{"id":1234,"search_text":"fallback","metadata":{"caption":"caption"}},"content":"payload bytes for chunk 1234"}`), "", "", nil
	}
	store, err := Open("/tmp/trace.mv2", WithBinary("/bin/memvid"), withRunner(runner))
	if err != nil {
		b.Fatalf("Open() error = %v", err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := store.Get(ctx, 1234); err != nil {
			b.Fatalf("Get() error = %v", err)
		}
	}
}

// BenchmarkResolveURI: URI-keyed resolve — the URI bundle lookup path
// used by ResolveURI consumers (manifest-style lookups).
func BenchmarkResolveURI(b *testing.B) {
	ctx := context.Background()
	runner := func(_ context.Context, _ []byte, _ string, _ ...string) ([]byte, string, string, error) {
		return []byte(`{"frame":{"id":7,"uri":"mlx://bundle/manifest","title":"manifest"},"content":"manifest text"}`), "", "", nil
	}
	store, err := Open("/tmp/trace.mv2", WithBinary("/bin/memvid"), withRunner(runner))
	if err != nil {
		b.Fatalf("Open() error = %v", err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := store.ResolveURI(ctx, "mlx://bundle/manifest"); err != nil {
			b.Fatalf("ResolveURI() error = %v", err)
		}
	}
}

// BenchmarkSearch_SingleHit: Search returns 1 hit, then Resolve is called
// once per hit. Tracks the Search → Resolve fan-out fold cost.
func BenchmarkSearch_SingleHit(b *testing.B) {
	ctx := context.Background()
	runner := func(_ context.Context, _ []byte, _ string, args ...string) ([]byte, string, string, error) {
		switch args[0] {
		case "find":
			return []byte(`{"hits":[{"rank":1,"score":0.75,"frame_id":0,"uri":"mlx://chunk/0","title":"trace","text":"payload"}]}`), "", "", nil
		case "view":
			return []byte(`{"frame":{"id":0,"uri":"mlx://chunk/0","search_text":"fallback"},"content":"payload"}`), "", "", nil
		}
		return nil, "", "", nil
	}
	store, err := Open("/tmp/trace.mv2", WithBinary("/bin/memvid"), withRunner(runner))
	if err != nil {
		b.Fatalf("Open() error = %v", err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := store.Search(ctx, "query", 1); err != nil {
			b.Fatalf("Search() error = %v", err)
		}
	}
}

// BenchmarkSearch_MultiHit: Search returns 8 hits — exercises the hit
// loop + 8× Resolve fan-out. Closer to real cross-segment Search load.
func BenchmarkSearch_MultiHit(b *testing.B) {
	ctx := context.Background()
	runner := func(_ context.Context, _ []byte, _ string, args ...string) ([]byte, string, string, error) {
		switch args[0] {
		case "find":
			return []byte(`{"hits":[
				{"rank":1,"score":0.95,"frame_id":0,"uri":"mlx://chunk/0","title":"alpha","text":"a"},
				{"rank":2,"score":0.85,"frame_id":1,"uri":"mlx://chunk/1","title":"beta","text":"b"},
				{"rank":3,"score":0.75,"frame_id":2,"uri":"mlx://chunk/2","title":"gamma","text":"c"},
				{"rank":4,"score":0.65,"frame_id":3,"uri":"mlx://chunk/3","title":"delta","text":"d"},
				{"rank":5,"score":0.55,"frame_id":4,"uri":"mlx://chunk/4","title":"epsilon","text":"e"},
				{"rank":6,"score":0.45,"frame_id":5,"uri":"mlx://chunk/5","title":"zeta","text":"f"},
				{"rank":7,"score":0.35,"frame_id":6,"uri":"mlx://chunk/6","title":"eta","text":"g"},
				{"rank":8,"score":0.25,"frame_id":7,"uri":"mlx://chunk/7","title":"theta","text":"h"}
			]}`), "", "", nil
		case "view":
			return []byte(`{"frame":{"id":0,"search_text":"x"},"content":"text"}`), "", "", nil
		}
		return nil, "", "", nil
	}
	store, err := Open("/tmp/trace.mv2", WithBinary("/bin/memvid"), withRunner(runner))
	if err != nil {
		b.Fatalf("Open() error = %v", err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := store.Search(ctx, "query", 8); err != nil {
			b.Fatalf("Search() error = %v", err)
		}
	}
}

// BenchmarkViewResponse_Text: bare resolution fallback — Search calls this
// N times per query (once per chunk). 96-byte struct via pointer receiver.
func BenchmarkViewResponse_Text(b *testing.B) {
	cases := []struct {
		name string
		view viewResponse
	}{
		{"content", viewResponse{Content: "from content"}},
		{"caption", func() viewResponse {
			v := viewResponse{}
			v.Frame.Metadata.Caption = "from caption"
			return v
		}()},
		{"search_text", func() viewResponse {
			v := viewResponse{}
			v.Frame.SearchText = "from search text"
			return v
		}()},
	}
	for _, c := range cases {
		c := c
		b.Run(c.name, func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = c.view.text()
			}
		})
	}
}

// BenchmarkLimitOutput: error-path output truncation. Both short (no copy)
// and long (slice + suffix) cases.
func BenchmarkLimitOutput(b *testing.B) {
	short := "memvid: simple error"
	long := make([]byte, 5000)
	for i := range long {
		long[i] = 'x'
	}
	b.Run("short", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = limitOutput(short)
		}
	})
	b.Run("long", func(b *testing.B) {
		s := string(long)
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = limitOutput(s)
		}
	})
}
