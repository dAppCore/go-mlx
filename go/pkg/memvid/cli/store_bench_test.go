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
