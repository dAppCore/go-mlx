// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"context"
	"testing"
)

// The CaptureKV* capture pipeline runs a real prefill pass and needs a loaded
// model — that path is exercised by the model-package suites. The nil-model
// guard each exported entry point defends is reachable here without a model.

func TestKVSnapshot_CaptureKV_Bad(t *testing.T) {
	var m *Model

	if got, err := m.CaptureKV(context.Background(), "hi"); err == nil || got != nil {
		t.Fatalf("(*Model)(nil).CaptureKV() = (%v,%v), want a nil-model error", got, err)
	}
	if got, err := m.CaptureKVWithOptions(context.Background(), "hi", KVSnapshotCaptureOptions{}); err == nil || got != nil {
		t.Fatalf("(*Model)(nil).CaptureKVWithOptions() = (%v,%v), want a nil-model error", got, err)
	}
}

func TestKVSnapshot_CaptureKVChunks_Bad(t *testing.T) {
	var m *Model
	chunks := func(yield func(string) bool) { yield("hi") }

	if got, err := m.CaptureKVChunks(context.Background(), chunks); err == nil || got != nil {
		t.Fatalf("(*Model)(nil).CaptureKVChunks() = (%v,%v), want a nil-model error", got, err)
	}
	if got, err := m.CaptureKVChunksWithOptions(context.Background(), chunks, KVSnapshotCaptureOptions{}); err == nil || got != nil {
		t.Fatalf("(*Model)(nil).CaptureKVChunksWithOptions() = (%v,%v), want a nil-model error", got, err)
	}
}
