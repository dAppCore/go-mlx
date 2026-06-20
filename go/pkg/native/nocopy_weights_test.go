// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

func TestShardBuffersEmptyWeightIsOptional(t *testing.T) {
	var sb shardBuffers
	got, err := sb.bufFor(nil)
	if err != nil {
		t.Fatalf("bufFor(nil): %v", err)
	}
	if got.buf != nil || got.off != 0 {
		t.Fatalf("bufFor(nil) = %+v, want zero bufView", got)
	}
}

func TestShardBuffersRejectsForeignWeight(t *testing.T) {
	weight := []byte{1, 2, 3, 4}
	sb := shardBuffers{bases: []uintptr{1}, ends: []uintptr{2}}
	if _, err := sb.bufFor(weight); err == nil {
		t.Fatal("expected bufFor to reject a weight outside mapped shards")
	}
}

func TestShardBuffersCloseIsNilSafe(t *testing.T) {
	var sb *shardBuffers
	if err := sb.Close(); err != nil {
		t.Fatalf("nil Close: %v", err)
	}
	if err := (&shardBuffers{}).Close(); err != nil {
		t.Fatalf("empty Close: %v", err)
	}
}
