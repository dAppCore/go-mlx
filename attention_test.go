//go:build darwin && arm64 && !nomlx

package mlx_test

import (
	"context"
	"testing"

	"dappco.re/go/core/inference"
	mlx "dappco.re/go/mlx"
)

func TestMetalAdapterImplementsAttentionInspector_Good(t *testing.T) {
	// Load a real model and verify the adapter implements AttentionInspector.
	b, ok := inference.Get("metal")
	if !ok {
		t.Fatal("metal backend not registered")
	}

	modelPath := gemma3ModelPath(t)
	m, err := b.LoadModel(modelPath)
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}
	defer func() { m.Close(); mlx.ClearCache() }()

	inspector, ok := m.(inference.AttentionInspector)
	if !ok {
		t.Fatal("metalAdapter does not implement AttentionInspector")
	}

	ctx := context.Background()
	snap, err := inspector.InspectAttention(ctx, "What is kindness?")
	if err != nil {
		t.Fatalf("InspectAttention: %v", err)
	}

	if snap.NumLayers == 0 {
		t.Error("NumLayers should be > 0")
	}
	if snap.NumHeads == 0 {
		t.Error("NumHeads should be > 0")
	}
	if snap.SeqLen == 0 {
		t.Error("SeqLen should be > 0")
	}
	if snap.HeadDim == 0 {
		t.Error("HeadDim should be > 0")
	}
	if snap.Architecture == "" {
		t.Error("Architecture should not be empty")
	}
	if len(snap.Keys) != snap.NumLayers {
		t.Errorf("Keys len = %d, want %d (NumLayers)", len(snap.Keys), snap.NumLayers)
	}

	// Verify at least the first layer has data
	if len(snap.Keys[0]) != snap.NumHeads {
		t.Errorf("Keys[0] len = %d, want %d (NumHeads)", len(snap.Keys[0]), snap.NumHeads)
	}

	expectedLen := snap.SeqLen * snap.HeadDim
	if len(snap.Keys[0][0]) != expectedLen {
		t.Errorf("Keys[0][0] len = %d, want %d (SeqLen*HeadDim)", len(snap.Keys[0][0]), expectedLen)
	}

	t.Logf("AttentionSnapshot: arch=%s layers=%d heads=%d seq=%d dim=%d",
		snap.Architecture, snap.NumLayers, snap.NumHeads, snap.SeqLen, snap.HeadDim)
}
