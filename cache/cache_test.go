//go:build darwin && arm64

package cache

import (
	"testing"

	"forge.lthn.ai/core/go-mlx"
)

// makeKV creates a small K/V pair with shape [B=1, H=2, L=seqLen, D=4].
func makeKV(seqLen int) (*mlx.Array, *mlx.Array) {
	size := 1 * 2 * seqLen * 4
	data := make([]float32, size)
	for i := range data {
		data[i] = float32(i) * 0.1
	}
	k := mlx.FromValues(data, 1, 2, seqLen, 4)
	v := mlx.FromValues(data, 1, 2, seqLen, 4)
	return k, v
}

// --- KVCache ---

func TestKVCache_New(t *testing.T) {
	c := NewKVCache()
	if c.Offset() != 0 {
		t.Errorf("offset = %d, want 0", c.Offset())
	}
	if c.Len() != 0 {
		t.Errorf("len = %d, want 0", c.Len())
	}
	if c.State() != nil {
		t.Error("state should be nil for empty cache")
	}
}

func TestKVCache_SingleUpdate(t *testing.T) {
	c := NewKVCache()
	k, v := makeKV(3) // 3 tokens

	outK, outV := c.Update(k, v, 3)
	mlx.Materialize(outK, outV)

	if c.Offset() != 3 {
		t.Errorf("offset = %d, want 3", c.Offset())
	}
	if c.Len() != 3 {
		t.Errorf("len = %d, want 3", c.Len())
	}

	// Output K should have shape [1, 2, 3, 4]
	shape := outK.Shape()
	if shape[0] != 1 || shape[1] != 2 || shape[2] != 3 || shape[3] != 4 {
		t.Errorf("outK shape = %v, want [1 2 3 4]", shape)
	}
}

func TestKVCache_MultipleUpdates(t *testing.T) {
	c := NewKVCache()

	// Prompt: 5 tokens
	k1, v1 := makeKV(5)
	outK, outV := c.Update(k1, v1, 5)
	mlx.Materialize(outK, outV)

	if c.Offset() != 5 {
		t.Errorf("offset = %d, want 5", c.Offset())
	}

	// Generate: 1 token at a time
	k2, v2 := makeKV(1)
	outK, outV = c.Update(k2, v2, 1)
	mlx.Materialize(outK, outV)

	if c.Offset() != 6 {
		t.Errorf("offset = %d, want 6", c.Offset())
	}

	shape := outK.Shape()
	if shape[2] != 6 {
		t.Errorf("outK L dim = %d, want 6", shape[2])
	}
}

func TestKVCache_Reset(t *testing.T) {
	c := NewKVCache()
	k, v := makeKV(3)
	c.Update(k, v, 3)

	c.Reset()

	if c.Offset() != 0 {
		t.Errorf("offset after reset = %d, want 0", c.Offset())
	}
	if c.State() != nil {
		t.Error("state should be nil after reset")
	}
}

func TestKVCache_State(t *testing.T) {
	c := NewKVCache()
	k, v := makeKV(2)
	c.Update(k, v, 2)

	state := c.State()
	if len(state) != 2 {
		t.Fatalf("state length = %d, want 2", len(state))
	}
	// state[0] = keys, state[1] = values
	if state[0] == nil || state[1] == nil {
		t.Error("state arrays should not be nil")
	}
}

// --- RotatingKVCache ---

func TestRotatingKVCache_New(t *testing.T) {
	c := NewRotatingKVCache(16)
	if c.Offset() != 0 {
		t.Errorf("offset = %d, want 0", c.Offset())
	}
	if c.Len() != 0 {
		t.Errorf("len = %d, want 0", c.Len())
	}
}

func TestRotatingKVCache_SingleToken(t *testing.T) {
	c := NewRotatingKVCache(8)
	k, v := makeKV(1)

	outK, outV := c.Update(k, v, 1)
	mlx.Materialize(outK, outV)

	if c.Offset() != 1 {
		t.Errorf("offset = %d, want 1", c.Offset())
	}
	if c.Len() != 1 {
		t.Errorf("len = %d, want 1", c.Len())
	}
}

func TestRotatingKVCache_MultiTokenPrompt(t *testing.T) {
	c := NewRotatingKVCache(16)
	k, v := makeKV(5)

	outK, outV := c.Update(k, v, 5)
	mlx.Materialize(outK, outV)

	if c.Offset() != 5 {
		t.Errorf("offset = %d, want 5", c.Offset())
	}
	if c.Len() != 5 {
		t.Errorf("len = %d, want 5", c.Len())
	}
}

func TestRotatingKVCache_Bounded(t *testing.T) {
	c := NewRotatingKVCache(4)

	// Fill with 4-token prompt (at max)
	k, v := makeKV(4)
	outK, outV := c.Update(k, v, 4)
	mlx.Materialize(outK, outV)

	if c.Len() != 4 {
		t.Errorf("len = %d, want 4 (at max)", c.Len())
	}

	// Add one more token — should trim to maxSize
	k2, v2 := makeKV(1)
	outK, outV = c.Update(k2, v2, 1)
	mlx.Materialize(outK, outV)

	if c.Offset() != 5 {
		t.Errorf("offset = %d, want 5", c.Offset())
	}
	// Len should be bounded by maxSize
	if c.Len() != 4 {
		t.Errorf("len = %d, want 4 (bounded)", c.Len())
	}
}

func TestRotatingKVCache_Reset(t *testing.T) {
	c := NewRotatingKVCache(8)
	k, v := makeKV(3)
	c.Update(k, v, 3)

	c.Reset()

	if c.Offset() != 0 {
		t.Errorf("offset after reset = %d, want 0", c.Offset())
	}
	if c.Len() != 0 {
		t.Errorf("len after reset = %d, want 0", c.Len())
	}
	if c.State() != nil {
		t.Error("state should be nil after reset")
	}
}
