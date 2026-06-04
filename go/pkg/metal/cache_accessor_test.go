// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import "testing"

func TestKVCache_Accessors_Good(t *testing.T) {
	c := &KVCache{offset: 7, step: 256}
	if got := c.Offset(); got != 7 {
		t.Fatalf("Offset() = %d, want 7", got)
	}
	if got := c.Step(); got != 256 {
		t.Fatalf("Step() = %d, want 256", got)
	}
}

func TestRotatingKVCache_Accessors_Good(t *testing.T) {
	c := &RotatingKVCache{maxSize: 1024}
	if got := c.MaxSize(); got != 1024 {
		t.Fatalf("MaxSize() = %d, want 1024", got)
	}
}

func TestFixedKVCache_Accessors_Good(t *testing.T) {
	c := &FixedKVCache{maxSize: 512}
	if got := c.MaxSize(); got != 512 {
		t.Fatalf("MaxSize() = %d, want 512", got)
	}
}

func TestPagedKVCache_Accessors_Good(t *testing.T) {
	c := &PagedKVCache{maxSize: 4096, pageSize: 256}
	if got := c.MaxSize(); got != 4096 {
		t.Fatalf("MaxSize() = %d, want 4096", got)
	}
	if got := c.PageSize(); got != 256 {
		t.Fatalf("PageSize() = %d, want 256", got)
	}
}

func TestQuantizedKVCache_Accessors_Good(t *testing.T) {
	c := &QuantizedKVCache{maxSize: 2048, step: 256, keyBits: 8, valueBits: 4}
	if got := c.MaxSize(); got != 2048 {
		t.Fatalf("MaxSize() = %d, want 2048", got)
	}
	if got := c.Step(); got != 256 {
		t.Fatalf("Step() = %d, want 256", got)
	}
	k, v := c.Bits()
	if k != 8 {
		t.Fatalf("Bits() key = %d, want 8", k)
	}
	if v != 4 {
		t.Fatalf("Bits() value = %d, want 4", v)
	}
}
