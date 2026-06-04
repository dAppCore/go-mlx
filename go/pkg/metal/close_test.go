// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"testing"
)

func TestClose_FreeLinear_Good(t *testing.T) {
	coverageTokens := "FreeLinear"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	w := FromValues([]float32{1, 2, 3, 4}, 2, 2)
	bias := FromValues([]float32{0.1, 0.2}, 2)
	Materialize(w, bias)

	l := NewLinear(w, bias)
	FreeLinear(l)

	if w.Valid() {
		t.Error("weight should be freed")
	}
	if bias.Valid() {
		t.Error("bias should be freed")
	}
}

func TestClose_FreeLinear_Nil_Good(t *testing.T) {
	coverageTokens := "FreeLinear Nil"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	defer func() {
		if recovered := recover(); recovered != nil {
			t.Fatalf("FreeLinear(nil) panicked: %v", recovered)
		}
	}()

	FreeLinear(nil)
}

func TestClose_FreeEmbedding_Good(t *testing.T) {
	coverageTokens := "FreeEmbedding"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	w := FromValues([]float32{1, 2, 3, 4, 5, 6}, 3, 2)
	Materialize(w)

	e := &Embedding{Weight: w}
	FreeEmbedding(e)

	if w.Valid() {
		t.Error("embedding weight should be freed")
	}
}

func TestClose_FreeRMSNorm_Good(t *testing.T) {
	coverageTokens := "FreeRMSNorm"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	w := FromValues([]float32{1, 1, 1, 1}, 4)
	Materialize(w)

	r := &RMSNormModule{Weight: w}
	FreeRMSNorm(r)

	if w.Valid() {
		t.Error("rmsnorm weight should be freed")
	}
}

// Qwen3 close coverage travels with the model in package metal/model/qwen3.

func TestClose_ModelClose_Idempotent_Good(t *testing.T) {
	coverageTokens := "ModelClose Idempotent"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	// Close on a model with nil internals should not panic.
	m := &Model{}
	if err := m.Close(); err != nil {
		t.Fatalf("Close on empty model: %v", err)
	}
	// Double close should be safe.
	if err := m.Close(); err != nil {
		t.Fatalf("Double close: %v", err)
	}
}

func TestClose_FreeCaches_Good(t *testing.T) {
	coverageTokens := "FreeCaches"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	c := NewKVCache()
	k := FromValues([]float32{1, 2, 3, 4}, 1, 1, 2, 2)
	v := FromValues([]float32{5, 6, 7, 8}, 1, 1, 2, 2)
	Materialize(k, v)
	c.Update(k, v, 2)

	state := c.State()
	if state == nil {
		t.Fatal("cache should have state after update")
	}

	FreeCaches([]Cache{c})
	// After freeing, the underlying arrays should be invalid.
	for _, arr := range state {
		if arr.Valid() {
			t.Error("cache array should be freed")
		}
	}
}

func TestClose_FreeCaches_NilCache_Ugly(t *testing.T) {
	coverageTokens := "FreeCaches NilCache"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	FreeCaches([]Cache{nil})
}

// Per-architecture close-helper nil coverage travels with each extracted model.
