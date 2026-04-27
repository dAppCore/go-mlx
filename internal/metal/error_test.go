// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"testing"
)

func TestMetal_Eval_Good(t *testing.T) {
	a := FromValues([]float32{1, 2, 3}, 3)
	b := FromValues([]float32{4, 5, 6}, 3)
	c := Add(a, b)

	if err := Eval(c); err != nil {
		t.Fatalf("Eval should succeed: %v", err)
	}

	got := c.Floats()
	want := []float32{5, 7, 9}
	for i := range got {
		if got[i] != want[i] {
			t.Errorf("got[%d] = %f, want %f", i, got[i], want[i])
		}
	}
}

func TestMetal_Eval_NilArray_Good(t *testing.T) {
	// Eval should handle nil arrays gracefully.
	if err := Eval(nil); err != nil {
		t.Fatalf("Eval(nil) should not error: %v", err)
	}
}

func TestMetal_LastError_NoError_Good(t *testing.T) {
	// When no error has occurred, lastError should return nil.
	if err := lastError(); err != nil {
		t.Errorf("lastError should be nil when no error occurred, got: %v", err)
	}
}

func TestMetal_NewCaches_ContextLen_Good(t *testing.T) {
	// When contextLen is set, unbounded KVCaches should become RotatingKVCaches.
	m := &Model{
		model: &fakeModel{numLayers: 4},
	}

	// Without contextLen — should get plain KVCaches.
	caches := m.newCaches()
	for i, c := range caches {
		if _, ok := c.(*KVCache); !ok {
			t.Errorf("cache[%d] without contextLen: got %T, want *KVCache", i, c)
		}
	}

	// With contextLen — should get RotatingKVCaches.
	m.contextLen = 2048
	caches = m.newCaches()
	for i, c := range caches {
		if _, ok := c.(*RotatingKVCache); !ok {
			t.Errorf("cache[%d] with contextLen=2048: got %T, want *RotatingKVCache", i, c)
		}
	}
}

// fakeModel is a minimal InternalModel for testing cache creation.
type fakeModel struct {
	numLayers int
}

func (f *fakeModel) Forward(_ *Array, _ []Cache) *Array                 { return nil }
func (f *fakeModel) ForwardMasked(_ *Array, _ *Array, _ []Cache) *Array { return nil }
func (f *fakeModel) NewCache() []Cache {
	caches := make([]Cache, f.numLayers)
	for i := range caches {
		caches[i] = NewKVCache()
	}
	return caches
}
func (f *fakeModel) NumLayers() int                      { return f.numLayers }
func (f *fakeModel) Tokenizer() *Tokenizer               { return nil }
func (f *fakeModel) ModelType() string                   { return "fake" }
func (f *fakeModel) ApplyLoRA(_ LoRAConfig) *LoRAAdapter { return nil }

func TestMetal_LoadAllSafetensors_MissingFile_Bad(t *testing.T) {
	_, err := LoadAllSafetensors("/nonexistent/path/model.safetensors")
	if err == nil {
		t.Fatal("LoadAllSafetensors should fail for missing file")
	}
}
