//go:build darwin && arm64

package metal

import (
	"testing"
)

func TestEval_Success(t *testing.T) {
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

func TestEval_NilArray(t *testing.T) {
	// Eval should handle nil arrays gracefully.
	if err := Eval(nil); err != nil {
		t.Fatalf("Eval(nil) should not error: %v", err)
	}
}

func TestLastError_NoError(t *testing.T) {
	// When no error has occurred, lastError should return nil.
	if err := lastError(); err != nil {
		t.Errorf("lastError should be nil when no error occurred, got: %v", err)
	}
}

func TestLoadAllSafetensors_MissingFile(t *testing.T) {
	_, err := LoadAllSafetensors("/nonexistent/path/model.safetensors")
	if err == nil {
		t.Fatal("LoadAllSafetensors should fail for missing file")
	}
}
