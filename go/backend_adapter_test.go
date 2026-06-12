// SPDX-Licence-Identifier: EUPL-1.2

// Tests for backend_adapter.go — NewMLXBackend's load-and-wrap path,
// driven through a stub inference backend.

package mlx

import (
	"testing"

	"dappco.re/go/inference"
)

// stubTextModel embeds the TextModel interface — NewMLXBackend's tests
// only identity-check the wrapped model, they never invoke it, so the
// nil method set is fine and keeps the fixture one line.
type stubTextModel struct {
	inference.TextModel
}

type stubBackend struct {
	model    inference.TextModel
	loadPath string
	loadErr  error
}

func (backend *stubBackend) Name() string { return "metal" }
func (backend *stubBackend) Available() bool {
	return true
}
func (backend *stubBackend) LoadModel(path string, _ ...inference.LoadOption) (inference.TextModel, error) {
	backend.loadPath = path
	if backend.loadErr != nil {
		return nil, backend.loadErr
	}
	return backend.model, nil
}

func TestNewMLXBackend_Good(t *testing.T) {
	oldBackend, hadOldBackend := inference.Get("metal")
	if hadOldBackend {
		defer inference.Register(oldBackend)
	}

	model := &stubTextModel{}
	backend := &stubBackend{model: model}
	inference.Register(backend)

	a, err := NewMLXBackend("/tmp/model-path", inference.WithContextLen(4096))
	if err != nil {
		t.Fatalf("NewMLXBackend() error = %v", err)
	}
	if a.Name() != "mlx" {
		t.Fatalf("adapter name = %q, want %q", a.Name(), "mlx")
	}
	if a.Model() != model {
		t.Fatal("adapter should expose the loaded model")
	}
	if backend.loadPath != "/tmp/model-path" {
		t.Fatalf("backend load path = %q, want %q", backend.loadPath, "/tmp/model-path")
	}
}
