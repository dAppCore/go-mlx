package mlx

import (
	"fmt"
	"sync"
)

// Backend is a named inference engine that can load models.
type Backend interface {
	// Name returns the backend identifier (e.g. "metal", "mlx_lm").
	Name() string

	// LoadModel loads a model from the given path.
	LoadModel(path string, opts ...LoadOption) (TextModel, error)
}

var (
	backendsMu sync.RWMutex
	backends   = map[string]Backend{}
)

// Register adds a backend to the registry.
func Register(b Backend) {
	backendsMu.Lock()
	defer backendsMu.Unlock()
	backends[b.Name()] = b
}

// Get returns a registered backend by name.
func Get(name string) (Backend, bool) {
	backendsMu.RLock()
	defer backendsMu.RUnlock()
	b, ok := backends[name]
	return b, ok
}

// Default returns the first available backend.
// Prefers "metal" if registered.
func Default() (Backend, error) {
	backendsMu.RLock()
	defer backendsMu.RUnlock()
	if b, ok := backends["metal"]; ok {
		return b, nil
	}
	for _, b := range backends {
		return b, nil
	}
	return nil, fmt.Errorf("mlx: no backends registered")
}

// LoadModel loads a model using the specified or default backend.
func LoadModel(path string, opts ...LoadOption) (TextModel, error) {
	cfg := ApplyLoadOpts(opts)
	if cfg.Backend != "" {
		b, ok := Get(cfg.Backend)
		if !ok {
			return nil, fmt.Errorf("mlx: backend %q not registered", cfg.Backend)
		}
		return b.LoadModel(path, opts...)
	}
	b, err := Default()
	if err != nil {
		return nil, err
	}
	return b.LoadModel(path, opts...)
}
