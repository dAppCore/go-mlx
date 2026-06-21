// SPDX-Licence-Identifier: EUPL-1.2

package model

import "dappco.re/go/mlx/pkg/safetensors"

// registry.go is the model-architecture loader registry — config.json model_type id → loader —
// mirroring pkg/metal's RegisterModelLoader (go-mlx #45). A model package registers its Load from
// init(), so a backend's neutral loader looks the arch up instead of hard-coding a switch over
// gemma4/mistral/…; adding an arch is a new package + an init() call, with no edit here and none to
// the backend. The backend-agnostic counterpart of metal's registry — both backends dispatch the
// same way.

// Loader reads a checkpoint directory and returns the neutral LoadedModel plus the DirMapping whose
// mmap the weight byte-views reference — the caller Closes it once the device buffers are bound (the
// views are zero-copy into the mmap, so it must outlive binding). One per architecture.
type Loader func(dir string) (*LoadedModel, *safetensors.DirMapping, error)

var loaders = map[string]Loader{}

// RegisterLoader registers fn for a config.json "model_type" id; a later call for the same id
// overrides. Call from a model package's init() so the dispatch needs no central switch. No-op on an
// empty id or nil fn.
func RegisterLoader(modelType string, fn Loader) {
	if modelType == "" || fn == nil {
		return
	}
	loaders[modelType] = fn
}

// LookupLoader returns the loader registered for a model_type, or nil if none is.
func LookupLoader(modelType string) Loader {
	return loaders[modelType]
}
