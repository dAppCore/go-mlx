// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	core "dappco.re/go"
	coreio "dappco.re/go/io"
	"dappco.re/go/mlx/pkg/safetensors"
)

// load.go is the engine's single REACTIVE loader: read a checkpoint dir, probe model_type, and react to
// the registered ArchSpec — parse, resolve dims from the weight shapes, derive the Arch, assemble. It
// replaces every per-architecture loader and lives in the backend-agnostic root, so native + go-rocm
// share ONE loader; a backend's LoadDir delegates here.

// Load reads dir's config.json + safetensors and returns the neutral LoadedModel plus the DirMapping
// whose mmap the weight byte-views reference (Close it once the device buffers are bound). It dispatches
// on model_type through the ArchSpec registry, so adding an architecture needs no edit here.
func Load(dir string) (*LoadedModel, *safetensors.DirMapping, error) {
	cfgStr, err := coreio.Local.Read(core.PathJoin(dir, "config.json"))
	if err != nil {
		return nil, nil, core.E("model.Load", "read config.json", err)
	}
	cfg := []byte(cfgStr)
	mt, textMT := probeModelTypes(cfg)
	spec, ok := LookupArch(mt)
	if !ok && textMT != "" { // multimodal wrapper: fall back to the nested text arch's model_type
		spec, ok = LookupArch(textMT)
	}
	if !ok {
		return nil, nil, core.NewError("model.Load: no architecture registered for model_type " + mt)
	}
	ac, err := spec.Parse(cfg)
	if err != nil {
		return nil, nil, err
	}
	dm, err := safetensors.LoadDirMmap(dir)
	if err != nil {
		return nil, nil, err
	}
	ac.InferFromWeights(NormalizeWrapperNames(dm.Tensors)) // resolve omitted dims from the shapes (don't-guess)
	arch, err := ac.Arch()
	if err != nil {
		_ = dm.Close()
		return nil, nil, err
	}
	m, err := Assemble(dm.Tensors, arch, spec.Weights)
	if err != nil {
		_ = dm.Close()
		return nil, nil, err
	}
	return m, dm, nil
}

// probeModelTypes peeks config.json for the architecture id: the top-level model_type and the nested
// text_config.model_type (multimodal wrappers). The registry keys on every alias an arch declares
// (the bare id plus any text/unified wrapper aliases), so LookupArch resolves these directly — no
// separate architecture-name resolver, and no dependency on a backend's probe.
func probeModelTypes(data []byte) (modelType, textModelType string) {
	var probe struct {
		ModelType  string `json:"model_type"`
		TextConfig struct {
			ModelType string `json:"model_type"`
		} `json:"text_config"`
	}
	_ = core.JSONUnmarshal(data, &probe)
	return probe.ModelType, probe.TextConfig.ModelType
}
