// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"maps"
	"os"

	"dappco.re/go"
)

func resolveModelRoot(modelPath string) string {
	if core.HasSuffix(modelPath, ".gguf") || core.HasSuffix(modelPath, ".safetensors") {
		return core.PathDir(modelPath)
	}
	info, err := os.Stat(modelPath)
	if err == nil && !info.IsDir() {
		return core.PathDir(modelPath)
	}
	return modelPath
}

func loadModelWeights(modelPath string) (map[string]*Array, error) {
	root := resolveModelRoot(modelPath)
	weights := make(map[string]*Array)

	if core.HasSuffix(modelPath, ".gguf") {
		return LoadAllGGUF(modelPath)
	}

	safetensors := core.PathGlob(core.JoinPath(root, "*.safetensors"))
	if len(safetensors) > 0 {
		for _, path := range safetensors {
			maps.Insert(weights, LoadSafetensors(path))
			if err := lastError(); err != nil {
				return nil, core.E("model.loadWeights", "load weights "+core.PathBase(path), err)
			}
		}
		return weights, nil
	}

	ggufs := core.PathGlob(core.JoinPath(root, "*.gguf"))
	switch len(ggufs) {
	case 0:
		return nil, core.E("model.loadWeights", "no .safetensors or .gguf files found in "+root, nil)
	case 1:
		return LoadAllGGUF(ggufs[0])
	default:
		return nil, core.E("model.loadWeights", "multiple .gguf files found in "+root, nil)
	}
}
