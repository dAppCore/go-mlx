// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"slices"

	core "dappco.re/go"
)

// LoRAAdapterInfo is the reproducible identity for an active inference adapter.
type LoRAAdapterInfo struct {
	Name       string   `json:"name,omitempty"`
	Path       string   `json:"path,omitempty"`
	Hash       string   `json:"hash,omitempty"`
	Rank       int      `json:"rank,omitempty"`
	Alpha      float32  `json:"alpha,omitempty"`
	Scale      float32  `json:"scale,omitempty"`
	TargetKeys []string `json:"target_keys,omitempty"`
}

type loraAdapterConfigJSON struct {
	Rank          int      `json:"rank"`
	R             int      `json:"r"`
	Alpha         float32  `json:"alpha"`
	LoRAAlpha     float32  `json:"lora_alpha"`
	Scale         float32  `json:"scale"`
	TargetKeys    []string `json:"target_keys"`
	TargetModules []string `json:"target_modules"`
	LoRALayers    []string `json:"lora_layers"`
}

// InspectLoRAAdapter reads adapter_config.json and hashes adapter files.
func InspectLoRAAdapter(path string) (LoRAAdapterInfo, error) {
	return inspectLoRAAdapter(path, path)
}

func inspectLoRAAdapter(path string, identityPath string) (LoRAAdapterInfo, error) {
	if path == "" {
		return LoRAAdapterInfo{}, core.NewError("mlx: LoRA adapter path is required")
	}
	configPath := loraAdapterConfigPath(path)
	read := core.ReadFile(configPath)
	if !read.OK {
		return LoRAAdapterInfo{}, core.E("InspectLoRAAdapter", "read adapter_config.json", loraAdapterResultError(read))
	}
	var cfg loraAdapterConfigJSON
	if result := core.JSONUnmarshal(read.Value.([]byte), &cfg); !result.OK {
		return LoRAAdapterInfo{}, core.E("InspectLoRAAdapter", "parse adapter_config.json", loraAdapterResultError(result))
	}
	info := LoRAAdapterInfo{
		Name:       core.PathBase(identityPath),
		Path:       identityPath,
		Rank:       firstNonZeroInt(cfg.Rank, cfg.R),
		Alpha:      firstNonZeroFloat32(cfg.Alpha, cfg.LoRAAlpha),
		Scale:      cfg.Scale,
		TargetKeys: firstNonEmptyStrings(cfg.TargetKeys, cfg.TargetModules, cfg.LoRALayers),
	}
	if info.Scale == 0 && info.Rank > 0 && info.Alpha != 0 {
		info.Scale = info.Alpha / float32(info.Rank)
	}
	if info.Alpha == 0 && info.Scale != 0 && info.Rank > 0 {
		info.Alpha = info.Scale * float32(info.Rank)
	}
	info.Hash = hashLoRAAdapter(path, read.Value.([]byte))
	return info, nil
}

func loraAdapterConfigPath(path string) string {
	if core.HasSuffix(path, ".safetensors") {
		return core.PathJoin(core.PathDir(path), "adapter_config.json")
	}
	return core.PathJoin(path, "adapter_config.json")
}

func hashLoRAAdapter(path string, config []byte) string {
	parts := []string{core.SHA256Hex(config)}
	paths := []string{path}
	if !core.HasSuffix(path, ".safetensors") {
		paths = core.PathGlob(core.PathJoin(path, "*.safetensors"))
	}
	slices.Sort(paths)
	for _, weightPath := range paths {
		read := core.ReadFile(weightPath)
		if read.OK {
			parts = append(parts, core.SHA256Hex(read.Value.([]byte)))
		}
	}
	return core.SHA256HexString(core.Join("\n", parts...))
}

func firstNonZeroInt(values ...int) int {
	for _, value := range values {
		if value != 0 {
			return value
		}
	}
	return 0
}

func firstNonZeroFloat32(values ...float32) float32 {
	for _, value := range values {
		if value != 0 {
			return value
		}
	}
	return 0
}

func firstNonEmptyStrings(values ...[]string) []string {
	for _, value := range values {
		if len(value) != 0 {
			return append([]string(nil), value...)
		}
	}
	return nil
}

func loraAdapterInfoEmpty(info LoRAAdapterInfo) bool {
	return info.Name == "" && info.Path == "" && info.Hash == "" && info.Rank == 0 && info.Alpha == 0 && info.Scale == 0 && len(info.TargetKeys) == 0
}

func loraAdapterResultError(result core.Result) error {
	if result.OK {
		return nil
	}
	if err, ok := result.Value.(error); ok {
		return err
	}
	return core.NewError("core result failed")
}
