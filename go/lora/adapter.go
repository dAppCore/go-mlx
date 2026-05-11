// SPDX-Licence-Identifier: EUPL-1.2

package lora

import (
	"slices"

	core "dappco.re/go"
)

// AdapterInfo is the reproducible identity for an active inference adapter.
type AdapterInfo struct {
	Name       string   `json:"name,omitempty"`
	Path       string   `json:"path,omitempty"`
	Hash       string   `json:"hash,omitempty"`
	Rank       int      `json:"rank,omitempty"`
	Alpha      float32  `json:"alpha,omitempty"`
	Scale      float32  `json:"scale,omitempty"`
	TargetKeys []string `json:"target_keys,omitempty"`
}

// IsEmpty reports whether the adapter info has no meaningful fields set.
func (info AdapterInfo) IsEmpty() bool {
	return info.Name == "" && info.Path == "" && info.Hash == "" && info.Rank == 0 && info.Alpha == 0 && info.Scale == 0 && len(info.TargetKeys) == 0
}

type adapterConfigJSON struct {
	Rank          int      `json:"rank"`
	R             int      `json:"r"`
	Alpha         float32  `json:"alpha"`
	LoRAAlpha     float32  `json:"lora_alpha"`
	Scale         float32  `json:"scale"`
	TargetKeys    []string `json:"target_keys"`
	TargetModules []string `json:"target_modules"`
	LoRALayers    []string `json:"lora_layers"`
}

// InspectAdapter reads adapter_config.json and hashes adapter files.
//
//	info, err := lora.InspectAdapter("/path/to/adapter")
func InspectAdapter(path string) (AdapterInfo, error) {
	return Inspect(path, path)
}

// Inspect reads adapter_config.json at path and records identityPath as the
// user-facing path (which may differ from path when the adapter was staged
// from a Medium).
//
//	info, err := lora.Inspect(stagedPath, originalPath)
func Inspect(path string, identityPath string) (AdapterInfo, error) {
	if path == "" {
		return AdapterInfo{}, core.NewError("mlx: LoRA adapter path is required")
	}
	configPath := adapterConfigPath(path)
	read := core.ReadFile(configPath)
	if !read.OK {
		return AdapterInfo{}, core.E("lora.Inspect", "read adapter_config.json", resultError(read))
	}
	var cfg adapterConfigJSON
	if result := core.JSONUnmarshal(read.Value.([]byte), &cfg); !result.OK {
		return AdapterInfo{}, core.E("lora.Inspect", "parse adapter_config.json", resultError(result))
	}
	info := AdapterInfo{
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
	info.Hash = hashAdapter(path, read.Value.([]byte))
	return info, nil
}

func adapterConfigPath(path string) string {
	if core.HasSuffix(path, ".safetensors") {
		return core.PathJoin(core.PathDir(path), "adapter_config.json")
	}
	return core.PathJoin(path, "adapter_config.json")
}

func hashAdapter(path string, config []byte) string {
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

func resultError(result core.Result) error {
	if result.OK {
		return nil
	}
	if err, ok := result.Value.(error); ok {
		return err
	}
	return core.NewError("core result failed")
}
