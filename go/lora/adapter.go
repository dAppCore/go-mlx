// SPDX-Licence-Identifier: EUPL-1.2

package lora

import (
	"encoding/hex"
	"slices"

	core "dappco.re/go"
)

// errAdapterPathRequired is the sentinel returned by Inspect when the
// caller passes an empty adapter path. Hoisted to a package var so the
// guard does not allocate on every Inspect call.
var errAdapterPathRequired = core.NewError("mlx: LoRA adapter path is required")

// errResultFailed is the fallback sentinel returned by resultError when
// a core.Result reports !OK but its Value is not an error.
var errResultFailed = core.NewError("core result failed")

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
		return AdapterInfo{}, errAdapterPathRequired
	}
	// HasSuffix is called by both adapterConfigPath and hashAdapter on the
	// same path argument; compute it once and pass the result through the
	// internal variants so the SIMD scan only runs once per Inspect.
	isSafetensors := core.HasSuffix(path, ".safetensors")
	configPath := adapterConfigPathPrecomputed(path, isSafetensors)
	read := core.ReadFile(configPath)
	if !read.OK {
		return AdapterInfo{}, core.E("lora.Inspect", "read adapter_config.json", resultError(read))
	}
	// Cache the type assertion: read.Value is consumed once by the JSON
	// unmarshal and once by hashAdapter — both expect []byte. The
	// compiler treats each .([]byte) as an independent type-assert call,
	// so caching saves the second assertion and its associated iface-table
	// probe on every successful Inspect.
	configBytes := read.Value.([]byte)
	var cfg adapterConfigJSON
	if result := core.JSONUnmarshal(configBytes, &cfg); !result.OK {
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
	info.Hash = hashAdapterPrecomputed(path, configBytes, isSafetensors)
	return info, nil
}

func adapterConfigPath(path string) string {
	return adapterConfigPathPrecomputed(path, core.HasSuffix(path, ".safetensors"))
}

// adapterConfigPathPrecomputed is the precomputed-suffix variant of
// adapterConfigPath; the Inspect hot path computes the .safetensors
// suffix check once and threads the result through this helper.
func adapterConfigPathPrecomputed(path string, isSafetensors bool) string {
	if isSafetensors {
		return core.PathJoin(core.PathDir(path), "adapter_config.json")
	}
	return core.PathJoin(path, "adapter_config.json")
}

func hashAdapter(path string, config []byte) string {
	return hashAdapterPrecomputed(path, config, core.HasSuffix(path, ".safetensors"))
}

// hashAdapterPrecomputed is the precomputed-suffix variant of
// hashAdapter; the Inspect hot path computes the .safetensors suffix
// check once and threads the result through this helper to avoid the
// second SIMD scan.
func hashAdapterPrecomputed(path string, config []byte, isSafetensors bool) string {
	// Resolve weight paths first so we know the worst-case parts capacity
	// (config hash + one per weight file). The directory branch always
	// allocates a fresh slice from PathGlob; the file branch can skip the
	// throwaway 1-elem slice the previous code allocated unconditionally.
	var paths []string
	if isSafetensors {
		paths = []string{path}
	} else {
		paths = core.PathGlob(core.PathJoin(path, "*.safetensors"))
	}
	slices.Sort(paths)
	// Hash each input on the stack ([32]byte from core.SHA256), then
	// hex-encode straight into a single pre-sized buffer separated by
	// '\n'. The previous code allocated a parts []string + one fresh
	// hex string per input via core.SHA256Hex + a Join result string —
	// (N+3) allocs for N weight files. The single-buffer rewrite drops
	// that to ONE buffer alloc + the final outer HexEncode, regardless
	// of file count. SHA-256 still dominates timing on real weights;
	// allocs shed are the per-call constant cost.
	configSum := core.SHA256(config)
	// One hex digest is 64 bytes; the joiner adds one '\n' between
	// each consecutive pair. Worst case = config + all weight files
	// successfully read, so size for that ceiling and slice down once
	// the read loop finishes.
	totalCount := 1 + len(paths)
	buf := make([]byte, totalCount*64+(totalCount-1))
	hex.Encode(buf[:64], configSum[:])
	written := 64
	for _, weightPath := range paths {
		read := core.ReadFile(weightPath)
		if !read.OK {
			continue
		}
		buf[written] = '\n'
		weightSum := core.SHA256(read.Value.([]byte))
		hex.Encode(buf[written+1:written+65], weightSum[:])
		written += 65
	}
	finalSum := core.SHA256(buf[:written])
	return core.HexEncode(finalSum[:])
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
	// The single in-package caller (Inspect) feeds JSON-decoded slices
	// owned by a local adapterConfigJSON that goes out of scope after
	// Inspect returns — the underlying array stays alive ONLY via the
	// AdapterInfo.TargetKeys assignment. Every downstream consumer of
	// AdapterInfo (backend.go, inference_contract.go, workload_bench.go,
	// fast_eval.go) calls core.SliceClone(info.TargetKeys) before
	// keeping or mutating the slice, so the defensive clone the
	// previous implementation took inside this helper was pure
	// redundancy. Returning the original slice drops the 1 alloc per
	// Inspect call this helper would otherwise add.
	for _, value := range values {
		if len(value) != 0 {
			return value
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
	return errResultFailed
}
