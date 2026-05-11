// SPDX-Licence-Identifier: EUPL-1.2

package kv

import "dappco.re/go/mlx/memory"

// BenchReportVersion is the current version of the cache-mode comparison report.
const BenchReportVersion = 1

const defaultBenchContextLength = 131072

// BenchConfig describes a model/context shape for cache-mode comparison.
type BenchConfig struct {
	ContextLength int                  `json:"context_length"`
	NumLayers     int                  `json:"num_layers"`
	HiddenSize    int                  `json:"hidden_size"`
	DTypeBytes    int                  `json:"dtype_bytes,omitempty"`
	Modes         []memory.KVCacheMode `json:"modes,omitempty"`
}

// BenchReport compares cache modes for one model/context shape.
type BenchReport struct {
	Version         int                 `json:"version"`
	Config          BenchConfig         `json:"config"`
	Modes           []ModeBench         `json:"modes"`
	RecommendedMode memory.KVCacheMode  `json:"recommended_mode,omitempty"`
	Notes           []string            `json:"notes,omitempty"`
}

// ModeBench is one mode's estimated memory and tradeoff profile.
type ModeBench struct {
	Mode                   memory.KVCacheMode `json:"mode"`
	KeyBits                int                `json:"key_bits,omitempty"`
	ValueBits              int                `json:"value_bits,omitempty"`
	StorageBytes           uint64             `json:"storage_bytes"`
	RelativeMemory         float64            `json:"relative_memory"`
	EstimatedDecodePenalty float64            `json:"estimated_decode_penalty,omitempty"`
	WinsWhen               string             `json:"wins_when,omitempty"`
}

// CompareModes estimates memory/performance tradeoffs for KV cache modes.
//
//	report := kv.CompareModes(kv.BenchConfig{ContextLength: 65536})
func CompareModes(cfg BenchConfig) BenchReport {
	cfg = normalizeBenchConfig(cfg)
	report := BenchReport{
		Version: BenchReportVersion,
		Config:  cfg,
	}
	fpBytes := modeStorageBytes(cfg, memory.KVCacheModeFP16)
	for _, mode := range cfg.Modes {
		report.Modes = append(report.Modes, modeBench(cfg, mode, fpBytes))
	}
	report.RecommendedMode = recommendMode(cfg)
	if cfg.NumLayers == 0 || cfg.HiddenSize == 0 {
		report.Notes = append(report.Notes, "using shape fallback; pass model metadata for sharper cache estimates")
	}
	return report
}

// ByMode returns the comparison row for mode, or a zero row when missing.
//
//	row := report.ByMode(memory.KVCacheModeQ8)
func (r BenchReport) ByMode(mode memory.KVCacheMode) ModeBench {
	for _, bench := range r.Modes {
		if bench.Mode == mode {
			return bench
		}
	}
	return ModeBench{}
}

func normalizeBenchConfig(cfg BenchConfig) BenchConfig {
	if cfg.ContextLength <= 0 {
		cfg.ContextLength = defaultBenchContextLength
	}
	if cfg.NumLayers <= 0 {
		cfg.NumLayers = 32
	}
	if cfg.HiddenSize <= 0 {
		cfg.HiddenSize = 3072
	}
	if cfg.DTypeBytes <= 0 {
		cfg.DTypeBytes = 2
	}
	if len(cfg.Modes) == 0 {
		cfg.Modes = []memory.KVCacheMode{memory.KVCacheModeFP16, memory.KVCacheModePaged, memory.KVCacheModeQ8, memory.KVCacheModeKQ8VQ4}
	}
	return cfg
}

func modeBench(cfg BenchConfig, mode memory.KVCacheMode, fpBytes uint64) ModeBench {
	keyBits, valueBits := modeBits(mode, cfg.DTypeBytes)
	storage := modeStorageBytes(cfg, mode)
	relative := float64(1)
	if fpBytes > 0 {
		relative = float64(storage) / float64(fpBytes)
	}
	return ModeBench{
		Mode:                   mode,
		KeyBits:                keyBits,
		ValueBits:              valueBits,
		StorageBytes:           storage,
		RelativeMemory:         relative,
		EstimatedDecodePenalty: modeDecodePenalty(mode),
		WinsWhen:               modeWinsWhen(mode),
	}
}

func modeBits(mode memory.KVCacheMode, dtypeBytes int) (keyBits, valueBits int) {
	switch mode {
	case memory.KVCacheModeQ8:
		return 8, 8
	case memory.KVCacheModeKQ8VQ4:
		return 8, 4
	default:
		bits := dtypeBytes * 8
		return bits, bits
	}
}

func modeStorageBytes(cfg BenchConfig, mode memory.KVCacheMode) uint64 {
	elements := uint64(cfg.ContextLength) * uint64(cfg.NumLayers) * uint64(cfg.HiddenSize) * 2
	switch mode {
	case memory.KVCacheModeQ8:
		return elements
	case memory.KVCacheModeKQ8VQ4:
		return elements * 3 / 4
	default:
		return elements * uint64(cfg.DTypeBytes)
	}
}

func modeDecodePenalty(mode memory.KVCacheMode) float64 {
	switch mode {
	case memory.KVCacheModeQ8:
		return 0.08
	case memory.KVCacheModeKQ8VQ4:
		return 0.14
	case memory.KVCacheModePaged:
		return 0.02
	default:
		return 0
	}
}

func modeWinsWhen(mode memory.KVCacheMode) string {
	switch mode {
	case memory.KVCacheModeQ8:
		return "memory pressure dominates and q4 value loss is not justified"
	case memory.KVCacheModeKQ8VQ4:
		return "small unified-memory machines need maximum KV savings"
	case memory.KVCacheModePaged:
		return "memory is available but long-context allocation churn hurts"
	default:
		return "quality and raw decode speed dominate memory pressure"
	}
}

func recommendMode(cfg BenchConfig) memory.KVCacheMode {
	fpBytes := modeStorageBytes(cfg, memory.KVCacheModeFP16)
	switch {
	case fpBytes >= 20*memory.GiB:
		return memory.KVCacheModeKQ8VQ4
	case fpBytes >= 2*memory.GiB:
		return memory.KVCacheModeQ8
	case cfg.ContextLength >= 65536:
		return memory.KVCacheModePaged
	default:
		return memory.KVCacheModeFP16
	}
}
