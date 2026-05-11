// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import "dappco.re/go/mlx/memory"

const KVCacheBenchReportVersion = 1

// KVCacheBenchConfig describes a model/context shape for cache-mode comparison.
type KVCacheBenchConfig struct {
	ContextLength int           `json:"context_length"`
	NumLayers     int           `json:"num_layers"`
	HiddenSize    int           `json:"hidden_size"`
	DTypeBytes    int           `json:"dtype_bytes,omitempty"`
	Modes         []memory.KVCacheMode `json:"modes,omitempty"`
}

// KVCacheBenchReport compares cache modes for one model/context shape.
type KVCacheBenchReport struct {
	Version         int                `json:"version"`
	Config          KVCacheBenchConfig `json:"config"`
	Modes           []KVCacheModeBench `json:"modes"`
	RecommendedMode memory.KVCacheMode        `json:"recommended_mode,omitempty"`
	Notes           []string           `json:"notes,omitempty"`
}

// KVCacheModeBench is one mode's estimated memory and tradeoff profile.
type KVCacheModeBench struct {
	Mode                   memory.KVCacheMode `json:"mode"`
	KeyBits                int         `json:"key_bits,omitempty"`
	ValueBits              int         `json:"value_bits,omitempty"`
	StorageBytes           uint64      `json:"storage_bytes"`
	RelativeMemory         float64     `json:"relative_memory"`
	EstimatedDecodePenalty float64     `json:"estimated_decode_penalty,omitempty"`
	WinsWhen               string      `json:"wins_when,omitempty"`
}

// CompareKVCacheModes estimates memory/performance tradeoffs for KV cache modes.
func CompareKVCacheModes(cfg KVCacheBenchConfig) KVCacheBenchReport {
	cfg = normalizeKVCacheBenchConfig(cfg)
	report := KVCacheBenchReport{
		Version: KVCacheBenchReportVersion,
		Config:  cfg,
	}
	fpBytes := kvCacheModeStorageBytes(cfg, memory.KVCacheModeFP16)
	for _, mode := range cfg.Modes {
		bench := kvCacheModeBench(cfg, mode, fpBytes)
		report.Modes = append(report.Modes, bench)
	}
	report.RecommendedMode = recommendKVCacheMode(cfg)
	if cfg.NumLayers == 0 || cfg.HiddenSize == 0 {
		report.Notes = append(report.Notes, "using shape fallback; pass model metadata for sharper cache estimates")
	}
	return report
}

// ByMode returns the comparison row for mode, or a zero row when missing.
func (r KVCacheBenchReport) ByMode(mode memory.KVCacheMode) KVCacheModeBench {
	for _, bench := range r.Modes {
		if bench.Mode == mode {
			return bench
		}
	}
	return KVCacheModeBench{}
}

func normalizeKVCacheBenchConfig(cfg KVCacheBenchConfig) KVCacheBenchConfig {
	if cfg.ContextLength <= 0 {
		cfg.ContextLength = DefaultLocalContextLength
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

func kvCacheModeBench(cfg KVCacheBenchConfig, mode memory.KVCacheMode, fpBytes uint64) KVCacheModeBench {
	keyBits, valueBits := kvCacheModeBits(mode, cfg.DTypeBytes)
	storage := kvCacheModeStorageBytes(cfg, mode)
	relative := float64(1)
	if fpBytes > 0 {
		relative = float64(storage) / float64(fpBytes)
	}
	return KVCacheModeBench{
		Mode:                   mode,
		KeyBits:                keyBits,
		ValueBits:              valueBits,
		StorageBytes:           storage,
		RelativeMemory:         relative,
		EstimatedDecodePenalty: kvCacheModeDecodePenalty(mode),
		WinsWhen:               kvCacheModeWinsWhen(mode),
	}
}

func kvCacheModeBits(mode memory.KVCacheMode, dtypeBytes int) (keyBits, valueBits int) {
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

func kvCacheModeStorageBytes(cfg KVCacheBenchConfig, mode memory.KVCacheMode) uint64 {
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

func kvCacheModeDecodePenalty(mode memory.KVCacheMode) float64 {
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

func kvCacheModeWinsWhen(mode memory.KVCacheMode) string {
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

func recommendKVCacheMode(cfg KVCacheBenchConfig) memory.KVCacheMode {
	fpBytes := kvCacheModeStorageBytes(cfg, memory.KVCacheModeFP16)
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
