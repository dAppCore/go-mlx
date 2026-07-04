// SPDX-Licence-Identifier: EUPL-1.2

package loraadapter

import core "dappco.re/go"

// Config is the shared adapter_config.json metadata surface understood by
// go-mlx adapter inspection and native Metal adapter loading.
type Config struct {
	Rank          int      `json:"rank"`
	R             int      `json:"r"`
	Alpha         float32  `json:"alpha"`
	LoRAAlpha     float32  `json:"lora_alpha"`
	Scale         float32  `json:"scale"`
	NumLayers     int      `json:"num_layers"`
	TargetKeys    []string `json:"target_keys"`
	TargetModules []string `json:"target_modules"`
	LoRALayers    []string `json:"lora_layers"`
}

// ParseConfig parses adapter_config.json bytes and applies lossless aliases.
// It does not fabricate required metadata such as rank; public inspection and
// fusion validation need to know when an adapter omitted those fields.
func ParseConfig(data []byte) (Config, error) {
	var cfg Config
	if result := core.JSONUnmarshal(data, &cfg); !result.OK {
		return Config{}, core.E("loraadapter.ParseConfig", "parse adapter_config.json", nil)
	}
	return NormalizeConfig(cfg), nil
}

// NormalizeConfig applies the adapter metadata aliases used by PEFT, mlx-lm,
// and go-mlx saved adapters without inventing missing required metadata.
func NormalizeConfig(cfg Config) Config {
	if cfg.Rank <= 0 && cfg.R > 0 {
		cfg.Rank = cfg.R
	}
	if cfg.Alpha == 0 {
		switch {
		case cfg.LoRAAlpha != 0:
			cfg.Alpha = cfg.LoRAAlpha
		case cfg.Scale != 0 && cfg.Rank > 0:
			cfg.Alpha = cfg.Scale * float32(cfg.Rank)
		}
	}
	if cfg.Scale == 0 && cfg.Rank > 0 && cfg.Alpha != 0 {
		cfg.Scale = cfg.Alpha / float32(cfg.Rank)
	}
	if len(cfg.TargetKeys) == 0 {
		switch {
		case len(cfg.TargetModules) > 0:
			cfg.TargetKeys = cfg.TargetModules
		case len(cfg.LoRALayers) > 0:
			cfg.TargetKeys = cfg.LoRALayers
		}
	}
	return cfg
}

// NormalizeForNativeLoad applies the default adapter values accepted by the
// native Metal loader. Keep this separate from ParseConfig so public metadata
// validation can still reject incomplete adapter_config.json files.
func NormalizeForNativeLoad(cfg Config) Config {
	cfg = NormalizeConfig(cfg)
	if cfg.Rank <= 0 {
		cfg.Rank = 8
	}
	if cfg.Alpha == 0 {
		switch {
		case cfg.Scale != 0:
			cfg.Alpha = cfg.Scale * float32(cfg.Rank)
		default:
			cfg.Alpha = float32(cfg.Rank) * 2
		}
	}
	if cfg.Scale == 0 && cfg.Rank > 0 && cfg.Alpha != 0 {
		cfg.Scale = cfg.Alpha / float32(cfg.Rank)
	}
	return cfg
}
