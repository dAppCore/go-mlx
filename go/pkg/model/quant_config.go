// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	core "dappco.re/go"
)

// QuantConfig is the checkpoint's mlx quantization block: the default group size + bit width,
// plus any per-module overrides (mixed-precision QAT packs — e.g. a pack may keep the experts
// 4-bit but the local MLP + router 8-bit). nil for bf16. Arch() is representation-agnostic;
// the assembler uses For(name) to get a tensor's actual (groupSize, bits).
type QuantConfig struct {
	GroupSize int                    `json:"group_size"` // tags drive MARSHALLING (round-trip); UnmarshalJSON reads the same keys
	Bits      int                    `json:"bits"`
	Mode      string                 `json:"mode"` // quant mode (affine/mxfp4/mxfp8/nvfp4); validated by the quant-config validator
	Overrides map[string]ModuleQuant `json:"-"`    // populated by UnmarshalJSON from the raw block; not marshalled
}

// ModuleQuant is one module's quant override.
type ModuleQuant struct {
	GroupSize int
	Bits      int
}

// UnmarshalJSON parses the mlx quantization block: the scalar group_size/bits are the default,
// and every object-valued key is a per-module override (its language_model. wrapper prefix
// stripped to match the normalised tensor names the assembler builds). Non-quant scalar keys
// (e.g. "mode") are ignored. Uses core.JSONUnmarshal so no encoding/json import is needed.
func (q *QuantConfig) UnmarshalJSON(b []byte) error {
	var m map[string]any
	if r := core.JSONUnmarshal(b, &m); !r.OK {
		return core.NewError("model.QuantConfig: quantization block parse failed")
	}
	toInt := func(v any) int {
		f, _ := v.(float64)
		return int(f)
	}
	const prefix = "language_model."
	q.Overrides = map[string]ModuleQuant{}
	for k, v := range m {
		switch k {
		case "group_size":
			q.GroupSize = toInt(v)
		case "bits":
			q.Bits = toInt(v)
		case "mode":
			q.Mode, _ = v.(string)
		default:
			mm, ok := v.(map[string]any)
			if !ok {
				continue // a non-quant scalar key (e.g. "mode")
			}
			bits := toInt(mm["bits"])
			if bits == 0 {
				continue
			}
			name := k
			if core.HasPrefix(name, prefix) {
				name = name[len(prefix):]
			}
			q.Overrides[name] = ModuleQuant{GroupSize: toInt(mm["group_size"]), Bits: bits}
		}
	}
	return nil
}

// For returns the (groupSize, bits) for a tensor by its NORMALISED name (no language_model.
// prefix) — the per-module override when present, else the default.
func (q *QuantConfig) For(name string) (groupSize, bits int) {
	if o, ok := q.Overrides[name]; ok {
		return o.GroupSize, o.Bits
	}
	return q.GroupSize, q.Bits
}

func NormalizeQuantizationMode(mode string) string {
	mode = core.Lower(core.Trim(mode))
	if mode == "" {
		return "affine"
	}
	return mode
}
