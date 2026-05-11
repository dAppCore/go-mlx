// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	core "dappco.re/go"
	"dappco.re/go/inference/quant/jang"
)

//	info := mlx.InferJANGFromHF(meta)
func InferJANGFromHF(meta HFModelMetadata) *jang.Info {
	needle := core.Lower(firstNonEmpty(meta.ID, meta.ModelID))
	for _, tag := range meta.Tags {
		needle = core.Concat(needle, " ", core.Lower(tag))
	}
	for _, file := range meta.Files {
		needle = core.Concat(needle, " ", core.Lower(file.filename()))
	}

	switch {
	case core.Contains(needle, "jangtq"):
		info := &jang.Info{
			Profile:          "JANGTQ",
			WeightFormat:     "mxtq",
			Method:           "affine+mxtq",
			GroupSize:        hfJANGGroupSize(meta),
			BitsDefault:      2,
			RoutedExpertBits: 2,
		}
		info.Packed = jang.BuildPackedProfile(info)
		return info
	case core.Contains(needle, "jang"):
		profile := inferJANGProfileName(needle)
		info := &jang.Info{
			Profile:     profile,
			GroupSize:   hfJANGGroupSize(meta),
			BitsDefault: firstPositive(jang.ProfileBits(profile), 0),
		}
		info.Packed = jang.BuildPackedProfile(info)
		return info
	default:
		return nil
	}
}

func hfJANGGroupSize(meta HFModelMetadata) int {
	if quant := meta.Config.QuantizationConfig; quant != nil && quant.GroupSize > 0 {
		return quant.GroupSize
	}
	if quant := meta.Config.Quantization; quant != nil && quant.GroupSize > 0 {
		return quant.GroupSize
	}
	return 64
}

func inferJANGProfileName(value string) string {
	for _, profile := range []string{"jang_1l", "jang_2s", "jang_2l", "jang_3l", "jang_4k", "jang_4m"} {
		if core.Contains(value, profile) {
			return core.Upper(profile)
		}
	}
	return "JANG"
}
