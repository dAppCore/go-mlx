// SPDX-Licence-Identifier: EUPL-1.2

package m2

import "dappco.re/go/inference/quant/jang"

// testJANGTQInfo returns a fixture JANGTQ info with packed profile for use
// across MiniMax M2 tensor-plan tests.
func testJANGTQInfo() *jang.Info {
	info := &jang.Info{
		Version:          2,
		WeightFormat:     "mxtq",
		Profile:          "JANGTQ",
		Method:           "affine+mxtq",
		GroupSize:        4,
		BitsDefault:      2,
		AttentionBits:    8,
		SharedExpertBits: 8,
		RoutedExpertBits: 2,
		EmbedTokensBits:  8,
		LMHeadBits:       8,
	}
	info.Packed = jang.BuildPackedProfile(info)
	return info
}
