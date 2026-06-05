// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import core "dappco.re/go"

func ExampleLoRAAdapter_Merge() {
	adapter := &LoRAAdapter{
		Layers: map[string]*LoRALinear{
			"model.layers.0.self_attn.q_proj": nil,
		},
	}
	adapter.Merge()
	core.Println(len(adapter.Layers))
	// Output: 0
}
