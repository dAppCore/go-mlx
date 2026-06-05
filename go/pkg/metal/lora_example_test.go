// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import core "dappco.re/go"

func ExampleDefaultLoRAConfig() {
	cfg := DefaultLoRAConfig()
	core.Println(cfg.Rank, cfg.Alpha, cfg.Scale, cfg.TargetKeys)
	// Output: 8 16 2 [q_proj v_proj]
}

func ExampleLoRAAdapter_SortedNames() {
	adapter := &LoRAAdapter{
		Layers: map[string]*LoRALinear{
			"model.layers.1.self_attn.v_proj": nil,
			"model.layers.0.self_attn.q_proj": nil,
		},
	}
	core.Println(adapter.SortedNames())
	// Output: [model.layers.0.self_attn.q_proj model.layers.1.self_attn.v_proj]
}

func ExampleLoRAAdapter_Unload() {
	adapter := &LoRAAdapter{
		Layers: map[string]*LoRALinear{
			"model.layers.0.self_attn.q_proj": nil,
		},
	}
	adapter.Unload()
	core.Println(len(adapter.Layers))
	// Output: 0
}
