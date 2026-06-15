// SPDX-Licence-Identifier: EUPL-1.2

package spine_test

import (
	"fmt"

	"dappco.re/go/mlx/spine"
)

func ExampleDefaultLoRAConfig() {
	cfg := spine.DefaultLoRAConfig()
	fmt.Println(cfg.Rank, cfg.Alpha, cfg.TargetKeys)
	// Output: 8 16 [q_proj v_proj]
}

func ExampleToMetalLoRAConfig() {
	cfg := spine.LoRAConfig{Rank: 4, Alpha: 8, TargetKeys: []string{"q_proj"}}
	mcfg := spine.ToMetalLoRAConfig(cfg)
	// TargetKeys is defensively cloned into the metal-side config.
	fmt.Println(mcfg.Rank, mcfg.Alpha, mcfg.TargetKeys)
	// Output: 4 8 [q_proj]
}

func ExampleLoRAConfigFromMetal() {
	// Round-trip the default through metal and back.
	cfg := spine.LoRAConfigFromMetal(spine.ToMetalLoRAConfig(spine.DefaultLoRAConfig()))
	fmt.Println(cfg.Rank, cfg.TargetKeys)
	// Output: 8 [q_proj v_proj]
}
