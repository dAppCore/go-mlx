// SPDX-Licence-Identifier: EUPL-1.2

package spine_test

import (
	"fmt"

	"dappco.re/go/mlx/spine"
)

func ExampleDefaultGenerateConfig() {
	cfg := spine.DefaultGenerateConfig()
	// MaxTokens 0 means "generate to the model's context window".
	fmt.Println(cfg.MaxTokens, cfg.Temperature, cfg.Thinking.Mode)
	// Output: 0 0 show
}

func ExampleApplyGenerateOptions() {
	opts := []spine.GenerateOption{
		func(c *spine.GenerateConfig) { c.MaxTokens = 256 },
		func(c *spine.GenerateConfig) { c.Temperature = 0.7 },
	}
	cfg := spine.ApplyGenerateOptions(opts)
	fmt.Println(cfg.MaxTokens, cfg.Temperature)
	// Output: 256 0.7
}
