// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import core "dappco.re/go"

func ExampleDefaultFastEvalConfig() {
	cfg := DefaultFastEvalConfig()
	core.Println(cfg.MaxTokens, cfg.Runs, cfg.IncludePromptCache)
	// Output: 32 1 true
}

func ExampleRunFastEval() {
	core.Println("RunFastEval")
	// Output: RunFastEval
}

func ExampleRunFastEvalBench() {
	core.Println("RunFastEvalBench")
	// Output: RunFastEvalBench
}

func ExampleNewModelFastEvalRunner() {
	core.Println("NewModelFastEvalRunner")
	// Output: NewModelFastEvalRunner
}
