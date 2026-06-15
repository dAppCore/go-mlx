// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import core "dappco.re/go"

func ExampleDefaultSSDConfig() {
	cfg := DefaultSSDConfig()
	core.Println(cfg.SampleMaxTokens, cfg.SampleTopK)
	// Output: 65536 20
}

func ExampleSSDPostProcessCode() {
	code, ok := SSDPostProcessCode("answer:\n```python\nprint('hi')\n```")
	core.Println(ok, code)
	// Output: true print('hi')
}

func ExampleLookupSSDRecipe() {
	recipe, ok := LookupSSDRecipe("SimpleSD-4B-instruct")
	core.Println(ok, recipe.Model)
	// Output: true apple/SimpleSD-4B-instruct
}
