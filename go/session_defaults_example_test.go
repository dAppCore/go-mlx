// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import core "dappco.re/go"

func ExampleDefaultLemmaNewSessionText() {
	core.Println(core.Contains(DefaultLemmaNewSessionText, "Lemma"))
	// Output: true
}

func ExampleDefaultNewSessionText() {
	core.Println(DefaultNewSessionText == DefaultLemmaNewSessionText)
	// Output: true
}
