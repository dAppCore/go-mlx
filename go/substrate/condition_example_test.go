// SPDX-Licence-Identifier: EUPL-1.2

package substrate

import core "dappco.re/go"

func ExampleNormalize() {
	condition, _ := Normalize("trad_no_replay")
	core.Println(condition)
	// Output: TRAD-no-replay
}

func ExampleCondition_RequiresReplay() {
	core.Println(TRAD.RequiresReplay())
	// Output: true
}
