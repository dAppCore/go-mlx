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

func ExampleMustNormalize() {
	// Unrecognised input falls back to CONT rather than erroring.
	core.Println(MustNormalize("not-a-condition"))
	// Output: CONT
}

func ExampleCondition_Valid() {
	core.Println(TRAD.Valid())
	core.Println(Condition("nope").Valid())
	// Output:
	// true
	// false
}

func ExampleAll() {
	for _, c := range All() {
		core.Println(c)
	}
	// Output:
	// TRAD
	// CONT
	// TRAD-no-replay
	// CONT-with-gap
}
