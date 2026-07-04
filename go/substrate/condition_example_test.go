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

func ExampleCondition_String() {
	core.Println(TRADNoReplay.String())
	// An unknown condition stringifies to empty, not its raw value.
	core.Println(Condition("nope").String() == "")
	// Output:
	// TRAD-no-replay
	// true
}

func ExampleCondition_UsesContinuousState() {
	// CONT mounts retained KV; TRAD re-prefills from scratch.
	core.Println(CONT.UsesContinuousState())
	core.Println(TRAD.UsesContinuousState())
	// Output:
	// true
	// false
}

func ExampleCondition_RequiresArtificialGap() {
	// CONT-with-gap waits for the prefill gap without doing replay work.
	core.Println(CONTWithGap.RequiresArtificialGap())
	core.Println(CONT.RequiresArtificialGap())
	// Output:
	// true
	// false
}

func ExampleCondition_MeasuresPrefillGap() {
	// Only TRAD's own replay work is the source for T_prefill samples.
	core.Println(TRAD.MeasuresPrefillGap())
	core.Println(CONT.MeasuresPrefillGap())
	// Output:
	// true
	// false
}
