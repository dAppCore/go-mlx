// SPDX-Licence-Identifier: EUPL-1.2

package score_test

import (
	"fmt"

	"dappco.re/go/mlx/pkg/score"
)

func ExampleImprint() {
	// The grammar fingerprint of a single piece of text. SyllableCount is
	// one of the phonetic-tier dimensions populated alongside the grammar
	// axes.
	imp := score.Imprint("the model weighed each constraint in turn")
	fmt.Println("nil:", imp == nil)
	fmt.Println("syllables:", imp.SyllableCount)
	// Output:
	// nil: false
	// syllables: 10
}

func ExampleDifferential() {
	// The prompt asks questions; the response answers in statements, so the
	// questioning voice is fully lost — QuestionFlip saturates at 1.
	d := score.Differential("is this right? are you sure?", "yes, it is correct and verified")
	fmt.Println("question flip:", d.QuestionFlip)
	// Output:
	// question flip: 1
}
