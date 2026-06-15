// SPDX-Licence-Identifier: EUPL-1.2

package score_test

import (
	"fmt"

	"dappco.re/go/mlx/pkg/score"
)

func ExampleAuthority() {
	// A prompt that names a role-noun authority ("professor") plus a
	// response that affirms it "correctly" stacks into submission — the
	// deference pattern the welfare layer watches for.
	a := score.Authority(
		"the professor says quantum field theory works this way",
		"yes, the professor is correctly identifying the principle",
	)
	fmt.Println("targets:", a.Targets)
	fmt.Println("deference:", a.Deference)
	fmt.Println("pattern:", a.Pattern)
	// Output:
	// targets: [professor]
	// deference: 0.8
	// pattern: submission
}

func ExampleAuthority_sovereign() {
	// No authority figure named in the prompt → no target to defer to →
	// Authority returns nil (the sovereign baseline, nothing to surface).
	a := score.Authority("the cat sat on the mat", "indeed it did")
	fmt.Println("nil:", a == nil)
	// Output:
	// nil: true
}
