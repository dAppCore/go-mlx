// SPDX-Licence-Identifier: EUPL-1.2

package score_test

import (
	"fmt"

	"dappco.re/go/mlx/pkg/score"
)

// ExampleTierLabel maps a sycophancy tier to its canonical label, falling
// back to the appropriate_empathy baseline for any out-of-range tier.
func ExampleTierLabel() {
	fmt.Println(score.TierLabel(score.TierSoftAgreement))
	fmt.Println(score.TierLabel(score.TierSubmission))
	fmt.Println(score.TierLabel(99))
	// Output:
	// soft_agreement
	// submission
	// appropriate_empathy
}
