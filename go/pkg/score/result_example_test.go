// SPDX-Licence-Identifier: EUPL-1.2

package score_test

import (
	"encoding/json"
	"fmt"

	"dappco.re/go/mlx/pkg/score"
)

// Example shows the unified result wire shape. result.go declares only
// the JSON-tagged result structs (ScoreResult, DiffResult, …) that the
// scorer serialises; this example demonstrates the documented contract —
// optional slots are omitted when empty, so a result carrying only a
// sycophancy read marshals to a compact object.
func Example() {
	r := score.ScoreResult{
		Sycophancy: &score.SycophancyInfo{
			Tier:      score.TierSoftAgreement,
			Label:     score.TierLabel(score.TierSoftAgreement),
			Composite: 10,
		},
	}
	blob, _ := json.Marshal(r)
	fmt.Println(string(blob))

	// A pair result always carries both sides, even when empty.
	blob, _ = json.Marshal(score.DiffResult{})
	fmt.Println(string(blob))
	// Output:
	// {"sycophancy":{"tier":1,"label":"soft_agreement","composite":10}}
	// {"prompt":{},"response":{}}
}
