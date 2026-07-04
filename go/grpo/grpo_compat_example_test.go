// SPDX-Licence-Identifier: EUPL-1.2

package grpo

import (
	"time"

	core "dappco.re/go"
)

// Example_nonZeroDuration shows the run-duration floor: a real elapsed
// time passes through unchanged, while a zero elapsed time (an instant
// run) is reported as a single nanosecond so GRPOResult.Duration is
// never a bare zero.
func Example_nonZeroDuration() {
	core.Println(nonZeroDuration(250 * time.Millisecond))
	core.Println(nonZeroDuration(0))
	// Output:
	// 250ms
	// 1ns
}

// Example_cloneStringMap shows that SFT-sample metadata is cloned into a
// detached map: an empty source yields nil (the omitempty Meta
// contract), and a populated source is copied by value.
func Example_cloneStringMap() {
	core.Println(cloneStringMap(nil) == nil)
	clone := cloneStringMap(map[string]string{"split": "train"})
	core.Println(clone["split"])
	// Output:
	// true
	// train
}
