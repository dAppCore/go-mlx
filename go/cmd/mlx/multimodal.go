// SPDX-Licence-Identifier: EUPL-1.2

package main

import "time"

// multimodalDecodeResult carries the shared verb decode-loop outcome.
type multimodalDecodeResult struct {
	Generated  []int32
	PrefillDur time.Duration
	DecodeDur  time.Duration
}

// countTokenID reports how many times id occurs in ids.
func countTokenID(ids []int32, id int32) int {
	n := 0
	for _, v := range ids {
		if v == id {
			n++
		}
	}
	return n
}
