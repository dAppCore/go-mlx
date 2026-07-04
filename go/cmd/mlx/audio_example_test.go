// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	core "dappco.re/go"
)

// Example_countTokenID shows the shared multimodal placeholder counter: how
// many times a soft-token id (audio/image/video) appears in the encoded
// prompt. The audio + vision verbs use it to verify the tokenizer produced
// exactly the soft-token count the encoder emitted before decoding.
func Example_countTokenID() {
	ids := []int32{2, 5, 3, 5, 5, 1}
	core.Println(countTokenID(ids, 5))
	core.Println(countTokenID(ids, 9))
	// Output:
	// 3
	// 0
}
