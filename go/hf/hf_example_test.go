// SPDX-Licence-Identifier: EUPL-1.2

package hf

import (
	"fmt"
)

// ExampleNewRemoteSource constructs a Hugging Face Hub metadata source. The
// constructor trims a trailing slash from the base URL and defaults the
// user-agent when none is supplied — no network is touched here.
func ExampleNewRemoteSource() {
	source := NewRemoteSource(RemoteConfig{
		BaseURL: "https://huggingface.co/",
	})
	fmt.Println(source.baseURL, source.userAgent)
	// Output: https://huggingface.co go-mlx
}
