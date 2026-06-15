// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal_test

import (
	"fmt"

	metal "dappco.re/go/mlx/pkg/metal"
)

// MixerLoaderFor lets a config-composed model resolve each layer's sequence
// mixer by the kind its config declares, with no central switch. The
// "full_attention" loader is registered by the engine's softmax mixer at init,
// so a composed model can always build a dense attention layer; an unknown kind
// is a clean miss the model refuses on.
func ExampleMixerLoaderFor() {
	_, attnOK := metal.MixerLoaderFor("full_attention")
	_, unknownOK := metal.MixerLoaderFor("no-such-mixer")
	fmt.Println(attnOK, unknownOK)
	// Output: true false
}
