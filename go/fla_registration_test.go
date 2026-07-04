// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"testing"

	"dappco.re/go/mlx/pkg/metal"
)

// TestFLAMixersRegisteredInServeBinary proves the serve binary (this package,
// which the serve/generate commands build on) carries every FLA sequence-mixer
// loader plus the generic softmax one — so the config-composed model can resolve
// a Mamba/RWKV/linear-attn/sparse layer by the kind a config declares. Without
// the blank imports in speculative.go these resolve to nil and a hybrid config is
// refused at load. (Weight-subpath matching against a real checkpoint is a
// separate, checkpoint-gated step; this only asserts the kinds are reachable.)
func TestFLAMixersRegisteredInServeBinary(t *testing.T) {
	for _, kind := range []string{
		"full_attention", // the generic softmax keystone (pkg/metal)
		"mamba2", "rwkv7",
		"gla", "retnet", "deltanet",
		"gsa", "nsa", "moba", "mla",
	} {
		if _, ok := metal.MixerLoaderFor(kind); !ok {
			t.Errorf("mixer kind %q not registered in the serve binary — the composed runner cannot build it", kind)
		}
	}
}
