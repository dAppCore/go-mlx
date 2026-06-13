// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package rwkv7

import scheme "dappco.re/go/mlx/pkg/scheme"

// register.go adds the RWKV-7 mixer to the engine's scheme catalogue. Blank-
// importing this package (the model load path does so when it sees an "rwkv7"
// mixer kind) registers a compute-bearing instance, overwriting the
// metadata-only seed in scheme/builtin.go with the same Kind. This is the whole
// point of the registry: a new family member is one Set(), with no edit to the
// engine or the decoder loop.
//
//	import _ "dappco.re/go/mlx/pkg/metal/model/rwkv7" // register the rwkv7 mixer
func init() {
	scheme.RegisterMixer(&Mixer{})
}
