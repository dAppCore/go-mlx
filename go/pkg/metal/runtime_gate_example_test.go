// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import core "dappco.re/go"

func ExampleSetRuntimeGate() {
	// SetRuntimeGate flips a typed fast-path gate and returns a restore func
	// that reverts it to its prior value — scoped, never via process env.
	before := RuntimeGateEnabled(GateNativeMLPMatVec)
	restore := SetRuntimeGate(GateNativeMLPMatVec, !before)
	core.Println(RuntimeGateEnabled(GateNativeMLPMatVec) == !before)
	restore()
	core.Println(RuntimeGateEnabled(GateNativeMLPMatVec) == before)
	// Output:
	// true
	// true
}

func ExampleRuntimeGateEnabled() {
	restore := SetRuntimeGate(GatePagedDecodeFastConcat, true)
	defer restore()
	core.Println(RuntimeGateEnabled(GatePagedDecodeFastConcat))
	// Output: true
}
