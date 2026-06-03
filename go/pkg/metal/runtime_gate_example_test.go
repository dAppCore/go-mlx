// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import core "dappco.re/go"

func ExampleSetRuntimeGate() {
	core.Println("SetRuntimeGate")
	// Output: SetRuntimeGate
}

func ExampleRuntimeGateValue() {
	core.Println("RuntimeGateValue")
	// Output: RuntimeGateValue
}

func ExampleRuntimeGateEnabled() {
	core.Println("RuntimeGateEnabled")
	// Output: RuntimeGateEnabled
}
