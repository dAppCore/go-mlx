// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import core "dappco.re/go"

func ExampleKVSnapshot() {
	core.Println("KVSnapshot")
	// Output: KVSnapshot
}

func ExampleKVLayerSnapshot() {
	core.Println("KVLayerSnapshot")
	// Output: KVLayerSnapshot
}

func ExampleKVHeadSnapshot() {
	core.Println("KVHeadSnapshot")
	// Output: KVHeadSnapshot
}
