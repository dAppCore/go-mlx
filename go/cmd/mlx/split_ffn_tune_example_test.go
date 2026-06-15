// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	core "dappco.re/go"
	mlx "dappco.re/go/mlx"
)

// Example_cliSplitFFNCacheLayers parses the comma-separated cache-layer
// counts the split-CPU-FFN tuning flag accepts. Whitespace is trimmed and
// negative values (streaming mode — no resident cache) are permitted.
//
//	lthn-mlx tune -model ... -split-cpu-ffn-cache "0,4,8"
func Example_cliSplitFFNCacheLayers() {
	layers, err := cliSplitFFNCacheLayers("0, 4, 8")
	core.Println(layers, err == nil)
	// Output: [0 4 8] true
}

// Example_cliSplitFFNReason documents the human-readable reasons the tuner
// attaches to a split-CPU-FFN candidate. The first line flips with the
// cache-layer count: a positive cache keeps N layers resident, zero caches
// every layer after first load, and a negative cache streams without a
// resident cache.
func Example_cliSplitFFNReason() {
	report := mlx.CPUSplitFFNMemoryReport{PeakResidentBytes: 4096}
	for _, cache := range []int{4, 0, -1} {
		core.Println(cliSplitFFNReason(cliSplitFFNEstimate{cache: cache, report: report})[0])
	}
	// Output:
	// split CPU FFN keeps up to 4 layers resident
	// split CPU FFN caches all layers after first load
	// split CPU FFN streams layer weights without retaining a resident cache
}
