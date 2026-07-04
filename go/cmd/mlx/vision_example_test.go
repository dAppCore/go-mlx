// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	core "dappco.re/go"
)

// Example_splitPathList shows how the -images / -video-frames comma lists are
// parsed: surrounding whitespace is trimmed, empty segments are dropped, and a
// blank input yields no paths (nil) so the caller's "need at least one input"
// guard fires.
//
//	lthn-mlx vision -images " a.png , b.jpg " <model>
func Example_splitPathList() {
	for _, p := range splitPathList(" a.png , ,b.jpg ") {
		core.Println(p)
	}
	core.Println(len(splitPathList("   ")))
	// Output:
	// a.png
	// b.jpg
	// 0
}
