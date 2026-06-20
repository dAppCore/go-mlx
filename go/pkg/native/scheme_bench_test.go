// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

func BenchmarkResolveGemma4Schemes(b *testing.B) {
	for i := 0; i < b.N; i++ {
		if err := resolveGemma4Schemes(); err != nil {
			b.Fatal(err)
		}
	}
}
