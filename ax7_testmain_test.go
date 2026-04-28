// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && !nomlx

package mlx

import (
	"fmt"
	"os"
	"testing"
)

func TestMain(m *testing.M) {
	if !MetalAvailable() {
		fmt.Fprintln(os.Stderr, "skipping root mlx tests: usable Metal device unavailable")
		os.Exit(0)
	}
	os.Exit(m.Run())
}
