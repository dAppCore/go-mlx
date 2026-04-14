//go:build darwin && arm64 && !nomlx

package metal

import (
	"fmt"
	"os"
	"testing"
)

func TestMain(m *testing.M) {
	if !MetalAvailable() {
		fmt.Fprintln(os.Stderr, "skipping internal/metal tests: usable Metal device unavailable")
		os.Exit(0)
	}
	os.Exit(m.Run())
}
