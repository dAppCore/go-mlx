// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"testing"

	core "dappco.re/go"
)

func TestMain(m *testing.M) {
	if !MetalAvailable() {
		core.Print(core.Stderr(), "skipping internal/metal tests: usable Metal device unavailable")
		core.Exit(0)
	}
	core.Exit(m.Run())
}
