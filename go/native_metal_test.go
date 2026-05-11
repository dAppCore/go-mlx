// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && !nomlx

package mlx

import (
	"testing"

	"dappco.re/go/mlx/internal/metal"
)

func skipIfNoUsableMetal(t *testing.T) {
	t.Helper()
	if !metal.MetalAvailable() {
		t.Skip("usable Metal device unavailable")
	}
}
