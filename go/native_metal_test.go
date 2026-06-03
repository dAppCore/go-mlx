// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"testing"

	"dappco.re/go/mlx/pkg/metal"
)

func skipIfNoUsableMetal(t *testing.T) {
	t.Helper()
	if !metal.MetalAvailable() {
		t.Skip("usable Metal device unavailable")
	}
}
