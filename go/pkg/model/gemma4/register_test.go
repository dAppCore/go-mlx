// SPDX-Licence-Identifier: EUPL-1.2

package gemma4

import (
	"testing"

	"dappco.re/go/mlx/pkg/model"
)

// TestRegistersLoaders pins that gemma4's init() registered Load for every model_type id the family
// uses, so a backend's neutral loader (model.LookupLoader) can dispatch to gemma4 with no central
// switch.
func TestRegistersLoaders(t *testing.T) {
	for _, mt := range []string{"gemma4", "gemma4_text", "gemma4_unified"} {
		if model.LookupLoader(mt) == nil {
			t.Fatalf("gemma4 init() should register a loader for model_type %q", mt)
		}
	}
}
