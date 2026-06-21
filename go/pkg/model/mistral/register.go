// SPDX-Licence-Identifier: EUPL-1.2

package mistral

import "dappco.re/go/mlx/pkg/model"

// init registers mistral.Load for the Mistral model_type ids, so a backend's neutral loader dispatches
// to it with no central switch (pkg/metal's #45 pattern). The Mistral3ForConditionalGeneration wrapper
// declares "mistral3" / "ministral3"; the bare text variants declare "mistral" / "ministral".
func init() {
	for _, mt := range []string{"mistral3", "ministral3", "mistral", "ministral"} {
		model.RegisterLoader(mt, Load)
	}
}
