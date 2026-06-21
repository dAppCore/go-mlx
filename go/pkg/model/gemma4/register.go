// SPDX-Licence-Identifier: EUPL-1.2

package gemma4

import "dappco.re/go/mlx/pkg/model"

// init registers gemma4.Load for the config model_type ids the gemma4 family declares, so a backend's
// neutral loader (model.LookupLoader) dispatches to it with no central switch — pkg/metal's #45
// pattern, now in the backend-agnostic pkg/model. (gemma4_unified is the multimodal wrapper; its
// nested text_config normalises to gemma4_text, handled by the same Load.)
func init() {
	for _, mt := range []string{"gemma4", "gemma4_text", "gemma4_unified"} {
		model.RegisterLoader(mt, Load)
	}
}
