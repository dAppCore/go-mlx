// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import "dappco.re/go/mlx/profile"

func isGemma4ModelArchitecture(architecture string) bool {
	return profile.IsGemma4TargetArchitecture(architecture)
}
