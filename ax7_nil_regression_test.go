// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && !nomlx

package mlx

import core "dappco.re/go"

func TestAX7_Array_Valid_Ugly(t *core.T) {
	var array *Array
	valid := array.Valid()

	core.AssertFalse(t, valid)
}
