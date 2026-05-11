// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	stdio "io"

	core "dappco.re/go"
)

func mlxMaxIntValue() int { return int(^uint(0) >> 1) }

func readSafetensorRefRaw(ref safetensorTensorRef) ([]byte, error) {
	if ref.ByteLen < 0 || ref.ByteLen > int64(mlxMaxIntValue()) {
		return nil, core.NewError("mlx: safetensors tensor byte length is invalid: " + ref.Name)
	}
	opened := core.Open(ref.Path)
	if !opened.OK {
		return nil, modelMergeResultError(opened)
	}
	file := opened.Value.(*core.OSFile)
	defer file.Close()

	raw := make([]byte, int(ref.ByteLen))
	n, err := file.ReadAt(raw, ref.DataStart)
	if err != nil && !(err == stdio.EOF && n == len(raw)) {
		return nil, err
	}
	if n != len(raw) {
		return nil, core.NewError("mlx: safetensors tensor payload is truncated: " + ref.Name)
	}
	return raw, nil
}
