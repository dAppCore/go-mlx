// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/mlx/adapter"
)

// NewMLXBackend loads the Metal backend and wraps it in an adapter.Adapter.
//
//	a, err := mlx.NewMLXBackend(modelPath, inference.WithContextLen(4096))
func NewMLXBackend(modelPath string, loadOpts ...inference.LoadOption) (*adapter.Adapter, error) {
	opts := make([]inference.LoadOption, len(loadOpts), len(loadOpts)+1)
	copy(opts, loadOpts)
	opts = append(opts, inference.WithBackend("metal"))
	r := inference.LoadModel(modelPath, opts...)
	if !r.OK {
		if err, ok := r.Value.(error); ok {
			return nil, err
		}
		return nil, core.E("mlx.NewMLXBackend", r.Error(), nil)
	}
	model, ok := r.Value.(inference.TextModel)
	if !ok {
		return nil, core.E("mlx.NewMLXBackend", "inference.LoadModel returned non-TextModel value", nil)
	}
	return adapter.New(model, "mlx"), nil
}
