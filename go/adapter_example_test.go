// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	core "dappco.re/go"
	"dappco.re/go/inference"
)

func ExampleNewMLXBackend() {
	oldBackend, hadOldBackend := inference.Get("metal")
	defer func() {
		if hadOldBackend {
			inference.Register(oldBackend)
			return
		}
		inference.Register(&metalbackend{})
	}()

	model := &stubTextModel{}
	backend := &stubBackend{model: model}
	inference.Register(backend)

	adapter, err := NewMLXBackend("/tmp/model-path", inference.WithContextLen(4096))

	core.Println(err == nil, adapter.Name(), adapter.Model() == model, backend.loadPath)
	// Output: true mlx true /tmp/model-path
}
