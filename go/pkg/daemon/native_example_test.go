// SPDX-Licence-Identifier: EUPL-1.2

package daemon

import core "dappco.re/go"

// Runnable example that invokes the native runner constructor and reports the
// resolved default model name (no model is loaded — the backend is stubbed).
func ExampleNewNativeGenerateRunner() {
	runner := NewNativeGenerateRunner(NativeGenerateConfig{
		ModelPaths: map[string]string{"default": "/models/main"},
	})
	defer func() { _ = runner.Close() }()
	core.Println(runner.defaultModel)
	// Output: default
}
