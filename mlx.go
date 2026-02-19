// Package mlx provides Apple Metal GPU inference via mlx-c bindings.
//
// This package implements the [inference.Backend] interface from
// forge.lthn.ai/core/go-inference for Apple Silicon (M1-M4) GPUs.
//
// Build mlx-c before use:
//
//	go generate ./...
//
// Load a model and generate text:
//
//	import "forge.lthn.ai/core/go-inference"
//	import _ "forge.lthn.ai/core/go-mlx" // register Metal backend
//
//	m, err := inference.LoadModel("/path/to/model/")
//	if err != nil { log.Fatal(err) }
//	defer m.Close()
//
//	for tok := range m.Generate(ctx, "What is 2+2?", inference.WithMaxTokens(128)) {
//	    fmt.Print(tok.Text)
//	}
package mlx

//go:generate cmake -S . -B build -DCMAKE_INSTALL_PREFIX=dist -DCMAKE_BUILD_TYPE=Release
//go:generate cmake --build build --parallel
//go:generate cmake --install build
