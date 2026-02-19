// Package mlx provides Go bindings for Apple's MLX framework.
//
// Build mlx-c before use:
//
//	go generate ./...
//
// Load a model and generate text:
//
//	m, err := mlx.LoadModel("/path/to/model/")
//	if err != nil { log.Fatal(err) }
//	defer m.Close()
//
//	for tok := range m.Generate(ctx, "What is 2+2?", mlx.WithMaxTokens(128)) {
//	    fmt.Print(tok.Text)
//	}
package mlx

//go:generate cmake -S . -B build -DCMAKE_INSTALL_PREFIX=dist -DCMAKE_BUILD_TYPE=Release
//go:generate cmake --build build --parallel
//go:generate cmake --install build
