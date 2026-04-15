// Package mlx provides Apple Metal GPU inference via mlx-c bindings.
//
// This package implements the [inference.Backend] interface from
// dappco.re/go/core/inference for Apple Silicon (M1-M4) GPUs.
// Import it blank to register the "metal" backend automatically:
//
//	import _ "dappco.re/go/mlx"
//
// Build mlx-c before use:
//
//	go generate ./...
//
// # Generate text
//
//	model, err := inference.LoadModel("/path/to/model/")
//	if err != nil { log.Fatal(err) }
//	defer model.Close()
//
//	ctx := context.Background()
//	for token := range model.Generate(ctx, "What is 2+2?", inference.WithMaxTokens(128)) {
//	    fmt.Print(token.Text)
//	}
//	if err := model.Err(); err != nil { log.Fatal(err) }
//
// # Multi-turn chat
//
// Chat applies the model's native template (Gemma3, Qwen3, Llama3):
//
//	for token := range model.Chat(ctx, []inference.Message{
//	    {Role: "system", Content: "You are a helpful assistant."},
//	    {Role: "user", Content: "Translate 'hello' to French."},
//	}, inference.WithMaxTokens(64)) {
//	    fmt.Print(token.Text)
//	}
//
// # Batch classification
//
// Classify runs a single forward pass per prompt (prefill only, no decoding):
//
//	results, err := model.Classify(ctx, []string{
//	    "Bonjour, comment allez-vous?",
//	    "The quarterly report shows growth.",
//	}, inference.WithTemperature(0))
//	for index, result := range results {
//	    fmt.Printf("prompt %d → %q\n", index, result.Token.Text)
//	}
//
// # Batch generation
//
//	results, err := model.BatchGenerate(ctx, []string{
//	    "The capital of France is",
//	    "Water boils at",
//	}, inference.WithMaxTokens(32))
//	for index, result := range results {
//	    for _, token := range result.Tokens {
//	        fmt.Print(token.Text)
//	    }
//	    fmt.Println()
//	}
//
// # Performance metrics
//
// After any inference call, retrieve timing and memory statistics:
//
//	for token := range model.Generate(ctx, prompt, inference.WithMaxTokens(128)) {
//	    fmt.Print(token.Text)
//	}
//	metrics := model.Metrics()
//	fmt.Printf("decode: %.0f tok/s, peak GPU: %d MB\n",
//	    metrics.DecodeTokensPerSec, metrics.PeakMemoryBytes/1024/1024)
//
// # Model info
//
//	modelInfo := model.Info()
//	fmt.Printf("%s %d-layer, %d-bit quantised\n",
//	    modelInfo.Architecture, modelInfo.NumLayers, modelInfo.QuantBits)
//
// # Model discovery
//
//	discoveredModels, err := inference.Discover("/path/to/models/")
//	for _, discoveredModel := range discoveredModels {
//	    fmt.Printf("%s (%s, %d-bit)\n", discoveredModel.Path, discoveredModel.ModelType, discoveredModel.QuantBits)
//	}
//
// # Metal memory controls
//
// These control the Metal allocator directly, not individual models:
//
//	mlx.SetCacheLimit(4 << 30)  // 4 GB cache limit
//	mlx.SetMemoryLimit(32 << 30) // 32 GB hard limit
//
//	// Between chat turns, reclaim prompt cache memory:
//	mlx.ClearCache()
//
//	fmt.Printf("active: %d MB, peak: %d MB\n",
//	    mlx.GetActiveMemory()/1024/1024, mlx.GetPeakMemory()/1024/1024)
package mlx

//go:generate cmake -S . -B build -DCMAKE_INSTALL_PREFIX=dist -DCMAKE_BUILD_TYPE=Release
//go:generate cmake --build build --parallel
//go:generate cmake --install build
