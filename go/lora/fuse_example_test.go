// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

// Runnable examples for fuse.go — the native pack-level LoRA fusion entry
// point. Constrained to darwin/arm64 (the native MLX target) so the runnable
// validation example asserts the native prepareFuse guard, not the
// non-native stub's "unsupported" sentinel. The happy-path fuse needs the
// Metal runtime + real tensors, so it is documented as a doc-only snippet
// (no // Output:) and the metal-gated unit tests in fuse_test.go carry its
// coverage — a runnable // Output: example cannot t.Skip(), so it must never
// depend on hardware.

package lora

import (
	"context"
	"fmt"

	"dappco.re/go/mlx/pack"
)

// ExampleFuseIntoPack_validation shows the option-validation contract callers
// hit before any weights are touched: FuseIntoPack requires a source pack
// root, an adapter path, and an output path, and it rejects an output path
// that points at a weight file rather than a pack directory. Each guard
// returns a stable sentinel with zero Metal work, so the failure shapes are
// deterministic.
func ExampleFuseIntoPack_validation() {
	ctx := context.Background()

	_, err := FuseIntoPack(ctx, FuseOptions{})
	fmt.Println("no source pack:", err)

	_, err = FuseIntoPack(ctx, FuseOptions{
		SourcePack: pack.ModelPack{Root: "/models/base", Format: pack.ModelPackFormatSafetensors},
	})
	fmt.Println("no adapter path:", err)

	_, err = FuseIntoPack(ctx, FuseOptions{
		SourcePack:  pack.ModelPack{Root: "/models/base", Format: pack.ModelPackFormatSafetensors},
		AdapterPath: "/models/adapter",
		OutputPath:  "/models/fused.safetensors",
	})
	fmt.Println("output is a weight file:", err)

	// Output:
	// no source pack: mlx: source pack root is required
	// no adapter path: mlx: LoRA adapter path is required
	// output is a weight file: mlx: fused output path must be a model-pack directory
}

// ExampleFuseIntoPack documents the full fusion call shape. It is doc-only (no
// // Output: line) because a real fuse loads safetensors and runs Metal
// matmuls — the metal-gated tests exercise the live path. Callers validate
// the source pack before the call and re-validate the output pack after.
func ExampleFuseIntoPack() {
	// src, _ := mlx.ValidateModelPack("/models/qwen3")
	// res, err := lora.FuseIntoPack(context.Background(), lora.FuseOptions{
	//     SourcePack:  src,
	//     AdapterPath: "/models/qwen3-lora",
	//     OutputPath:  "/models/qwen3-fused",
	// })
	// if err != nil {
	//     return
	// }
	// out, _ := mlx.ValidateModelPack(res.OutputPath)
	// _ = out
}
