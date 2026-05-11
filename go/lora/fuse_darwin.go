// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && !nomlx

package lora

import (
	"context"
	"slices"

	core "dappco.re/go"
	"dappco.re/go/mlx/internal/metal"
)

type fusePair struct {
	MatrixA *metal.Array
	MatrixB *metal.Array
}

// FuseIntoPack merges a LoRA adapter into dense safetensors base weights
// and writes a go-mlx-loadable model pack. Callers validate
// opts.SourcePack with mlx.ValidateModelPack before invoking, and
// validate the OutputPath after the call returns.
//
//	src, err := mlx.ValidateModelPack(path)
//	res, err := lora.FuseIntoPack(ctx, lora.FuseOptions{SourcePack: src, AdapterPath: a, OutputPath: o})
//	out, err := mlx.ValidateModelPack(res.OutputPath)
func FuseIntoPack(ctx context.Context, opts FuseOptions) (*FuseResult, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	prepared, err := prepareFuse(ctx, opts)
	if err != nil {
		return nil, err
	}

	adapterWeights, err := loadFuseAdapterWeights(opts.AdapterPath)
	if err != nil {
		return nil, err
	}
	defer freeMetalMap(adapterWeights)

	pairs, err := buildFusePairs(adapterWeights)
	if err != nil {
		return nil, err
	}

	weightFiles, fusedKeys, err := fuseModelWeightFiles(ctx, prepared.Model.WeightFiles, prepared.Output, pairs, prepared.Adapter.Scale)
	if err != nil {
		return nil, err
	}

	provenancePath := core.PathJoin(prepared.Output, FuseProvenanceFile)
	if err := writeFuseProvenance(provenancePath, FuseProvenance{
		Version:         1,
		SourceModel:     prepared.Model,
		Adapter:         prepared.Adapter,
		OutputWeight:    core.PathBase(weightFiles[0]),
		OutputWeights:   outputWeightFileNames(weightFiles),
		FusedWeightKeys: fusedKeys,
		Labels:          opts.Labels,
	}); err != nil {
		return nil, err
	}

	return &FuseResult{
		OutputPath:      prepared.Output,
		WeightPath:      weightFiles[0],
		WeightFiles:     weightFiles,
		ProvenancePath:  provenancePath,
		Adapter:         prepared.Adapter,
		FusedWeights:    len(fusedKeys),
		FusedWeightKeys: fusedKeys,
	}, nil
}

func loadFuseAdapterWeights(path string) (map[string]*metal.Array, error) {
	paths, err := fuseAdapterWeightFiles(path)
	if err != nil {
		return nil, err
	}
	weights := make(map[string]*metal.Array)
	for _, path := range paths {
		loaded, err := metal.LoadAllSafetensors(path)
		if err != nil {
			freeMetalMap(weights)
			return nil, core.E("lora.FuseIntoPack", "load adapter weights "+core.PathBase(path), err)
		}
		for name, tensor := range loaded {
			if previous := weights[name]; previous != nil {
				metal.Free(previous)
			}
			weights[name] = tensor
		}
	}
	return weights, nil
}

func buildFusePairs(weights map[string]*metal.Array) (map[string]fusePair, error) {
	pairs := make(map[string]fusePair)
	for name, tensor := range weights {
		pairName, suffix, ok := fusePairName(name)
		if !ok {
			continue
		}
		pair := pairs[pairName]
		switch suffix {
		case "a":
			pair.MatrixA = tensor
		case "b":
			pair.MatrixB = tensor
		}
		pairs[pairName] = pair
	}
	if len(pairs) == 0 {
		return nil, core.NewError("mlx: no LoRA tensor pairs found")
	}
	for name, pair := range pairs {
		if pair.MatrixA == nil || pair.MatrixB == nil {
			return nil, core.NewError("mlx: incomplete LoRA tensor pair: " + name)
		}
	}
	return pairs, nil
}

func fuseModelWeightFiles(ctx context.Context, sourceFiles []string, outputRoot string, pairs map[string]fusePair, scale float32) ([]string, []string, error) {
	if len(sourceFiles) == 0 {
		return nil, nil, core.NewError("mlx: no base weight files available for LoRA fusion")
	}

	fusedPairs := map[string]struct{}{}
	weightFiles := make([]string, 0, len(sourceFiles))
	fusedKeys := make([]string, 0, len(pairs))
	for _, sourceFile := range sourceFiles {
		if err := ctx.Err(); err != nil {
			return nil, nil, err
		}
		baseWeights, err := metal.LoadAllSafetensors(sourceFile)
		if err != nil {
			return nil, nil, core.E("lora.FuseIntoPack", "load base weights "+core.PathBase(sourceFile), err)
		}

		shardFusedKeys, err := fuseWeightPairs(ctx, baseWeights, pairs, fusedPairs, scale)
		if err != nil {
			freeMetalMap(baseWeights)
			return nil, nil, err
		}
		fusedKeys = append(fusedKeys, shardFusedKeys...)

		outputName := fuseOutputWeights
		if len(sourceFiles) > 1 {
			outputName = core.PathBase(sourceFile)
		}
		weightPath := core.PathJoin(outputRoot, outputName)
		if err := metal.SaveSafetensors(weightPath, baseWeights); err != nil {
			freeMetalMap(baseWeights)
			return nil, nil, core.E("lora.FuseIntoPack", "save fused safetensors", err)
		}
		freeMetalMap(baseWeights)
		weightFiles = append(weightFiles, weightPath)
	}

	for name := range pairs {
		if _, ok := fusedPairs[name]; ok {
			continue
		}
		return nil, nil, core.NewError("mlx: base weight not found for LoRA target: " + fuseBaseWeightKey(name))
	}
	return weightFiles, fusedKeys, nil
}

func fuseWeightPairs(ctx context.Context, baseWeights map[string]*metal.Array, pairs map[string]fusePair, fusedPairs map[string]struct{}, scale float32) ([]string, error) {
	names := make([]string, 0, len(pairs))
	for name := range pairs {
		names = append(names, name)
	}
	slices.Sort(names)

	fusedKeys := make([]string, 0, len(names))
	for _, name := range names {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if _, ok := fusedPairs[name]; ok {
			continue
		}
		baseKey := fuseBaseWeightKey(name)
		base := baseWeights[baseKey]
		if base == nil {
			continue
		}

		pair := pairs[name]
		delta := metal.Matmul(pair.MatrixB, pair.MatrixA)
		scaled := metal.MulScalar(delta, scale)
		fused := metal.Add(base, scaled)
		metal.Materialize(fused)
		metal.Free(delta, scaled, base)
		baseWeights[baseKey] = fused
		fusedKeys = append(fusedKeys, baseKey)
		fusedPairs[name] = struct{}{}
	}
	return fusedKeys, nil
}

func outputWeightFileNames(paths []string) []string {
	names := make([]string, 0, len(paths))
	for _, path := range paths {
		names = append(names, core.PathBase(path))
	}
	return names
}

func freeMetalMap(weights map[string]*metal.Array) {
	for _, tensor := range weights {
		metal.Free(tensor)
	}
}
