// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"slices"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
)

// FuseLoRAIntoWeights folds a trained LoRA adapter's deltas into a dense base
// weights map, returning a NEW map where each targeted base weight W is replaced
// by W + (B·A)·scale (dense) and every other tensor is carried through unchanged.
// It also returns the sorted names of the layers it fused, for diagnostics.
//
// This is the disk-fuse core (the P-phase on-disk fuse): it works at the
// weights-map level so a fused checkpoint serialises straight back out with
// SaveSafetensors, without walking a live model. scale is the LoRA alpha/rank.
//
// The base must be DENSE (unquantized): a fused layer becomes a dense matrix, and
// a single config-level quantization block cannot describe a checkpoint that
// mixes fused-dense layers with quantized neighbours. A quantized base is refused
// here — fusing one means dequantize-merge-then-requantize the whole model, a
// separate path. A targeted layer missing its base weight, or a shape that does
// not match the delta, is a loud error rather than a silent skip.
//
//	scale := alpha / float32(rank)
//	merged, fused, err := metal.FuseLoRAIntoWeights(base, adapter, scale)
func FuseLoRAIntoWeights(base, adapter map[string]*Array, scale float32) (map[string]*Array, []string, error) {
	const op = "metal.FuseLoRAIntoWeights"

	merged := make(map[string]*Array, len(base))
	for name, arr := range base {
		merged[name] = arr
	}

	var fused []string
	for key, a := range adapter {
		if !core.HasSuffix(key, ".lora_a") {
			continue
		}
		layer := key[:len(key)-len(".lora_a")]
		b := adapter[layer+".lora_b"]
		if a == nil || b == nil {
			return nil, nil, core.E(op, core.Sprintf("layer %q: lora_a/lora_b pair incomplete", layer), nil)
		}

		baseName := layer + ".weight"
		baseW := ResolveWeight(base, baseName)
		if baseW == nil {
			return nil, nil, core.E(op, core.Sprintf("layer %q: no base weight %q for the adapter delta", layer, baseName), nil)
		}
		// A dense base only — refuse a quantized base rather than mis-merging a
		// packed weight against a dense delta.
		if ResolveWeight(base, layer+".scales") != nil {
			return nil, nil, core.E(op, core.Sprintf("layer %q: base is quantized; fuse requires a dense (bf16/fp32) base", layer), nil)
		}

		// delta = (B·A)·scale, shape [out, in] to match the [out, in] base weight.
		ba := Matmul(b, a)
		delta := MulScalar(ba, scale)
		Free(ba)

		if !slices.Equal(baseW.Shape(), delta.Shape()) {
			Free(delta)
			return nil, nil, core.E(op, core.Sprintf("layer %q: base weight shape %v != lora delta shape %v", layer, baseW.Shape(), delta.Shape()), nil)
		}

		mergedW := Add(baseW, delta)
		Materialize(mergedW)
		Detach(mergedW)
		Free(delta)

		merged[baseName] = mergedW
		fused = append(fused, layer)
	}

	slices.Sort(fused)
	return merged, fused, nil
}

// FuseModelDir is the on-disk fuse (the P-phase serializer): it reads a dense
// base model dir and a trained LoRA adapter dir, folds the adapter into the base
// at the weights-map level, writes the fused dense model to outDir, and carries
// the base's servable sidecars (config.json + tokenizer files) across. It returns
// the layers it fused. The adapter is the engine's own format
// (adapter.safetensors + adapter_config.json, alpha/rank scaling) — the output of
// LoRA training's adapter.Save.
//
//	fused, err := metal.FuseModelDir("/models/lemma-base", "/lora/lek2", "/models/lemma-fused")
func FuseModelDir(baseDir, adapterDir, outDir string) ([]string, error) {
	const op = "metal.FuseModelDir"

	// Discard any stale process-global MLX error left by a prior unrelated
	// operation: LoadModelWeights reads the (sticky) LastError after a successful
	// LoadSafetensors, so a leftover error would falsely fail this fresh load.
	_ = LastError()

	base, err := LoadModelWeights(baseDir)
	if err != nil {
		return nil, core.E(op, "load base weights", err)
	}
	config, err := parseAdapterConfig(core.JoinPath(adapterDir, "adapter_config.json"))
	if err != nil {
		return nil, core.E(op, "parse adapter config", err)
	}
	if config.Rank <= 0 {
		return nil, core.E(op, "adapter config rank must be > 0", nil)
	}
	adapterWeights, err := loadAdapterWeights(adapterDir)
	if err != nil {
		return nil, core.E(op, "load adapter weights", err)
	}
	scale := config.Alpha / float32(config.Rank)

	merged, fused, err := FuseLoRAIntoWeights(base, adapterWeights, scale)
	if err != nil {
		return nil, core.E(op, "fuse adapter into base", err)
	}
	if len(fused) == 0 {
		return nil, core.E(op, "adapter fused no layers — its targets matched no base weight", nil)
	}

	if result := core.MkdirAll(outDir, 0o755); !result.OK {
		return nil, core.E(op, "ensure output dir "+outDir, nil)
	}
	// Materialise so the save reads real data, not an unexecuted graph.
	all := make([]*Array, 0, len(merged))
	for _, a := range merged {
		all = append(all, a)
	}
	Materialize(all...)
	if err := SaveSafetensors(core.JoinPath(outDir, "model.safetensors"), merged); err != nil {
		return nil, core.E(op, "write fused model.safetensors", err)
	}

	// Carry the base's servable sidecars across — a fused dense model keeps the
	// base's config + tokenizer. A safetensors shard index is intentionally not
	// copied: it would be stale for the single re-consolidated model.safetensors.
	for _, name := range []string{"config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "generation_config.json"} {
		copyDirFile(baseDir, outDir, name)
	}
	return fused, nil
}

// copyDirFile copies one sidecar from srcDir to dstDir when it exists; a missing
// optional sidecar is silently skipped (not every model carries every file).
func copyDirFile(srcDir, dstDir, name string) {
	src := core.JoinPath(srcDir, name)
	info := core.Stat(src)
	if !info.OK || info.Value.(core.FsFileInfo).IsDir() {
		return
	}
	data, err := coreio.Local.Read(src)
	if err != nil {
		return
	}
	_ = coreio.Local.Write(core.JoinPath(dstDir, name), data)
}
