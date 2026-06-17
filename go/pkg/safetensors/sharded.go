// SPDX-Licence-Identifier: EUPL-1.2

package safetensors

import (
	core "dappco.re/go"
	coreio "dappco.re/go/io"
)

// Standard HF checkpoint file names within a model directory.
const (
	indexName  = "model.safetensors.index.json"
	singleName = "model.safetensors"
)

// shardIndex is the subset of model.safetensors.index.json LoadDir consumes: weight_map names
// each tensor to the shard file holding it. (The metadata/total_size block is informational
// and ignored.)
type shardIndex struct {
	WeightMap map[string]string `json:"weight_map"`
}

// LoadDir loads a gemma4 checkpoint directory, handling BOTH layouts HF emits: a SHARDED
// checkpoint (model.safetensors.index.json + model-NNNNN-of-NNNNN.safetensors shards) or a
// SINGLE model.safetensors. It returns the merged name→Tensor map — the same shape Parse/Load
// give for one blob — so the gemma4 assembler is identical however the weights were split.
// Each shard is read+parsed ONCE (cached by file name), not once per tensor; this is the thin
// I/O layer over Parse the single-blob LoadGemma4BF16 doc flags. Loading a real multi-GB
// checkpoint is a deliberate, memory-heavy step — each shard's bytes stay resident, sub-sliced
// by its tensors (no copy), so the whole model is in memory once merged.
func LoadDir(dir string) (map[string]Tensor, error) {
	idxPath := core.PathJoin(dir, indexName)
	if coreio.Local.IsFile(idxPath) {
		idxStr, err := coreio.Local.Read(idxPath)
		if err != nil {
			return nil, core.E("safetensors.LoadDir", "read "+idxPath, err)
		}
		var idx shardIndex
		if r := core.JSONUnmarshal([]byte(idxStr), &idx); !r.OK {
			return nil, core.NewError("safetensors.LoadDir: " + indexName + " parse failed")
		}
		if len(idx.WeightMap) == 0 {
			return nil, core.NewError("safetensors.LoadDir: " + indexName + " has an empty weight_map")
		}
		shards := make(map[string]map[string]Tensor) // each shard parsed once, keyed by file name
		out := make(map[string]Tensor, len(idx.WeightMap))
		for name, shard := range idx.WeightMap {
			parsed, ok := shards[shard]
			if !ok {
				p, err := Load(core.PathJoin(dir, shard))
				if err != nil {
					return nil, core.E("safetensors.LoadDir", "load shard "+shard, err)
				}
				shards[shard] = p
				parsed = p
			}
			t, ok := parsed[name]
			if !ok {
				return nil, core.NewError("safetensors.LoadDir: index maps " + name + " to " + shard + " but that shard lacks it")
			}
			out[name] = t
		}
		return out, nil
	}
	singlePath := core.PathJoin(dir, singleName)
	if coreio.Local.IsFile(singlePath) {
		return Load(singlePath)
	}
	return nil, core.NewError("safetensors.LoadDir: neither " + indexName + " nor " + singleName + " found in " + dir)
}
