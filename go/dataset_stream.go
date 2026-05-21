// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	core "dappco.re/go"
	"dappco.re/go/mlx/dataset"
)

// BuildDatasetBatches tokenizes a dataset with optional sequence packing.
//
//	batches, err := mlx.BuildDatasetBatches(tok, ds, dataset.BatchConfig{BatchSize: 4, MaxSeqLen: 1024})
func BuildDatasetBatches(tok *Tokenizer, ds dataset.Dataset, cfg dataset.BatchConfig) ([]SFTBatch, error) {
	if !cfg.SequencePacking {
		return BuildSFTBatches(tok, ds, SFTConfig{
			BatchSize: cfg.BatchSize,
			MaxSeqLen: cfg.MaxSeqLen,
			NoEOS:     cfg.NoEOS,
		})
	}
	if tok == nil || tok.tok == nil {
		return nil, core.NewError("mlx: tokenizer is nil")
	}
	if ds == nil {
		return nil, core.NewError("mlx: dataset is nil")
	}
	cfg = normalizeDatasetBatchConfig(cfg)
	builder := newSFTBatchBuilder(cfg.BatchSize)
	packer := newDatasetPacker(cfg.MaxSeqLen, builder)
	for {
		sample, ok, err := ds.Next()
		if err != nil {
			return nil, err
		}
		if !ok {
			break
		}
		example, usable, err := buildSFTExample(tok, sample, SFTConfig{MaxSeqLen: cfg.MaxSeqLen, NoEOS: cfg.NoEOS})
		if err != nil {
			return nil, err
		}
		if usable {
			packer.add(example)
		}
	}
	packer.finish()
	return builder.finish(), nil
}

func normalizeDatasetBatchConfig(cfg dataset.BatchConfig) dataset.BatchConfig {
	if cfg.BatchSize <= 0 {
		cfg.BatchSize = 1
	}
	return cfg
}

type datasetPacker struct {
	maxSeqLen int
	builder   *sftBatchBuilder
	current   sftExample
}

func newDatasetPacker(maxSeqLen int, builder *sftBatchBuilder) *datasetPacker {
	p := &datasetPacker{maxSeqLen: maxSeqLen, builder: builder}
	// When maxSeqLen is known, p.current can never grow past it — packer
	// flushes the moment another row would overflow. Pre-allocate the
	// staging buffers at maxSeqLen capacity so the per-add appends never
	// trigger reallocation. Re-allocated to fresh maxSeqLen after each
	// flush() so the builder owns the previous backing array.
	if maxSeqLen > 0 {
		p.current.inputs = make([]int, 0, maxSeqLen)
		p.current.targets = make([]int, 0, maxSeqLen)
		p.current.mask = make([]float32, 0, maxSeqLen)
	}
	return p
}

func (p *datasetPacker) add(example sftExample) {
	if p == nil || p.builder == nil {
		return
	}
	if len(example.inputs) == 0 {
		return
	}
	if p.maxSeqLen > 0 && len(p.current.inputs) > 0 && len(p.current.inputs)+len(example.inputs) > p.maxSeqLen {
		p.flush()
	}
	if p.maxSeqLen > 0 && len(example.inputs) > p.maxSeqLen {
		start := len(example.inputs) - p.maxSeqLen
		// Pre-sized clones avoid the append-grow cascade — each tail is
		// exactly maxSeqLen long.
		tailLen := p.maxSeqLen
		inputs := make([]int, tailLen)
		copy(inputs, example.inputs[start:])
		targets := make([]int, tailLen)
		copy(targets, example.targets[start:])
		mask := make([]float32, tailLen)
		copy(mask, example.mask[start:])
		example.inputs = inputs
		example.targets = targets
		example.mask = mask
	}
	p.current.inputs = append(p.current.inputs, example.inputs...)
	p.current.targets = append(p.current.targets, example.targets...)
	p.current.mask = append(p.current.mask, example.mask...)
}

func (p *datasetPacker) finish() {
	if p != nil {
		p.flush()
	}
}

func (p *datasetPacker) flush() {
	if p == nil || p.builder == nil || len(p.current.inputs) == 0 {
		return
	}
	// Hand the builder a fully-sized clone — len is known so each make
	// allocates exactly once instead of going through append grow stages.
	n := len(p.current.inputs)
	inputs := make([]int, n)
	copy(inputs, p.current.inputs)
	targets := make([]int, n)
	copy(targets, p.current.targets)
	mask := make([]float32, n)
	copy(mask, p.current.mask)
	p.builder.add(sftExample{
		inputs:  inputs,
		targets: targets,
		mask:    mask,
	})
	// Truncate staging buffers in place — the cloned slices above are
	// what the builder now owns, so the original backing arrays are
	// still ours and ready for the next pack cycle without a fresh make.
	p.current.inputs = p.current.inputs[:0]
	p.current.targets = p.current.targets[:0]
	p.current.mask = p.current.mask[:0]
}
