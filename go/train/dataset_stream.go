// SPDX-Licence-Identifier: EUPL-1.2

package train

import (
	"unsafe"

	core "dappco.re/go"
	"dappco.re/go/mlx/dataset"
	"dappco.re/go/mlx/spine"
)

// BuildDatasetBatches tokenizes a dataset with optional sequence packing.
//
//	batches, err := train.BuildDatasetBatches(tok, ds, dataset.BatchConfig{BatchSize: 4, MaxSeqLen: 1024})
func BuildDatasetBatches(tok *spine.Tokenizer, ds dataset.Dataset, cfg dataset.BatchConfig) ([]SFTBatch, error) {
	if !cfg.SequencePacking {
		return BuildSFTBatches(tok, ds, SFTConfig{
			BatchSize: cfg.BatchSize,
			MaxSeqLen: cfg.MaxSeqLen,
			NoEOS:     cfg.NoEOS,
		})
	}
	if !tok.Valid() {
		return nil, core.NewError("mlx: tokenizer is nil")
	}
	if ds == nil {
		return nil, core.NewError("mlx: dataset is nil")
	}
	cfg = normalizeDatasetBatchConfig(cfg)
	builder := newSFTBatchBuilder(cfg.BatchSize)
	packer := newDatasetPacker(cfg.MaxSeqLen, builder)
	// Hoist per-sample SFTConfig out of the loop — buildSFTExample only
	// reads MaxSeqLen + NoEOS and never mutates, so the same value is
	// safe to share across every sample.
	exampleCfg := SFTConfig{MaxSeqLen: cfg.MaxSeqLen, NoEOS: cfg.NoEOS}
	for {
		sample, ok, err := ds.Next()
		if err != nil {
			return nil, err
		}
		if !ok {
			break
		}
		example, usable, err := buildSFTExample(tok, sample, exampleCfg)
		if err != nil {
			return nil, err
		}
		if usable {
			packer.add(&example)
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
	// Lazy first-add allocation — see add() for the why. Upfront
	// pre-sizing is wasted work for the NoPack path (newDatasetPacker
	// is unreachable, but kept symmetric with sftStreamingPacker) and
	// would force a second per-flush allocation pair every time the
	// previous flush handed staging to the builder.
	return &datasetPacker{maxSeqLen: maxSeqLen, builder: builder}
}

func (p *datasetPacker) add(example *sftExample) {
	if p == nil || p.builder == nil || example == nil {
		return
	}
	if len(example.inputs) == 0 {
		return
	}
	if p.maxSeqLen > 0 && len(p.current.inputs) > 0 && len(p.current.inputs)+len(example.inputs) > p.maxSeqLen {
		p.flush()
	}
	// Source slices for the per-add append. When truncating an oversized
	// example we just narrow the source range — the previous code copied
	// the tail into fresh slices first, but the subsequent appends into
	// p.current already do that copy, so the intermediate make+copy was
	// wasted work.
	srcInputs := example.inputs
	srcTargets := example.targets
	srcMask := example.mask
	if p.maxSeqLen > 0 && len(srcInputs) > p.maxSeqLen {
		start := len(srcInputs) - p.maxSeqLen
		srcInputs = srcInputs[start:]
		srcTargets = srcTargets[start:]
		srcMask = srcMask[start:]
	}
	// First add into an empty accumulator: pre-size to maxSeqLen (when
	// known) so the doubling cascade across subsequent appends collapses
	// into a single allocation per accumulator field. Inputs + Targets +
	// Mask all share ONE backing of 2*maxSeqLen+(maxSeqLen+1)/2 ints. The
	// inputs/targets are two cap-maxSeqLen []int views over the first
	// 2*maxSeqLen ints; mask is a cap-maxSeqLen []float32 view reinterpreted
	// over the trailing (maxSeqLen+1)/2 ints. All three grow in lockstep —
	// mask appends the same count as inputs on every add (see the appends
	// below) and the caller flushes before any add would push past maxSeqLen
	// — so none ever grows past its cap and the in-place appends never
	// realloc or stomp a neighbour's region. []int is 8-byte aligned
	// (exceeds float32's 4-byte requirement) and neither type contains
	// pointers, so the float32 reinterpret is sound and GC scanning of the
	// combined allocation stays straightforward. Folding mask into the
	// shared backing drops the separate []float32 make — one allocation per
	// first-add instead of two. This is the same shared-backing-with-mask
	// pattern buildSFTExamplePromptResponse uses for the per-example case.
	if p.maxSeqLen > 0 && cap(p.current.inputs) == 0 {
		maskInts := (p.maxSeqLen + 1) / 2
		intBacking := make([]int, 2*p.maxSeqLen+maskInts)
		p.current.inputs = intBacking[:0:p.maxSeqLen]
		p.current.targets = intBacking[p.maxSeqLen : p.maxSeqLen : 2*p.maxSeqLen]
		p.current.mask = unsafe.Slice((*float32)(unsafe.Pointer(&intBacking[2*p.maxSeqLen])), p.maxSeqLen)[:0:p.maxSeqLen]
	}
	p.current.inputs = append(p.current.inputs, srcInputs...)
	p.current.targets = append(p.current.targets, srcTargets...)
	p.current.mask = append(p.current.mask, srcMask...)
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
	// Hand the builder p.current's backing arrays directly — the
	// immediately-following p.current = sftExample{} drops our last
	// reference to them, so the builder is the sole owner. The previous
	// form cloned all three slices then nuked the originals, paying three
	// copy()-sized memory writes per flush (up to maxSeqLen elements
	// each). The next add() re-allocates fresh buffers via the
	// cap(p.current.inputs) == 0 branch, same allocation count as the
	// previous in-place truncate-and-reuse path. Mirrors the ownership
	// flip already in sftStreamingPacker.flush.
	example := p.current
	p.current = sftExample{}
	p.builder.add(example)
}
