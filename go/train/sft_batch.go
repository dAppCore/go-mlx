// SPDX-Licence-Identifier: EUPL-1.2

// Package train holds the native training machinery — SFT batch building,
// sequence packing, checkpoint metadata, the LoRA epoch loop, and the SSD
// (sampling-and-fine-tuning) pipeline with its code benchmark. The root mlx
// package aliases the exported types and keeps the Model-bound entry points
// (Model.TrainSFT / Model.RunSSD), which delegate here.
package train

import (
	"unsafe"

	core "dappco.re/go"
	"dappco.re/go/mlx/dataset"
	"dappco.re/go/mlx/pkg/metal"
	"dappco.re/go/mlx/spine"
)

type sftExample struct {
	inputs  []int
	targets []int
	mask    []float32
}

// BuildSFTTrainingBatches tokenizes an SFT dataset using runner-level batching settings.
func BuildSFTTrainingBatches(tok *spine.Tokenizer, ds dataset.Dataset, cfg SFTConfig) ([]SFTBatch, error) {
	if !tok.Valid() {
		return nil, core.NewError("mlx: tokenizer is nil")
	}
	if ds == nil {
		return nil, core.NewError("mlx: SFT dataset is nil")
	}
	cfg = normalizeSFTConfig(cfg)
	return BuildDatasetBatches(tok, ds, dataset.BatchConfig{
		BatchSize:       SFTEffectiveBatchSize(cfg),
		MaxSeqLen:       cfg.MaxSeqLen,
		SequencePacking: cfg.SequencePacking,
		NoEOS:           cfg.NoEOS,
	})
}

// BuildSFTBatches tokenizes an SFT dataset into response-masked training batches.
func BuildSFTBatches(tok *spine.Tokenizer, ds dataset.Dataset, cfg SFTConfig) ([]SFTBatch, error) {
	if !tok.Valid() {
		return nil, core.NewError("mlx: tokenizer is nil")
	}
	if ds == nil {
		return nil, core.NewError("mlx: SFT dataset is nil")
	}

	cfg = normalizeSFTConfig(cfg)
	builder := newSFTBatchBuilder(cfg.BatchSize)
	// Hoist a small per-call SFTConfig for buildSFTExample — it only
	// reads MaxSeqLen + NoEOS and never mutates, so the same value is
	// safe to share across every sample. Passing the full SFTConfig by
	// value copied 18 fields (including embedded LoRAConfig with two
	// []string slices) per sample; the narrowed struct strips that
	// per-iteration copy. Mirrors BuildDatasetBatches's existing hoist.
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
		if !usable {
			continue
		}
		builder.add(example)
	}
	return builder.finish(), nil
}

type sftBatchBuilder struct {
	batchSize int
	current   []sftExample
	out       []SFTBatch
}

func newSFTBatchBuilder(batchSize int) *sftBatchBuilder {
	if batchSize <= 0 {
		batchSize = 1
	}
	// Pre-size current to batchSize — every flush truncates back to :0
	// with the same backing, so the doubling cascade across the first
	// batch's appends collapses to a single allocation that gets reused
	// for every subsequent batch.
	//
	// Pre-size out to cap=4 — short SFT runs (single-epoch over a small
	// dataset) flush 1-4 batches, hitting the 0→1→2→4 doubling cascade
	// on every Build call. The 4-element pre-size collapses two
	// reallocations into one upfront ~384 B allocation. Larger runs
	// still grow exponentially from 4 onward (4→8→16…), trading two
	// fewer reallocations for the same upfront cost.
	return &sftBatchBuilder{
		batchSize: batchSize,
		current:   make([]sftExample, 0, batchSize),
		out:       make([]SFTBatch, 0, 4),
	}
}

func (b *sftBatchBuilder) add(example sftExample) {
	b.current = append(b.current, example)
	if len(b.current) >= b.batchSize {
		b.flush()
	}
}

func (b *sftBatchBuilder) finish() []SFTBatch {
	b.flush()
	// Hand b.out directly to the caller — finish() is the terminal
	// builder call and b is discarded immediately by every existing
	// caller (BuildSFTBatches / BuildDatasetBatches). The defensive
	// core.SliceClone the original form paid only trimmed the slice
	// from append-grown cap down to exact len, providing no isolation
	// (the SFTBatch elements still share their inner []]int slices).
	// Caller-side memory hygiene from cap == len is not worth one
	// per-build allocation.
	return b.out
}

func (b *sftBatchBuilder) flush() {
	if len(b.current) == 0 {
		return
	}
	b.out = append(b.out, sftBatchFromExamples(b.current))
	b.current = b.current[:0]
}

func sftBatchFromExamples(examples []sftExample) SFTBatch {
	n := len(examples)
	// Share one 3n-wide slice-header backing across Tokens + Targets +
	// LossMask. [][]int and [][]float32 have identical 24-byte slice
	// header layout (data ptr + len + cap) and identical GC scan masks
	// (one pointer field at offset 0), so reinterpreting a trailing
	// stretch of [][]int as [][]float32 via unsafe.Slice is sound. The
	// caller-side semantics (Tokens[i] is []int, LossMask[i] is
	// []float32) stay intact because the assignment fully overwrites
	// each header with the correct typed slice from the source example.
	// Length stays []int (different element layout — 8 B int vs 24 B
	// slice header). Net: 3 allocs → 2 allocs per batch.
	headers := make([][]int, 3*n)
	lossMaskBacking := headers[2*n : 3*n : 3*n]
	var lossMask [][]float32
	if n > 0 {
		lossMask = unsafe.Slice((*[]float32)(unsafe.Pointer(&lossMaskBacking[0])), n)
	}
	batch := SFTBatch{
		Batch: metal.Batch{
			Tokens:   headers[:n:n],
			Length:   make([]int, n),
			LossMask: lossMask,
		},
		Targets: headers[n : 2*n : 2*n],
	}
	// Transfer ownership of each example's slices into the batch — the
	// callers (sftBatchBuilder.flush and runSFTDatasetEpoch.flushCurrent)
	// truncate the examples slice immediately after this call, dropping
	// their last live reference to the struct values. Every sftExample
	// originates from buildSFTExample which always returns fresh
	// allocations (no aliasing), or from sftStreamingPacker.flush which
	// already transferred ownership exclusively to the example. The
	// previous per-element SliceClone trio was three pointless
	// allocations per example per batch — gone now that the batch is the
	// sole owner.
	for i := range examples {
		example := &examples[i]
		batch.Batch.Tokens[i] = example.inputs
		batch.Batch.Length[i] = len(example.inputs)
		batch.Batch.LossMask[i] = example.mask
		batch.Targets[i] = example.targets
	}
	return batch
}

func buildSFTExample(tok *spine.Tokenizer, sample dataset.Sample, cfg SFTConfig) (sftExample, bool, error) {
	trainWholeText := sample.Text != ""
	if trainWholeText {
		return buildSFTExampleText(tok, sample, cfg)
	}
	return buildSFTExamplePromptResponse(tok, sample, cfg)
}

// buildSFTExamplePromptResponse handles the dominant prompt+response path.
// It never materialises the concatenated int32 sequence the obvious form
// would build (promptIDs ++ responseIDs ++ EOS) — that intermediate was a
// per-sample allocation whose only use was being walked once to widen each
// element into the int inputs/targets. Instead the virtual sequence
// V = promptIDs | responseIDs | [EOS] is described by its three pieces and
// the inputs/targets are filled directly from them: inputs = V[0..n),
// targets = V[1..n+1). Net: drops the seq make on every prompt/response row
// (2 allocs → 1 on this path), leaving only the shared inputs/targets/mask
// backing. Held byte-identical to buildSFTExampleRef by the differential
// matrix in sft_buildexample_test.go.
func buildSFTExamplePromptResponse(tok *spine.Tokenizer, sample dataset.Sample, cfg SFTConfig) (sftExample, bool, error) {
	promptIDs, err := tok.Encode(sample.Prompt)
	if err != nil {
		return sftExample{}, false, err
	}
	responseIDs, err := tok.Encode(sample.Response)
	if err != nil {
		return sftExample{}, false, err
	}
	promptLen := len(promptIDs)
	extra := 0
	var eos int32
	if !cfg.NoEOS {
		extra = 1
		eos = tok.EOS()
	}
	// vLen is the length of the virtual sequence V; n = vLen-1 is the number
	// of (input, target) pairs (each target is the next virtual element).
	vLen := len(promptIDs) + len(responseIDs) + extra
	if vLen < 2 {
		return sftExample{}, false, nil
	}
	n := vLen - 1

	// inputs + targets + mask share ONE backing: 2n+(n+1)/2 ints worth, where
	// the trailing (n+1)/2 ints host n float32s via unsafe.Slice reinterpret.
	// []int is 8-byte aligned (guaranteed by Go's allocator) which exceeds
	// float32's 4-byte alignment requirement, so the reinterpret is safe.
	// Neither []int nor []float32 contains pointers so GC scanning of the
	// combined allocation is straightforward (one base pointer kept alive
	// while any of the three views is referenced).
	maskInts := (n + 1) / 2
	combined := make([]int, 2*n+maskInts)
	inputs := combined[:n:n]
	targets := combined[n : 2*n : 2*n]
	// inputs[i] = int(V[i]); targets[i] = int(V[i+1]). Fill both directly
	// from the source pieces with segment copies — sftWidenVirtual walks the
	// prompt/response/EOS pieces once per destination, widening int32→int as
	// it goes, so the dropped seq slice's single use is folded in here.
	sftWidenVirtual(inputs, promptIDs, responseIDs, eos, extra, 0)
	sftWidenVirtual(targets, promptIDs, responseIDs, eos, extra, 1)

	var mask []float32
	if n > 0 {
		mask = unsafe.Slice((*float32)(unsafe.Pointer(&combined[2*n])), n)
		// combined is freshly allocated and zero-initialised; the
		// reinterpreted mask view inherits that zero state byte-for-byte
		// (n floats of all-zero bytes is the +0.0 representation).
	}
	// mask is zero-initialised by make — only write the trailing 1s starting
	// where the response begins (i+1 >= promptLen).
	start := max(promptLen-1, 0)
	if start < len(mask) {
		tail := mask[start:]
		for i := range tail {
			tail[i] = 1
		}
	}
	return sftTruncateAndReturn(inputs, targets, mask, cfg)
}

// sftWidenVirtual fills dst[i] = int(V[i+shift]) where V is the virtual
// concatenation prompt | response | [eos] (eos present iff extra==1). It
// copies each overlapping source segment in turn with an int32→int widen,
// so no intermediate []int32 sequence is allocated. dst length plus shift
// must not exceed the virtual length.
func sftWidenVirtual(dst []int, prompt, response []int32, eos int32, extra, shift int) {
	pos := 0 // write cursor into dst
	// Prompt segment: virtual indices [0, len(prompt)).
	for vi := shift; vi < len(prompt) && pos < len(dst); vi++ {
		dst[pos] = int(prompt[vi])
		pos++
	}
	// Response segment: virtual indices [len(prompt), len(prompt)+len(response)).
	rStart := shift - len(prompt)
	if rStart < 0 {
		rStart = 0
	}
	for ri := rStart; ri < len(response) && pos < len(dst); ri++ {
		dst[pos] = int(response[ri])
		pos++
	}
	// EOS segment: single virtual index len(prompt)+len(response).
	if extra == 1 && pos < len(dst) {
		dst[pos] = int(eos)
		pos++
	}
}

// buildSFTExampleText handles the free-form text path (whole sequence is the
// training target). Left as the original allocation-naive form: it reuses the
// tokenizer's freshly allocated ids directly and appends EOS in place, so the
// seq slice is already owned exclusively — there is no intermediate to drop.
func buildSFTExampleText(tok *spine.Tokenizer, sample dataset.Sample, cfg SFTConfig) (sftExample, bool, error) {
	ids, err := tok.Encode(sample.Text)
	if err != nil {
		return sftExample{}, false, err
	}
	// Reuse ids directly — Tokenizer.Encode allocates a fresh slice per call
	// (internal tokenizer.Encode + stripImplicitBOS), so we own it
	// exclusively. The downstream EOS append usually fits the existing cap
	// (inner Encode over-allocates len(text)+1); if not, append falls back to
	// a single re-alloc — strictly no worse than the previous unconditional
	// make+copy.
	seq := ids
	if !cfg.NoEOS {
		seq = append(seq, tok.EOS())
	}
	if len(seq) < 2 {
		return sftExample{}, false, nil
	}
	n := len(seq) - 1
	maskInts := (n + 1) / 2
	combined := make([]int, 2*n+maskInts)
	inputs := combined[:n:n]
	targets := combined[n : 2*n : 2*n]
	for i := range n {
		inputs[i] = int(seq[i])
		targets[i] = int(seq[i+1])
	}
	var mask []float32
	if n > 0 {
		mask = unsafe.Slice((*float32)(unsafe.Pointer(&combined[2*n])), n)
	}
	for i := range mask {
		mask[i] = 1
	}
	return sftTruncateAndReturn(inputs, targets, mask, cfg)
}

// sftTruncateAndReturn applies the MaxSeqLen tail-truncation (shared by both
// build paths) and gates on the presence of a training target.
func sftTruncateAndReturn(inputs, targets []int, mask []float32, cfg SFTConfig) (sftExample, bool, error) {
	if cfg.MaxSeqLen > 0 && len(inputs) > cfg.MaxSeqLen {
		start := len(inputs) - cfg.MaxSeqLen
		// Combined-backing carve for the truncated inputs+targets — same
		// share trick the construction path uses, except now the original
		// 2n backing is being trimmed to 2*MaxSeqLen. One alloc covers
		// both slices instead of two SliceClones. The mask clone stays
		// separate (different element type).
		truncLen := cfg.MaxSeqLen
		truncBacking := make([]int, 2*truncLen)
		copy(truncBacking[:truncLen], inputs[start:])
		copy(truncBacking[truncLen:], targets[start:])
		inputs = truncBacking[:truncLen:truncLen]
		targets = truncBacking[truncLen : 2*truncLen : 2*truncLen]
		mask = core.SliceClone(mask[start:])
	}
	if !hasTrainingTarget(mask) {
		return sftExample{}, false, nil
	}
	return sftExample{inputs: inputs, targets: targets, mask: mask}, true, nil
}

func hasTrainingTarget(mask []float32) bool {
	// Scan backward — the SFT response mask is zero across the prompt
	// region and one across the response region, with the response
	// region at the tail. A backward scan finds the first 1 in O(1)
	// for typical inputs; the original forward scan walked the entire
	// prompt prefix every time. For trainWholeText the mask is all
	// ones so direction doesn't matter (O(1) either way). The
	// no-training-target case still costs O(N) but that's the rare
	// path filtered out by the caller.
	for i := len(mask) - 1; i >= 0; i-- {
		if mask[i] != 0 {
			return true
		}
	}
	return false
}
