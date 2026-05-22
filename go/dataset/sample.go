// SPDX-Licence-Identifier: EUPL-1.2

// Package dataset holds dataset-shaped types and JSONL ingestion for the
// go-mlx training and evaluation stacks.
package dataset

import core "dappco.re/go"

// Sentinel errors hoisted from the nil-guard call sites so they
// allocate exactly once at package init instead of one *Err per
// nil-receiver call. These are cold paths (only fire when a caller
// has passed a nil receiver) but the package contract is the same
// either way.
var (
	errFuncDatasetNil  = core.NewError("dataset: dataset func is nil")
	errSliceDatasetNil = core.NewError("dataset: slice dataset is nil")
)

// Sample is one supervised fine-tuning record.
type Sample struct {
	Prompt   string
	Response string
	Text     string
	Meta     map[string]string
}

// Dataset streams supervised fine-tuning records.
type Dataset interface {
	Next() (Sample, bool, error)
}

// Resetter marks datasets that can be replayed for multiple epochs.
type Resetter interface {
	Reset() error
}

// Func adapts a function into a Dataset.
type Func func() (Sample, bool, error)

// Next returns the next sample from the wrapped function.
//
//	dataset := dataset.Func(func() (dataset.Sample, bool, error) { ... })
func (fn Func) Next() (Sample, bool, error) {
	if fn == nil {
		return Sample{}, false, errFuncDatasetNil
	}
	return fn()
}

// SliceDataset is an in-memory replayable dataset.
type SliceDataset struct {
	samples []Sample
	index   int
}

// NewSliceDataset returns a replayable dataset backed by samples.
//
//	d := dataset.NewSliceDataset(samples)
func NewSliceDataset(samples []Sample) *SliceDataset {
	return &SliceDataset{samples: core.SliceClone(samples)}
}

// Next returns the next sample.
func (d *SliceDataset) Next() (Sample, bool, error) {
	if d == nil {
		return Sample{}, false, errSliceDatasetNil
	}
	if d.index >= len(d.samples) {
		return Sample{}, false, nil
	}
	sample := d.samples[d.index]
	d.index++
	return sample, true, nil
}

// Reset rewinds the dataset.
func (d *SliceDataset) Reset() error {
	if d == nil {
		return errSliceDatasetNil
	}
	d.index = 0
	return nil
}

// CloneSample returns a defensive deep copy of sample including Meta.
//
//	copy := dataset.CloneSample(sample)
func CloneSample(sample Sample) Sample {
	sample.Meta = cloneStringMap(sample.Meta)
	return sample
}

// CloneSamples returns a defensive deep copy of samples.
//
//	copies := dataset.CloneSamples(samples)
func CloneSamples(samples []Sample) []Sample {
	if len(samples) == 0 {
		return nil
	}
	out := make([]Sample, len(samples))
	for i, sample := range samples {
		out[i] = CloneSample(sample)
	}
	return out
}

func cloneStringMap(values map[string]string) map[string]string {
	// core.MapClone wraps maps.Clone which uses runtime internals to
	// pre-size the destination and bulk-copy entries, skipping the
	// per-key hash/insert ceremony of a range-copy loop. Returns nil
	// for an empty input (matching the prior nil-fast-path).
	if len(values) == 0 {
		return nil
	}
	return core.MapClone(values)
}
