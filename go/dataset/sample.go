// SPDX-Licence-Identifier: EUPL-1.2

// Package dataset holds dataset-shaped types and JSONL ingestion for the
// go-mlx training and evaluation stacks.
package dataset

import core "dappco.re/go"

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
		return Sample{}, false, core.NewError("dataset: dataset func is nil")
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
	return &SliceDataset{samples: append([]Sample(nil), samples...)}
}

// Next returns the next sample.
func (d *SliceDataset) Next() (Sample, bool, error) {
	if d == nil {
		return Sample{}, false, core.NewError("dataset: slice dataset is nil")
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
		return core.NewError("dataset: slice dataset is nil")
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
	if len(values) == 0 {
		return nil
	}
	out := make(map[string]string, len(values))
	for key, value := range values {
		out[key] = value
	}
	return out
}
