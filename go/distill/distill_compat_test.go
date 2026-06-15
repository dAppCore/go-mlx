// SPDX-Licence-Identifier: EUPL-1.2

package distill

import (
	"testing"
	"time"

	"dappco.re/go/mlx/dataset"
)

// nonZeroDuration and normalizeDatasetBatchConfig are the unexported leaf
// helpers carried with the package on extraction. They have no exported
// surface and no genuine standalone usage example, so they are covered
// with Good/Bad unit assertions only (no theatre Example).

func TestNonZeroDuration_FloorsToNanosecond_Good(t *testing.T) {
	// A positive duration passes through untouched.
	if got := nonZeroDuration(5 * time.Millisecond); got != 5*time.Millisecond {
		t.Fatalf("nonZeroDuration(5ms) = %v, want 5ms passthrough", got)
	}
}

func TestNonZeroDuration_ZeroAndNegativeFloored_Bad(t *testing.T) {
	// A zero or negative measured duration (clock granularity on a
	// sub-nanosecond run) is floored to one nanosecond so callers never
	// record a zero-length run.
	for _, d := range []time.Duration{0, -time.Second} {
		if got := nonZeroDuration(d); got != time.Nanosecond {
			t.Fatalf("nonZeroDuration(%v) = %v, want 1ns floor", d, got)
		}
	}
}

func TestNormalizeDatasetBatchConfig_DefaultsBatchSize_Good(t *testing.T) {
	// A configured positive batch size is preserved.
	got := normalizeDatasetBatchConfig(dataset.BatchConfig{BatchSize: 8})
	if got.BatchSize != 8 {
		t.Fatalf("BatchSize = %d, want 8 preserved", got.BatchSize)
	}
}

func TestNormalizeDatasetBatchConfig_NonPositiveBatchSize_Bad(t *testing.T) {
	// A zero or negative batch size is normalised up to one so the batch
	// builder never divides by a non-positive size.
	for _, size := range []int{0, -4} {
		got := normalizeDatasetBatchConfig(dataset.BatchConfig{BatchSize: size})
		if got.BatchSize != 1 {
			t.Fatalf("normalizeDatasetBatchConfig(BatchSize=%d) = %d, want 1", size, got.BatchSize)
		}
	}
}
