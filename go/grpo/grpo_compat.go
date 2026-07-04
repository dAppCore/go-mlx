// SPDX-Licence-Identifier: EUPL-1.2

package grpo

import (
	"time"

	core "dappco.re/go"
	mlx "dappco.re/go/mlx"
)

// ModelInfo and Tokenizer are the root model-metadata and tokenizer types this
// GRPO training package operates on. Aliased here so the extracted package reads
// against the same types the engine exposes; grpo depends on mlx one-way (the
// root never imports grpo), so there is no import cycle.
type (
	ModelInfo = mlx.ModelInfo
	Tokenizer = mlx.Tokenizer
)

// nonZeroDuration / cloneStringMap are small leaf helpers carried with the
// package on extraction (they were unexported root helpers in training.go /
// helpers.go, not importable across the package boundary).
func nonZeroDuration(duration time.Duration) time.Duration {
	if duration <= 0 {
		return time.Nanosecond
	}
	return duration
}

func cloneStringMap(values map[string]string) map[string]string {
	if len(values) == 0 {
		return nil
	}
	return core.MapClone(values)
}
