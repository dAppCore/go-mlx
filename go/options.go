// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"reflect"

	"dappco.re/go/inference"
	"dappco.re/go/mlx/internal/metal"
)

// inferenceMinPFieldIndex / inferenceMinPFieldPresent cache the structural
// offset of the MinP field on the linked inference.GenerateConfig so the
// forward-compatibility lookup walks the struct fields once at package
// init rather than once per Generate / Chat / Classify call.
//
// reflect.Type.FieldByName performs a linear scan with no internal cache
// in Go 1.21-1.26. Resolving the probe in init() instead of the prior
// sync.Once-guarded helper drops the per-call cost from "atomic load +
// function call + branch + return tuple" to a single package-var read on
// the hot path — when MinP is absent (the current shape of
// inference.GenerateConfig), the predicate short-circuits before any
// reflect.ValueOf work runs at all.
var (
	inferenceMinPFieldIndex   []int
	inferenceMinPFieldPresent bool
)

func init() {
	field, ok := reflect.TypeOf(inference.GenerateConfig{}).FieldByName("MinP")
	if !ok {
		return
	}
	switch field.Type.Kind() {
	case reflect.Float32, reflect.Float64:
		inferenceMinPFieldIndex = field.Index
		inferenceMinPFieldPresent = true
	}
}

func inferenceGenerateConfigToMetal(cfg inference.GenerateConfig) metal.GenerateConfig {
	out := metal.GenerateConfig{
		MaxTokens:      cfg.MaxTokens,
		Temperature:    cfg.Temperature,
		TopK:           cfg.TopK,
		TopP:           cfg.TopP,
		StopTokens:     cfg.StopTokens,
		RepeatPenalty:  cfg.RepeatPenalty,
		EnableThinking: cfg.EnableThinking,
	}
	// Keep go-mlx forward-compatible with inference.GenerateConfig versions
	// that expose MinP without requiring a synchronized dependency update
	// here. The reflect FieldByName scan is amortised through the package-
	// init probe so we pay it once per process and the per-call cost is a
	// single bool load on the absent-field hot path.
	if inferenceMinPFieldPresent {
		out.MinP = float32(reflect.ValueOf(cfg).FieldByIndex(inferenceMinPFieldIndex).Float())
	}
	return out
}
