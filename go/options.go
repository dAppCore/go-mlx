// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"reflect"
	"sync"

	"dappco.re/go/inference"
	"dappco.re/go/mlx/internal/metal"
)

// inferenceMinPProbe caches the structural offset of the MinP field on the
// linked inference.GenerateConfig so the forward-compatibility lookup walks
// the struct fields once per program lifetime instead of once per Generate
// call. reflect.Type.FieldByName performs a linear scan with no internal
// cache in Go 1.21-1.26, so amortising it lifts the per-call cost from
// ~25ns to a single atomic load.
var inferenceMinPProbe struct {
	once  sync.Once
	index []int // nil → field absent
	isFP  bool  // true when the field is a float kind we can read
}

// loadInferenceMinPProbe returns the cached MinP field index plus a flag
// reporting whether the field exists and is a float on the linked
// inference.GenerateConfig type. Safe for concurrent callers.
func loadInferenceMinPProbe() ([]int, bool) {
	inferenceMinPProbe.once.Do(func() {
		field, ok := reflect.TypeOf(inference.GenerateConfig{}).FieldByName("MinP")
		if !ok {
			return
		}
		switch field.Type.Kind() {
		case reflect.Float32, reflect.Float64:
			inferenceMinPProbe.index = field.Index
			inferenceMinPProbe.isFP = true
		}
	})
	return inferenceMinPProbe.index, inferenceMinPProbe.isFP
}

func inferenceGenerateConfigToMetal(cfg inference.GenerateConfig) metal.GenerateConfig {
	out := metal.GenerateConfig{
		MaxTokens:     cfg.MaxTokens,
		Temperature:   cfg.Temperature,
		TopK:          cfg.TopK,
		TopP:          cfg.TopP,
		StopTokens:    cfg.StopTokens,
		RepeatPenalty: cfg.RepeatPenalty,
	}
	// Keep go-mlx forward-compatible with inference.GenerateConfig versions
	// that expose MinP without requiring a synchronized dependency update
	// here. The reflect FieldByName scan is amortised through
	// inferenceMinPProbe so we pay it once per process instead of once
	// per Generate/Chat/Classify request.
	if index, isFP := loadInferenceMinPProbe(); isFP {
		out.MinP = float32(reflect.ValueOf(cfg).FieldByIndex(index).Float())
	}
	return out
}
