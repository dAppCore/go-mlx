// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"dappco.re/go/inference"
	"dappco.re/go/inference/scheduler"
)

// Legacy aliases — the canonical scheduler lives at
// dappco.re/go/inference/scheduler/. mlx-root callers keep their
// existing Scheduled* surface via these aliases.
type (
	ScheduledModel  = scheduler.Model
	SchedulerConfig = scheduler.Config
)

// NewScheduledModel returns a scheduler wrapper for model. Nil models
// are accepted so callers can construct package surfaces before a
// backend loads.
//
//	model := mlx.NewScheduledModel(backend, mlx.SchedulerConfig{MaxConcurrent: 4})
func NewScheduledModel(model inference.TextModel, cfg SchedulerConfig) *ScheduledModel {
	return scheduler.New(model, cfg)
}
