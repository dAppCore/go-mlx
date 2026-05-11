// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && !nomlx

package mlx

import (
	"context"

	"dappco.re/go/inference"
	"dappco.re/go/inference/scheduler"
)

func (adapter *metaladapter) Schedule(ctx context.Context, req inference.ScheduledRequest) (inference.RequestHandle, <-chan inference.ScheduledToken, error) {
	return adapter.schedulerModel().Schedule(ctx, req)
}

func (adapter *metaladapter) CancelRequest(ctx context.Context, id string) (inference.RequestCancelResult, error) {
	return adapter.schedulerModel().CancelRequest(ctx, id)
}

func (adapter *metaladapter) schedulerModel() *scheduler.Model {
	if adapter == nil {
		return scheduler.New(nil, scheduler.Config{})
	}
	adapter.schedulerMu.Lock()
	defer adapter.schedulerMu.Unlock()
	if adapter.scheduler == nil {
		maxConcurrent := adapter.schedulerMaxConcurrent
		if maxConcurrent <= 0 {
			maxConcurrent = DefaultLocalParallelSlots
		}
		adapter.scheduler = scheduler.New(adapter, scheduler.Config{
			MaxConcurrent:   maxConcurrent,
			MaxQueue:        maxConcurrent * 4,
			StreamBuffer:    0,
			RequestIDPrefix: "mlx-metal",
			ProbeSink:       adapter.probeSink,
		})
	}
	return adapter.scheduler
}
