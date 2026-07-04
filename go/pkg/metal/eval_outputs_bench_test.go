// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import "testing"

// evalOutputs per-token allocation bench (AX-11).
//
// Generate's decode loop calls Eval/EvalAsync on every step to materialise the
// freshly-computed logits (and, on the pipelined path, the staged cache state),
// so evalOutputs runs once per decoded token. The real-e2b Pass-2 profile
// flagged it as the #3 per-token engine allocator. The cost is the Go-side
// bookkeeping around the cgo eval call, NOT the GPU work: the handle buffer is
// pooled (evalOutputCtxPool), and the remaining per-call heap traffic is the
// boxed return code plus the eval-worker closure.
//
// This bench isolates that bookkeeping with no model load (AX-11 synthetic): a
// single tiny array is materialised ONCE outside the loop, then Eval'd every
// iteration. Re-evaluating an already-resident array is a GPU no-op but drives
// the full Go path — append into the pooled buffer, onEvalWorker handoff, read
// rc — so allocs/op is the per-token engine bookkeeping the change targets,
// free of any array-creation churn that would swamp it if allocated in-loop.

// BenchmarkEvalOutputs_Sync measures the per-call allocation of the synchronous
// eval path (metal.Eval), the form Generate uses to materialise logits before
// sampling. allocs/op here is the per-token evalOutputs bookkeeping cost.
func BenchmarkEvalOutputs_Sync(b *testing.B) {
	a := FromValues([]float32{1, 2, 3, 4}, 2, 2)
	defer Free(a)
	Materialize(a)

	b.ReportAllocs()
	for b.Loop() {
		if err := Eval(a); err != nil {
			b.Fatalf("Eval: %v", err)
		}
	}
}

// BenchmarkEvalOutputs_Async measures the asynchronous eval path
// (metal.EvalAsync), used on the pipelined decode loop to queue the staged
// cache/logits without blocking. Same bookkeeping, async cgo entry.
func BenchmarkEvalOutputs_Async(b *testing.B) {
	a := FromValues([]float32{1, 2, 3, 4}, 2, 2)
	defer Free(a)
	Materialize(a)

	b.ReportAllocs()
	for b.Loop() {
		if err := EvalAsync(a); err != nil {
			b.Fatalf("EvalAsync: %v", err)
		}
	}
}
