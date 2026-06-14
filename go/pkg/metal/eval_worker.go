// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

/*
#include <pthread.h>
static unsigned long long go_mlx_thread_id(void) {
    return (unsigned long long)pthread_self();
}
*/
import "C"

import (
	"runtime"
	"sync"
	"sync/atomic"
)

// Eval worker — one dedicated, OS-locked encoding thread.
//
// MLX 0.31.2 encodes GPU graphs on the calling thread with per-thread
// command encoders, and ASYNC evals deliberately leave encoder state
// uncommitted for op batching. A Go goroutine that migrates OS threads
// between async evals therefore splits one logical op stream across two
// encoders whose commit order is scheduler-dependent — occasionally
// reordering GPU work against the same cache buffers (observed as a rare
// late greedy fork). v0.31.1 had one device-global encoder, which made
// calling-thread identity irrelevant.
//
// Routing every eval-class entry through a single LockOSThread worker
// restores those ordering semantics exactly: one thread, one encoder map,
// one commit order. The channel hop costs ~100ns against eval costs in
// the hundreds of microseconds.
var (
	evalWorkerOnce sync.Once
	evalWorkerJobs chan *evalJob
	evalWorkerTID  atomic.Uint64
)

// evalJob carries one unit of work to the encoding thread. It is pooled
// (evalJobPool) so the per-eval handoff — taken on every Eval, the decode hot
// path — allocates neither a fresh channel nor a closure. done is a buffered(1)
// channel signalled by a SEND (not close) so the same channel survives reuse;
// panicked carries a recovered panic back to the caller's goroutine.
type evalJob struct {
	fn       func()
	done     chan struct{}
	panicked any
}

var evalJobPool = sync.Pool{New: func() any { return &evalJob{done: make(chan struct{}, 1)} }}

func startEvalWorker() {
	evalWorkerJobs = make(chan *evalJob, 64)
	ready := make(chan struct{})
	go func() {
		runtime.LockOSThread()
		evalWorkerTID.Store(uint64(C.go_mlx_thread_id()))
		seenGen := ensureThreadStreams()
		close(ready)
		for job := range evalWorkerJobs {
			// Streams can be created AFTER the worker is born — the one-shot
			// CLI path loads weights (first evals start the worker) before
			// the engine builds its decode streams. Replay registrations
			// whenever the registry generation moves; otherwise this is a
			// single atomic load per job.
			if gen := threadStreamRegistryGen.Load(); gen != seenGen {
				seenGen = ensureThreadStreams()
			}
			runEvalJob(job)
		}
	}()
	<-ready
}

// runEvalJob runs one job on the worker thread, capturing a panic so the worker
// survives (the compiled-layer poison paths rely on panics propagating to the
// caller), and signalling completion with a send so done stays reusable.
func runEvalJob(job *evalJob) {
	defer func() {
		job.panicked = recover()
		job.done <- struct{}{}
	}()
	job.fn()
}

// onEvalWorker runs fn on the dedicated encoding thread and waits for it.
// Calls already executing on the worker run fn directly — a job enqueueing
// another job would deadlock waiting on its own completion. The job and its
// done channel are pooled, so the handoff is allocation-free on the hot path. A
// panic inside fn is captured and re-raised on the caller's goroutine so the
// worker survives (the compiled-layer poison paths rely on panics propagating).
func onEvalWorker(fn func()) {
	evalWorkerOnce.Do(startEvalWorker)
	if evalWorkerTID.Load() == uint64(C.go_mlx_thread_id()) {
		fn()
		return
	}
	job := evalJobPool.Get().(*evalJob)
	job.fn = fn
	job.panicked = nil
	evalWorkerJobs <- job
	<-job.done
	panicked := job.panicked
	job.fn = nil
	job.panicked = nil
	evalJobPool.Put(job)
	if panicked != nil {
		panic(panicked)
	}
}
