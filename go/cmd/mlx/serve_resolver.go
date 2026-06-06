// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"sync"
	"sync/atomic"

	core "dappco.re/go"
	"dappco.re/go/inference"
	openaicompat "dappco.re/go/inference/openai"
	mlx "dappco.re/go/mlx"
)

// loadedModel is the snapshot the hotSwapResolver hands back to
// callers. modelPath stamps which weights are in use so
// /v1/admin/serve/status + reload audit lines can name the source.
type loadedModel struct {
	model     inference.TextModel
	modelPath string
}

// errNoModelLoaded is returned by ResolveModel when the driver started
// model-less (serve with no --model) and nothing has been loaded via
// /v1/admin/serve/reload yet. The openai mux surfaces it to inference
// callers; admin + health endpoints stay reachable so a model can be
// loaded.
var errNoModelLoaded = core.NewError("no model loaded — POST /v1/admin/serve/reload to load a model")

// hotSwapResolver is the openaicompat.Resolver that backs
// /v1/admin/serve/reload (F-7). The active model lives in an
// atomic.Pointer so ResolveModel reads are lock-free on the hot
// path (every chat/completions call hits this); Replace serialises
// swaps under swapMu so two concurrent reloads can't race.
//
// First-call lazy load: the boot-time model isn't loaded eagerly —
// the first ResolveModel triggers the load via initial.Do. That keeps
// `serve --model X` from blocking on a multi-GB load before binding
// the listener, matching the pre-F-7 closure behaviour.
//
// Drain policy (audited per §4.F-7.5): in-flight Generate/Chat calls
// keep their TextModel reference and complete on old weights. New
// calls hit new weights. Old model is NOT explicitly Closed — Go GC
// reclaims when the last in-flight reference drops. Operator running
// many reloads should restart serve to reclaim GPU memory
// deterministically.
//
//	r := newHotSwapResolver(modelPath, opts)
//	openaiMux := openai.NewMuxWithAdmin(r, adminCfg)
//	// later, on /v1/admin/serve/reload:
//	old, err := r.Replace(newPath, newOpts)
type hotSwapResolver struct {
	active        atomic.Pointer[loadedModel]
	initial       sync.Once
	initErr       error
	initPath      string
	initDraftPath string
	initOpts      []mlx.LoadOption
	swapMu        sync.Mutex
}

// newHotSwapResolver returns a resolver staged with the initial model
// path + options. The model is NOT loaded until first ResolveModel
// call.
func newHotSwapResolver(modelPath, draftPath string, opts []mlx.LoadOption) *hotSwapResolver {
	return &hotSwapResolver{
		initPath:      modelPath,
		initDraftPath: draftPath,
		initOpts:      opts,
	}
}

// ResolveModel returns the active model. First call loads the initial
// model; subsequent calls return whatever's currently active
// (possibly swapped via Replace). modelName is the OpenAI-API
// `model` field from the request, ignored — lthn-mlx serves one
// model at a time.
func (r *hotSwapResolver) ResolveModel(_ context.Context, _ string) (inference.TextModel, error) {
	// Already-active model wins — covers both the lazy-loaded boot model
	// and one swapped in via Replace (/v1/admin/serve/reload). Lock-free
	// hot path: every chat/completions call lands here. Checked first so a
	// reload-loaded model is never shadowed by a stale boot-load initErr.
	if cur := r.active.Load(); cur != nil {
		return cur.model, nil
	}
	// Model-less start: no boot model was staged. Inference is unavailable
	// until a model is loaded via Replace; admin + health stay reachable.
	if r.initPath == "" {
		return nil, errNoModelLoaded
	}
	// First call with a staged boot model: load it now. Lazy so
	// `serve --model X` binds the listener before paying the multi-GB
	// load; initial.Do guarantees exactly one load attempt.
	r.initial.Do(func() {
		var m inference.TextModel
		var err error
		if r.initDraftPath != "" {
			// Native Gemma-4 MTP speculative lane: target + assistant drafter.
			m, err = mlx.LoadSpeculativePairAsTextModel(r.initPath, r.initDraftPath, r.initOpts...)
		} else {
			m, err = mlx.LoadModelAsTextModel(r.initPath, r.initOpts...)
		}
		if err != nil {
			r.initErr = err
			return
		}
		r.active.Store(&loadedModel{model: m, modelPath: r.initPath})
	})
	if r.initErr != nil {
		return nil, r.initErr
	}
	if cur := r.active.Load(); cur != nil {
		return cur.model, nil
	}
	return nil, r.initErr
}

// Replace loads a new model at newPath with newOpts and atomically
// swaps it in. Returns the previously-active loadedModel (caller may
// inspect modelPath for audit logging; do NOT Close it — see drain
// policy above) plus the new active path. swapMu serialises swaps so
// two concurrent reloads can't race.
//
//	prev, newPath, err := r.Replace(modelPath, opts)
//	if err != nil { return err }
//	core.Print(stderr, "reload %s → %s", prev.modelPath, newPath)
//
// The auto-tuned boot options (initOpts — CacheMode, BatchSize,
// PromptCache, allocator limits, etc. from the tuning profile) are
// preserved across reload (Mantis #1785 F-7 N-7): newOpts is overlaid
// on top of initOpts so a reload that only carries ContextLength +
// AdapterPath keeps every tuned field rather than reloading the model
// with bare defaults. LoadOption application is last-wins, so the
// overlay correctly overrides any base field it sets.
func (r *hotSwapResolver) Replace(newPath string, newOpts []mlx.LoadOption) (prev *loadedModel, newActive string, err error) {
	r.swapMu.Lock()
	defer r.swapMu.Unlock()
	loaded, err := mlx.LoadModelAsTextModel(newPath, r.reloadLoadOpts(newOpts)...)
	if err != nil {
		return nil, "", err
	}
	next := &loadedModel{model: loaded, modelPath: newPath}
	prev = r.active.Swap(next)
	return prev, newPath, nil
}

// reloadLoadOpts overlays the per-reload options on top of the auto-tuned
// boot options (Mantis #1785 F-7 N-7). LoadOption application is last-wins,
// so initOpts establishes the tuned baseline (CacheMode, BatchSize,
// PromptCache, allocator limits, …) and newOpts overrides only the fields
// the reload explicitly carries.
//
//	merged := r.reloadLoadOpts([]mlx.LoadOption{mlx.WithContextLength(8192)})
func (r *hotSwapResolver) reloadLoadOpts(newOpts []mlx.LoadOption) []mlx.LoadOption {
	merged := make([]mlx.LoadOption, 0, len(r.initOpts)+len(newOpts))
	merged = append(merged, r.initOpts...)
	merged = append(merged, newOpts...)
	return merged
}

// CurrentPath returns the modelPath of the active model, or the
// initial path if no load has happened yet. Used by handlers that
// need to render the active source (e.g. /v1/admin/serve/status).
func (r *hotSwapResolver) CurrentPath() string {
	if cur := r.active.Load(); cur != nil {
		return cur.modelPath
	}
	return r.initPath
}

// openaiResolver returns r as an openaicompat.Resolver. Useful at
// wire-up sites that want to keep the interface narrow without
// exposing the hot-swap surface.
func (r *hotSwapResolver) openaiResolver() openaicompat.Resolver {
	return openaicompat.ResolverFunc(r.ResolveModel)
}
