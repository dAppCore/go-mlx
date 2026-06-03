// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"

	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/metal"
)

// NativeSplitLocalRuntime is the local Metal-side runtime handle for split
// inference. It validates and retains the materialised slice now; attention
// and logits execution are wired behind the SplitLocalRuntime interface.
type NativeSplitLocalRuntime struct {
	slicePath  string
	cfg        LoadConfig
	inspection ModelSliceInspection
	tokenizer  *metal.Tokenizer
	model      nativeSplitModel
	state      *metal.SplitState
}

type nativeSplitModel interface {
	SplitPrefill(context.Context, string) (*metal.SplitState, error)
	SplitForwardAttention(context.Context, *metal.SplitState, metal.SplitAttentionRequest) (metal.SplitAttentionResult, error)
	SplitSample(context.Context, *metal.SplitState, metal.SplitSampleRequest) (metal.SplitSampleResult, error)
	Close() error
}

var loadNativeSplitModel = func(path string, cfg metal.LoadConfig) (nativeSplitModel, error) {
	return metal.LoadAndInit(path, cfg)
}

// LoadNativeSplitLocalRuntime prepares the local attention/logits runtime for a
// materialised slice. The current implementation keeps construction cheap and
// explicit; actual Metal attention kernels attach through the runtime methods.
func LoadNativeSplitLocalRuntime(ctx context.Context, slicePath string, cfg LoadConfig) (*NativeSplitLocalRuntime, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	// Trim once at construction so the stored slicePath is in the
	// final canonical form. Every downstream readiness check then
	// reduces to a len() against the receiver field instead of a
	// per-call Trim that walked the same string repeatedly.
	slicePath = core.Trim(slicePath)
	if slicePath == "" {
		return nil, core.NewError("mlx: native split local runtime requires a slice path")
	}
	normalised, err := normalizeLoadConfig(cfg)
	if err != nil {
		return nil, err
	}
	inspection, err := InspectModelSlice(slicePath)
	if err != nil {
		return nil, err
	}
	tokenizer, err := metal.LoadTokenizer(core.PathJoin(slicePath, "tokenizer.json"))
	if err != nil {
		return nil, err
	}
	return &NativeSplitLocalRuntime{
		slicePath:  slicePath,
		cfg:        normalised,
		inspection: inspection,
		tokenizer:  tokenizer,
	}, nil
}

// Prefill starts a native split decode session.
func (runtime *NativeSplitLocalRuntime) Prefill(ctx context.Context, req SplitPrefillRequest) (SplitPrefillResult, error) {
	if err := nativeSplitLocalRuntimeReady(ctx, runtime); err != nil {
		return SplitPrefillResult{}, err
	}
	model, err := runtime.nativeModel(ctx)
	if err != nil {
		return SplitPrefillResult{}, err
	}
	state, err := model.SplitPrefill(ctx, req.Prompt)
	if err != nil {
		return SplitPrefillResult{}, err
	}
	if state == nil {
		return SplitPrefillResult{}, errNativeSplitPrefillNilState
	}
	runtime.state = state
	return SplitPrefillResult{
		// Tokens stays as a defensive copy: subsequent Sample calls
		// mutate runtime.state.Tokens in place via
		//   state.Tokens = append(state.Tokens, id)
		// which can grow the existing backing array if capacity
		// allows — aliasing here would let the caller observe new
		// IDs appearing in their slice view.
		Tokens: append([]int32(nil), state.Tokens...),
		// Hidden can alias safely. Sample replaces runtime.state.Hidden
		// with a freshly-allocated slice
		//   state.Hidden = append([]float32(nil), nextHidden...)
		// rather than mutating the existing backing array, so the
		// prefill-time backing stays pinned and unchanged for the life
		// of the caller's slice header. The previous defensive clone
		// duplicated the float32 buffer for no behaviour gain.
		Hidden: state.Hidden,
		Layers: state.Layers,
	}, nil
}

// ForwardAttention runs one native local attention layer.
func (runtime *NativeSplitLocalRuntime) ForwardAttention(ctx context.Context, req SplitAttentionRequest) (SplitAttentionResult, error) {
	if err := nativeSplitLocalRuntimeReady(ctx, runtime); err != nil {
		return SplitAttentionResult{}, err
	}
	model, err := runtime.nativeModel(ctx)
	if err != nil {
		return SplitAttentionResult{}, err
	}
	if runtime.state == nil {
		return SplitAttentionResult{}, errNativeSplitNoPrefillAttn
	}
	// metal.SplitForwardAttention copies the request hidden / shape
	// slices into Metal arrays via FromValues, which performs a
	// binary.Encode into a fresh []byte buffer before handing the
	// pointer to mlx_array_new_data. Neither slice is retained past
	// the call, so the previous append([]T(nil), src...) defensive
	// clones served no contract — aliasing the caller's slice and
	// the receiver's HiddenShape saves two allocations + two N-element
	// copies per layer attention call.
	result, err := model.SplitForwardAttention(ctx, runtime.state, metal.SplitAttentionRequest{
		Layer:       req.Layer,
		Hidden:      req.Hidden,
		HiddenShape: runtime.state.HiddenShape,
	})
	if err != nil {
		return SplitAttentionResult{}, err
	}
	// metal.Model.SplitForwardAttention already allocates a fresh
	// result.Hidden via out.Floats() and stores an independent state
	// copy separately, so the slice handed back to us is exclusively
	// owned. The previous append([]float32(nil), result.Hidden...) was
	// a redundant second clone over the freshly-allocated data —
	// transferring ownership directly saves the per-call copy.
	return SplitAttentionResult{Hidden: result.Hidden}, nil
}

// Sample projects local logits and samples one token.
func (runtime *NativeSplitLocalRuntime) Sample(ctx context.Context, req SplitSampleRequest) (SplitSampleResult, error) {
	if err := nativeSplitLocalRuntimeReady(ctx, runtime); err != nil {
		return SplitSampleResult{}, err
	}
	model, err := runtime.nativeModel(ctx)
	if err != nil {
		return SplitSampleResult{}, err
	}
	if runtime.state == nil {
		return SplitSampleResult{}, errNativeSplitNoPrefillSample
	}
	// metal.SplitSample iterates req.Tokens (for repeat-penalty), then
	// FromValues-copies req.Hidden and req.HiddenShape into Metal byte
	// buffers; no slice is retained past the call. The previous
	// append([]T(nil), src...) defensive clones each pre-allocated a
	// duplicate Go-side buffer of the same data the Metal binding was
	// about to copy anyway — drop them to save three allocations +
	// three N-element copies per sample.
	result, err := model.SplitSample(ctx, runtime.state, metal.SplitSampleRequest{
		Tokens:      req.Tokens,
		Hidden:      req.Hidden,
		HiddenShape: runtime.state.HiddenShape,
		Config:      toMetalGenerateConfig(req.Config),
	})
	if err != nil {
		return SplitSampleResult{}, err
	}
	// metal.Model.SplitSample returns result.Hidden as the freshly
	// allocated embedding slice and stores an independent
	// state.Hidden = append([]float32(nil), nextHidden...) for itself.
	// The slice handed to us has a single owner, so re-cloning it
	// here was redundant — alias the result.Hidden directly.
	return SplitSampleResult{
		TokenID: result.TokenID,
		Hidden:  result.Hidden,
	}, nil
}

// DecodeToken converts a generated token to text.
func (runtime *NativeSplitLocalRuntime) DecodeToken(ctx context.Context, id int32) (string, error) {
	if err := nativeSplitLocalRuntimeReady(ctx, runtime); err != nil {
		return "", err
	}
	if runtime.tokenizer == nil {
		return "", errNativeSplitTokenizerNil
	}
	return runtime.tokenizer.DecodeToken(id), nil
}

// Sentinel errors reused across native split runtime guards. Built once
// at package init so the runtime-readiness check never allocates a new
// error wrapper when a guard fires, and the steady-state ready path has
// no allocations at all.
var (
	errNativeSplitRuntimeNil      = core.NewError("mlx: native split local runtime is nil")
	errNativeSplitRuntimeNoPath   = core.NewError("mlx: native split local runtime has no slice path")
	errNativeSplitPrefillNilState = core.NewError("mlx: native split local runtime prefill returned nil state")
	errNativeSplitNoPrefillAttn   = core.NewError("mlx: native split local runtime requires prefill before attention")
	errNativeSplitNoPrefillSample = core.NewError("mlx: native split local runtime requires prefill before sample")
	errNativeSplitTokenizerNil    = core.NewError("mlx: native split local runtime tokenizer is nil")
)

func nativeSplitLocalRuntimeReady(ctx context.Context, runtime *NativeSplitLocalRuntime) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if runtime == nil {
		return errNativeSplitRuntimeNil
	}
	// LoadNativeSplitLocalRuntime already trimmed the slice path and
	// rejected an empty value before the runtime exists. Re-running
	// core.Trim on every call (Prefill/ForwardAttention/Sample/Decode
	// each go through this helper) walked the slice path string for a
	// guarantee the constructor had already proven; cheaper to assert
	// non-empty via len() on the stored, already-trimmed value.
	if len(runtime.slicePath) == 0 {
		return errNativeSplitRuntimeNoPath
	}
	return nil
}

func (runtime *NativeSplitLocalRuntime) nativeModel(ctx context.Context) (nativeSplitModel, error) {
	// Every public method (Prefill / ForwardAttention / Sample /
	// DecodeToken) already gated on nativeSplitLocalRuntimeReady before
	// calling nativeModel — re-running ctx.Err + nil + path checks here
	// repeated the same ctx-channel cas + receiver deref on every call.
	// Fast-path the cached model and skip the duplicate readiness work.
	if runtime.model != nil {
		return runtime.model, nil
	}
	if err := nativeSplitLocalRuntimeReady(ctx, runtime); err != nil {
		return nil, err
	}
	model, err := loadNativeSplitModel(runtime.slicePath, toMetalSplitLoadConfig(runtime.cfg))
	if err != nil {
		return nil, err
	}
	runtime.model = model
	return model, nil
}

func toMetalSplitLoadConfig(cfg LoadConfig) metal.LoadConfig {
	return metal.LoadConfig{
		ContextLen:           cfg.ContextLength,
		ParallelSlots:        cfg.ParallelSlots,
		DisablePromptCache:   !cfg.PromptCache,
		PromptCacheMinTokens: cfg.PromptCacheMinTokens,
		AdapterPath:          cfg.AdapterPath,
		Device:               metal.DeviceType(cfg.Device),
		CachePolicy:          string(cfg.CachePolicy),
		KVCacheMode:          string(cfg.CacheMode),
		BatchSize:            cfg.BatchSize,
		PrefillChunkSize:     cfg.PrefillChunkSize,
		ExpectedQuantization: cfg.ExpectedQuantization,
		MemoryLimitBytes:     cfg.MemoryLimitBytes,
		CacheLimitBytes:      cfg.CacheLimitBytes,
		WiredLimitBytes:      cfg.WiredLimitBytes,
	}
}
