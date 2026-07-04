// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/bundle"
	"dappco.re/go/inference/kv"
	infspine "dappco.re/go/inference/spine"
	session "dappco.re/go/inference/state/session"
	"dappco.re/go/mlx/pkg/metal"
	"dappco.re/go/mlx/spine"
)

// session.go: the root constructors for persistent sessions. The session
// machinery itself (prefill, generation, KV capture/restore, sleep/wake)
// lives in dappco.re/go/inference/state/session (lifted from go-mlx during the
// engine merge); root keeps the Model-side entry points, aliases the type so
// the public API is unchanged, and bridges the engine handle + model
// description onto the engine-neutral session contract.

var (
	errModelNil         = core.NewError("mlx: model is nil")
	errStateBundleNil   = core.NewError("mlx: state bundle is nil")
	errNativeNilSession = core.NewError("mlx: native model returned nil session")
	errNativeNoSessions = core.NewError("mlx: native model does not support sessions")
)

// neutralSessionFactory is the native-lane probe: the no-cgo engine opens an
// engine-neutral inference.SessionHandle directly.
type neutralSessionFactory interface {
	NewSession() inference.SessionHandle
}

// metalSessionFactory is the cgo-lane probe: pkg/metal opens a metal-typed
// handle that the root wraps in metalSessionAdapter to speak the contract.
type metalSessionFactory interface {
	NewSession() metal.SessionHandle
}

// ModelSession is a persistent model-state handle with retained KV cache.
type ModelSession = session.Session

// NewSession creates a persistent session for prefill, generation, KV capture, and forking.
func (m *Model) NewSession() (*ModelSession, error) {
	if m == nil || m.model == nil {
		return nil, errModelNil
	}
	handle, err := m.newSessionHandle()
	if err != nil {
		return nil, err
	}
	return session.New(handle, inferenceSpineModelInfo(m.Info()), inferenceSpineTokenizer(m.Tokenizer())), nil
}

// newSessionHandle opens the engine session and returns it as the neutral
// contract: the native engine already speaks it; the metal engine's handle is
// wrapped in the kvconv-backed adapter.
func (m *Model) newSessionHandle() (inference.SessionHandle, error) {
	switch factory := m.model.(type) {
	case neutralSessionFactory:
		handle := factory.NewSession()
		if handle == nil {
			return nil, errNativeNilSession
		}
		return handle, nil
	case metalSessionFactory:
		handle := factory.NewSession()
		if handle == nil {
			return nil, errNativeNilSession
		}
		return newMetalSessionAdapter(handle), nil
	default:
		return nil, errNativeNoSessions
	}
}

// inferenceSpineModelInfo re-expresses the root (mlx/spine) ModelInfo as the
// lifted package's inference/spine.ModelInfo. The two structs are field-
// identical (mlx/lora.AdapterInfo aliases inference/lora.AdapterInfo), so this
// is a direct struct conversion, not a hand copy.
func inferenceSpineModelInfo(info ModelInfo) infspine.ModelInfo {
	return infspine.ModelInfo(info)
}

// inferenceSpineTokenizer rebuilds the root tokenizer under the lifted
// package's inference/spine.Tokenizer. The wrapped TokenizerImpl is engine-
// neutral (identical method set in both spine packages), so the raw impl
// satisfies inference/spine.TokenizerImpl directly.
func inferenceSpineTokenizer(tok *Tokenizer) *infspine.Tokenizer {
	if tok == nil {
		return nil
	}
	return infspine.NewTokenizer(tok.Impl())
}

// NewSessionFromKV creates a persistent session restored from a KV snapshot.
func (m *Model) NewSessionFromKV(snapshot *kv.Snapshot) (*ModelSession, error) {
	sess, err := m.NewSession()
	if err != nil {
		return nil, err
	}
	if err := sess.RestoreKV(snapshot); err != nil {
		if closeErr := sess.Close(); closeErr != nil {
			return nil, core.ErrorJoin(err, closeErr)
		}
		return nil, err
	}
	return sess, nil
}

// NewSessionFromBundle creates a persistent session restored from a state bundle.
func (m *Model) NewSessionFromBundle(b *bundle.Bundle) (*ModelSession, error) {
	if b == nil {
		return nil, errStateBundleNil
	}
	if err := bundle.CheckCompatibility(spine.ModelInfoToBundle(m.Info()), b); err != nil {
		return nil, err
	}
	snapshot, err := b.Snapshot()
	if err != nil {
		return nil, err
	}
	return m.NewSessionFromKV(snapshot)
}
