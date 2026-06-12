// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	core "dappco.re/go"
	"dappco.re/go/mlx/bundle"
	"dappco.re/go/mlx/kv"
	"dappco.re/go/mlx/pkg/metal"
	"dappco.re/go/mlx/session"
	"dappco.re/go/mlx/spine"
)

// session.go: the root constructors for persistent sessions. The session
// machinery itself (prefill, generation, KV capture/restore, sleep/wake)
// lives in dappco.re/go/mlx/session; root keeps the Model-side entry
// points and aliases the type so the public API is unchanged.

var (
	errModelNil         = core.NewError("mlx: model is nil")
	errStateBundleNil   = core.NewError("mlx: state bundle is nil")
	errNativeNilSession = core.NewError("mlx: native model returned nil session")
	errNativeNoSessions = core.NewError("mlx: native model does not support sessions")
)

type nativeModelSessionFactory interface {
	NewSession() metal.SessionHandle
}

// ModelSession is a persistent model-state handle with retained KV cache.
type ModelSession = session.Session

// NewSession creates a persistent session for prefill, generation, KV capture, and forking.
func (m *Model) NewSession() (*ModelSession, error) {
	if m == nil || m.model == nil {
		return nil, errModelNil
	}
	factory, ok := m.model.(nativeModelSessionFactory)
	if !ok {
		return nil, errNativeNoSessions
	}
	handle := factory.NewSession()
	if handle == nil {
		return nil, errNativeNilSession
	}
	return session.New(handle, m.Info(), m.Tokenizer()), nil
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
