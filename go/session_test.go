// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"testing"

	"dappco.re/go/mlx/internal/sessionfake"
	"dappco.re/go/mlx/kv"
)

func TestModelNewSession_Good(t *testing.T) {
	nativeSession := &sessionfake.Handle{}
	model := &Model{model: &fakeNativeModel{session: nativeSession}}

	session, err := model.NewSession()

	if err != nil {
		t.Fatalf("NewSession() error = %v", err)
	}
	if session == nil {
		t.Fatal("NewSession() = nil, want session")
	}
	if !session.Valid() {
		t.Fatal("NewSession() returned an invalid session")
	}
}

func TestModelNewSession_Bad(t *testing.T) {
	var model *Model

	session, err := model.NewSession()

	if err == nil {
		t.Fatal("expected nil model error")
	}
	if session != nil {
		t.Fatalf("session = %v, want nil", session)
	}
}

func TestModelNewSession_Ugly(t *testing.T) {
	model := &Model{model: nativeWithoutPromptCache{}}

	session, err := model.NewSession()

	if err == nil {
		t.Fatal("expected unsupported native session error")
	}
	if session != nil {
		t.Fatalf("session = %v, want nil", session)
	}
}

func TestModelNewSession_ReturnedNilAndBundleErrors_Bad(t *testing.T) {
	model := &Model{model: &fakeNativeModel{}}
	if session, err := model.NewSession(); err == nil || session != nil {
		t.Fatalf("NewSession(nil native session) = %+v/%v, want error", session, err)
	}
	if session, err := model.NewSessionFromBundle(nil); err == nil || session != nil {
		t.Fatalf("NewSessionFromBundle(nil) = %+v/%v, want error", session, err)
	}
}

func TestModelNewSessionFromKV_Good(t *testing.T) {
	nativeSession := &sessionfake.Handle{}
	model := &Model{model: &fakeNativeModel{session: nativeSession}}
	snapshot := &kv.Snapshot{
		Version:      kv.SnapshotVersion,
		Architecture: "gemma4_text",
		Tokens:       []int32{1},
		TokenOffset:  1,
		SeqLen:       1,
		HeadDim:      1,
		LogitShape:   []int32{1, 1, 2},
		Logits:       []float32{0.1, 0.9},
		Layers: []kv.LayerSnapshot{{
			Layer:      0,
			CacheIndex: 0,
			Heads: []kv.HeadSnapshot{{
				Key:   []float32{1},
				Value: []float32{2},
			}},
		}},
	}

	session, err := model.NewSessionFromKV(snapshot)

	if err != nil {
		t.Fatalf("NewSessionFromKV() error = %v", err)
	}
	if !session.Valid() {
		t.Fatalf("NewSessionFromKV() = %#v, want wrapped native session", session)
	}
	if nativeSession.RestoredKV == nil || nativeSession.RestoredKV.Logits[1] != 0.9 {
		t.Fatalf("restored KV = %+v", nativeSession.RestoredKV)
	}
}
