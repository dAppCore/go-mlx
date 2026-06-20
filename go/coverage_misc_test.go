// SPDX-Licence-Identifier: EUPL-1.2

// Coverage for assorted reachable guard/error prologues that need no model:
// session constructors' early returns, model-pack inspection delegation,
// FuseLoRAIntoModelPack's ctx + validate guards, and LoadModelAsTextModel's
// config-normalisation error path.

package mlx

import (
	"context"
	"testing"
)

func TestNewSessionFromKV_NoSessionSupport(t *testing.T) {
	// fakeNativeModel is not a nativeModelSessionFactory, so NewSession (and
	// therefore NewSessionFromKV) errors before any restore.
	model := &Model{model: &fakeNativeModel{}}
	if _, err := model.NewSessionFromKV(nil); err == nil {
		t.Fatalf("NewSessionFromKV on a non-session model should error")
	}

	// Nil model.
	if _, err := (*Model)(nil).NewSessionFromKV(nil); err == nil {
		t.Fatalf("NewSessionFromKV(nil model) should error")
	}
}

func TestNewSessionFromBundle_NilBundleGuard(t *testing.T) {
	model := &Model{model: &fakeNativeModel{}}
	if _, err := model.NewSessionFromBundle(nil); err == nil {
		t.Fatalf("NewSessionFromBundle(nil) should error")
	}
}

func TestInspectModelPack_BadPathErrors(t *testing.T) {
	// Pure delegation to modelinspect.Inspect — a missing dir surfaces the
	// inspector error.
	if _, err := InspectModelPack("/nonexistent/model/pack"); err == nil {
		t.Fatalf("InspectModelPack on a missing path should error")
	}
}

func TestFuseLoRAIntoModelPack_Guards(t *testing.T) {
	// Cancelled context errors before any pack work.
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := FuseLoRAIntoModelPack(cancelled, FuseLoRAOptions{ModelPath: "/x"}); err == nil {
		t.Fatalf("cancelled context should error")
	}

	// Bad source path fails inside ValidateModelPack.
	if _, err := FuseLoRAIntoModelPack(context.Background(), FuseLoRAOptions{ModelPath: "/nonexistent/source"}); err == nil {
		t.Fatalf("missing source pack should error")
	}
}

func TestLoadModelAsTextModel_BadConfigErrors(t *testing.T) {
	// A negative context length fails in normalizeLoadConfig, before any
	// native load is attempted.
	if _, err := LoadModelAsTextModel("/any/path", WithContextLength(-1)); err == nil {
		t.Fatalf("LoadModelAsTextModel with invalid config should error")
	}
}
