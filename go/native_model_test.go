// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import "testing"

// The NativeModel seam (public interface + NewModel constructor + Native
// accessor) is the floor that lets subpackages build on the root Model without
// reaching its unexported field. It must round-trip the engine and be nil-safe.
func TestNativeModel_Seam_Good(t *testing.T) {
	engine := &fakeNativeModel{}
	m := NewModel(engine)
	if m.Native() != engine {
		t.Fatal("NewModel(engine).Native() did not return the same engine")
	}

	var nilModel *Model
	if nilModel.Native() != nil {
		t.Fatal("(*Model)(nil).Native() = non-nil, want nil")
	}
}
