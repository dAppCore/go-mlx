// SPDX-Licence-Identifier: EUPL-1.2

package loraadapter

import "testing"

// TestNormalizeForNativeLoad_ScaleOnly_Good covers the native-load arm where an
// adapter supplies only scale (no rank, no alpha). NormalizeConfig cannot derive
// alpha from scale without a rank, so the native loader defaults rank to 8 first
// and then back-fills alpha as scale*rank — the `case cfg.Scale != 0` branch.
func TestNormalizeForNativeLoad_ScaleOnly_Good(t *testing.T) {
	cfg := NormalizeForNativeLoad(Config{Scale: 3})
	if cfg.Rank != 8 || cfg.Alpha != 24 || cfg.Scale != 3 {
		t.Fatalf("scale-only native rank/alpha/scale = %d/%f/%f, want 8/24/3", cfg.Rank, cfg.Alpha, cfg.Scale)
	}
}
