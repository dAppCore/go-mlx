// SPDX-Licence-Identifier: EUPL-1.2

package agent

import "testing"

// The trusted flag must reach the block options — the continuity lane's
// declaration rides SleepOptions into kv.StateBlockOptions.
func TestSleepBlockOptions_TrustedFlagPlumbs_Good(t *testing.T) {
	blockOpts := SleepBlockOptions(SleepOptions{ReuseParentPrefixTrusted: true}, "mlx://bundle")
	if !blockOpts.ReusePrefixTrusted {
		t.Fatal("ReusePrefixTrusted did not plumb through SleepBlockOptions")
	}
	if SleepBlockOptions(SleepOptions{}, "mlx://bundle").ReusePrefixTrusted {
		t.Fatal("ReusePrefixTrusted set without the SleepOptions declaration")
	}
}
