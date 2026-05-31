// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"testing"

	core "dappco.re/go"
)

func TestRunCommand_ProductionTurboQuantPolicyJSON_Good(t *testing.T) {
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"production-turboquant", "-json"}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	for _, want := range []string{
		`"kind": "production-turboquant-policy"`,
		`"cache_mode": "turboquant"`,
		`"target_effective_bits_milli": 3500`,
		`"enabled_by_default": false`,
		`"requires_explicit_opt_in": true`,
		`"requires_normal_context_validation": true`,
		`"requires_stress_context_validation": true`,
		`"compare_against_cache_modes": [`,
		`"fp16"`,
		`"paged"`,
		`"q8"`,
		`"k-q8-v-q4"`,
		`"estimated_power_watts"`,
		`"quality_flags"`,
	} {
		if !core.Contains(stdout.String(), want) {
			t.Fatalf("stdout = %q, want %s", stdout.String(), want)
		}
	}
	if stderr.Len() != 0 {
		t.Fatalf("stderr = %q, want empty", stderr.String())
	}
}
