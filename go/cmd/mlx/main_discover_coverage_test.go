// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"strings"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	mlx "dappco.re/go/mlx"
)

// discover with an unexpected positional argument prints the usage block (the
// fs.Usage closure) and exits 2.
func TestRunDiscover_UnexpectedPositional_Bad(t *testing.T) {
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{"discover", "stray-arg"}, stdout, stderr)
	if code != 2 {
		t.Fatalf("exit = %d, want 2 (unexpected positional)", code)
	}
	out := stderr.String()
	if !strings.Contains(out, "unexpected positional") || !strings.Contains(out, "Usage:") {
		t.Fatalf("stderr = %q, want the positional notice + usage", out)
	}
	// The usage closure also lists the examples block.
	if !strings.Contains(out, "discover -probe-device -json") {
		t.Fatalf("stderr = %q, want the discover usage examples", out)
	}
}

// discover with an unknown -workload errors out at workload validation (exit 2).
func TestRunDiscover_BadWorkload_Bad(t *testing.T) {
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{"discover", "-workload", "not-a-workload"}, stdout, stderr)
	if code != 2 {
		t.Fatalf("exit = %d, want 2 (bad workload)", code)
	}
	if !strings.Contains(stderr.String(), "unsupported workload") {
		t.Fatalf("stderr = %q, want the unsupported-workload error", stderr.String())
	}
}

// discover -h prints usage and exits 0 (flag.ErrHelp short-circuit).
func TestRunDiscover_Help_Good(t *testing.T) {
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{"discover", "-h"}, stdout, stderr)
	if code != 0 {
		t.Fatalf("exit = %d, want 0 for -h", code)
	}
	if !strings.Contains(stderr.String(), "Report what MLX runtime") {
		t.Fatalf("stderr = %q, want the discover help body", stderr.String())
	}
}

// discover surfaces a discovery-runtime error as exit 1 (mocked seam).
func TestRunDiscover_DiscoveryError_Bad(t *testing.T) {
	orig := runDiscoverLocalRuntime
	t.Cleanup(func() { runDiscoverLocalRuntime = orig })
	runDiscoverLocalRuntime = func(_ context.Context, _ mlx.LocalDiscoveryConfig) (inference.MachineDiscoveryReport, error) {
		return inference.MachineDiscoveryReport{}, core.NewError("discover failed")
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{"discover"}, stdout, stderr)
	if code != 1 {
		t.Fatalf("exit = %d, want 1 (discovery error)", code)
	}
	if !strings.Contains(stderr.String(), "discover failed") {
		t.Fatalf("stderr = %q, want the surfaced discovery error", stderr.String())
	}
}

// discover (human-readable, no JSON) over a mocked runtime prints the summary
// lines — the non-JSON output lane.
func TestRunDiscover_HumanSummary_Good(t *testing.T) {
	orig := runDiscoverLocalRuntime
	t.Cleanup(func() { runDiscoverLocalRuntime = orig })
	runDiscoverLocalRuntime = func(_ context.Context, _ mlx.LocalDiscoveryConfig) (inference.MachineDiscoveryReport, error) {
		return inference.MachineDiscoveryReport{
			Runtime:   inference.RuntimeIdentity{Backend: "metal"},
			Available: true,
			Device:    inference.MachineDeviceInfo{Architecture: "apple9", MemorySize: 1 << 30},
		}, nil
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{"discover", "-workload", "chat"}, stdout, stderr)
	if code != 0 {
		t.Fatalf("exit = %d, want 0; stderr=%q", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "runtime discovery: metal") {
		t.Fatalf("stdout = %q, want the human summary", stdout.String())
	}
}
