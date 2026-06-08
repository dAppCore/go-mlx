// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"testing"

	core "dappco.re/go"
)

func TestMain(m *testing.M) {
	if !MetalAvailable() {
		core.Print(core.Stderr(), "skipping pkg/metal tests: usable Metal device unavailable")
		core.Exit(0)
	}
	// Bound MLX allocator memory for the whole test/bench binary, mirroring what
	// LoadAndInit does for the production model path. Synthetic benches never
	// call LoadAndInit, so without this the freed-buffer pool (GetCacheMemory)
	// grows unbounded across iterations — a high -benchtime kv-cache bench
	// ballooned past the 96 GiB machine (≈143 GiB) and OOM'd it. SetCacheLimit
	// caps the buffer-reuse pool that ballooned; SetMemoryLimit is a hard ceiling
	// so any runaway (a leak, or a too-large live model bench) fails its
	// allocation gracefully instead of taking the machine down. The biggest real
	// bench (31B q6 ≈ 24 GiB resident) sits well under the ceiling.
	SetMemoryLimit(48 << 30) // 48 GiB hard ceiling (machine has 96 GiB)
	SetCacheLimit(4 << 30)   // 4 GiB allocator-cache pool (LoadAndInit floors at 1 GiB)
	core.Exit(m.Run())
}
