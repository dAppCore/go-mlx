// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"testing"

	core "dappco.re/go"
)

func TestOfficialPlatform_DefaultAPILocks_Good(t *testing.T) {
	locks := DefaultOfficialPlatformAPILocks()
	if len(locks) < 5 {
		t.Fatalf("DefaultOfficialPlatformAPILocks() = %d locks, want macOS 26 plus Metal 4 source links", len(locks))
	}

	seen := map[string]bool{}
	for _, lock := range locks {
		if lock.MinimumOS != "macOS 26.0" || lock.SourceCheckedAt != "2026-05-31" {
			t.Fatalf("lock provenance = %+v, want macOS 26.0 checked source", lock)
		}
		if lock.IntroducedIn != "macOS 26.0" {
			t.Fatalf("lock IntroducedIn = %q, want macOS 26.0 API-generation provenance", lock.IntroducedIn)
		}
		if lock.SourceURL == "" || lock.Name == "" || lock.APIClass == "" {
			t.Fatalf("lock is incomplete: %+v", lock)
		}
		seen[lock.SourceURL] = true
	}

	for _, want := range []string{
		"https://developer.apple.com/documentation/macos-release-notes/macos-26-release-notes",
		"https://developer.apple.com/macos/whats-new/",
		"https://developer.apple.com/metal/whats-new/",
		"https://developer.apple.com/documentation/metal/understanding-the-metal-4-core-api",
		"https://developer.apple.com/documentation/metal/using-the-metal-4-compilation-api",
		"https://developer.apple.com/documentation/metal/machine-learning-passes",
		"https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf",
	} {
		if !seen[want] {
			t.Fatalf("locks = %+v, want source URL %s", locks, want)
		}
	}
	if !core.Contains(locks[1].Notes, "Metal 4") {
		t.Fatalf("overview notes = %q, want Metal 4 provenance", locks[1].Notes)
	}
}
