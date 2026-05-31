// SPDX-Licence-Identifier: EUPL-1.2

package mlx

const officialPlatformSourceCheckedAt = "2026-05-31"

// OfficialPlatformAPILock records the OS/API provenance for native Metal
// features that define the production binary floor.
type OfficialPlatformAPILock struct {
	Name            string `json:"name"`
	MinimumOS       string `json:"minimum_os"`
	SDK             string `json:"sdk"`
	IntroducedIn    string `json:"introduced_in"`
	SourceCheckedAt string `json:"source_checked_at"`
	SourceURL       string `json:"source_url"`
	APIClass        string `json:"api_class"`
	Notes           string `json:"notes"`
}

// DefaultOfficialPlatformAPILocks returns the Apple platform source links that
// justify the macOS 26.0+ production floor. These are documentation locks, not
// runtime feature probes; runtime checks still come from the Metal backend.
func DefaultOfficialPlatformAPILocks() []OfficialPlatformAPILock {
	return []OfficialPlatformAPILock{
		{
			Name:            "macOS Tahoe 26 release notes",
			MinimumOS:       "macOS 26.0",
			SDK:             "macOS 26 SDK",
			IntroducedIn:    "macOS 26.0",
			SourceCheckedAt: officialPlatformSourceCheckedAt,
			SourceURL:       "https://developer.apple.com/documentation/macos-release-notes/macos-26-release-notes",
			APIClass:        "operating-system-floor",
			Notes:           "Apple documents the macOS 26 SDK and Metal 4 support on the Tahoe 26 release; this is the production OS floor for APIs introduced with that release.",
		},
		{
			Name:            "What's new in macOS 26",
			MinimumOS:       "macOS 26.0",
			SDK:             "macOS 26 SDK",
			IntroducedIn:    "macOS 26.0",
			SourceCheckedAt: officialPlatformSourceCheckedAt,
			SourceURL:       "https://developer.apple.com/macos/whats-new/",
			APIClass:        "macos-26-api-generation",
			Notes:           "Apple's macOS 26 overview ties the platform release to the new Metal 4 API generation used by the native runner.",
		},
		{
			Name:            "What's new in Metal",
			MinimumOS:       "macOS 26.0",
			SDK:             "Metal 4",
			IntroducedIn:    "macOS 26.0",
			SourceCheckedAt: officialPlatformSourceCheckedAt,
			SourceURL:       "https://developer.apple.com/metal/whats-new/",
			APIClass:        "metal-4-overview",
			Notes:           "Metal 4 is the source generation for lower-overhead command encoding, explicit compilation, tensors, and machine-learning integration.",
		},
		{
			Name:            "Understanding the Metal 4 core API",
			MinimumOS:       "macOS 26.0",
			SDK:             "Metal 4",
			IntroducedIn:    "macOS 26.0",
			SourceCheckedAt: officialPlatformSourceCheckedAt,
			SourceURL:       "https://developer.apple.com/documentation/metal/understanding-the-metal-4-core-api",
			APIClass:        "metal-core-api",
			Notes:           "The native runner tracks the Metal 4 core API generation for command and resource-management work.",
		},
		{
			Name:            "Using the Metal 4 compilation API",
			MinimumOS:       "macOS 26.0",
			SDK:             "Metal 4",
			IntroducedIn:    "macOS 26.0",
			SourceCheckedAt: officialPlatformSourceCheckedAt,
			SourceURL:       "https://developer.apple.com/documentation/metal/using-the-metal-4-compilation-api",
			APIClass:        "metal-compilation-api",
			Notes:           "The production path records this source because explicit compilation is part of the macOS 26 Metal 4 API generation.",
		},
		{
			Name:            "Metal machine learning passes",
			MinimumOS:       "macOS 26.0",
			SDK:             "Metal 4",
			IntroducedIn:    "macOS 26.0",
			SourceCheckedAt: officialPlatformSourceCheckedAt,
			SourceURL:       "https://developer.apple.com/documentation/metal/machine-learning-passes",
			APIClass:        "metal-machine-learning",
			Notes:           "Metal 4 machine-learning passes and tensor resources are the relevant Apple API family for future native ML integration.",
		},
		{
			Name:            "Metal feature set tables",
			MinimumOS:       "macOS 26.0",
			SDK:             "Metal 4",
			IntroducedIn:    "macOS 26.0",
			SourceCheckedAt: officialPlatformSourceCheckedAt,
			SourceURL:       "https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf",
			APIClass:        "metal-4-feature-availability",
			Notes:           "Apple's feature tables list command allocators, decoupled command queues, dedicated compilation contexts, machine-learning encoding, and tensors as Metal 4 feature-family entries.",
		},
	}
}
