// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"testing"

	"dappco.re/go/inference"
)

func TestAlgorithmProfile_BuiltinStatuses_Good(t *testing.T) {
	coverageTokens := "AlgorithmProfile BuiltinStatuses"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	cases := []struct {
		id      inference.CapabilityID
		runtime AlgorithmRuntimeStatus
		status  inference.CapabilityStatus
	}{
		{id: inference.CapabilityScheduler, runtime: AlgorithmRuntimeNative, status: inference.CapabilityStatusSupported},
		{id: inference.CapabilityCacheBlocks, runtime: AlgorithmRuntimeNative, status: inference.CapabilityStatusSupported},
		{id: inference.CapabilityReasoningParse, runtime: AlgorithmRuntimeNative, status: inference.CapabilityStatusSupported},
		{id: inference.CapabilityJANGTQ, runtime: AlgorithmRuntimeMetadataOnly, status: inference.CapabilityStatusExperimental},
		{id: inference.CapabilityCodebookVQ, runtime: AlgorithmRuntimeExperimental, status: inference.CapabilityStatusExperimental},
		{id: inference.CapabilityEmbeddings, runtime: AlgorithmRuntimeMetadataOnly, status: inference.CapabilityStatusPlanned},
		{id: inference.CapabilityMoERouting, runtime: AlgorithmRuntimeMetadataOnly, status: inference.CapabilityStatusPlanned},
		{id: inference.CapabilityMoELazyExperts, runtime: AlgorithmRuntimeExperimental, status: inference.CapabilityStatusExperimental},
		{id: inference.CapabilitySpeculativeDecode, runtime: AlgorithmRuntimeExperimental, status: inference.CapabilityStatusExperimental},
		{id: inference.CapabilityPromptLookupDecode, runtime: AlgorithmRuntimeExperimental, status: inference.CapabilityStatusExperimental},
	}

	for _, tc := range cases {
		t.Run(string(tc.id), func(t *testing.T) {
			profile, ok := LookupAlgorithmProfile(tc.id)
			if !ok {
				t.Fatalf("LookupAlgorithmProfile(%q) ok = false", tc.id)
			}
			if profile.RuntimeStatus != tc.runtime || profile.CapabilityStatus != tc.status {
				t.Fatalf("profile = %+v, want runtime/status %q/%q", profile, tc.runtime, tc.status)
			}
			if profile.Group == "" || profile.Detail == "" {
				t.Fatalf("profile = %+v, want group and detail", profile)
			}
		})
	}
}

func TestAlgorithmProfile_LazyExpertsExperimental_Good(t *testing.T) {
	profile, ok := LookupAlgorithmProfile(inference.CapabilityMoELazyExperts)
	if !ok {
		t.Fatal("missing lazy expert profile")
	}
	if profile.RuntimeStatus != AlgorithmRuntimeExperimental || profile.CapabilityStatus != inference.CapabilityStatusExperimental {
		t.Fatalf("lazy expert status = runtime:%q capability:%q, want experimental", profile.RuntimeStatus, profile.CapabilityStatus)
	}
	if !containsCapabilityProvide(profile.Provides, "expert.page_in") || !containsCapabilityProvide(profile.Provides, "expert.residency.probe") {
		t.Fatalf("lazy expert provides = %+v, want page-in and probe labels", profile.Provides)
	}
}

func containsCapabilityProvide(values []string, want string) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}

func TestAlgorithmProfile_CapabilityLabels_Good(t *testing.T) {
	profile, ok := LookupAlgorithmProfile(inference.CapabilityPromptLookupDecode)
	if !ok {
		t.Fatal("missing prompt lookup decode profile")
	}

	capability := profile.Capability()

	if capability.ID != inference.CapabilityPromptLookupDecode || capability.Status != inference.CapabilityStatusExperimental {
		t.Fatalf("capability = %+v, want experimental prompt lookup decode", capability)
	}
	if capability.Labels["runtime_status"] != string(AlgorithmRuntimeExperimental) || capability.Labels["algorithm"] != "prompt-lookup" {
		t.Fatalf("labels = %+v, want runtime_status and algorithm", capability.Labels)
	}
}

func TestAlgorithmProfile_CapabilityListHasNoDuplicateIDs_Good(t *testing.T) {
	capabilities := algorithmProfileCapabilities()
	seen := map[inference.CapabilityID]bool{}
	for _, capability := range capabilities {
		if seen[capability.ID] {
			t.Fatalf("duplicate algorithm capability %q", capability.ID)
		}
		seen[capability.ID] = true
		if capability.Labels["runtime_status"] == "" {
			t.Fatalf("capability = %+v, want runtime_status label", capability)
		}
	}
	for _, id := range []inference.CapabilityID{
		inference.CapabilitySpeculativeDecode,
		inference.CapabilityPromptLookupDecode,
		inference.CapabilityEmbeddings,
		inference.CapabilityRerank,
		inference.CapabilityMoERouting,
		inference.CapabilityMoELazyExperts,
		inference.CapabilityCodebookVQ,
	} {
		if !seen[id] {
			t.Fatalf("missing algorithm capability %q", id)
		}
	}
}

func TestAlgorithmProfile_BuiltinProfilesAreCloned_Bad(t *testing.T) {
	profiles := BuiltinAlgorithmProfiles()
	if len(profiles) == 0 {
		t.Fatal("BuiltinAlgorithmProfiles() returned no profiles")
	}
	profiles[0].Algorithm = "mutated"
	again := BuiltinAlgorithmProfiles()
	if again[0].Algorithm == "mutated" {
		t.Fatal("BuiltinAlgorithmProfiles returned aliased profile data")
	}
	if _, ok := LookupAlgorithmProfile("missing-capability"); ok {
		t.Fatal("LookupAlgorithmProfile(missing) ok = true")
	}
}
