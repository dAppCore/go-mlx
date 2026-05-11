// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"testing"

	"dappco.re/go/inference"
	prof "dappco.re/go/mlx/profile"
)

func TestAlgorithmProfile_BuiltinStatuses_Good(t *testing.T) {
	coverageTokens := "AlgorithmProfile BuiltinStatuses"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	cases := []struct {
		id      inference.CapabilityID
		runtime prof.AlgorithmRuntimeStatus
		status  inference.CapabilityStatus
	}{
		{id: inference.CapabilityScheduler, runtime: prof.AlgorithmRuntimeNative, status: inference.CapabilityStatusSupported},
		{id: inference.CapabilityCacheBlocks, runtime: prof.AlgorithmRuntimeNative, status: inference.CapabilityStatusSupported},
		{id: inference.CapabilityReasoningParse, runtime: prof.AlgorithmRuntimeNative, status: inference.CapabilityStatusSupported},
		{id: inference.CapabilityJANGTQ, runtime: prof.AlgorithmRuntimeMetadataOnly, status: inference.CapabilityStatusExperimental},
		{id: inference.CapabilityCodebookVQ, runtime: prof.AlgorithmRuntimeExperimental, status: inference.CapabilityStatusExperimental},
		{id: inference.CapabilityEmbeddings, runtime: prof.AlgorithmRuntimeMetadataOnly, status: inference.CapabilityStatusPlanned},
		{id: inference.CapabilityMoERouting, runtime: prof.AlgorithmRuntimeMetadataOnly, status: inference.CapabilityStatusPlanned},
		{id: inference.CapabilityMoELazyExperts, runtime: prof.AlgorithmRuntimeExperimental, status: inference.CapabilityStatusExperimental},
		{id: inference.CapabilitySpeculativeDecode, runtime: prof.AlgorithmRuntimeExperimental, status: inference.CapabilityStatusExperimental},
		{id: inference.CapabilityPromptLookupDecode, runtime: prof.AlgorithmRuntimeExperimental, status: inference.CapabilityStatusExperimental},
	}

	for _, tc := range cases {
		t.Run(string(tc.id), func(t *testing.T) {
			p, ok := prof.LookupAlgorithmProfile(tc.id)
			if !ok {
				t.Fatalf("prof.LookupAlgorithmProfile(%q) ok = false", tc.id)
			}
			if p.RuntimeStatus != tc.runtime || p.CapabilityStatus != tc.status {
				t.Fatalf("profile = %+v, want runtime/status %q/%q", p, tc.runtime, tc.status)
			}
			if p.Group == "" || p.Detail == "" {
				t.Fatalf("profile = %+v, want group and detail", p)
			}
		})
	}
}

func TestAlgorithmProfile_LazyExpertsExperimental_Good(t *testing.T) {
	p, ok := prof.LookupAlgorithmProfile(inference.CapabilityMoELazyExperts)
	if !ok {
		t.Fatal("missing lazy expert profile")
	}
	if p.RuntimeStatus != prof.AlgorithmRuntimeExperimental || p.CapabilityStatus != inference.CapabilityStatusExperimental {
		t.Fatalf("lazy expert status = runtime:%q capability:%q, want experimental", p.RuntimeStatus, p.CapabilityStatus)
	}
	if !containsCapabilityProvide(p.Provides, "expert.page_in") || !containsCapabilityProvide(p.Provides, "expert.residency.probe") {
		t.Fatalf("lazy expert provides = %+v, want page-in and probe labels", p.Provides)
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
	p, ok := prof.LookupAlgorithmProfile(inference.CapabilityPromptLookupDecode)
	if !ok {
		t.Fatal("missing prompt lookup decode profile")
	}

	capability := p.Capability()

	if capability.ID != inference.CapabilityPromptLookupDecode || capability.Status != inference.CapabilityStatusExperimental {
		t.Fatalf("capability = %+v, want experimental prompt lookup decode", capability)
	}
	if capability.Labels["runtime_status"] != string(prof.AlgorithmRuntimeExperimental) || capability.Labels["algorithm"] != "prompt-lookup" {
		t.Fatalf("labels = %+v, want runtime_status and algorithm", capability.Labels)
	}
}

func TestAlgorithmProfile_CapabilityListHasNoDuplicateIDs_Good(t *testing.T) {
	capabilities := prof.AlgorithmCapabilities()
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
	profiles := prof.BuiltinAlgorithmProfiles()
	if len(profiles) == 0 {
		t.Fatal("prof.BuiltinAlgorithmProfiles() returned no profiles")
	}
	profiles[0].Algorithm = "mutated"
	again := prof.BuiltinAlgorithmProfiles()
	if again[0].Algorithm == "mutated" {
		t.Fatal("prof.BuiltinAlgorithmProfiles returned aliased profile data")
	}
	if _, ok := prof.LookupAlgorithmProfile("missing-capability"); ok {
		t.Fatal("prof.LookupAlgorithmProfile(missing) ok = true")
	}
}
