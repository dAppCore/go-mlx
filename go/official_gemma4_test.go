// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import "testing"

func TestOfficialGemma4E2BLocks_Good(t *testing.T) {
	locks := DefaultOfficialGemma4E2BLocks()

	if len(locks) != 2 {
		t.Fatalf("DefaultOfficialGemma4E2BLocks() = %d locks, want target plus assistant", len(locks))
	}

	byRole := map[string]OfficialGemma4E2BLock{}
	for _, lock := range locks {
		byRole[lock.Role] = lock
		if lock.Licence != "apache-2.0" || lock.LicenceURL != "https://ai.google.dev/gemma/docs/gemma_4_license" {
			t.Fatalf("%s licence = %q %q, want Apache-2.0 Gemma 4 licence link", lock.ModelID, lock.Licence, lock.LicenceURL)
		}
		if lock.Gated {
			t.Fatalf("%s Gated = true, want current public/ungated HF snapshot lock", lock.ModelID)
		}
		if lock.AccessNotes == "" {
			t.Fatalf("%s AccessNotes empty, want gating/access evidence", lock.ModelID)
		}
		if lock.ConfigSHA256 == "" || lock.TokenizerSHA256 == "" || lock.TokenizerConfigSHA256 == "" || lock.WeightSHA256 == "" {
			t.Fatalf("%s hashes incomplete: %+v", lock.ModelID, lock)
		}
		if lock.SafetensorsIndexPresent || lock.SafetensorsIndexSHA256 != "" || lock.SafetensorsIndexNotes == "" {
			t.Fatalf("%s safetensors index = present:%v hash:%q notes:%q, want explicit absent index evidence", lock.ModelID, lock.SafetensorsIndexPresent, lock.SafetensorsIndexSHA256, lock.SafetensorsIndexNotes)
		}
	}

	target := byRole[OfficialGemma4E2BRoleTarget]
	if target.ModelID != "google/gemma-4-E2B-it" || target.Revision != "905e84b50c4d2a365ebde34e685027578e6728db" {
		t.Fatalf("target identity = %+v", target)
	}
	if target.Architecture != "Gemma4ForConditionalGeneration" || target.ModelType != "gemma4" || target.ChatTemplateSHA256 == "" {
		t.Fatalf("target model contract = %+v, want Gemma4 conditional generation with chat template hash", target)
	}
	if target.ConfigSHA256 != "1b28f3d2c3100f6c594754b81107428bd7b822a7f48272ca681dae9d2ec38330" ||
		target.TokenizerSHA256 != "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f" ||
		target.TokenizerConfigSHA256 != "90c3a3ba5bf53818383a58e1a776cbcacd2a038d4812eaa373e1522f2d06f3df" ||
		target.ChatTemplateSHA256 != "2f1b4d75d067bae3fe44e676721c7f077d243bc007156cb9c2f8b5836613d082" ||
		target.WeightSHA256 != "2db5482b20d746879bb3ef79b5203e9075a2e2b98f54ec7c2f281c1477ddc550" {
		t.Fatalf("target hashes = %+v", target)
	}

	assistant := byRole[OfficialGemma4E2BRoleAssistant]
	if assistant.ModelID != "google/gemma-4-E2B-it-assistant" || assistant.Revision != "5810c41a67974da9c7bd6f3e6c69d5d13854d9f0" {
		t.Fatalf("assistant identity = %+v", assistant)
	}
	if assistant.Architecture != "Gemma4AssistantForCausalLM" || assistant.ModelType != "gemma4_assistant" || assistant.ChatTemplateSHA256 != "" {
		t.Fatalf("assistant model contract = %+v, want Gemma4 assistant causal LM without standalone chat template", assistant)
	}
	if assistant.ConfigSHA256 != "7f42f559a6a69ffaeaf6b61a1ece3a562a2ed5ad00b8d30f16917ba5ab1bcbe9" ||
		assistant.TokenizerSHA256 != "75a6583c1a418e2bbd79c60d95d28e0f5bf549ad3f2990b5bdb5238c6c2bf70c" ||
		assistant.TokenizerConfigSHA256 != "089594a3924fcfd4cb1c596a7906fbf476193519e5198f780912eed02b177e42" ||
		assistant.WeightSHA256 != "93682eb1c97639d18f007704dc880bd74cbe530adaf7b1bb561213863fdad2a6" {
		t.Fatalf("assistant hashes = %+v", assistant)
	}
}

func TestOfficialGemma4E2BLocks_ByRoleAndModelID_Good(t *testing.T) {
	target, ok := OfficialGemma4E2BLockByRole(OfficialGemma4E2BRoleTarget)
	if !ok {
		t.Fatal("OfficialGemma4E2BLockByRole(target) = false, want official target lock")
	}
	if target != OfficialGemma4E2BTargetLock() {
		t.Fatalf("OfficialGemma4E2BTargetLock() = %+v, want role lookup target", OfficialGemma4E2BTargetLock())
	}
	if target.ModelID != DefaultProductionQuantizationPolicy().TargetModelID {
		t.Fatalf("target ModelID = %q, want production policy target %q", target.ModelID, DefaultProductionQuantizationPolicy().TargetModelID)
	}

	assistant, ok := OfficialGemma4E2BLockByModelID("google/gemma-4-E2B-it-assistant")
	if !ok {
		t.Fatal("OfficialGemma4E2BLockByModelID(assistant) = false, want official assistant lock")
	}
	if assistant != OfficialGemma4E2BAssistantLock() {
		t.Fatalf("OfficialGemma4E2BAssistantLock() = %+v, want model lookup assistant", OfficialGemma4E2BAssistantLock())
	}
	if assistant.ModelType != "gemma4_assistant" || assistant.Role != OfficialGemma4E2BRoleAssistant {
		t.Fatalf("assistant lock = %+v, want assistant role/model type", assistant)
	}

	if _, ok := OfficialGemma4E2BLockByRole("draft"); ok {
		t.Fatal("OfficialGemma4E2BLockByRole(draft) = true, want false for non-official role")
	}
	if _, ok := OfficialGemma4E2BLockByModelID("mlx-community/gemma-4-e2b-it-6bit"); ok {
		t.Fatal("OfficialGemma4E2BLockByModelID(mlx-community q6) = true, want false for derived quant pack")
	}
}

func BenchmarkOfficialGemma4E2BLockByRole_Target(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		lock, ok := OfficialGemma4E2BLockByRole(OfficialGemma4E2BRoleTarget)
		if !ok || lock.ModelID != "google/gemma-4-E2B-it" {
			b.Fatalf("OfficialGemma4E2BLockByRole(target) = %+v %v", lock, ok)
		}
	}
}
