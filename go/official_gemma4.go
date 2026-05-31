// SPDX-Licence-Identifier: EUPL-1.2

package mlx

const (
	// OfficialGemma4E2BRoleTarget identifies the model that produces final
	// user-visible tokens in the official Google E2B lane.
	OfficialGemma4E2BRoleTarget = "target"
	// OfficialGemma4E2BRoleAssistant identifies the MTP drafter paired with
	// the target model.
	OfficialGemma4E2BRoleAssistant = "assistant"

	officialGemma4E2BSourceCheckedAt = "2026-05-31"
	officialGemma4E2BLicenceURL      = "https://ai.google.dev/gemma/docs/gemma_4_license"
)

// OfficialGemma4E2BLock pins the exact Hugging Face snapshot identity used by
// the official Google E2B target+assistant production lane. Hashes are SHA-256
// for downloaded artefacts; BlobID keeps the HF git blob identity when useful
// for API cross-checking, but SHA-256 remains the runtime verification value.
type OfficialGemma4E2BLock struct {
	Role                    string `json:"role"`
	ModelID                 string `json:"model_id"`
	Revision                string `json:"revision"`
	LastModified            string `json:"last_modified"`
	SourceCheckedAt         string `json:"source_checked_at"`
	SourceURL               string `json:"source_url"`
	Licence                 string `json:"licence"`
	LicenceURL              string `json:"licence_url"`
	Gated                   bool   `json:"gated"`
	AccessNotes             string `json:"access_notes"`
	Architecture            string `json:"architecture"`
	ModelType               string `json:"model_type"`
	ConfigBlobID            string `json:"config_blob_id,omitempty"`
	ConfigSHA256            string `json:"config_sha256"`
	TokenizerBlobID         string `json:"tokenizer_blob_id,omitempty"`
	TokenizerSHA256         string `json:"tokenizer_sha256"`
	TokenizerConfigBlobID   string `json:"tokenizer_config_blob_id,omitempty"`
	TokenizerConfigSHA256   string `json:"tokenizer_config_sha256"`
	GenerationConfigBlobID  string `json:"generation_config_blob_id,omitempty"`
	GenerationConfigSHA256  string `json:"generation_config_sha256,omitempty"`
	ChatTemplateBlobID      string `json:"chat_template_blob_id,omitempty"`
	ChatTemplateSHA256      string `json:"chat_template_sha256,omitempty"`
	WeightFile              string `json:"weight_file"`
	WeightBlobID            string `json:"weight_blob_id,omitempty"`
	WeightSHA256            string `json:"weight_sha256"`
	WeightBytes             uint64 `json:"weight_bytes"`
	SafetensorsIndexPresent bool   `json:"safetensors_index_present"`
	SafetensorsIndexSHA256  string `json:"safetensors_index_sha256,omitempty"`
	SafetensorsIndexNotes   string `json:"safetensors_index_notes"`
}

// DefaultOfficialGemma4E2BLocks returns the official Google target and MTP
// assistant snapshot locks. These are metadata locks, not model-load proof; the
// native-load gate must still verify the runtime contracts against these exact
// revisions before replacing the archived q4 baseline.
func DefaultOfficialGemma4E2BLocks() []OfficialGemma4E2BLock {
	return []OfficialGemma4E2BLock{
		{
			Role:            OfficialGemma4E2BRoleTarget,
			ModelID:         "google/gemma-4-E2B-it",
			Revision:        "905e84b50c4d2a365ebde34e685027578e6728db",
			LastModified:    "2026-05-18T16:24:52.000Z",
			SourceCheckedAt: officialGemma4E2BSourceCheckedAt,
			SourceURL:       "https://huggingface.co/google/gemma-4-E2B-it",
			Licence:         "apache-2.0",
			LicenceURL:      officialGemma4E2BLicenceURL,
			Gated:           false,
			AccessNotes:     "HF API reported private=false and gated=false on 2026-05-31; metadata and listed artefacts were readable without an auth token.",
			Architecture:    "Gemma4ForConditionalGeneration",
			ModelType:       "gemma4",

			ConfigBlobID:           "923b5e9405e7d319572b0c1b1a89291512262aa3",
			ConfigSHA256:           "1b28f3d2c3100f6c594754b81107428bd7b822a7f48272ca681dae9d2ec38330",
			TokenizerBlobID:        "1ff9f3e3439a939b971f9919e821bf87e835a503",
			TokenizerSHA256:        "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f",
			TokenizerConfigBlobID:  "375b25dc8be85705251e41be1c25310d24932051",
			TokenizerConfigSHA256:  "90c3a3ba5bf53818383a58e1a776cbcacd2a038d4812eaa373e1522f2d06f3df",
			GenerationConfigBlobID: "e605bb4523b1462ea9d9a3810b9e3ecf7ab7b1f6",
			GenerationConfigSHA256: "d4226bbe3117d2d253ba4609720ba82c6c4ce4627a9a6ae05387c78983ac03de",
			ChatTemplateBlobID:     "c19999a347da729cf62806a8ddb7eb8e315223b5",
			ChatTemplateSHA256:     "2f1b4d75d067bae3fe44e676721c7f077d243bc007156cb9c2f8b5836613d082",

			WeightFile:              "model.safetensors",
			WeightBlobID:            "f293405c7515215112c31a164f4cb738040cc69d",
			WeightSHA256:            "2db5482b20d746879bb3ef79b5203e9075a2e2b98f54ec7c2f281c1477ddc550",
			WeightBytes:             10246621918,
			SafetensorsIndexPresent: false,
			SafetensorsIndexNotes:   "HF snapshot lists a single model.safetensors file and no model.safetensors.index.json.",
		},
		{
			Role:            OfficialGemma4E2BRoleAssistant,
			ModelID:         "google/gemma-4-E2B-it-assistant",
			Revision:        "5810c41a67974da9c7bd6f3e6c69d5d13854d9f0",
			LastModified:    "2026-05-11T07:51:55.000Z",
			SourceCheckedAt: officialGemma4E2BSourceCheckedAt,
			SourceURL:       "https://huggingface.co/google/gemma-4-E2B-it-assistant",
			Licence:         "apache-2.0",
			LicenceURL:      officialGemma4E2BLicenceURL,
			Gated:           false,
			AccessNotes:     "HF API reported private=false and gated=false on 2026-05-31; metadata and listed artefacts were readable without an auth token.",
			Architecture:    "Gemma4AssistantForCausalLM",
			ModelType:       "gemma4_assistant",

			ConfigBlobID:           "b4c30e888c89b39c8f106b5015307fb7830f0bb2",
			ConfigSHA256:           "7f42f559a6a69ffaeaf6b61a1ece3a562a2ed5ad00b8d30f16917ba5ab1bcbe9",
			TokenizerBlobID:        "24aa4244652e010036db5fdd29ed39b9428e6e19",
			TokenizerSHA256:        "75a6583c1a418e2bbd79c60d95d28e0f5bf549ad3f2990b5bdb5238c6c2bf70c",
			TokenizerConfigBlobID:  "1a6bee041ca75778c514a071efbdb568b0f3d7b0",
			TokenizerConfigSHA256:  "089594a3924fcfd4cb1c596a7906fbf476193519e5198f780912eed02b177e42",
			GenerationConfigBlobID: "c699930448995c777880df16f5ceb94e477a4acf",
			GenerationConfigSHA256: "8e58004dc0e2407b63410b190bb8470efbdcfeb71533f1770e09c20abe193a6f",

			WeightFile:              "model.safetensors",
			WeightBlobID:            "9649e2286efcda6fae0387b8aeec33f11d0de960",
			WeightSHA256:            "93682eb1c97639d18f007704dc880bd74cbe530adaf7b1bb561213863fdad2a6",
			WeightBytes:             157565344,
			SafetensorsIndexPresent: false,
			SafetensorsIndexNotes:   "HF snapshot lists a single model.safetensors file and no model.safetensors.index.json.",
		},
	}
}
