// SPDX-Licence-Identifier: EUPL-1.2

package mlx

// ProductionQuantizationFileLock pins one file inside a quantized MLX target
// pack. BlobID records the Hugging Face cache/git blob identity; SHA256 is the
// content hash used for local verification.
type ProductionQuantizationFileLock struct {
	Name   string `json:"name"`
	BlobID string `json:"blob_id,omitempty"`
	SHA256 string `json:"sha256"`
	Bytes  uint64 `json:"bytes,omitempty"`
}

// ProductionQuantizationPackLock records the q8/q6/q4 MLX-community
// derivatives that sit beside the official Google E2B source locks. These are
// not a promotion signal; they make the app quantisation ladder auditable.
type ProductionQuantizationPackLock struct {
	Name              string `json:"name"`
	ModelID           string `json:"model_id"`
	Revision          string `json:"revision"`
	SourceCheckedAt   string `json:"source_checked_at"`
	SourceURL         string `json:"source_url"`
	BaseModelID       string `json:"base_model_id"`
	BaseRevision      string `json:"base_revision"`
	ConversionTool    string `json:"conversion_tool"`
	ConversionCommand string `json:"conversion_command"`
	AccuracySmoke     string `json:"accuracy_smoke"`
	Licence           string `json:"licence"`
	LicenceURL        string `json:"licence_url"`

	QuantBits  int    `json:"quant_bits"`
	QuantGroup int    `json:"quant_group"`
	QuantMode  string `json:"quant_mode"`

	ReadmeBlobID            string                           `json:"readme_blob_id,omitempty"`
	ReadmeSHA256            string                           `json:"readme_sha256"`
	ConfigBlobID            string                           `json:"config_blob_id,omitempty"`
	ConfigSHA256            string                           `json:"config_sha256"`
	ProcessorConfigBlobID   string                           `json:"processor_config_blob_id,omitempty"`
	ProcessorConfigSHA256   string                           `json:"processor_config_sha256"`
	TokenizerBlobID         string                           `json:"tokenizer_blob_id,omitempty"`
	TokenizerSHA256         string                           `json:"tokenizer_sha256"`
	TokenizerConfigBlobID   string                           `json:"tokenizer_config_blob_id,omitempty"`
	TokenizerConfigSHA256   string                           `json:"tokenizer_config_sha256"`
	GenerationConfigBlobID  string                           `json:"generation_config_blob_id,omitempty"`
	GenerationConfigSHA256  string                           `json:"generation_config_sha256"`
	ChatTemplateBlobID      string                           `json:"chat_template_blob_id,omitempty"`
	ChatTemplateSHA256      string                           `json:"chat_template_sha256"`
	SafetensorsIndexPresent bool                             `json:"safetensors_index_present"`
	SafetensorsIndexBlobID  string                           `json:"safetensors_index_blob_id,omitempty"`
	SafetensorsIndexSHA256  string                           `json:"safetensors_index_sha256"`
	SafetensorsIndexBytes   uint64                           `json:"safetensors_index_bytes,omitempty"`
	WeightFiles             []ProductionQuantizationFileLock `json:"weight_files"`
}

// DefaultProductionQuantizationPackLocks returns the exact local q8/q6/q4
// derivatives that back the app-facing Gemma 4 E2B quantisation ladder.
func DefaultProductionQuantizationPackLocks() []ProductionQuantizationPackLock {
	return []ProductionQuantizationPackLock{
		{
			Name:              "quality",
			ModelID:           "mlx-community/gemma-4-e2b-it-8bit",
			Revision:          "48ef0737faea4e72556670e49da0ba421027a545",
			SourceCheckedAt:   officialGemma4E2BSourceCheckedAt,
			SourceURL:         "https://huggingface.co/mlx-community/gemma-4-e2b-it-8bit",
			BaseModelID:       OfficialGemma4E2BTargetLock().ModelID,
			BaseRevision:      OfficialGemma4E2BTargetLock().Revision,
			ConversionTool:    "mlx-vlm 0.4.3",
			ConversionCommand: "mlx_vlm.convert --hf-path google/gemma-4-E2B-it --mlx-path mlx-community/gemma-4-e2b-it-8bit --q-bits 8 --q-group-size 64",
			AccuracySmoke:     "metadata lock only; official target native-load, retained-state, and long-output quality gates remain pending",
			Licence:           "apache-2.0",
			LicenceURL:        officialGemma4E2BLicenceURL,
			QuantBits:         ProductionLaneQualityQuantBits,
			QuantGroup:        64,
			QuantMode:         "affine",

			ReadmeBlobID:           "bcc32ab6721f82fbe0a9fdd078f4a91dfa1c68ab",
			ReadmeSHA256:           "306177431807e9ff28450b718b022ce411c422f34d44e8d64461901b99beb13d",
			ConfigBlobID:           "5bc9d70ecfeaa8da4d0ad174d088bb96e86d24f9",
			ConfigSHA256:           "5cdd5627ab3ecf52086cc79b2c14c45a277d273069f1d73bf17a3a5136afe3db",
			ProcessorConfigBlobID:  "13e92a44d19566f334d7450e7898935e16e16f3d",
			ProcessorConfigSHA256:  "1bd0d00776284f369c1eff5fb631e865dfcdca861e0b7d60dbef27fcf37436a8",
			TokenizerBlobID:        "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f",
			TokenizerSHA256:        "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f",
			TokenizerConfigBlobID:  "375b25dc8be85705251e41be1c25310d24932051",
			TokenizerConfigSHA256:  "90c3a3ba5bf53818383a58e1a776cbcacd2a038d4812eaa373e1522f2d06f3df",
			GenerationConfigBlobID: "e605bb4523b1462ea9d9a3810b9e3ecf7ab7b1f6",
			GenerationConfigSHA256: "d4226bbe3117d2d253ba4609720ba82c6c4ce4627a9a6ae05387c78983ac03de",
			ChatTemplateBlobID:     "c19999a347da729cf62806a8ddb7eb8e315223b5",
			ChatTemplateSHA256:     "2f1b4d75d067bae3fe44e676721c7f077d243bc007156cb9c2f8b5836613d082",

			SafetensorsIndexPresent: true,
			SafetensorsIndexBlobID:  "d95167d34932a42ea08c502c0a8dec0060f7c15e",
			SafetensorsIndexSHA256:  "cba1620cfe01e35a14cbebddcc32415d55292529795565d1d11e9cb9cf669f50",
			SafetensorsIndexBytes:   270064,
			WeightFiles: []ProductionQuantizationFileLock{
				{
					Name:   "model-00001-of-00002.safetensors",
					BlobID: "fe889fb027f0b79758af4a7da6a27c6c7bc715680bbdd5af9797bd8355d86820",
					SHA256: "fe889fb027f0b79758af4a7da6a27c6c7bc715680bbdd5af9797bd8355d86820",
					Bytes:  5367135201,
				},
				{
					Name:   "model-00002-of-00002.safetensors",
					BlobID: "83bb2a3420d473d416ffcb3cf9c93bacce064981fb22ea20cb6111a178d2679b",
					SHA256: "83bb2a3420d473d416ffcb3cf9c93bacce064981fb22ea20cb6111a178d2679b",
					Bytes:  532432577,
				},
			},
		},
		{
			Name:              "default",
			ModelID:           ProductionLaneModelID,
			Revision:          "40d43b05f94ee798c0e40fe19fcd9ef49928486b",
			SourceCheckedAt:   officialGemma4E2BSourceCheckedAt,
			SourceURL:         "https://huggingface.co/mlx-community/gemma-4-e2b-it-6bit",
			BaseModelID:       OfficialGemma4E2BTargetLock().ModelID,
			BaseRevision:      OfficialGemma4E2BTargetLock().Revision,
			ConversionTool:    "mlx-vlm 0.4.3",
			ConversionCommand: "mlx_vlm.convert --hf-path google/gemma-4-E2B-it --mlx-path mlx-community/gemma-4-e2b-it-6bit --q-bits 6 --q-group-size 64",
			AccuracySmoke:     "metadata lock only; official target native-load, retained-state, and long-output quality gates remain pending",
			Licence:           "apache-2.0",
			LicenceURL:        officialGemma4E2BLicenceURL,
			QuantBits:         ProductionLaneProductDefaultQuantBits,
			QuantGroup:        64,
			QuantMode:         "affine",

			ReadmeBlobID:           "3f9b6be9d37f54da4e4e4b22d932c3a567da4244",
			ReadmeSHA256:           "9293f5a79db1e170557902c0a7b87d309a8f70c28be42f3a298ee6f2ce006ca4",
			ConfigBlobID:           "541def7346234957712da69bcf118b8ab82fb4e1",
			ConfigSHA256:           "32e50a33a18172e79c86b7a78aff7e79c7544031199d672a2a65e526a8bf0199",
			ProcessorConfigBlobID:  "13e92a44d19566f334d7450e7898935e16e16f3d",
			ProcessorConfigSHA256:  "1bd0d00776284f369c1eff5fb631e865dfcdca861e0b7d60dbef27fcf37436a8",
			TokenizerBlobID:        "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f",
			TokenizerSHA256:        "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f",
			TokenizerConfigBlobID:  "375b25dc8be85705251e41be1c25310d24932051",
			TokenizerConfigSHA256:  "90c3a3ba5bf53818383a58e1a776cbcacd2a038d4812eaa373e1522f2d06f3df",
			GenerationConfigBlobID: "e605bb4523b1462ea9d9a3810b9e3ecf7ab7b1f6",
			GenerationConfigSHA256: "d4226bbe3117d2d253ba4609720ba82c6c4ce4627a9a6ae05387c78983ac03de",
			ChatTemplateBlobID:     "c19999a347da729cf62806a8ddb7eb8e315223b5",
			ChatTemplateSHA256:     "2f1b4d75d067bae3fe44e676721c7f077d243bc007156cb9c2f8b5836613d082",

			SafetensorsIndexPresent: true,
			SafetensorsIndexBlobID:  "26a5c56f5fa221a4ffa87179a8607f70410d75ac",
			SafetensorsIndexSHA256:  "7e6bdf16f05a9d296179d9fe93ae18b52177e84a6e78d46f126e2fa6f6b02414",
			SafetensorsIndexBytes:   230329,
			WeightFiles: []ProductionQuantizationFileLock{
				{
					Name:   "model.safetensors",
					BlobID: "1ce6f5c8d5daf306e71824cfc752020b70fc9262ff201a577d18d62cc446d5bc",
					SHA256: "1ce6f5c8d5daf306e71824cfc752020b70fc9262ff201a577d18d62cc446d5bc",
					Bytes:  4740335854,
				},
			},
		},
		{
			Name:              "constrained",
			ModelID:           ProductionLaneArchivedBaselineModelID,
			Revision:          "99d9a53ff828d365a8ecae538e45f80a08d612cd",
			SourceCheckedAt:   officialGemma4E2BSourceCheckedAt,
			SourceURL:         "https://huggingface.co/mlx-community/gemma-4-e2b-it-4bit",
			BaseModelID:       OfficialGemma4E2BTargetLock().ModelID,
			BaseRevision:      OfficialGemma4E2BTargetLock().Revision,
			ConversionTool:    "mlx-vlm 0.4.3",
			ConversionCommand: "mlx_vlm.convert --hf-path google/gemma-4-E2B-it --mlx-path mlx-community/gemma-4-e2b-it-4bit --q-bits 4 --q-group-size 64",
			AccuracySmoke:     "archived q4 control; historical retained-state benchmark baseline accepted before official q6/q8 promotion",
			Licence:           "apache-2.0",
			LicenceURL:        officialGemma4E2BLicenceURL,
			QuantBits:         ProductionLaneConstrainedQuantBits,
			QuantGroup:        64,
			QuantMode:         "affine",

			ReadmeBlobID:           "b30b13e8d835165e92b1de220c7e371398278266",
			ReadmeSHA256:           "0d0e79f7c5427656411c4ce41fb2a69889bd4f5011ef1885a3b8af9cf6ce8167",
			ConfigBlobID:           "e4f9de994fcdf7a8c104e4f5aafa0d137474837c",
			ConfigSHA256:           "6d12c87861fff3871d3a745011b0d852be6513f3ce594ae1e8d643dae9d3b9a8",
			ProcessorConfigBlobID:  "13e92a44d19566f334d7450e7898935e16e16f3d",
			ProcessorConfigSHA256:  "1bd0d00776284f369c1eff5fb631e865dfcdca861e0b7d60dbef27fcf37436a8",
			TokenizerBlobID:        "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f",
			TokenizerSHA256:        "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f",
			TokenizerConfigBlobID:  "375b25dc8be85705251e41be1c25310d24932051",
			TokenizerConfigSHA256:  "90c3a3ba5bf53818383a58e1a776cbcacd2a038d4812eaa373e1522f2d06f3df",
			GenerationConfigBlobID: "e605bb4523b1462ea9d9a3810b9e3ecf7ab7b1f6",
			GenerationConfigSHA256: "d4226bbe3117d2d253ba4609720ba82c6c4ce4627a9a6ae05387c78983ac03de",
			ChatTemplateBlobID:     "07e50e69a8c445f2c31a089b828e85b2a93942bf",
			ChatTemplateSHA256:     "781d10940fbc44be40064b5d43a056fc486c84ceaa55538226368b57314132bf",

			SafetensorsIndexPresent: true,
			SafetensorsIndexBlobID:  "cbba8cce606b3549efd993cdc055372bcc9cb42d",
			SafetensorsIndexSHA256:  "a8aa7359c747a0d59368dbff9a1029da86bda139ccc0ae1f1e938db75de7d5ce",
			SafetensorsIndexBytes:   230329,
			WeightFiles: []ProductionQuantizationFileLock{
				{
					Name:   "model.safetensors",
					BlobID: "e9bea0584546fafb5ff83a1132a6c4662a8498cc6a5bcda52fc6ca562b7bafab",
					SHA256: "e9bea0584546fafb5ff83a1132a6c4662a8498cc6a5bcda52fc6ca562b7bafab",
					Bytes:  3581101896,
				},
			},
		},
	}
}
