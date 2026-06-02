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

// ProductionQuantizationPackLock records MLX-community Gemma 4 E2B derivatives
// that sit beside the official Google E2B source locks. These are not a
// promotion signal; they make the app quantisation ladder and bench/R&D pack
// matrix auditable.
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

// DefaultProductionQuantizationPackLocks returns the exact local MLX-community
// derivatives that back the app-facing Gemma 4 E2B quantisation ladder and
// seven-format bench matrix.
func DefaultProductionQuantizationPackLocks() []ProductionQuantizationPackLock {
	return []ProductionQuantizationPackLock{
		{
			Name:              "research-mxfp4",
			ModelID:           "mlx-community/gemma-4-e2b-it-mxfp4",
			Revision:          "6505f8b409be66c5a6d767e21b7d2bed277fcaa4",
			SourceCheckedAt:   officialGemma4E2BSourceCheckedAt,
			SourceURL:         "https://huggingface.co/mlx-community/gemma-4-e2b-it-mxfp4",
			BaseModelID:       OfficialGemma4E2BTargetLock().ModelID,
			BaseRevision:      OfficialGemma4E2BTargetLock().Revision,
			ConversionTool:    "mlx-vlm 0.4.3",
			ConversionCommand: "mlx_vlm.convert --hf-path google/gemma-4-E2B-it --mlx-path mlx-community/gemma-4-e2b-it-mxfp4 (MXFP4; exact upstream conversion flags not recorded)",
			AccuracySmoke:     "bench/R&D lock only; MXFP4 remains a research pack until retained-workflow quality and memory evidence promote it",
			Licence:           "apache-2.0",
			LicenceURL:        officialGemma4E2BLicenceURL,
			QuantBits:         4,
			QuantGroup:        32,
			QuantMode:         "mxfp4",

			ReadmeBlobID:           "c5b8a3aae52a8a1848b25f1a9b0644f8ea4f8e09",
			ReadmeSHA256:           "a77b4db96f0e1067216103be91d53b544c7e96bae001736226a2a15fa851be82",
			ConfigBlobID:           "d706dfb12b81ea5d844d3cc0a7000a3b51496dd9",
			ConfigSHA256:           "614e876b4efcaff13ce4c7a3f96a5b9de86325e3d2ab9c622606ced688f1b8b7",
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
			SafetensorsIndexBlobID:  "4172298f4f32c8988cf4e7b99d2545b0723d3e8c",
			SafetensorsIndexSHA256:  "682ab3c507de77072844c5dff4fbb35dfa46fec9fc4b6f3ae014b3f42e78d51b",
			SafetensorsIndexBytes:   211538,
			WeightFiles: []ProductionQuantizationFileLock{
				{
					Name:   "model.safetensors",
					BlobID: "d9209536088aa473de0f28bc5d590a15f2af845d59b32e38bbb0a45e8750889c",
					SHA256: "d9209536088aa473de0f28bc5d590a15f2af845d59b32e38bbb0a45e8750889c",
					Bytes:  4263396466,
				},
			},
		},
		{
			Name:              "research-mxfp8",
			ModelID:           "mlx-community/gemma-4-e2b-it-mxfp8",
			Revision:          "58034520e7459bf1e5be508e46906aa943683ee4",
			SourceCheckedAt:   officialGemma4E2BSourceCheckedAt,
			SourceURL:         "https://huggingface.co/mlx-community/gemma-4-e2b-it-mxfp8",
			BaseModelID:       OfficialGemma4E2BTargetLock().ModelID,
			BaseRevision:      OfficialGemma4E2BTargetLock().Revision,
			ConversionTool:    "mlx-vlm 0.4.3",
			ConversionCommand: "mlx_vlm.convert --hf-path google/gemma-4-E2B-it --mlx-path mlx-community/gemma-4-e2b-it-mxfp8 (MXFP8; exact upstream conversion flags not recorded)",
			AccuracySmoke:     "bench/R&D lock only; MXFP8 remains a research pack until retained-workflow quality and memory evidence promote it",
			Licence:           "apache-2.0",
			LicenceURL:        officialGemma4E2BLicenceURL,
			QuantBits:         8,
			QuantGroup:        32,
			QuantMode:         "mxfp8",

			ReadmeBlobID:           "074b4d6efb3958c64b8ffd9c23aa4acc3f51f35f",
			ReadmeSHA256:           "e26522311415e53896517e66fe70be411012327cc5275e48067170119dc07756",
			ConfigBlobID:           "3f3831386be423acaf28914c9e2303d127f3cd94",
			ConfigSHA256:           "d6be5b24cbc974d492804737716ade8d2575eb849ec90a1d316bb64e99838104",
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
			SafetensorsIndexBlobID:  "5783959ebbd9f1cfe9351051f1aa3d41cc5705f3",
			SafetensorsIndexSHA256:  "3dd5efc67da447bc266f6f9e727450b54377cb8563181a947ff727dbf9d1eae1",
			SafetensorsIndexBytes:   237768,
			WeightFiles: []ProductionQuantizationFileLock{
				{
					Name:   "model-00001-of-00002.safetensors",
					BlobID: "d6e4ec568ad5301f74e46772b745aeeffedf4f4cc3f87e2eeeab5e0cba812592",
					SHA256: "d6e4ec568ad5301f74e46772b745aeeffedf4f4cc3f87e2eeeab5e0cba812592",
					Bytes:  5367071866,
				},
				{
					Name:   "model-00002-of-00002.safetensors",
					BlobID: "56ab229f33c37fc325c6c07cad8bbf87e3306ead53b90f36ebf34a1353530629",
					SHA256: "56ab229f33c37fc325c6c07cad8bbf87e3306ead53b90f36ebf34a1353530629",
					Bytes:  387549560,
				},
			},
		},
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
			Name:              "bench-5bit",
			ModelID:           "mlx-community/gemma-4-e2b-it-5bit",
			Revision:          "9604b4538ef64c05790d1d94305487ca6fcb17ba",
			SourceCheckedAt:   officialGemma4E2BSourceCheckedAt,
			SourceURL:         "https://huggingface.co/mlx-community/gemma-4-e2b-it-5bit",
			BaseModelID:       OfficialGemma4E2BTargetLock().ModelID,
			BaseRevision:      OfficialGemma4E2BTargetLock().Revision,
			ConversionTool:    "mlx-vlm 0.4.3",
			ConversionCommand: "mlx_vlm.convert --hf-path google/gemma-4-E2B-it --mlx-path mlx-community/gemma-4-e2b-it-5bit --q-bits 5 --q-group-size 64",
			AccuracySmoke:     "bench lock only; q5 is measured in the seven-format matrix but has no app-facing product role",
			Licence:           "apache-2.0",
			LicenceURL:        officialGemma4E2BLicenceURL,
			QuantBits:         5,
			QuantGroup:        64,
			QuantMode:         "affine",

			ReadmeBlobID:           "590f3f1f64c43861746401919b5ee85d043f49a5",
			ReadmeSHA256:           "5e3a8c155ca21b0b8235e980472304e743cb9c7b0370cfcd4047a262f63a93c2",
			ConfigBlobID:           "dcb66abab2c470965053425254601806641fe5f7",
			ConfigSHA256:           "7bf8329ef9605396b93bf9fee4c590a8320cf5eae3f569763507e434b16a1a26",
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
			SafetensorsIndexBlobID:  "cc6e99079f57df24fa933b8445f73bf3925fc62f",
			SafetensorsIndexSHA256:  "dee9f3492acd7d43330f4ca7a9541a6bdab6bec21c8f1f9eca37fb7a8a2c0010",
			SafetensorsIndexBytes:   230329,
			WeightFiles: []ProductionQuantizationFileLock{
				{
					Name:   "model.safetensors",
					BlobID: "9dd8a7988bc2c8a693dc00f1a742c11d255634ed4259b29a5394126db7b7ab11",
					SHA256: "9dd8a7988bc2c8a693dc00f1a742c11d255634ed4259b29a5394126db7b7ab11",
					Bytes:  4160719027,
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
		{
			Name:              "quality-control-bf16",
			ModelID:           "mlx-community/gemma-4-e2b-it-bf16",
			Revision:          "22a2753af6114b0c364f09921771b458e40b9e09",
			SourceCheckedAt:   officialGemma4E2BSourceCheckedAt,
			SourceURL:         "https://huggingface.co/mlx-community/gemma-4-e2b-it-bf16",
			BaseModelID:       OfficialGemma4E2BTargetLock().ModelID,
			BaseRevision:      OfficialGemma4E2BTargetLock().Revision,
			ConversionTool:    "mlx-vlm 0.4.3",
			ConversionCommand: "mlx_vlm.convert --hf-path google/gemma-4-E2B-it --mlx-path mlx-community/gemma-4-e2b-it-bf16",
			AccuracySmoke:     "quality-control lock only; BF16 is the unquantised comparison target and requires native validation before promotion",
			Licence:           "apache-2.0",
			LicenceURL:        officialGemma4E2BLicenceURL,
			QuantBits:         16,
			QuantGroup:        0,
			QuantMode:         "bf16",

			ReadmeBlobID:           "26b776a67cb07bbe6a6bf732d721c940aef5a90c",
			ReadmeSHA256:           "157c751ee86bfe06c986860228d6500d2719a36d8696d43e166279eed67a6c50",
			ConfigBlobID:           "2955d57831a441b2eab07ce1575f622015e69df1",
			ConfigSHA256:           "29b810ed760b55104943a3cc3b6f8b9ca079e6e00b09585d85aec54863a42fb4",
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
			SafetensorsIndexBlobID:  "350bb838190a6563cb42bb7781cead17894c3a6b",
			SafetensorsIndexSHA256:  "3c147c85c7d2d964452007af9056a78c0ca916dffc06fec1e7c218f28b30bd4f",
			SafetensorsIndexBytes:   205473,
			WeightFiles: []ProductionQuantizationFileLock{
				{
					Name:   "model-00001-of-00003.safetensors",
					BlobID: "ff4c28c7f1b0a841697cdd10fc7b45d434c2edeb6e02360e8a56ed88fa7b1cef",
					SHA256: "ff4c28c7f1b0a841697cdd10fc7b45d434c2edeb6e02360e8a56ed88fa7b1cef",
					Bytes:  4569831590,
				},
				{
					Name:   "model-00002-of-00003.safetensors",
					BlobID: "b2d44b0ee3454db90d6d10b4006b0270be0729094809570c9b366f3a35ca7655",
					SHA256: "b2d44b0ee3454db90d6d10b4006b0270be0729094809570c9b366f3a35ca7655",
					Bytes:  5366705230,
				},
				{
					Name:   "model-00003-of-00003.safetensors",
					BlobID: "2fb5cbee871ebe7dcfaebef771c3013dd6cee51d9c8e0023d5d7c32cb0e9e244",
					SHA256: "2fb5cbee871ebe7dcfaebef771c3013dd6cee51d9c8e0023d5d7c32cb0e9e244",
					Bytes:  310074804,
				},
			},
		},
	}
}
