// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"encoding/binary"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/mlx/safetensors"
)

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

func TestOfficialGemma4E2BSourceLockArtifact_MatchesRuntimeLocks_Good(t *testing.T) {
	var artifact struct {
		Version              int                              `json:"version"`
		Kind                 string                           `json:"kind"`
		SourceCheckedAt      string                           `json:"source_checked_at"`
		ArchivedBaseline     string                           `json:"archived_baseline"`
		DefaultTargetBits    int                              `json:"default_target_bits"`
		QualityTargetBits    int                              `json:"quality_target_bits"`
		FallbackTargetBits   int                              `json:"fallback_target_bits"`
		OfficialLanePromoted bool                             `json:"official_lane_promoted"`
		Locks                []OfficialGemma4E2BLock          `json:"locks"`
		QuantizedTargetLocks []ProductionQuantizationPackLock `json:"quantized_target_locks"`
		PlatformAPILocks     []OfficialPlatformAPILock        `json:"platform_api_locks"`
	}
	read := core.ReadFile(core.PathJoin("..", "docs", "runtime", "2026-05-31-official-gemma4-e2b-source-lock.json"))
	if !read.OK {
		t.Fatalf("ReadFile(source-lock artifact): %v", read.Value)
	}
	if result := core.JSONUnmarshal(read.Value.([]byte), &artifact); !result.OK {
		t.Fatalf("JSONUnmarshal(source-lock artifact): %v", result.Value)
	}
	if artifact.Version != 1 || artifact.Kind != "official-gemma4-e2b-source-lock" {
		t.Fatalf("artifact identity = version:%d kind:%q, want v1 official Gemma 4 E2B source lock", artifact.Version, artifact.Kind)
	}
	if artifact.SourceCheckedAt != officialGemma4E2BSourceCheckedAt {
		t.Fatalf("artifact SourceCheckedAt = %q, want %q", artifact.SourceCheckedAt, officialGemma4E2BSourceCheckedAt)
	}
	if artifact.ArchivedBaseline != ProductionLaneArchivedBaselineModelID || artifact.DefaultTargetBits != 6 || artifact.QualityTargetBits != 8 || artifact.FallbackTargetBits != 4 {
		t.Fatalf("artifact policy = baseline:%q q%d/q%d/q%d, want archived q4 baseline plus q8/q6/q4 ladder", artifact.ArchivedBaseline, artifact.QualityTargetBits, artifact.DefaultTargetBits, artifact.FallbackTargetBits)
	}
	if artifact.OfficialLanePromoted {
		t.Fatal("artifact OfficialLanePromoted = true, want false until native-load, retained-state, and MTP benchmark gates pass")
	}

	expected := DefaultOfficialGemma4E2BLocks()
	if len(artifact.Locks) != len(expected) {
		t.Fatalf("artifact locks = %d, want %d", len(artifact.Locks), len(expected))
	}
	byRole := make(map[string]OfficialGemma4E2BLock, len(artifact.Locks))
	for _, lock := range artifact.Locks {
		byRole[lock.Role] = lock
	}
	for _, want := range expected {
		got, ok := byRole[want.Role]
		if !ok {
			t.Fatalf("artifact missing role %q", want.Role)
		}
		if got != want {
			t.Fatalf("artifact lock[%s] = %+v, want %+v", want.Role, got, want)
		}
	}

	expectedQuantLocks := DefaultProductionQuantizationPackLocks()
	if len(artifact.QuantizedTargetLocks) != len(expectedQuantLocks) {
		t.Fatalf("artifact quantized locks = %d, want %d q8/q6/q4 locks", len(artifact.QuantizedTargetLocks), len(expectedQuantLocks))
	}
	byBits := make(map[int]ProductionQuantizationPackLock, len(artifact.QuantizedTargetLocks))
	for _, lock := range artifact.QuantizedTargetLocks {
		byBits[lock.QuantBits] = lock
	}
	for _, want := range expectedQuantLocks {
		got, ok := byBits[want.QuantBits]
		if !ok {
			t.Fatalf("artifact missing quantized q%d lock", want.QuantBits)
		}
		if got.ModelID != want.ModelID || got.Revision != want.Revision || got.ConfigSHA256 != want.ConfigSHA256 || len(got.WeightFiles) != len(want.WeightFiles) {
			t.Fatalf("artifact q%d lock = %+v, want %+v", want.QuantBits, got, want)
		}
		if got.BaseRevision != want.BaseRevision || got.ConversionCommand != want.ConversionCommand || got.AccuracySmoke != want.AccuracySmoke {
			t.Fatalf("artifact q%d conversion record = base:%q command:%q smoke:%q, want %+v", want.QuantBits, got.BaseRevision, got.ConversionCommand, got.AccuracySmoke, want)
		}
	}

	expectedPlatformLocks := DefaultOfficialPlatformAPILocks()
	if len(artifact.PlatformAPILocks) != len(expectedPlatformLocks) {
		t.Fatalf("artifact platform locks = %d, want %d macOS 26 API locks", len(artifact.PlatformAPILocks), len(expectedPlatformLocks))
	}
	byURL := make(map[string]OfficialPlatformAPILock, len(artifact.PlatformAPILocks))
	for _, lock := range artifact.PlatformAPILocks {
		byURL[lock.SourceURL] = lock
	}
	for _, want := range expectedPlatformLocks {
		got, ok := byURL[want.SourceURL]
		if !ok {
			t.Fatalf("artifact missing platform source %q", want.SourceURL)
		}
		if got != want {
			t.Fatalf("artifact platform lock[%s] = %+v, want %+v", want.SourceURL, got, want)
		}
	}
}

func TestOfficialGemma4E2BLocalSnapshot_VerifiesHashes_Good(t *testing.T) {
	lock, dir := officialGemma4TestSnapshot(t)

	if err := VerifyOfficialGemma4E2BLocalSnapshot(dir, lock); err != nil {
		t.Fatalf("VerifyOfficialGemma4E2BLocalSnapshot() error = %v", err)
	}
}

func TestOfficialGemma4E2BLocalSnapshot_VerifiesCacheRoot_Good(t *testing.T) {
	lock, cacheRoot, snapshotDir := officialGemma4TestCacheRoot(t)

	if err := VerifyOfficialGemma4E2BLocalSnapshot(cacheRoot, lock); err != nil {
		t.Fatalf("VerifyOfficialGemma4E2BLocalSnapshot(cache root) error = %v", err)
	}

	inspectLock, inspectCacheRoot, inspectSnapshotDir := officialGemma4InspectableTargetCacheRoot(t)
	report, err := InspectOfficialGemma4E2BLocalSnapshot(inspectCacheRoot, inspectLock)
	if err != nil {
		t.Fatalf("InspectOfficialGemma4E2BLocalSnapshot(cache root) error = %v", err)
	}
	if !report.Verified {
		t.Fatalf("report verified = false, want verified report")
	}
	if core.PathBase(snapshotDir) != lock.Revision {
		t.Fatalf("test cache root snapshot = %q, want basename to match lock revision %q", snapshotDir, lock.Revision)
	}
	if report.SnapshotDir != inspectSnapshotDir {
		t.Fatalf("report SnapshotDir = %q, want resolved locked snapshot %q", report.SnapshotDir, inspectSnapshotDir)
	}
}

func TestOfficialGemma4E2BLocalSnapshotReport_TargetPreflight_Good(t *testing.T) {
	lock, dir := officialGemma4InspectableTargetSnapshot(t)

	report, err := InspectOfficialGemma4E2BLocalSnapshot(dir, lock)
	if err != nil {
		t.Fatalf("InspectOfficialGemma4E2BLocalSnapshot() error = %v", err)
	}
	if !report.Verified || report.Error != "" {
		t.Fatalf("report verified/error = %v/%q, want verified clean report", report.Verified, report.Error)
	}
	if report.Role != OfficialGemma4E2BRoleTarget || report.ModelID != lock.ModelID || report.Revision != lock.Revision {
		t.Fatalf("report identity = %+v, want target lock identity", report)
	}
	if report.ExpectedArchitecture != "gemma4_text" || !report.ArchitectureOK {
		t.Fatalf("report architecture = %q ok=%v, want gemma4_text match", report.ExpectedArchitecture, report.ArchitectureOK)
	}
	if !report.Pack.Valid() || !report.Pack.NativeLoadable || report.Pack.Architecture != "gemma4_text" {
		t.Fatalf("pack = %+v, want valid native Gemma 4 text path", report.Pack)
	}
	if report.Pack.QuantBits != 6 || report.Pack.ContextLength != ProductionLaneHyperLongContextLength {
		t.Fatalf("pack quant/context = %d/%d, want q6 128Ki", report.Pack.QuantBits, report.Pack.ContextLength)
	}
}

func TestOfficialGemma4E2BPairPreflight_TargetAssistantContract_Good(t *testing.T) {
	targetLock, targetDir := officialGemma4InspectableTargetSnapshot(t)
	assistantLock, assistantDir := officialGemma4InspectableAssistantSnapshot(t)

	report, err := InspectOfficialGemma4E2BPairLocalSnapshots(targetDir, assistantDir, targetLock, assistantLock)
	if err != nil {
		t.Fatalf("InspectOfficialGemma4E2BPairLocalSnapshots() error = %v", err)
	}
	if !report.PairOK || report.Error != "" {
		t.Fatalf("pair report ok/error = %v/%q, want clean pair contract", report.PairOK, report.Error)
	}
	if !report.Target.Verified || !report.Assistant.Verified {
		t.Fatalf("verified = target:%v assistant:%v, want both verified", report.Target.Verified, report.Assistant.Verified)
	}
	if !report.SameVocabSize || !report.SameContextLength || !report.AssistantBackboneMatchesTarget {
		t.Fatalf("pair metadata = vocab:%v context:%v backbone:%v, want compatible official E2B pair", report.SameVocabSize, report.SameContextLength, report.AssistantBackboneMatchesTarget)
	}
	if report.TargetHiddenSize != 1536 || report.AssistantBackboneHiddenSize != 1536 {
		t.Fatalf("hidden sizes = target:%d assistant_backbone:%d, want 1536/1536", report.TargetHiddenSize, report.AssistantBackboneHiddenSize)
	}
	if !report.AssistantOrderedEmbeddings || report.AssistantNumCentroids != 2048 || report.AssistantCentroidIntermediateTopK != 32 {
		t.Fatalf("assistant ordered embedding = ordered:%v centroids:%d topk:%d, want official ordered centroid path", report.AssistantOrderedEmbeddings, report.AssistantNumCentroids, report.AssistantCentroidIntermediateTopK)
	}
	if !report.AssistantProjectionTensorsOK || !report.AssistantOrderedEmbeddingTensorsOK || len(report.AssistantMissingTensorNames) != 0 || len(report.AssistantInvalidTensorShapes) != 0 {
		t.Fatalf("assistant tensor evidence = projection:%v ordered:%v missing:%v invalid:%v, want clean pre/post and ordered-embedding tensor evidence", report.AssistantProjectionTensorsOK, report.AssistantOrderedEmbeddingTensorsOK, report.AssistantMissingTensorNames, report.AssistantInvalidTensorShapes)
	}
	if report.AssistantTokenOrderingDType != "I64" || !intSliceEqual(report.AssistantTokenOrderingShape, []int{262144}) {
		t.Fatalf("assistant token ordering evidence = dtype:%q shape:%v, want I64 [262144]", report.AssistantTokenOrderingDType, report.AssistantTokenOrderingShape)
	}
	if report.AssistantLayerCount != 4 || !report.AssistantFourLayerDrafter {
		t.Fatalf("assistant layer shape = count:%d four:%v, want official four-layer drafter", report.AssistantLayerCount, report.AssistantFourLayerDrafter)
	}
	if !report.AssistantLayerTypesCoveredByTarget {
		t.Fatalf("AssistantLayerTypesCoveredByTarget = false, want target K/V streams to cover assistant layer types")
	}
	if len(report.TargetKVLayerTypes) != 2 || report.TargetKVLayerTypes[0] != "sliding_attention" || report.TargetKVLayerTypes[1] != "full_attention" {
		t.Fatalf("TargetKVLayerTypes = %v, want sliding/full target K/V streams", report.TargetKVLayerTypes)
	}
	if len(report.AssistantLayerTypes) != 4 || report.AssistantLayerTypes[0] != "sliding_attention" || report.AssistantLayerTypes[3] != "full_attention" {
		t.Fatalf("AssistantLayerTypes = %v, want three sliding layers then one full layer", report.AssistantLayerTypes)
	}
	if !report.AssistantAttachable {
		t.Fatalf("AssistantAttachable = false, want attached-native MTP drafter contract")
	}
}

func TestOfficialGemma4E2BPairPreflight_CacheRoots_Good(t *testing.T) {
	targetLock, targetCacheRoot, targetSnapshotDir := officialGemma4InspectableTargetCacheRoot(t)
	assistantLock, assistantCacheRoot, assistantSnapshotDir := officialGemma4InspectableAssistantCacheRoot(t)

	if core.PathBase(targetCacheRoot) != "models--google--gemma-4-E2B-it" {
		t.Fatalf("target cache root = %q, want official target HF cache basename", targetCacheRoot)
	}
	if core.PathBase(assistantCacheRoot) != "models--google--gemma-4-E2B-it-assistant" {
		t.Fatalf("assistant cache root = %q, want official assistant HF cache basename", assistantCacheRoot)
	}
	report, err := InspectOfficialGemma4E2BPairLocalSnapshots(targetCacheRoot, assistantCacheRoot, targetLock, assistantLock)
	if err != nil {
		t.Fatalf("InspectOfficialGemma4E2BPairLocalSnapshots(cache roots) error = %v", err)
	}
	if !report.PairOK {
		t.Fatalf("PairOK = false, want cache-root official pair report to pass")
	}
	if report.TargetPath != targetSnapshotDir || report.AssistantPath != assistantSnapshotDir {
		t.Fatalf("pair paths = %q %q, want resolved snapshots %q %q", report.TargetPath, report.AssistantPath, targetSnapshotDir, assistantSnapshotDir)
	}
}

func TestOfficialGemma4E2BPairPreflight_RejectsZeroCentroidTokenOrdering_Bad(t *testing.T) {
	evidence := officialGemma4AssistantTensorEvidence{}
	ok := officialGemma4TokenOrderingHasShape(safetensors.Index{
		Tensors: map[string]safetensors.TensorRef{
			"masked_embedding.token_ordering": {
				Name:  "masked_embedding.token_ordering",
				Shape: []uint64{16},
			},
		},
	}, &evidence, "masked_embedding.token_ordering", 0, 16)

	if ok || len(evidence.InvalidTensorShapes) != 1 {
		t.Fatalf("token ordering ok=%v invalid=%v, want fail-closed invalid shape for zero centroids", ok, evidence.InvalidTensorShapes)
	}
}

func TestOfficialGemma4E2BPairPreflight_RejectsFloatTokenOrdering_Bad(t *testing.T) {
	evidence := officialGemma4AssistantTensorEvidence{}
	ok := officialGemma4TokenOrderingHasShape(safetensors.Index{
		Tensors: map[string]safetensors.TensorRef{
			"masked_embedding.token_ordering": {
				Name:  "masked_embedding.token_ordering",
				DType: "F32",
				Shape: []uint64{2048, 128},
			},
		},
	}, &evidence, "masked_embedding.token_ordering", 2048, 262144)

	if ok || len(evidence.InvalidTensorShapes) != 1 || !core.Contains(evidence.InvalidTensorShapes[0], "dtype") {
		t.Fatalf("token ordering ok=%v invalid=%v, want fail-closed integer dtype rejection", ok, evidence.InvalidTensorShapes)
	}
}

func TestOfficialGemma4E2BLocalSnapshot_RejectsHashMismatch_Bad(t *testing.T) {
	lock, dir := officialGemma4TestSnapshot(t)
	writeOfficialGemma4TestFile(t, dir, "config.json", []byte("changed"))

	err := VerifyOfficialGemma4E2BLocalSnapshot(dir, lock)
	if err == nil || !core.Contains(err.Error(), "config.json") || !core.Contains(err.Error(), "SHA-256") {
		t.Fatalf("VerifyOfficialGemma4E2BLocalSnapshot(hash mismatch) error = %v, want config SHA-256 mismatch", err)
	}
}

func TestOfficialGemma4E2BLocalSnapshot_RejectsUnexpectedSafetensorsIndex_Ugly(t *testing.T) {
	lock, dir := officialGemma4TestSnapshot(t)
	writeOfficialGemma4TestFile(t, dir, "model.safetensors.index.json", []byte("{}"))

	err := VerifyOfficialGemma4E2BLocalSnapshot(dir, lock)
	if err == nil || !core.Contains(err.Error(), "model.safetensors.index.json") {
		t.Fatalf("VerifyOfficialGemma4E2BLocalSnapshot(unexpected index) error = %v, want explicit index rejection", err)
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

func officialGemma4TestSnapshot(t *testing.T) (OfficialGemma4E2BLock, string) {
	t.Helper()
	lock := OfficialGemma4E2BLock{
		Role:                   OfficialGemma4E2BRoleTarget,
		ModelID:                "google/gemma-4-E2B-it",
		Revision:               "test-revision",
		ConfigSHA256:           core.SHA256Hex([]byte("config")),
		TokenizerSHA256:        core.SHA256Hex([]byte("tokenizer")),
		TokenizerConfigSHA256:  core.SHA256Hex([]byte("tokenizer-config")),
		GenerationConfigSHA256: core.SHA256Hex([]byte("generation-config")),
		ChatTemplateSHA256:     core.SHA256Hex([]byte("chat-template")),
		WeightFile:             "model.safetensors",
		WeightSHA256:           core.SHA256Hex([]byte("weights")),
		WeightBytes:            uint64(len("weights")),
	}
	dir := core.PathJoin(t.TempDir(), lock.Revision)
	if result := core.MkdirAll(dir, 0o755); !result.OK {
		t.Fatalf("MkdirAll snapshot: %v", result.Value)
	}
	writeOfficialGemma4TestFile(t, dir, "config.json", []byte("config"))
	writeOfficialGemma4TestFile(t, dir, "tokenizer.json", []byte("tokenizer"))
	writeOfficialGemma4TestFile(t, dir, "tokenizer_config.json", []byte("tokenizer-config"))
	writeOfficialGemma4TestFile(t, dir, "generation_config.json", []byte("generation-config"))
	writeOfficialGemma4TestFile(t, dir, "chat_template.jinja", []byte("chat-template"))
	writeOfficialGemma4TestFile(t, dir, lock.WeightFile, []byte("weights"))
	return lock, dir
}

func officialGemma4TestCacheRoot(t *testing.T) (OfficialGemma4E2BLock, string, string) {
	t.Helper()
	lock, sourceDir := officialGemma4TestSnapshot(t)
	return officialGemma4TestCacheRootFrom(t, lock, sourceDir)
}

func officialGemma4InspectableTargetCacheRoot(t *testing.T) (OfficialGemma4E2BLock, string, string) {
	t.Helper()
	lock, sourceDir := officialGemma4InspectableTargetSnapshot(t)
	return officialGemma4TestCacheRootFrom(t, lock, sourceDir)
}

func officialGemma4InspectableAssistantCacheRoot(t *testing.T) (OfficialGemma4E2BLock, string, string) {
	t.Helper()
	lock, sourceDir := officialGemma4InspectableAssistantSnapshot(t)
	return officialGemma4TestCacheRootFrom(t, lock, sourceDir)
}

func officialGemma4TestCacheRootFrom(t *testing.T, lock OfficialGemma4E2BLock, sourceDir string) (OfficialGemma4E2BLock, string, string) {
	t.Helper()
	cacheRoot := core.PathJoin(t.TempDir(), "models--"+core.Replace(lock.ModelID, "/", "--"))
	snapshotDir := core.PathJoin(cacheRoot, "snapshots", lock.Revision)
	if result := core.MkdirAll(snapshotDir, 0o755); !result.OK {
		t.Fatalf("MkdirAll cache snapshot: %v", result.Value)
	}
	for _, name := range []string{
		"config.json",
		"tokenizer.json",
		"tokenizer_config.json",
		"generation_config.json",
		lock.WeightFile,
	} {
		read := core.ReadFile(core.PathJoin(sourceDir, name))
		if !read.OK {
			t.Fatalf("ReadFile %s: %v", name, read.Value)
		}
		writeOfficialGemma4TestFile(t, snapshotDir, name, read.Value.([]byte))
	}
	if lock.ChatTemplateSHA256 != "" {
		read := core.ReadFile(core.PathJoin(sourceDir, "chat_template.jinja"))
		if !read.OK {
			t.Fatalf("ReadFile chat_template.jinja: %v", read.Value)
		}
		writeOfficialGemma4TestFile(t, snapshotDir, "chat_template.jinja", read.Value.([]byte))
	}
	return lock, cacheRoot, snapshotDir
}

func officialGemma4InspectableTargetSnapshot(t *testing.T) (OfficialGemma4E2BLock, string) {
	t.Helper()
	config := []byte(`{
		"model_type": "gemma4",
		"architectures": ["Gemma4ForConditionalGeneration"],
		"text_config": {
			"model_type": "gemma4_text",
			"vocab_size": 262144,
			"hidden_size": 1536,
			"hidden_size_per_layer_input": 256,
			"num_hidden_layers": 35,
			"num_attention_heads": 8,
			"num_key_value_heads": 1,
			"num_kv_shared_layers": 20,
			"head_dim": 256,
			"global_head_dim": 512,
			"max_position_embeddings": 131072,
			"sliding_window": 512,
			"layer_types": [
				"sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
				"sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
				"sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
				"sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
				"sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
				"sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
				"sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention"
			],
			"rope_parameters": {
				"full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0, "rope_type": "proportional"},
				"sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}
			}
		},
		"vision_config": {
			"hidden_size": 768
		},
		"quantization_config": {"bits": 6, "group_size": 64}
	}`)
	tokenizer := []byte(`{
		"model": {
			"type": "BPE",
			"vocab": {"h": 0, "e": 1, "l": 2, "o": 3},
			"merges": ["h e"],
			"byte_fallback": false
		},
		"added_tokens": [
			{"id": 100, "content": "<bos>", "special": true},
			{"id": 101, "content": "<eos>", "special": true}
		]
	}`)
	tokenizerConfig := []byte(`{"model_max_length": 131072}`)
	generationConfig := []byte(`{"max_new_tokens": 8192}`)
	chatTemplate := []byte(`{{ bos_token }}{% for message in messages %}{{ message["content"] }}{% endfor %}`)
	weights := []byte("weights")
	lock := OfficialGemma4E2BLock{
		Role:                   OfficialGemma4E2BRoleTarget,
		ModelID:                "google/gemma-4-E2B-it",
		Revision:               "test-inspect-revision",
		Architecture:           "Gemma4ForConditionalGeneration",
		ModelType:              "gemma4",
		ConfigSHA256:           core.SHA256Hex(config),
		TokenizerSHA256:        core.SHA256Hex(tokenizer),
		TokenizerConfigSHA256:  core.SHA256Hex(tokenizerConfig),
		GenerationConfigSHA256: core.SHA256Hex(generationConfig),
		ChatTemplateSHA256:     core.SHA256Hex(chatTemplate),
		WeightFile:             "model.safetensors",
		WeightSHA256:           core.SHA256Hex(weights),
		WeightBytes:            uint64(len(weights)),
	}
	dir := core.PathJoin(t.TempDir(), lock.Revision)
	if result := core.MkdirAll(dir, 0o755); !result.OK {
		t.Fatalf("MkdirAll snapshot: %v", result.Value)
	}
	writeOfficialGemma4TestFile(t, dir, "config.json", config)
	writeOfficialGemma4TestFile(t, dir, "tokenizer.json", tokenizer)
	writeOfficialGemma4TestFile(t, dir, "tokenizer_config.json", tokenizerConfig)
	writeOfficialGemma4TestFile(t, dir, "generation_config.json", generationConfig)
	writeOfficialGemma4TestFile(t, dir, "chat_template.jinja", chatTemplate)
	writeOfficialGemma4TestFile(t, dir, lock.WeightFile, weights)
	return lock, dir
}

func officialGemma4InspectableAssistantSnapshot(t *testing.T) (OfficialGemma4E2BLock, string) {
	t.Helper()
	config := []byte(`{
		"model_type": "gemma4_assistant",
		"architectures": ["Gemma4AssistantForCausalLM"],
		"backbone_hidden_size": 1536,
		"num_centroids": 2048,
		"centroid_intermediate_top_k": 32,
		"use_ordered_embeddings": true,
		"text_config": {
			"model_type": "gemma4_text",
			"vocab_size": 262144,
			"hidden_size": 256,
			"num_hidden_layers": 4,
			"num_attention_heads": 4,
			"num_key_value_heads": 1,
			"num_kv_shared_layers": 4,
			"head_dim": 256,
			"global_head_dim": 512,
			"max_position_embeddings": 131072,
			"sliding_window": 512,
			"layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"],
			"rope_parameters": {
				"full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0, "rope_type": "proportional"},
				"sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}
			}
		}
	}`)
	tokenizer := []byte(`{
		"model": {
			"type": "BPE",
			"vocab": {"h": 0, "e": 1, "l": 2, "o": 3},
			"merges": ["h e"],
			"byte_fallback": false
		},
		"added_tokens": [
			{"id": 100, "content": "<bos>", "special": true},
			{"id": 101, "content": "<eos>", "special": true}
		]
	}`)
	tokenizerConfig := []byte(`{"model_max_length": 131072}`)
	generationConfig := []byte(`{"max_new_tokens": 8192}`)
	weights := officialGemma4AssistantTensorFixture(t)
	lock := OfficialGemma4E2BLock{
		Role:                   OfficialGemma4E2BRoleAssistant,
		ModelID:                "google/gemma-4-E2B-it-assistant",
		Revision:               "test-assistant-inspect-revision",
		Architecture:           "Gemma4AssistantForCausalLM",
		ModelType:              "gemma4_assistant",
		ConfigSHA256:           core.SHA256Hex(config),
		TokenizerSHA256:        core.SHA256Hex(tokenizer),
		TokenizerConfigSHA256:  core.SHA256Hex(tokenizerConfig),
		GenerationConfigSHA256: core.SHA256Hex(generationConfig),
		WeightFile:             "model.safetensors",
		WeightSHA256:           core.SHA256Hex(weights),
		WeightBytes:            uint64(len(weights)),
	}
	dir := core.PathJoin(t.TempDir(), lock.Revision)
	if result := core.MkdirAll(dir, 0o755); !result.OK {
		t.Fatalf("MkdirAll snapshot: %v", result.Value)
	}
	writeOfficialGemma4TestFile(t, dir, "config.json", config)
	writeOfficialGemma4TestFile(t, dir, "tokenizer.json", tokenizer)
	writeOfficialGemma4TestFile(t, dir, "tokenizer_config.json", tokenizerConfig)
	writeOfficialGemma4TestFile(t, dir, "generation_config.json", generationConfig)
	writeOfficialGemma4TestFile(t, dir, lock.WeightFile, weights)
	return lock, dir
}

func officialGemma4AssistantTensorFixture(t *testing.T) []byte {
	t.Helper()
	return officialGemma4SafetensorsHeaderOnly(t, map[string][]int64{
		"pre_projection.weight":                  {256, 3072},
		"post_projection.weight":                 {1536, 256},
		"masked_embedding.centroids.weight":      {2048, 256},
		"masked_embedding.token_ordering":        {262144},
		"model.layers.0.self_attn.q_proj.weight": {1024, 256},
	})
}

func officialGemma4SafetensorsHeaderOnly(t *testing.T, shapes map[string][]int64) []byte {
	t.Helper()
	type headerEntry struct {
		DType       string  `json:"dtype"`
		Shape       []int64 `json:"shape"`
		DataOffsets []int64 `json:"data_offsets"`
	}
	header := make(map[string]headerEntry, len(shapes))
	for name, shape := range shapes {
		dtype := "F32"
		if name == "masked_embedding.token_ordering" {
			dtype = "I64"
		}
		header[name] = headerEntry{DType: dtype, Shape: shape, DataOffsets: []int64{0, 0}}
	}
	encoded := core.JSONMarshal(header)
	if !encoded.OK {
		t.Fatalf("JSONMarshal safetensors fixture: %v", encoded.Value)
	}
	headerBytes := encoded.Value.([]byte)
	out := make([]byte, 8+len(headerBytes))
	binary.LittleEndian.PutUint64(out[:8], uint64(len(headerBytes)))
	copy(out[8:], headerBytes)
	return out
}

func writeOfficialGemma4TestFile(t *testing.T, dir, name string, data []byte) {
	t.Helper()
	if result := core.WriteFile(core.PathJoin(dir, name), data, 0o644); !result.OK {
		t.Fatalf("WriteFile %s: %v", name, result.Value)
	}
}
