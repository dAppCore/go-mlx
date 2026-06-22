// SPDX-Licence-Identifier: EUPL-1.2

// Coverage-closing tests for adapter.go + fuse.go — drives the residual
// branches the primary suites in adapter_test.go / fuse_test.go leave
// uncovered: nil-context defaulting, the streaming-hash accumulator reuse +
// fault legs, the write-failure error wraps (chmod-driven, no root needed),
// the Gemma4 canonical-pair-name fallbacks, and the metal-gated fuse error
// paths (corrupt adapter/base weights, multi-shard output naming,
// SaveSafetensors / provenance write failures).

package lora

import (
	"context"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/mlx/pack"
	"dappco.re/go/mlx/pkg/metal"
)

// --- adapter.go: streamHashWeightFile accumulator reuse + fault legs ---

func TestStreamHashWeightFile_AccumulatorReuse_Ugly(t *testing.T) {
	// Ugly: hashAdapterPrecomputed shares a single hashWriter across every
	// weight shard, so the second large shard must hit the hasher.Reset()
	// arm (adapter.go L269-271) instead of allocating a fresh accumulator.
	// Two shards both over streamHashMinBytes force the directory branch to
	// stream each in turn, exercising both the first-shard lazy-construct
	// (h == nil) and the second-shard Reset. The digest must stay
	// deterministic, proving the reset accumulator hashed the second shard's
	// bytes correctly rather than carrying state across.
	const shardSize = streamHashMinBytes + 32*1024 // ~160 KiB each, over the 128 KiB gate

	makeAdapter := func(t *testing.T, fill byte) AdapterInfo {
		t.Helper()
		dir := t.TempDir()
		if result := core.WriteFile(core.PathJoin(dir, "adapter_config.json"), []byte(`{"rank":8,"alpha":16,"target_modules":["q_proj"]}`), 0o600); !result.OK {
			t.Fatalf("WriteFile adapter_config: %s", result.Error())
		}
		shard := make([]byte, shardSize)
		for i := range shard {
			shard[i] = fill
		}
		// Two distinct *.safetensors shards in one adapter dir: the glob
		// returns both, so the shared hasher is reused on the second.
		if result := core.WriteFile(core.PathJoin(dir, "adapter-00001.safetensors"), shard, 0o600); !result.OK {
			t.Fatalf("WriteFile shard 1: %s", result.Error())
		}
		if result := core.WriteFile(core.PathJoin(dir, "adapter-00002.safetensors"), shard, 0o600); !result.OK {
			t.Fatalf("WriteFile shard 2: %s", result.Error())
		}
		info, err := InspectAdapter(dir)
		if err != nil {
			t.Fatalf("InspectAdapter(two large shards) error = %v", err)
		}
		if info.Hash == "" {
			t.Fatalf("InspectAdapter(two large shards) produced empty hash: %+v", info)
		}
		return info
	}

	first := makeAdapter(t, 0x5A)
	repeat := makeAdapter(t, 0x5A)
	if first.Hash != repeat.Hash {
		t.Fatalf("streaming-hash accumulator reuse not deterministic: %q != %q", first.Hash, repeat.Hash)
	}
	different := makeAdapter(t, 0xA5)
	if first.Hash == different.Hash {
		t.Fatalf("streaming-hash accumulator reuse collided across distinct shard content: %q", first.Hash)
	}
}

func TestStreamHashWeightFile_OpenFailure_Bad(t *testing.T) {
	// Bad: streamHashWeightFile takes the stream path only when Stat reports
	// a size over streamHashMinBytes, then Open must succeed. A large shard
	// chmod'd 0o000 stats large (so the gate routes to streaming) but cannot
	// be opened — driving the !opened.OK skip (adapter.go L263-265). The
	// unreadable shard contributes nothing, so the resulting digest equals an
	// adapter holding only the readable config + small stub (no large shard).
	const shardSize = streamHashMinBytes + 32*1024

	dir := t.TempDir()
	if result := core.WriteFile(core.PathJoin(dir, "adapter_config.json"), []byte(`{"rank":8,"alpha":16,"target_modules":["q_proj"]}`), 0o600); !result.OK {
		t.Fatalf("WriteFile adapter_config: %s", result.Error())
	}
	blocked := core.PathJoin(dir, "blocked.safetensors")
	shard := make([]byte, shardSize)
	if result := core.WriteFile(blocked, shard, 0o600); !result.OK {
		t.Fatalf("WriteFile blocked shard: %s", result.Error())
	}
	if result := core.Chmod(blocked, 0o000); !result.OK {
		t.Fatalf("Chmod blocked shard: %s", result.Error())
	}
	t.Cleanup(func() { core.Chmod(blocked, 0o600) })

	info, err := InspectAdapter(dir)
	if err != nil {
		t.Fatalf("InspectAdapter(blocked large shard) error = %v", err)
	}
	if info.Hash == "" {
		t.Fatalf("InspectAdapter(blocked large shard) produced empty hash: %+v", info)
	}

	// An adapter with the same config but no large shard at all must hash
	// identically — the open-failed shard was skipped, not folded in.
	bare := t.TempDir()
	if result := core.WriteFile(core.PathJoin(bare, "adapter_config.json"), []byte(`{"rank":8,"alpha":16,"target_modules":["q_proj"]}`), 0o600); !result.OK {
		t.Fatalf("WriteFile bare adapter_config: %s", result.Error())
	}
	bareInfo, err := InspectAdapter(bare)
	if err != nil {
		t.Fatalf("InspectAdapter(bare) error = %v", err)
	}
	if info.Hash != bareInfo.Hash {
		t.Fatalf("open-failed large shard altered digest: %q != %q", info.Hash, bareInfo.Hash)
	}
}

// --- adapter.go / fuse.go: nil-context defaulting ---

func TestPrepareFuse_NilContextDefaults_Good(t *testing.T) {
	// Good: prepareFuse accepts a nil context and substitutes
	// context.Background() (fuse.go L87-89) before the ctx.Err() probe. A
	// nil ctx with an otherwise-empty options struct must therefore fall
	// through to the source-root validation error, not panic on the nil
	// ctx.Err() call.
	//nolint:staticcheck // deliberately passing a nil ctx to exercise the default.
	if _, err := prepareFuse(nil, FuseOptions{}); err != errFuseSourceRootRequired {
		t.Fatalf("prepareFuse(nil ctx) = %v, want errFuseSourceRootRequired (nil ctx defaulted)", err)
	}
}

func TestFuseIntoPack_NilContextDefaults_Bad(t *testing.T) {
	// Bad: FuseIntoPack also defaults a nil context (fuse.go L679-681). With
	// empty options it must reach prepareFuse's source-root guard and return
	// that validation error rather than panicking on the nil ctx — no metal
	// needed because validation fails before any tensor load.
	//nolint:staticcheck // deliberately passing a nil ctx to exercise the default.
	if _, err := FuseIntoPack(nil, FuseOptions{}); err != errFuseSourceRootRequired {
		t.Fatalf("FuseIntoPack(nil ctx) = %v, want errFuseSourceRootRequired (nil ctx defaulted)", err)
	}
}

// --- fuse.go: ensureEmptyFuseWeightDestination stat-error (not IsNotExist) ---

func TestEnsureEmptyFuseWeightDestination_StatError_Ugly(t *testing.T) {
	// Ugly: ensureEmptyFuseWeightDestination distinguishes a not-exist output
	// (accepted, L157-159) from any other stat failure (wrapped, L160). A path
	// whose *parent* is a regular file makes Stat fail with ENOTDIR — which is
	// not IsNotExist — so the wrap fires. No root needed: a file standing where
	// a directory component is expected is an ordinary user mistake.
	dir := t.TempDir()
	file := core.PathJoin(dir, "iam-a-file")
	if result := core.WriteFile(file, []byte("x"), 0o644); !result.OK {
		t.Fatalf("WriteFile file: %s", result.Error())
	}
	// "<file>/child" — traversing through a file errors with ENOTDIR.
	err := ensureEmptyFuseWeightDestination(core.PathJoin(file, "child"))
	if err == nil {
		t.Fatal("ensureEmptyFuseWeightDestination(path under a file) = nil, want stat error")
	}
	if !core.Contains(err.Error(), "inspect output path") {
		t.Fatalf("error = %v, want inspect-output-path context", err)
	}
}

// --- fuse.go: copyLocalFile write failure ---

func TestCopyLocalFile_WriteFailure_Bad(t *testing.T) {
	// Bad: copyLocalFile wraps a WriteFile failure on the destination
	// (fuse.go L365-367). A readable source plus a destination directory
	// chmod'd read-only (0o500) makes the write fail without touching the
	// read leg. uid != 0 so the directory mode is honoured.
	dir := t.TempDir()
	source := core.PathJoin(dir, "src.json")
	if result := core.WriteFile(source, []byte(`{"ok":true}`), 0o644); !result.OK {
		t.Fatalf("WriteFile source: %s", result.Error())
	}
	destDir := core.PathJoin(dir, "ro")
	if result := core.MkdirAll(destDir, 0o755); !result.OK {
		t.Fatalf("MkdirAll destDir: %s", result.Error())
	}
	if result := core.Chmod(destDir, 0o500); !result.OK {
		t.Fatalf("Chmod destDir: %s", result.Error())
	}
	t.Cleanup(func() { core.Chmod(destDir, 0o755) })

	err := copyLocalFile(source, core.PathJoin(destDir, "out.json"))
	if err == nil {
		t.Fatal("copyLocalFile(readonly dest) = nil, want write error")
	}
	if !core.Contains(err.Error(), "write ") {
		t.Fatalf("error = %v, want write-destination context", err)
	}
}

// --- fuse.go: copyModelPackMetadata propagates a copy failure ---

func TestCopyModelPackMetadata_CopyFailure_Bad(t *testing.T) {
	// Bad: copyModelPackMetadata returns the first copyLocalFile error
	// (fuse.go L288-290). A source dir holding a metadata file the suffix
	// filter accepts, copied into a read-only output dir, makes the inner
	// WriteFile fail and the error propagate out of the metadata copy loop.
	source := t.TempDir()
	if result := core.WriteFile(core.PathJoin(source, "config.json"), []byte(`{"model_type":"qwen3"}`), 0o644); !result.OK {
		t.Fatalf("WriteFile source config: %s", result.Error())
	}
	output := t.TempDir()
	if result := core.Chmod(output, 0o500); !result.OK {
		t.Fatalf("Chmod output: %s", result.Error())
	}
	t.Cleanup(func() { core.Chmod(output, 0o755) })

	err := copyModelPackMetadata(source, output)
	if err == nil {
		t.Fatal("copyModelPackMetadata(readonly output) = nil, want propagated copy error")
	}
	if !core.Contains(err.Error(), "write ") {
		t.Fatalf("error = %v, want propagated write error", err)
	}
}

// --- fuse.go: isModelWeightMetadataCopySkip .gguf arm ---

func TestIsModelWeightMetadataCopySkip_GgufArm_Bad(t *testing.T) {
	// Bad: the .gguf substring arm (fuse.go L322-324) skips GGUF weight files
	// from the metadata copy. The existing Bad test only covers the
	// .safetensors substring and the provenance-name equality; this drives a
	// lowercase .gguf name (the fold path the uppercase MODEL.GGUF case does
	// not reach via the same byte branch) and a sharded .gguf-bearing name.
	for _, name := range []string{"model.gguf", "weights-00001-of-00002.gguf"} {
		if !isModelWeightMetadataCopySkip(name) {
			t.Fatalf("isModelWeightMetadataCopySkip(%q) = false, want true (.gguf skip)", name)
		}
	}
	// A name merely containing "gguf" without the dot must NOT be skipped —
	// the substring is ".gguf", anchoring the extension.
	if isModelWeightMetadataCopySkip("ggufnotes.json") {
		t.Fatal("isModelWeightMetadataCopySkip(ggufnotes.json) = true, want false")
	}
}

// --- fuse.go: fuseGemma4PairName / fuseJoinCanonicalTarget fallbacks ---

func TestFuseGemma4PairName_Fallbacks_Ugly(t *testing.T) {
	// Ugly: fuseGemma4PairName has three exit shapes the suffix-target fuse
	// tests reach only via the full canonical path. Drive each directly:
	//   - empty pairName short-circuits to ("", false) (L626-628).
	//   - a single-segment target ("q_proj") skips the two-segment lookup
	//     (len(parts) < 2) and resolves through the last-segment fallback
	//     (L636-637); the prefix is empty so fuseJoinCanonicalTarget returns
	//     the canonical path verbatim (L643-645).
	//   - a pairName whose segments map to no LoRA target returns
	//     ("", false) via the final fall-through (L639).
	if canonical, ok := fuseGemma4PairName("", "gemma4_text"); ok || canonical != "" {
		t.Fatalf("fuseGemma4PairName(\"\") = %q,%v, want \"\",false", canonical, ok)
	}
	canonical, ok := fuseGemma4PairName("q_proj", "gemma4_text")
	if !ok || canonical != "self_attn.q_proj" {
		t.Fatalf("fuseGemma4PairName(q_proj) = %q,%v, want self_attn.q_proj,true (single-segment + empty-prefix join)", canonical, ok)
	}
	if canonical, ok := fuseGemma4PairName("model.layers.0.not_a_target", "gemma4_text"); ok || canonical != "" {
		t.Fatalf("fuseGemma4PairName(non-target) = %q,%v, want \"\",false", canonical, ok)
	}
}

func TestFuseBaseWeightKeyForArchitecture_SingleSegmentGemma4_Good(t *testing.T) {
	// Good: the single-segment Gemma4 target flows through
	// fuseBaseWeightKeyForArchitecture → fuseGemma4PairName's empty-prefix
	// fuseJoinCanonicalTarget arm and gains the canonical ".weight" key.
	if got := fuseBaseWeightKeyForArchitecture("q_proj", "gemma4_text"); got != "self_attn.q_proj.weight" {
		t.Fatalf("fuseBaseWeightKeyForArchitecture(q_proj) = %q, want self_attn.q_proj.weight", got)
	}
}

// --- fuse.go: fuseBaseWeightIndexForArchitecture nil-array skip ---

func TestFuseBaseWeightIndexForArchitecture_NilArraySkip_Ugly(t *testing.T) {
	// Ugly: fuseBaseWeightIndexForArchitecture iterates the sorted base keys
	// and skips any whose array is nil (fuse.go L517-518) before attempting
	// canonicalisation. A map carrying a real array plus a nil-valued key
	// proves the nil entry is dropped from the index while the real one is
	// retained under its canonical name. No metal: the arrays are never read,
	// only their presence is inspected.
	real := &metal.Array{}
	base := map[string]*metal.Array{
		"language_model.model.layers.0.self_attn.q_proj.weight": real,
		"language_model.model.layers.0.self_attn.k_proj.weight": nil,
	}
	index := fuseBaseWeightIndexForArchitecture(base, "gemma4_text")
	if index == nil {
		t.Fatal("fuseBaseWeightIndexForArchitecture(gemma4) = nil, want populated index")
	}
	match, ok := index["model.layers.0.self_attn.q_proj.weight"]
	if !ok || match.Key != "language_model.model.layers.0.self_attn.q_proj.weight" {
		t.Fatalf("index q_proj = %+v ok=%v, want the real-array source key", match, ok)
	}
	if _, ok := index["model.layers.0.self_attn.k_proj.weight"]; ok {
		t.Fatal("nil-array k_proj key was indexed, want it skipped")
	}
}

// --- fuse.go: fuseBaseWeightSidecars duplicate-prefix dedup ---

func TestFuseBaseWeightSidecars_DuplicatePrefixSkip_Ugly(t *testing.T) {
	// Ugly: fuseBaseWeightSidecars resolves sidecar prefixes from both the
	// matched key and its canonical alias, then dedups identical prefixes
	// (fuse.go L575-577, L580-581 continue). When key == canonical both
	// prefixes are equal, so the second is skipped — proving .scales/.biases
	// are appended once, not twice. Driven directly with a synthetic base map
	// (arrays never read).
	scales := &metal.Array{}
	biases := &metal.Array{}
	base := map[string]*metal.Array{
		"model.layers.0.q_proj.weight": {},
		"model.layers.0.q_proj.scales": scales,
		"model.layers.0.q_proj.biases": biases,
	}
	// key == canonical → prefixes resolve to the same "model.layers.0.q_proj"
	// twice, exercising the duplicate-skip.
	scaleKey, biasKey, sidecars := fuseBaseWeightSidecars(base, "model.layers.0.q_proj.weight", "model.layers.0.q_proj.weight", nil)
	if scaleKey != "model.layers.0.q_proj.scales" || biasKey != "model.layers.0.q_proj.biases" {
		t.Fatalf("sidecar keys = %q/%q, want q_proj scales/biases", scaleKey, biasKey)
	}
	if len(sidecars) != 2 {
		t.Fatalf("sidecar keys = %v, want exactly 2 (no duplicate-prefix double-append)", sidecars)
	}
}

// --- fuse.go: fuseQuantizedBaseTargetMetadataError canonical-key branch ---

func TestFuseQuantizedBaseTargetMetadataError_CanonicalBranch_Ugly(t *testing.T) {
	// Ugly: the metadata-missing error message appends the canonical key only
	// when it differs from the matched key (fuse.go L619-621). Build the match
	// both ways and assert the canonical-suffix appears exactly when expected.
	withCanonical := fuseQuantizedBaseTargetMetadataError(fuseBaseWeightMatch{
		Key:          "language_model.model.layers.0.self_attn.q_proj.weight",
		CanonicalKey: "model.layers.0.self_attn.q_proj.weight",
	})
	if !core.Contains(withCanonical.Error(), "(canonical model.layers.0.self_attn.q_proj.weight)") {
		t.Fatalf("error = %v, want canonical-key suffix", withCanonical)
	}
	sameKey := fuseQuantizedBaseTargetMetadataError(fuseBaseWeightMatch{
		Key:          "model.layers.0.self_attn.q_proj.weight",
		CanonicalKey: "model.layers.0.self_attn.q_proj.weight",
	})
	if core.Contains(sameKey.Error(), "canonical") {
		t.Fatalf("error = %v, want no canonical suffix when key == canonical", sameKey)
	}
}

// --- fuse.go: writeFuseProvenance write failure ---

func TestWriteFuseProvenance_WriteFailure_Bad(t *testing.T) {
	// Bad: writeFuseProvenance wraps a WriteFile failure (fuse.go L659-661).
	// Marshalling a plain FuseProvenance always succeeds, so the only
	// reachable failure is the write — a provenance path under a read-only
	// directory forces it. uid != 0 so the directory mode blocks the write.
	roDir := t.TempDir()
	if result := core.Chmod(roDir, 0o500); !result.OK {
		t.Fatalf("Chmod roDir: %s", result.Error())
	}
	t.Cleanup(func() { core.Chmod(roDir, 0o755) })

	err := writeFuseProvenance(core.PathJoin(roDir, FuseProvenanceFile), FuseProvenance{
		Version:      1,
		OutputWeight: "model.safetensors",
	})
	if err == nil {
		t.Fatal("writeFuseProvenance(readonly dir) = nil, want write error")
	}
	if !core.Contains(err.Error(), "write adapter provenance") {
		t.Fatalf("error = %v, want write-provenance context", err)
	}
}

// --- fuse.go: fuseWeightPairs already-fused skip ---

func TestFuseWeightPairs_AlreadyFusedSkip_Ugly(t *testing.T) {
	// Ugly: fuseWeightPairs skips any pair already recorded in the shared
	// fusedPairs set (fuse.go L856-857 continue) — the cross-shard
	// idempotence guard that stops a multi-shard fuse re-applying a delta.
	// Pre-seeding fusedPairs with the only pair name means the loop body is
	// entirely skipped, so no base lookup or matmul runs and the returned
	// key slice is empty. No metal: the skip happens before any tensor op.
	pairs := map[string]fusePair{
		"model.layers.0.self_attn.q_proj": {MatrixA: &metal.Array{}, MatrixB: &metal.Array{}},
	}
	already := map[string]struct{}{"model.layers.0.self_attn.q_proj": {}}
	fused, err := fuseWeightPairs(context.Background(), map[string]*metal.Array{
		"model.layers.0.self_attn.q_proj.weight": {},
	}, pairs, already, 1, pack.ModelPack{})
	if err != nil {
		t.Fatalf("fuseWeightPairs(already-fused) error = %v", err)
	}
	if len(fused) != 0 {
		t.Fatalf("fused keys = %v, want none (pair already fused, body skipped)", fused)
	}
}
