// SPDX-Licence-Identifier: EUPL-1.2

package kvconv

import (
	"context"
	"testing"

	core "dappco.re/go"
	state "dappco.re/go/inference/state"
	statefile "dappco.re/go/inference/state/filestore"
	"dappco.re/go/mlx/kv"
	"dappco.re/go/mlx/pkg/metal"
	trix "forge.lthn.ai/Snider/Enchantrix/pkg/trix"
)

func TestMetalKVSnapshotBlockSourcePartialPrefix_Good(t *testing.T) {
	bundle := &kv.StateBlockBundle{
		Version:    kv.StateBlockVersion,
		Kind:       kv.StateBlockBundleKind,
		TokenCount: 6,
		Blocks: []kv.StateBlockRef{
			{Index: 0, TokenStart: 0, TokenCount: 2},
			{Index: 1, TokenStart: 2, TokenCount: 2},
			{Index: 2, TokenStart: 4, TokenCount: 2},
		},
	}

	source, err := MetalKVSnapshotBlockSource(context.Background(), state.NewInMemoryStore(nil), bundle, 3)
	if err != nil {
		t.Fatalf("MetalKVSnapshotBlockSource() error = %v", err)
	}
	if source.BlockCount != 2 || source.PrefixTokens != 3 || source.TokenCount != 6 {
		t.Fatalf("source = %+v, want two covering blocks for three-token prefix", source)
	}
}

func TestMetalKVSnapshotBlockSourceRejectsNonContiguousBundle_Bad(t *testing.T) {
	bundle := &kv.StateBlockBundle{
		Version:    kv.StateBlockVersion,
		Kind:       kv.StateBlockBundleKind,
		TokenCount: 4,
		Blocks: []kv.StateBlockRef{
			{Index: 0, TokenStart: 0, TokenCount: 2},
			{Index: 1, TokenStart: 3, TokenCount: 1},
		},
	}

	if _, err := MetalKVSnapshotBlockSource(context.Background(), state.NewInMemoryStore(nil), bundle, 4); err != errStateKVBlockMetaMismatch {
		t.Fatalf("MetalKVSnapshotBlockSource() error = %v, want metadata mismatch", err)
	}
}

func TestMetalKVSnapshotBlockSourceNilStore_Bad(t *testing.T) {
	bundle := &kv.StateBlockBundle{
		Version:    kv.StateBlockVersion,
		Kind:       kv.StateBlockBundleKind,
		TokenCount: 2,
		Blocks:     []kv.StateBlockRef{{Index: 0, TokenStart: 0, TokenCount: 2}},
	}
	if _, err := MetalKVSnapshotBlockSource(context.Background(), nil, bundle, 2); err != errStateKVStoreNil {
		t.Fatalf("MetalKVSnapshotBlockSource(nil store) error = %v, want store-nil", err)
	}
}

func TestMetalKVSnapshotBlockSourcePrefixExceedsTokenCount_Bad(t *testing.T) {
	bundle := &kv.StateBlockBundle{
		Version:    kv.StateBlockVersion,
		Kind:       kv.StateBlockBundleKind,
		TokenCount: 4,
		Blocks: []kv.StateBlockRef{
			{Index: 0, TokenStart: 0, TokenCount: 2},
			{Index: 1, TokenStart: 2, TokenCount: 2},
		},
	}
	// A prefix larger than the bundle's own token count is unsatisfiable.
	if _, err := MetalKVSnapshotBlockSource(context.Background(), state.NewInMemoryStore(nil), bundle, 5); err != errStateKVPrefixExceeds {
		t.Fatalf("MetalKVSnapshotBlockSource(prefix>count) error = %v, want prefix-exceeds", err)
	}
}

// TestMetalKVSnapshotBlockSourceLoadOutOfRange_Ugly drives the Load closure's
// bounds guard — an index outside [0, BlockCount) must error rather than index
// past the covering blocks.
func TestMetalKVSnapshotBlockSourceLoadOutOfRange_Ugly(t *testing.T) {
	bundle := &kv.StateBlockBundle{
		Version:    kv.StateBlockVersion,
		Kind:       kv.StateBlockBundleKind,
		TokenCount: 4,
		Blocks: []kv.StateBlockRef{
			{Index: 0, TokenStart: 0, TokenCount: 2},
			{Index: 1, TokenStart: 2, TokenCount: 2},
		},
	}
	source, err := MetalKVSnapshotBlockSource(context.Background(), state.NewInMemoryStore(nil), bundle, 4)
	if err != nil {
		t.Fatalf("MetalKVSnapshotBlockSource() error = %v", err)
	}
	if _, err := source.Load(context.Background(), -1); err != errStateKVBlockOutOfRange {
		t.Fatalf("Load(-1) error = %v, want out-of-range", err)
	}
	if _, err := source.Load(context.Background(), source.BlockCount); err != errStateKVBlockOutOfRange {
		t.Fatalf("Load(%d) error = %v, want out-of-range", source.BlockCount, err)
	}
}

// TestMetalKVSnapshotBlockSourceTrimsPartialBlock_Ugly exercises the Load
// closure's mid-block trim path: a prefix that lands inside a covering block
// forces SliceBlock to clip the trailing tokens (lines 85-99) and the
// non-terminal block to have its terminal state cleared. Uses a real saved
// native bundle so the load + slice run end to end.
func TestMetalKVSnapshotBlockSourceTrimsPartialBlock_Ugly(t *testing.T) {
	// 512 tokens / 128 block size -> four 128-token blocks.
	fixture := newStateKVContainerFixture(t, 512, 128)
	region := fixture.openRegion(t)
	defer region.Close()

	// Prefix 200 covers block 0 fully (128) and trims block 1 to 72 tokens.
	const prefix = 200
	source, err := MetalKVSnapshotBlockSource(fixture.Context, region, fixture.Bundle, prefix)
	if err != nil {
		t.Fatalf("MetalKVSnapshotBlockSource(prefix=%d) error = %v", prefix, err)
	}
	if source.BlockCount != 2 || source.PrefixTokens != prefix {
		t.Fatalf("source = %+v, want two covering blocks for a %d-token prefix", source, prefix)
	}

	// Block 0 loads whole (128 tokens), block 1 is trimmed to 72.
	whole, err := source.Load(fixture.Context, 0)
	if err != nil {
		t.Fatalf("Load(0) error = %v", err)
	}
	if whole.TokenCount != 128 {
		t.Fatalf("block 0 TokenCount = %d, want 128 (untrimmed)", whole.TokenCount)
	}
	trimmed, err := source.Load(fixture.Context, 1)
	if err != nil {
		t.Fatalf("Load(1) error = %v", err)
	}
	if trimmed.TokenCount != prefix-128 {
		t.Fatalf("block 1 TokenCount = %d, want %d (trimmed to the prefix)", trimmed.TokenCount, prefix-128)
	}
	if trimmed.Snapshot == nil || len(trimmed.Snapshot.Layers) != 1 {
		t.Fatalf("trimmed block snapshot = %+v, want one native layer", trimmed.Snapshot)
	}
	if trimmed.TokenStart != 128 {
		t.Fatalf("trimmed block TokenStart = %d, want 128", trimmed.TokenStart)
	}
}

// --- merged from the root state_kv_test.go (orphan sweep: exercises
// MetalKVSnapshotBlockSource against region/MVLog state containers) ---
const (
	stateKVTestMagic = "KVST"
	stateKVTestKind  = "go-mlx/state-kv"
)

var stateKVRegionBenchmarkTokens int

type stateKVContainerFixture struct {
	Context       context.Context
	SourcePath    string
	ContainerPath string
	Bundle        *kv.StateBlockBundle
	PayloadOffset int64
	PayloadBytes  int64
}

func TestStateKVRegionBlockSourceLoadsWithoutOriginalMVLog_Good(t *testing.T) {
	fixture := newStateKVContainerFixture(t, 512, 128)
	if result := core.Remove(fixture.SourcePath); !result.OK {
		t.Fatalf("remove source State log: %v", result.Value)
	}
	region := fixture.openRegion(t)
	defer region.Close()
	source, err := MetalKVSnapshotBlockSource(fixture.Context, region, fixture.Bundle, fixture.Bundle.TokenCount)
	if err != nil {
		t.Fatalf("MetalKVSnapshotBlockSource(region) error = %v", err)
	}
	if source.BlockCount != 4 {
		t.Fatalf("block count = %d, want 4", source.BlockCount)
	}
	loadedTokens := 0
	for i := 0; i < source.BlockCount; i++ {
		block, err := source.Load(fixture.Context, i)
		if err != nil {
			t.Fatalf("Load(region block %d) error = %v", i, err)
		}
		if block.Snapshot == nil || len(block.Snapshot.Layers) != 1 {
			t.Fatalf("block %d snapshot = %+v, want one native layer", i, block.Snapshot)
		}
		layer := block.Snapshot.Layers[0]
		if len(layer.KeyBytes) == 0 || len(layer.ValueBytes) == 0 {
			t.Fatalf("block %d raw bytes = key:%d value:%d, want native bytes", i, len(layer.KeyBytes), len(layer.ValueBytes))
		}
		loadedTokens += block.TokenCount
	}
	if loadedTokens != fixture.Bundle.TokenCount {
		t.Fatalf("loaded tokens = %d, want %d", loadedTokens, fixture.Bundle.TokenCount)
	}
}

func BenchmarkStateKVRegionBlockSource_LoadNativeSlab4Blocks(b *testing.B) {
	fixture := newStateKVContainerFixture(b, 4096, 1024)
	region := fixture.openRegion(b)
	defer region.Close()
	source, err := MetalKVSnapshotBlockSource(fixture.Context, region, fixture.Bundle, fixture.Bundle.TokenCount)
	if err != nil {
		b.Fatalf("MetalKVSnapshotBlockSource(region): %v", err)
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		stateKVRegionBenchmarkTokens += loadStateKVBenchmarkBlocks(b, fixture.Context, source)
	}
}

func BenchmarkStateMVLogBlockSource_LoadNativeSlab4Blocks(b *testing.B) {
	fixture := newStateKVContainerFixture(b, 4096, 1024)
	store, err := statefile.Open(fixture.Context, fixture.SourcePath)
	if err != nil {
		b.Fatalf("Open(source): %v", err)
	}
	defer store.Close()
	source, err := MetalKVSnapshotBlockSource(fixture.Context, store, fixture.Bundle, fixture.Bundle.TokenCount)
	if err != nil {
		b.Fatalf("MetalKVSnapshotBlockSource(source): %v", err)
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		stateKVRegionBenchmarkTokens += loadStateKVBenchmarkBlocks(b, fixture.Context, source)
	}
}

func loadStateKVBenchmarkBlocks(tb testing.TB, ctx context.Context, source metal.KVSnapshotBlockSource) int {
	tb.Helper()
	tokens := 0
	for blockIndex := 0; blockIndex < source.BlockCount; blockIndex++ {
		block, err := source.Load(ctx, blockIndex)
		if err != nil {
			tb.Fatalf("Load(block %d): %v", blockIndex, err)
		}
		tokens += block.TokenCount
	}
	return tokens
}

func newStateKVContainerFixture(tb testing.TB, tokenCount, blockSize int) stateKVContainerFixture {
	tb.Helper()
	ctx := context.Background()
	dir := tb.TempDir()
	sourcePath := core.PathJoin(dir, "session.mvlog")
	containerPath := core.PathJoin(dir, "session.kv")
	store, err := statefile.Create(ctx, sourcePath)
	if err != nil {
		tb.Fatalf("Create(source): %v", err)
	}
	snapshot := stateKVNativeLayerSlabSnapshot(tokenCount, 2, 64)
	bundle, err := snapshot.SaveStateBlocks(ctx, store, kv.StateBlockOptions{
		BlockSize:  blockSize,
		KVEncoding: kv.EncodingNative,
	})
	if err != nil {
		_ = store.Close()
		tb.Fatalf("SaveStateBlocks(source): %v", err)
	}
	if err := store.Close(); err != nil {
		tb.Fatalf("Close(source): %v", err)
	}
	payloadBytes := stateKVFileSize(tb, sourcePath)
	stateKVWriteContainer(tb, containerPath, sourcePath, map[string]any{
		"kind":             stateKVTestKind,
		"state_store_path": sourcePath,
		"payload_bytes":    payloadBytes,
		"token_count":      bundle.TokenCount,
	})
	payloadOffset, payloadBytes := stateKVReadContainerPayloadWindow(tb, containerPath, payloadBytes)
	return stateKVContainerFixture{
		Context:       ctx,
		SourcePath:    sourcePath,
		ContainerPath: containerPath,
		Bundle:        bundle,
		PayloadOffset: payloadOffset,
		PayloadBytes:  payloadBytes,
	}
}

func (f stateKVContainerFixture) openRegion(tb testing.TB) *statefile.Store {
	tb.Helper()
	region, err := statefile.OpenRegionWithSegmentAlias(f.Context, f.ContainerPath, f.PayloadOffset, f.PayloadBytes, f.SourcePath)
	if err != nil {
		tb.Fatalf("OpenRegionWithSegmentAlias(container): %v", err)
	}
	return region
}

func stateKVWriteContainer(tb testing.TB, containerPath, sourcePath string, header map[string]any) {
	tb.Helper()
	payload := core.Open(sourcePath)
	if !payload.OK {
		tb.Fatalf("Open(source payload): %v", payload.Value)
	}
	payloadFile := payload.Value.(*core.OSFile)
	defer payloadFile.Close()
	output := core.OpenFile(containerPath, core.O_CREATE|core.O_TRUNC|core.O_WRONLY, 0o600)
	if !output.OK {
		tb.Fatalf("OpenFile(container): %v", output.Value)
	}
	outputFile := output.Value.(*core.OSFile)
	defer outputFile.Close()
	if _, err := trix.EncodeStream(header, stateKVTestMagic, payloadFile, outputFile); err != nil {
		tb.Fatalf("EncodeStream(container): %v", err)
	}
}

func stateKVReadContainerPayloadWindow(tb testing.TB, containerPath string, wantPayloadBytes int64) (int64, int64) {
	tb.Helper()
	input := core.Open(containerPath)
	if !input.OK {
		tb.Fatalf("Open(container): %v", input.Value)
	}
	file := input.Value.(*core.OSFile)
	defer file.Close()
	info, err := trix.ReadHeaderInfo(file, stateKVTestMagic)
	if err != nil {
		tb.Fatalf("ReadHeaderInfo(container): %v", err)
	}
	if kind, _ := info.Header["kind"].(string); kind != stateKVTestKind {
		tb.Fatalf("container kind = %q, want %q", kind, stateKVTestKind)
	}
	if info.PayloadBytes != wantPayloadBytes {
		tb.Fatalf("payload bytes = %d, want %d", info.PayloadBytes, wantPayloadBytes)
	}
	if info.PayloadOffset <= 0 {
		tb.Fatalf("payload offset = %d, want Trix payload offset", info.PayloadOffset)
	}
	return info.PayloadOffset, info.PayloadBytes
}

func stateKVFileSize(tb testing.TB, path string) int64 {
	tb.Helper()
	stat := core.Stat(path)
	if !stat.OK {
		tb.Fatalf("Stat(%s): %v", path, stat.Value)
	}
	return stat.Value.(core.FsFileInfo).Size()
}

func stateKVNativeLayerSlabSnapshot(tokenCount, heads, headDim int) *kv.Snapshot {
	tokens := make([]int32, tokenCount)
	B, H, L, D := 1, heads, tokenCount, headDim
	bytesPerValue := 2
	slabBytes := B * H * L * D * bytesPerValue
	keyBytes := make([]byte, slabBytes)
	valueBytes := make([]byte, slabBytes)
	for i := range tokenCount {
		tokens[i] = int32(i + 1)
	}
	for i := range keyBytes {
		keyBytes[i] = byte(i)
		valueBytes[i] = byte(i + 31)
	}
	return &kv.Snapshot{
		Version:       kv.SnapshotVersion,
		Architecture:  "gemma4_text",
		Tokens:        tokens,
		TokenOffset:   tokenCount,
		NumLayers:     1,
		NumHeads:      heads,
		SeqLen:        tokenCount,
		HeadDim:       headDim,
		NumQueryHeads: heads,
		Layers: []kv.LayerSnapshot{{
			Layer:      0,
			CacheIndex: 0,
			KeyDType:   "float16",
			KeyBytes:   keyBytes,
			KeyShape:   []int32{int32(B), int32(H), int32(L), int32(D)},
			ValueDType: "float16",
			ValueBytes: valueBytes,
			ValueShape: []int32{int32(B), int32(H), int32(L), int32(D)},
			Heads:      make([]kv.HeadSnapshot, heads),
		}},
	}
}
