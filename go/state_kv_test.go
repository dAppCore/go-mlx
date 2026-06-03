// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"testing"

	core "dappco.re/go"
	statefile "dappco.re/go/inference/state/filestore"
	"dappco.re/go/mlx/kv"
	"dappco.re/go/mlx/pkg/metal"
	trix "forge.lthn.ai/Snider/Enchantrix/pkg/trix"
)

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
	coverageTokens := "StateKVRegion BlockSourceLoadsWithoutOriginalMVLog"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	fixture := newStateKVContainerFixture(t, 512, 128)
	if result := core.Remove(fixture.SourcePath); !result.OK {
		t.Fatalf("remove source State log: %v", result.Value)
	}
	region := fixture.openRegion(t)
	defer region.Close()
	source, err := metalKVSnapshotBlockSource(fixture.Context, region, fixture.Bundle, fixture.Bundle.TokenCount)
	if err != nil {
		t.Fatalf("metalKVSnapshotBlockSource(region) error = %v", err)
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
	source, err := metalKVSnapshotBlockSource(fixture.Context, region, fixture.Bundle, fixture.Bundle.TokenCount)
	if err != nil {
		b.Fatalf("metalKVSnapshotBlockSource(region): %v", err)
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
	source, err := metalKVSnapshotBlockSource(fixture.Context, store, fixture.Bundle, fixture.Bundle.TokenCount)
	if err != nil {
		b.Fatalf("metalKVSnapshotBlockSource(source): %v", err)
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
