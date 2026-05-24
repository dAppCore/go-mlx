// SPDX-Licence-Identifier: EUPL-1.2

package kv

import (
	"context"
	stdio "io"
	"math"
	"testing"

	core "dappco.re/go"
	state "dappco.re/go/inference/state"
	filestore "dappco.re/go/inference/state/filestore"
)

func TestKVSnapshotBlocks_Good_SplitAndAssemble(t *testing.T) {
	snapshot := kvSnapshotBlocksTestSnapshot()

	blocks, err := snapshot.SplitBlocks(2)
	if err != nil {
		t.Fatalf("SplitBlocks() error = %v", err)
	}
	if len(blocks) != 2 {
		t.Fatalf("blocks len = %d, want 2", len(blocks))
	}
	if blocks[0].Index != 0 || blocks[0].TokenStart != 0 || blocks[0].TokenCount != 2 {
		t.Fatalf("block[0] metadata = %+v", blocks[0])
	}
	if got := blocks[0].Snapshot.Tokens; len(got) != 2 || got[0] != 1 || got[1] != 2 {
		t.Fatalf("block[0] tokens = %v, want [1 2]", got)
	}
	if got := blocks[0].Snapshot.Layers[0].Heads[0].Key; len(got) != 4 || got[0] != 10 || got[3] != 13 {
		t.Fatalf("block[0] key = %v, want first token range", got)
	}
	if len(blocks[0].Snapshot.Logits) != 0 {
		t.Fatalf("block[0] logits = %v, want logits only on final block", blocks[0].Snapshot.Logits)
	}
	if got := blocks[1].Snapshot.Layers[0].Heads[0].Value; len(got) != 4 || got[0] != 24 || got[3] != 27 {
		t.Fatalf("block[1] value = %v, want second token range", got)
	}

	assembled, err := AssembleBlocks(blocks)
	if err != nil {
		t.Fatalf("AssembleBlocks() error = %v", err)
	}
	if assembled.SeqLen != snapshot.SeqLen || assembled.TokenOffset != snapshot.TokenOffset {
		t.Fatalf("assembled seq/offset = %d/%d, want %d/%d", assembled.SeqLen, assembled.TokenOffset, snapshot.SeqLen, snapshot.TokenOffset)
	}
	if len(assembled.Tokens) != 4 || assembled.Tokens[0] != 1 || assembled.Tokens[3] != 4 {
		t.Fatalf("assembled tokens = %v, want original tokens", assembled.Tokens)
	}
	head, ok := assembled.Head(0, 0)
	if !ok {
		t.Fatal("assembled Head(0,0) ok = false")
	}
	if len(head.Key) != 8 || head.Key[0] != 10 || head.Key[7] != 17 || head.Value[0] != 20 || head.Value[7] != 27 {
		t.Fatalf("assembled head = %+v, want original key/value", head)
	}
	if len(assembled.Logits) != 3 || assembled.Logits[2] != 0.7 {
		t.Fatalf("assembled logits = %v, want final logits", assembled.Logits)
	}
}

func TestKVSnapshotBlocks_Good_RangeBlocksStopsEarly(t *testing.T) {
	snapshot := kvSnapshotBlocksTestSnapshot()
	seen := []int{}

	err := snapshot.RangeBlocks(1, func(block Block) bool {
		seen = append(seen, block.Index)
		return len(seen) < 2
	})

	if err != nil {
		t.Fatalf("RangeBlocks() error = %v", err)
	}
	if len(seen) != 2 || seen[0] != 0 || seen[1] != 1 {
		t.Fatalf("seen blocks = %v, want [0 1]", seen)
	}
}

func TestKVSnapshotBlocks_Good_SplitsMixedHeadDims(t *testing.T) {
	snapshot := kvSnapshotBlocksTestSnapshot()
	snapshot.Layers[0].Heads[0].Key = []float32{
		10, 11, 12,
		13, 14, 15,
		16, 17, 18,
		19, 20, 21,
	}
	snapshot.Layers[0].Heads[0].Value = []float32{
		30,
		31,
		32,
		33,
	}

	blocks, err := snapshot.SplitBlocks(2)
	if err != nil {
		t.Fatalf("SplitBlocks() error = %v", err)
	}
	if got := blocks[0].Snapshot.Layers[0].Heads[0].Key; len(got) != 6 || got[0] != 10 || got[5] != 15 {
		t.Fatalf("block[0] mixed key = %v, want first two 3-wide tokens", got)
	}
	if got := blocks[1].Snapshot.Layers[0].Heads[0].Value; len(got) != 2 || got[0] != 32 || got[1] != 33 {
		t.Fatalf("block[1] mixed value = %v, want final two 1-wide tokens", got)
	}
}

func TestKVSnapshotBlocks_Good_SplitsLayerSuffixWindows(t *testing.T) {
	snapshot := kvSnapshotBlocksTestSnapshot()
	snapshot.Tokens = []int32{1, 2, 3, 4, 5}
	snapshot.TokenOffset = 5
	snapshot.SeqLen = 5
	snapshot.Layers[0].Heads[0].Key = []float32{10, 11, 12, 13, 14, 15, 16, 17, 18, 19}
	snapshot.Layers[0].Heads[0].Value = []float32{20, 21, 22, 23, 24, 25, 26, 27, 28, 29}
	snapshot.NumLayers = 2
	snapshot.Layers = append(snapshot.Layers, LayerSnapshot{
		Layer:      1,
		CacheIndex: 1,
		Heads: []HeadSnapshot{{
			Key:   []float32{100, 101, 102, 103},
			Value: []float32{200, 201, 202, 203},
		}},
	})

	blocks, err := snapshot.SplitBlocks(2)
	if err != nil {
		t.Fatalf("SplitBlocks() error = %v", err)
	}
	if len(blocks[0].Snapshot.Layers[1].Heads) != 0 {
		t.Fatalf("block[0] layer 1 heads = %d, want omitted before suffix window", len(blocks[0].Snapshot.Layers[1].Heads))
	}
	last := blocks[len(blocks)-1]
	if got := last.Snapshot.Layers[1].Heads[0].Key; len(got) != 2 || got[0] != 102 || got[1] != 103 {
		t.Fatalf("last block suffix key = %v, want final suffix token", got)
	}

	assembled, err := AssembleBlocks(blocks)
	if err != nil {
		t.Fatalf("AssembleBlocks() error = %v", err)
	}
	if assembled.SeqLen != 5 || len(assembled.Tokens) != 5 {
		t.Fatalf("assembled metadata = %+v, want global sequence retained", assembled)
	}
	head, ok := assembled.Head(1, 0)
	if !ok {
		t.Fatal("assembled Head(1,0) ok = false")
	}
	if len(head.Key) != 4 || head.Key[0] != 100 || head.Value[3] != 203 {
		t.Fatalf("assembled suffix head = %+v, want retained local cache", head)
	}
}

func TestKVSnapshotBlocks_Good_SplitAndAssembleNativeDType(t *testing.T) {
	snapshot := kvSnapshotBlocksTestSnapshot()
	head := &snapshot.Layers[0].Heads[0]
	head.KeyDType = "float16"
	head.ValueDType = "bfloat16"
	for _, value := range head.Key {
		head.KeyBytes = appendUint16LE(head.KeyBytes, float32ToFloat16(value))
	}
	for _, value := range head.Value {
		head.ValueBytes = appendUint16LE(head.ValueBytes, uint16(math.Float32bits(value)>>16))
	}

	blocks, err := snapshot.SplitBlocks(2)
	if err != nil {
		t.Fatalf("SplitBlocks() error = %v", err)
	}

	if got := len(blocks[0].Snapshot.Layers[0].Heads[0].KeyBytes); got != 8 {
		t.Fatalf("block[0] key bytes = %d, want two tokens x dim two x f16", got)
	}
	if blocks[0].Snapshot.Layers[0].Heads[0].KeyDType != "float16" {
		t.Fatalf("block[0] key dtype = %q, want float16", blocks[0].Snapshot.Layers[0].Heads[0].KeyDType)
	}
	assembled, err := AssembleBlocks(blocks)
	if err != nil {
		t.Fatalf("AssembleBlocks() error = %v", err)
	}
	assembledHead := assembled.Layers[0].Heads[0]
	if !equalBytes(assembledHead.KeyBytes, head.KeyBytes) || !equalBytes(assembledHead.ValueBytes, head.ValueBytes) {
		t.Fatalf("assembled native bytes = %d/%d, want original %d/%d", len(assembledHead.KeyBytes), len(assembledHead.ValueBytes), len(head.KeyBytes), len(head.ValueBytes))
	}
}

func TestKVSnapshotBlocks_Bad_RejectsInvalidHeadShape(t *testing.T) {
	snapshot := kvSnapshotBlocksTestSnapshot()
	snapshot.Layers[0].Heads[0].Key = snapshot.Layers[0].Heads[0].Key[:7]

	_, err := snapshot.SplitBlocks(2)

	if err == nil {
		t.Fatal("SplitBlocks() error = nil, want invalid head shape error")
	}
}

func TestKVSnapshotStateBlocks_Good_SaveLoadRoundTrip(t *testing.T) {
	store := state.NewInMemoryStore(nil)
	snapshot := kvSnapshotBlocksTestSnapshot()

	bundle, err := snapshot.SaveStateBlocks(context.Background(), store, StateBlockOptions{
		BlockSize:  2,
		KVEncoding: EncodingQ8,
		URI:        "mlx://session/blocks",
		Labels:     []string{"session-kv-block"},
	})
	if err != nil {
		t.Fatalf("SaveStateBlocks() error = %v", err)
	}
	if bundle.Kind != StateBlockBundleKind || len(bundle.Blocks) != 2 || bundle.BlockSize != 2 {
		t.Fatalf("bundle = %+v, want two State KV blocks", bundle)
	}
	if bundle.Blocks[0].State.ChunkID == bundle.Blocks[1].State.ChunkID {
		t.Fatalf("block refs = %+v, want distinct State chunks", bundle.Blocks)
	}
	if bundle.Blocks[0].PayloadEncoding != kvSnapshotStatePayloadRaw || bundle.Blocks[0].PayloadByteCount == 0 {
		t.Fatalf("block payload metadata = %+v, want raw binary payload", bundle.Blocks[0])
	}
	chunk, err := state.ResolveBytes(context.Background(), store, bundle.Blocks[0].State.ChunkID)
	if err != nil {
		t.Fatalf("ResolveBytes(block chunk) error = %v", err)
	}
	if len(chunk.Data) != bundle.Blocks[0].PayloadByteCount || core.Contains(chunk.Text, `"block_index":0`) {
		t.Fatalf("block chunk = text %q data %d, want raw binary payload", chunk.Text, len(chunk.Data))
	}

	loaded, err := LoadFromStateBlocks(context.Background(), store, bundle)
	if err != nil {
		t.Fatalf("LoadFromStateBlocks() error = %v", err)
	}
	if loaded.TokenOffset != snapshot.TokenOffset || len(loaded.Tokens) != len(snapshot.Tokens) {
		t.Fatalf("loaded metadata = %+v, want original token state", loaded)
	}
	head, ok := loaded.Head(0, 0)
	if !ok {
		t.Fatal("loaded Head(0,0) ok = false")
	}
	if len(head.Key) != 8 || head.Key[0] < 9.99 || head.Key[7] < 16.99 || head.Value[7] < 26.99 {
		t.Fatalf("loaded head = %+v, want original q8-ish values", head)
	}
}

func TestKVSnapshotStateBlocks_Good_TextStoreUsesEnvelopeFallback(t *testing.T) {
	store := &textOnlyStateStore{store: state.NewInMemoryStore(nil)}
	snapshot := kvSnapshotBlocksTestSnapshot()

	bundle, err := snapshot.SaveStateBlocks(context.Background(), store, StateBlockOptions{
		BlockSize:  2,
		KVEncoding: EncodingQ8,
		URI:        "mlx://session/text-blocks",
	})
	if err != nil {
		t.Fatalf("SaveStateBlocks(text store) error = %v", err)
	}
	if bundle.Blocks[0].PayloadEncoding != kvSnapshotStatePayloadJSONBase64 {
		t.Fatalf("payload encoding = %q, want JSON/base64 fallback", bundle.Blocks[0].PayloadEncoding)
	}
	chunk, err := state.Resolve(context.Background(), store, bundle.Blocks[0].State.ChunkID)
	if err != nil {
		t.Fatalf("Resolve(block chunk) error = %v", err)
	}
	if !core.Contains(chunk.Text, `"kind":"`+KVSnapshotStateBlockKind+`"`) || !core.Contains(chunk.Text, `"block_index":0`) {
		t.Fatalf("block chunk = %s, want block envelope", chunk.Text)
	}
	loaded, err := LoadFromStateBlocks(context.Background(), store, bundle)
	if err != nil {
		t.Fatalf("LoadFromStateBlocks(text store) error = %v", err)
	}
	if loaded.TokenOffset != snapshot.TokenOffset || len(loaded.Tokens) != len(snapshot.Tokens) {
		t.Fatalf("loaded metadata = %+v, want original token state", loaded)
	}
}

func TestKVSnapshotStateBlocks_Good_SaveNativeRawOnlyWithoutFloat32(t *testing.T) {
	store := state.NewInMemoryStore(nil)
	snapshot := kvSnapshotBlocksTestSnapshot()
	head := &snapshot.Layers[0].Heads[0]
	for _, value := range head.Key {
		head.KeyBytes = appendUint16LE(head.KeyBytes, float32ToFloat16(value))
	}
	for _, value := range head.Value {
		head.ValueBytes = appendUint16LE(head.ValueBytes, uint16(math.Float32bits(value)>>16))
	}
	head.Key = nil
	head.Value = nil
	head.KeyDType = "float16"
	head.ValueDType = "bfloat16"

	blocks, err := snapshot.SplitBlocks(2)
	if err != nil {
		t.Fatalf("SplitBlocks(native raw-only) error = %v", err)
	}
	if len(blocks) != 2 || blocks[0].Hash == "" {
		t.Fatalf("raw-only split blocks = %+v, want hashed streamed blocks", blocks)
	}

	bundle, err := snapshot.SaveStateBlocks(context.Background(), store, StateBlockOptions{
		BlockSize:  2,
		KVEncoding: EncodingNative,
	})
	if err != nil {
		t.Fatalf("SaveStateBlocks(native raw-only) error = %v", err)
	}
	loaded, err := LoadFromStateBlocksWithOptions(context.Background(), store, bundle, LoadOptions{RawKVOnly: true})
	if err != nil {
		t.Fatalf("LoadFromStateBlocksWithOptions(raw-only) error = %v", err)
	}
	loadedHead := loaded.Layers[0].Heads[0]
	if len(loadedHead.Key) != 0 || len(loadedHead.Value) != 0 {
		t.Fatalf("loaded float32 key/value lengths = %d/%d, want raw-only", len(loadedHead.Key), len(loadedHead.Value))
	}
	if loadedHead.KeyDType != "float16" || loadedHead.ValueDType != "bfloat16" {
		t.Fatalf("loaded dtypes = %q/%q, want float16/bfloat16", loadedHead.KeyDType, loadedHead.ValueDType)
	}
	if len(loadedHead.KeyBytes) != 16 || len(loadedHead.ValueBytes) != 16 {
		t.Fatalf("loaded raw bytes = %d/%d, want four tokens x dim two x two bytes", len(loadedHead.KeyBytes), len(loadedHead.ValueBytes))
	}
}

func TestKVSnapshotStateBlocks_Good_SaveNativeLayerRawOnlyWithoutHeadDuplication(t *testing.T) {
	store := state.NewInMemoryStore(nil)
	keyBytes := []byte{
		1, 0, 2, 0, 3, 0, 4, 0,
		5, 0, 6, 0, 7, 0, 8, 0,
	}
	valueBytes := []byte{
		11, 0, 12, 0, 13, 0, 14, 0,
		15, 0, 16, 0, 17, 0, 18, 0,
	}
	snapshot := &Snapshot{
		Version:       SnapshotVersion,
		Architecture:  "gemma4_text",
		Tokens:        []int32{1, 2, 3, 4},
		TokenOffset:   4,
		NumLayers:     1,
		NumHeads:      2,
		SeqLen:        4,
		HeadDim:       1,
		NumQueryHeads: 2,
		Layers: []LayerSnapshot{{
			Layer:      0,
			CacheIndex: 0,
			KeyDType:   "float16",
			KeyBytes:   keyBytes,
			KeyShape:   []int32{1, 2, 4, 1},
			ValueDType: "float16",
			ValueBytes: valueBytes,
			ValueShape: []int32{1, 2, 4, 1},
			Heads:      make([]HeadSnapshot, 2),
		}},
	}

	blocks, err := snapshot.SplitBlocks(2)
	if err != nil {
		t.Fatalf("SplitBlocks(native layer raw-only) error = %v", err)
	}
	if got := blocks[0].Snapshot.Layers[0].KeyBytes; !equalBytes(got, []byte{1, 0, 2, 0, 5, 0, 6, 0}) {
		t.Fatalf("block[0] layer key bytes = %v, want first two tokens for both heads", got)
	}
	bundle, err := snapshot.SaveStateBlocks(context.Background(), store, StateBlockOptions{
		BlockSize:  2,
		KVEncoding: EncodingNative,
	})
	if err != nil {
		t.Fatalf("SaveStateBlocks(native layer raw-only) error = %v", err)
	}
	loaded, err := LoadFromStateBlocksWithOptions(context.Background(), store, bundle, LoadOptions{RawKVOnly: true})
	if err != nil {
		t.Fatalf("LoadFromStateBlocksWithOptions(native layer raw-only) error = %v", err)
	}
	layer := loaded.Layers[0]
	if !equalBytes(layer.KeyBytes, keyBytes) || !equalBytes(layer.ValueBytes, valueBytes) {
		t.Fatalf("assembled layer bytes = %v/%v, want original slabs", layer.KeyBytes, layer.ValueBytes)
	}
	if len(layer.Heads) != 2 || len(layer.Heads[0].KeyBytes) != 0 {
		t.Fatalf("assembled heads = %+v, want no duplicated per-head bytes", layer.Heads)
	}
}

func TestKVSnapshotStateBlocks_Good_SaveNativeLayerSingleHeadRawOnly(t *testing.T) {
	store := state.NewInMemoryStore(nil)
	keyBytes := []byte{1, 0, 2, 0, 3, 0, 4, 0}
	valueBytes := []byte{11, 0, 12, 0, 13, 0, 14, 0}
	snapshot := &Snapshot{
		Version:       SnapshotVersion,
		Architecture:  "gemma4_text",
		Tokens:        []int32{1, 2, 3, 4},
		TokenOffset:   4,
		NumLayers:     1,
		NumHeads:      1,
		SeqLen:        4,
		HeadDim:       1,
		NumQueryHeads: 1,
		Layers: []LayerSnapshot{{
			Layer:      0,
			CacheIndex: 0,
			KeyDType:   "float16",
			KeyBytes:   keyBytes,
			KeyShape:   []int32{1, 1, 4, 1},
			ValueDType: "float16",
			ValueBytes: valueBytes,
			ValueShape: []int32{1, 1, 4, 1},
			Heads:      make([]HeadSnapshot, 1),
		}},
	}

	bundle, err := snapshot.SaveStateBlocks(context.Background(), store, StateBlockOptions{
		BlockSize:  2,
		KVEncoding: EncodingNative,
	})
	if err != nil {
		t.Fatalf("SaveStateBlocks(native single-head layer raw-only) error = %v", err)
	}
	loaded, err := LoadFromStateBlocksWithOptions(context.Background(), store, bundle, LoadOptions{RawKVOnly: true})
	if err != nil {
		t.Fatalf("LoadFromStateBlocksWithOptions(native single-head layer raw-only) error = %v", err)
	}
	layer := loaded.Layers[0]
	if !equalBytes(layer.KeyBytes, keyBytes) || !equalBytes(layer.ValueBytes, valueBytes) {
		t.Fatalf("assembled single-head layer bytes = %v/%v, want original slabs", layer.KeyBytes, layer.ValueBytes)
	}
	if len(layer.Heads) != 1 || len(layer.Heads[0].KeyBytes) != 0 {
		t.Fatalf("assembled heads = %+v, want no duplicated per-head bytes", layer.Heads)
	}
}

func TestKVSnapshotStateBlocks_Good_SaveNativeRawOnlyToFileStore(t *testing.T) {
	ctx := context.Background()
	path := core.PathJoin(t.TempDir(), "kv-blocks.mvlog")
	store, err := filestore.Create(ctx, path)
	if err != nil {
		t.Fatalf("filestore.Create() error = %v", err)
	}
	snapshot := kvSnapshotBlocksTestSnapshot()
	head := &snapshot.Layers[0].Heads[0]
	for _, value := range head.Key {
		head.KeyBytes = appendUint16LE(head.KeyBytes, float32ToFloat16(value))
	}
	for _, value := range head.Value {
		head.ValueBytes = appendUint16LE(head.ValueBytes, uint16(math.Float32bits(value)>>16))
	}
	head.Key = nil
	head.Value = nil
	head.KeyDType = "float16"
	head.ValueDType = "bfloat16"

	bundle, err := snapshot.SaveStateBlocks(ctx, store, StateBlockOptions{
		BlockSize:  2,
		KVEncoding: EncodingNative,
	})
	if err != nil {
		t.Fatalf("SaveStateBlocks(file native raw-only) error = %v", err)
	}
	if len(bundle.Blocks) != 2 || bundle.Blocks[0].State.Codec != filestore.CodecFile {
		t.Fatalf("bundle refs = %+v, want file-backed block refs", bundle.Blocks)
	}
	if bundle.Blocks[0].PayloadEncoding != kvSnapshotStatePayloadRaw || bundle.Blocks[0].PayloadByteCount == 0 {
		t.Fatalf("bundle payload = %+v, want raw file-backed payload", bundle.Blocks[0])
	}
	rawChunk, err := state.ResolveBytes(ctx, store, bundle.Blocks[0].State.ChunkID)
	if err != nil {
		t.Fatalf("ResolveBytes(file block) error = %v", err)
	}
	if len(rawChunk.Data) != bundle.Blocks[0].PayloadByteCount || core.Contains(rawChunk.Text, `"data"`) {
		t.Fatalf("raw file chunk = text %q data %d, want binary payload", rawChunk.Text, len(rawChunk.Data))
	}
	if err := store.Close(); err != nil {
		t.Fatalf("filestore.Close() error = %v", err)
	}
	if stat := core.Stat(path); !stat.OK || stat.Value.(core.FsFileInfo).Size() == 0 {
		t.Fatalf("file-backed store stat = %+v, want non-empty file", stat)
	}

	reopened, err := filestore.Open(ctx, path)
	if err != nil {
		t.Fatalf("filestore.Open() error = %v", err)
	}
	defer reopened.Close()
	loaded, err := LoadFromStateBlocksWithOptions(ctx, reopened, bundle, LoadOptions{RawKVOnly: true})
	if err != nil {
		t.Fatalf("LoadFromStateBlocksWithOptions(file raw-only) error = %v", err)
	}
	loadedHead := loaded.Layers[0].Heads[0]
	if len(loadedHead.Key) != 0 || len(loadedHead.Value) != 0 {
		t.Fatalf("loaded float32 key/value lengths = %d/%d, want raw-only", len(loadedHead.Key), len(loadedHead.Value))
	}
	if len(loadedHead.KeyBytes) != 16 || len(loadedHead.ValueBytes) != 16 {
		t.Fatalf("loaded raw bytes = %d/%d, want file-backed native bytes", len(loadedHead.KeyBytes), len(loadedHead.ValueBytes))
	}
}

func TestKVSnapshotStateBlocks_Good_LoadNativeRawOnlyFromRegionStore(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	sourcePath := core.PathJoin(dir, "kv-blocks.mvlog")
	containerPath := core.PathJoin(dir, "session.kv")
	store, err := filestore.Create(ctx, sourcePath)
	if err != nil {
		t.Fatalf("filestore.Create() error = %v", err)
	}
	snapshot := kvSnapshotBlocksTestSnapshot()
	head := &snapshot.Layers[0].Heads[0]
	for _, value := range head.Key {
		head.KeyBytes = appendUint16LE(head.KeyBytes, float32ToFloat16(value))
	}
	for _, value := range head.Value {
		head.ValueBytes = appendUint16LE(head.ValueBytes, uint16(math.Float32bits(value)>>16))
	}
	head.Key = nil
	head.Value = nil
	head.KeyDType = "float16"
	head.ValueDType = "bfloat16"

	bundle, err := snapshot.SaveStateBlocks(ctx, store, StateBlockOptions{
		BlockSize:  2,
		KVEncoding: EncodingNative,
	})
	if err != nil {
		t.Fatalf("SaveStateBlocks(region source) error = %v", err)
	}
	if err := store.Close(); err != nil {
		t.Fatalf("filestore.Close() error = %v", err)
	}
	read := core.ReadFile(sourcePath)
	if !read.OK {
		t.Fatalf("ReadFile(source) error = %s", read.Error())
	}
	prefix := []byte("KVST-region-head")
	payload := read.Value.([]byte)
	container := append(append(append([]byte(nil), prefix...), payload...), []byte("tail")...)
	if write := core.WriteFile(containerPath, container, 0o600); !write.OK {
		t.Fatalf("WriteFile(container) error = %s", write.Error())
	}

	region, err := filestore.OpenRegionWithSegmentAlias(ctx, containerPath, int64(len(prefix)), int64(len(payload)), sourcePath)
	if err != nil {
		t.Fatalf("OpenRegionWithSegmentAlias() error = %v", err)
	}
	defer region.Close()
	loaded, err := LoadFromStateBlocksWithOptions(ctx, region, bundle, LoadOptions{RawKVOnly: true})
	if err != nil {
		t.Fatalf("LoadFromStateBlocksWithOptions(region raw-only) error = %v", err)
	}
	loadedHead := loaded.Layers[0].Heads[0]
	if len(loadedHead.Key) != 0 || len(loadedHead.Value) != 0 {
		t.Fatalf("loaded region float32 key/value lengths = %d/%d, want raw-only", len(loadedHead.Key), len(loadedHead.Value))
	}
	if len(loadedHead.KeyBytes) != 16 || len(loadedHead.ValueBytes) != 16 {
		t.Fatalf("loaded region raw bytes = %d/%d, want file-backed native bytes", len(loadedHead.KeyBytes), len(loadedHead.ValueBytes))
	}
}

func TestKVSnapshotStateBlocks_Good_UsesStreamingBinaryWriter(t *testing.T) {
	store := &streamRecordingStateStore{store: state.NewInMemoryStore(nil)}
	snapshot := kvSnapshotBlocksTestSnapshot()

	bundle, err := snapshot.SaveStateBlocks(context.Background(), store, StateBlockOptions{
		BlockSize:  2,
		KVEncoding: EncodingNative,
	})
	if err != nil {
		t.Fatalf("SaveStateBlocks(streaming) error = %v", err)
	}
	if store.streamPuts != len(bundle.Blocks) || store.textPuts != 0 {
		t.Fatalf("writes = stream %d text %d for %d blocks, want streaming raw block writes", store.streamPuts, store.textPuts, len(bundle.Blocks))
	}
	if bundle.Blocks[0].PayloadEncoding != kvSnapshotStatePayloadRaw || bundle.Blocks[0].PayloadByteCount == 0 {
		t.Fatalf("block payload = %+v, want raw streamed payload", bundle.Blocks[0])
	}
	if len(store.streamOpts) != len(bundle.Blocks) {
		t.Fatalf("stream opts = %d, want one per block", len(store.streamOpts))
	}
	if _, ok := store.streamOpts[0].Tags["kv_hash"]; ok {
		t.Fatalf("stream metadata tags = %+v, want no blank kv_hash before payload is hashed", store.streamOpts[0].Tags)
	}
	if store.streamOpts[0].Tags["payload_encoding"] != kvSnapshotStatePayloadRaw {
		t.Fatalf("stream metadata payload_encoding = %q, want raw", store.streamOpts[0].Tags["payload_encoding"])
	}
	chunk, err := state.ResolveBytes(context.Background(), store, bundle.Blocks[0].State.ChunkID)
	if err != nil {
		t.Fatalf("ResolveBytes(streamed block) error = %v", err)
	}
	if len(chunk.Data) != bundle.Blocks[0].PayloadByteCount {
		t.Fatalf("streamed payload bytes = %d, want %d", len(chunk.Data), bundle.Blocks[0].PayloadByteCount)
	}
	loaded, err := LoadFromStateBlocksWithOptions(context.Background(), store, bundle, LoadOptions{RawKVOnly: true})
	if err != nil {
		t.Fatalf("LoadFromStateBlocksWithOptions(streaming) error = %v", err)
	}
	if len(loaded.Tokens) != len(snapshot.Tokens) || loaded.TokenOffset != snapshot.TokenOffset {
		t.Fatalf("loaded metadata = %+v, want original token state", loaded)
	}
}

func TestKVSnapshotStateBlocks_Good_SaveStreamInfersBundleMetadata(t *testing.T) {
	store := &streamRecordingStateStore{store: state.NewInMemoryStore(nil)}
	snapshot := kvSnapshotBlocksTestSnapshot()

	bundle, err := SaveStateBlocksFromStream(context.Background(), store, StateBlockOptions{
		BlockSize:  2,
		KVEncoding: EncodingNative,
		URI:        "mlx://streamed/session",
	}, func(yield func(Block) (bool, error)) error {
		return snapshot.walkBlocks(2, false, yield)
	})

	if err != nil {
		t.Fatalf("SaveStateBlocksFromStream() error = %v", err)
	}
	if bundle.Architecture != snapshot.Architecture || bundle.TokenCount != len(snapshot.Tokens) || bundle.TokenOffset != snapshot.TokenOffset {
		t.Fatalf("bundle metadata = %+v, want snapshot metadata", bundle)
	}
	if bundle.NumLayers != snapshot.NumLayers || bundle.NumHeads != snapshot.NumHeads || bundle.HeadDim != snapshot.HeadDim || bundle.SeqLen != snapshot.SeqLen {
		t.Fatalf("bundle shape = %+v, want snapshot shape", bundle)
	}
	if len(bundle.Blocks) != 2 || store.streamPuts != 2 {
		t.Fatalf("bundle blocks = %d stream writes = %d, want two streamed blocks", len(bundle.Blocks), store.streamPuts)
	}
	if bundle.SnapshotHash == "" {
		t.Fatal("bundle SnapshotHash is empty")
	}
	loaded, err := LoadFromStateBlocksWithOptions(context.Background(), store, bundle, LoadOptions{RawKVOnly: true})
	if err != nil {
		t.Fatalf("LoadFromStateBlocksWithOptions(stream bundle) error = %v", err)
	}
	if len(loaded.Tokens) != len(snapshot.Tokens) || loaded.TokenOffset != snapshot.TokenOffset {
		t.Fatalf("loaded metadata = %+v, want original token state", loaded)
	}
}

func TestKVSnapshotStateBlocks_Good_StreamReusesPrefixBlocks(t *testing.T) {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)
	parent := kvSnapshotBlocksTestSnapshot()
	parentBundle, err := parent.SaveStateBlocks(ctx, store, StateBlockOptions{
		BlockSize:  2,
		KVEncoding: EncodingNative,
		URI:        "mlx://parent",
	})
	if err != nil {
		t.Fatalf("SaveStateBlocks(parent) error = %v", err)
	}
	child := kvSnapshotBlocksTestSnapshot()
	child.Tokens[2] = 9
	child.Tokens[3] = 10
	child.Generated = []int32{10}
	child.Layers[0].Heads[0].Key[4] = 90
	child.Layers[0].Heads[0].Key[5] = 91
	child.Layers[0].Heads[0].Key[6] = 92
	child.Layers[0].Heads[0].Key[7] = 93
	child.Layers[0].Heads[0].Value[4] = 100
	child.Layers[0].Heads[0].Value[5] = 101
	child.Layers[0].Heads[0].Value[6] = 102
	child.Layers[0].Heads[0].Value[7] = 103

	childBundle, err := SaveStateBlocksFromStream(ctx, store, StateBlockOptions{
		BlockSize:         2,
		KVEncoding:        EncodingNative,
		URI:               "mlx://child",
		ReusePrefix:       parentBundle,
		ReusePrefixTokens: 2,
	}, func(yield func(Block) (bool, error)) error {
		return child.walkBlocks(2, false, yield)
	})
	if err != nil {
		t.Fatalf("SaveStateBlocksFromStream(child reuse) error = %v", err)
	}
	if childBundle.ReusedBlocks != 1 {
		t.Fatalf("child reused blocks = %d, want 1", childBundle.ReusedBlocks)
	}
	if childBundle.Blocks[0].State.ChunkID != parentBundle.Blocks[0].State.ChunkID {
		t.Fatalf("child first block ref = %+v, want parent first ref %+v", childBundle.Blocks[0], parentBundle.Blocks[0])
	}
	if childBundle.Blocks[1].State.ChunkID == parentBundle.Blocks[1].State.ChunkID {
		t.Fatalf("child second block reused parent ref %+v, want new suffix block", childBundle.Blocks[1])
	}
	loaded, err := LoadFromStateBlocksWithOptions(ctx, store, childBundle, LoadOptions{RawKVOnly: true})
	if err != nil {
		t.Fatalf("LoadFromStateBlocksWithOptions(child reuse) error = %v", err)
	}
	if len(loaded.Tokens) != 4 || loaded.Tokens[0] != 1 || loaded.Tokens[2] != 9 || loaded.Tokens[3] != 10 {
		t.Fatalf("loaded child tokens = %v, want reused prefix plus new suffix", loaded.Tokens)
	}
}

func TestKVSnapshotStateBlocks_Bad_SaveStreamErrors(t *testing.T) {
	snapshot := kvSnapshotBlocksTestSnapshot()
	store := &streamRecordingStateStore{store: state.NewInMemoryStore(nil)}
	if _, err := SaveStateBlocksFromStream(context.Background(), nil, StateBlockOptions{}, func(func(Block) (bool, error)) error {
		return nil
	}); err == nil {
		t.Fatal("SaveStateBlocksFromStream(nil store) error = nil")
	}
	if _, err := SaveStateBlocksFromStream(context.Background(), store, StateBlockOptions{}, nil); err == nil {
		t.Fatal("SaveStateBlocksFromStream(nil stream) error = nil")
	}
	if _, err := SaveStateBlocksFromStream(context.Background(), store, StateBlockOptions{}, func(func(Block) (bool, error)) error {
		return nil
	}); err == nil {
		t.Fatal("SaveStateBlocksFromStream(empty stream) error = nil")
	}
	if _, err := SaveStateBlocksFromStream(context.Background(), store, StateBlockOptions{}, func(yield func(Block) (bool, error)) error {
		_, err := yield(Block{Index: 0, TokenStart: 0, TokenCount: 1})
		return err
	}); err == nil {
		t.Fatal("SaveStateBlocksFromStream(nil block snapshot) error = nil")
	}

	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := SaveStateBlocksFromStream(cancelled, store, StateBlockOptions{}, func(yield func(Block) (bool, error)) error {
		return snapshot.walkBlocks(2, false, yield)
	}); err == nil {
		t.Fatal("SaveStateBlocksFromStream(cancelled context) error = nil")
	}

	writerStore := &failingStreamStateStore{}
	if _, err := SaveStateBlocksFromStream(context.Background(), writerStore, StateBlockOptions{}, func(yield func(Block) (bool, error)) error {
		return snapshot.walkBlocks(2, false, yield)
	}); err == nil {
		t.Fatal("SaveStateBlocksFromStream(writer failure) error = nil")
	}
}

func TestKVSnapshotStateBlocks_Bad_ValidationAndLoadErrors(t *testing.T) {
	if _, err := LoadFromStateBlocks(context.Background(), nil, &StateBlockBundle{}); err == nil {
		t.Fatal("LoadFromStateBlocks(nil store) error = nil")
	}
	if _, err := LoadFromStateBlocks(context.Background(), state.NewInMemoryStore(nil), nil); err == nil {
		t.Fatal("LoadFromStateBlocks(nil bundle) error = nil")
	}
	for _, bundle := range []*StateBlockBundle{
		{Version: StateBlockVersion + 1, Kind: StateBlockBundleKind, TokenCount: 1, Blocks: []StateBlockRef{{}}},
		{Version: StateBlockVersion, Kind: "wrong", TokenCount: 1, Blocks: []StateBlockRef{{}}},
		{Version: StateBlockVersion, Kind: StateBlockBundleKind, Blocks: []StateBlockRef{{}}},
		{Version: StateBlockVersion, Kind: StateBlockBundleKind, TokenCount: 1},
	} {
		if err := ValidateStateBlockBundle(bundle); err == nil {
			t.Fatalf("ValidateStateBlockBundle(%+v) error = nil", bundle)
		}
	}
	if err := ValidateStateBlockBundle(nil); err == nil {
		t.Fatal("ValidateStateBlockBundle(nil) error = nil")
	}
	if _, err := LoadPrefixFromStateBlocks(context.Background(), nil, &StateBlockBundle{}, 1); err == nil {
		t.Fatal("LoadPrefixFromStateBlocks(nil store) error = nil")
	}
}

func TestKVSnapshotStateBlocks_Bad_RawBlockIntegrity(t *testing.T) {
	store := state.NewInMemoryStore(nil)
	ref, err := store.PutBytes(context.Background(), []byte(kvSnapshotMagic), state.PutOptions{})
	if err != nil {
		t.Fatalf("PutBytes() error = %v", err)
	}
	blockRef := StateBlockRef{
		Index:            0,
		TokenStart:       0,
		TokenCount:       1,
		KVHash:           "not-the-hash",
		PayloadEncoding:  kvSnapshotStatePayloadRaw,
		PayloadByteCount: len(kvSnapshotMagic),
		State:            ref,
	}
	if _, err := loadRawKVSnapshotStateBlockWithOptions(context.Background(), store, blockRef, LoadOptions{}); err == nil {
		t.Fatal("loadRawKVSnapshotStateBlockWithOptions(hash mismatch) error = nil")
	}
	blockRef.KVHash = ""
	blockRef.PayloadByteCount++
	if _, err := loadRawKVSnapshotStateBlockWithOptions(context.Background(), store, blockRef, LoadOptions{}); err == nil {
		t.Fatal("loadRawKVSnapshotStateBlockWithOptions(length mismatch) error = nil")
	}
}

func TestKVSnapshotStateBlocks_Bad_EnvelopeIntegrity(t *testing.T) {
	for _, envelope := range []kvSnapshotStateBlockEnvelope{
		{Version: StateBlockVersion + 1, Kind: KVSnapshotStateBlockKind, BinaryEncoding: "base64"},
		{Version: StateBlockVersion, Kind: "wrong", BinaryEncoding: "base64"},
		{Version: StateBlockVersion, Kind: KVSnapshotStateBlockKind, BinaryEncoding: "hex"},
		{Version: StateBlockVersion, Kind: KVSnapshotStateBlockKind, BinaryEncoding: "base64", Data: "not base64"},
		{Version: StateBlockVersion, Kind: KVSnapshotStateBlockKind, BinaryEncoding: "base64", Data: core.Base64Encode([]byte("x")), PayloadByteCount: 2},
		{Version: StateBlockVersion, Kind: KVSnapshotStateBlockKind, BinaryEncoding: "base64", Data: core.Base64Encode([]byte("x")), KVHash: "bad"},
	} {
		if _, err := decodeKVSnapshotStateBlockEnvelope(envelope, ""); err == nil {
			t.Fatalf("decodeKVSnapshotStateBlockEnvelope(%+v) error = nil", envelope)
		}
	}
	data := []byte("x")
	envelope := kvSnapshotStateBlockEnvelope{
		Version:        StateBlockVersion,
		Kind:           KVSnapshotStateBlockKind,
		BinaryEncoding: "base64",
		Data:           core.Base64Encode(data),
	}
	if _, err := decodeKVSnapshotStateBlockEnvelope(envelope, "wrong-ref-hash"); err == nil {
		t.Fatal("decodeKVSnapshotStateBlockEnvelope(ref hash mismatch) error = nil")
	}
}

func TestKVSnapshotStateBlocks_Good_LoadPrefixOnlyReadsNeededBlocks(t *testing.T) {
	source := state.NewInMemoryStore(nil)
	snapshot := kvSnapshotBlocksTestSnapshot()
	bundle, err := snapshot.SaveStateBlocks(context.Background(), source, StateBlockOptions{BlockSize: 2})
	if err != nil {
		t.Fatalf("SaveStateBlocks() error = %v", err)
	}
	store := &recordingStateStore{store: source}

	loaded, err := LoadPrefixFromStateBlocks(context.Background(), store, bundle, 2)
	if err != nil {
		t.Fatalf("LoadPrefixFromStateBlocks() error = %v", err)
	}

	if len(store.resolved) != 1 || store.resolved[0] != bundle.Blocks[0].State.ChunkID {
		t.Fatalf("resolved chunks = %v, want only first block chunk %d", store.resolved, bundle.Blocks[0].State.ChunkID)
	}
	if loaded.TokenOffset != 2 || loaded.SeqLen != 2 || len(loaded.Tokens) != 2 || loaded.Tokens[0] != 1 || loaded.Tokens[1] != 2 {
		t.Fatalf("loaded prefix metadata = %+v, want first two tokens", loaded)
	}
	head, ok := loaded.Head(0, 0)
	if !ok {
		t.Fatal("loaded Head(0,0) ok = false")
	}
	if len(head.Key) != 4 || head.Key[0] < 9.99 || head.Key[3] < 12.99 {
		t.Fatalf("loaded prefix head = %+v, want first block key/value tensors", head)
	}
	if len(loaded.Logits) != 0 {
		t.Fatalf("loaded prefix logits = %v, want no logits for non-final prefix", loaded.Logits)
	}
}

func TestKVSnapshotStateBlocks_Good_LoadPartialPrefixSlicesCoveringBlock(t *testing.T) {
	source := state.NewInMemoryStore(nil)
	snapshot := kvSnapshotBlocksTestSnapshot()
	bundle, err := snapshot.SaveStateBlocks(context.Background(), source, StateBlockOptions{BlockSize: 2})
	if err != nil {
		t.Fatalf("SaveStateBlocks() error = %v", err)
	}

	loaded, err := LoadPrefixFromStateBlocks(context.Background(), source, bundle, 3)
	if err != nil {
		t.Fatalf("LoadPrefixFromStateBlocks() error = %v", err)
	}

	if loaded.TokenOffset != 3 || loaded.SeqLen != 3 || len(loaded.Tokens) != 3 || loaded.Tokens[2] != 3 {
		t.Fatalf("loaded prefix metadata = %+v, want first three tokens", loaded)
	}
	head, ok := loaded.Head(0, 0)
	if !ok {
		t.Fatal("loaded Head(0,0) ok = false")
	}
	if len(head.Key) != 6 || head.Key[0] < 9.99 || head.Key[5] < 14.99 {
		t.Fatalf("loaded prefix head = %+v, want sliced first three tokens", head)
	}
	if len(loaded.Logits) != 0 {
		t.Fatalf("loaded prefix logits = %v, want no logits for partial final block", loaded.Logits)
	}
}

func TestKVSnapshotStateBlocks_Good_LoadPrefixTokensSkipsKVAssembly(t *testing.T) {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)
	first := stateTokenOnlyTestSnapshot([]int32{1, 2}, 2, 2)
	second := stateTokenOnlyTestSnapshot([]int32{3, 4}, 4, 1)
	bundle, err := SaveStateBlocksFromStream(ctx, store, StateBlockOptions{
		BlockSize:  2,
		KVEncoding: EncodingNative,
	}, func(yield func(Block) (bool, error)) error {
		ok, err := yield(Block{Index: 0, TokenStart: 0, TokenCount: 2, Snapshot: first})
		if err != nil || !ok {
			return err
		}
		_, err = yield(Block{Index: 1, TokenStart: 2, TokenCount: 2, Snapshot: second})
		return err
	})
	if err != nil {
		t.Fatalf("SaveStateBlocksFromStream() error = %v", err)
	}

	if _, err := LoadPrefixFromStateBlocksWithOptions(ctx, store, bundle, 4, LoadOptions{RawKVOnly: true}); err == nil {
		t.Fatal("LoadPrefixFromStateBlocksWithOptions(mismatched shapes) error = nil")
	}
	tokens, err := LoadPrefixTokensFromStateBlocksWithOptions(ctx, store, bundle, 4, LoadOptions{RawKVOnly: true})
	if err != nil {
		t.Fatalf("LoadPrefixTokensFromStateBlocksWithOptions() error = %v", err)
	}
	if len(tokens) != 4 || tokens[0] != 1 || tokens[3] != 4 {
		t.Fatalf("tokens = %v, want [1 2 3 4]", tokens)
	}
}

type recordingStateStore struct {
	store    state.Store
	resolved []int
}

func (s *recordingStateStore) Get(ctx context.Context, chunkID int) (string, error) {
	s.resolved = append(s.resolved, chunkID)
	return s.store.Get(ctx, chunkID)
}

func (s *recordingStateStore) Resolve(ctx context.Context, chunkID int) (state.Chunk, error) {
	s.resolved = append(s.resolved, chunkID)
	return state.Resolve(ctx, s.store, chunkID)
}

type textOnlyStateStore struct {
	store *state.InMemoryStore
}

func (s *textOnlyStateStore) Get(ctx context.Context, chunkID int) (string, error) {
	return s.store.Get(ctx, chunkID)
}

func (s *textOnlyStateStore) Resolve(ctx context.Context, chunkID int) (state.Chunk, error) {
	return s.store.Resolve(ctx, chunkID)
}

func (s *textOnlyStateStore) ResolveURI(ctx context.Context, uri string) (state.Chunk, error) {
	return s.store.ResolveURI(ctx, uri)
}

func (s *textOnlyStateStore) Put(ctx context.Context, text string, opts state.PutOptions) (state.ChunkRef, error) {
	return s.store.Put(ctx, text, opts)
}

type streamRecordingStateStore struct {
	store      *state.InMemoryStore
	streamPuts int
	textPuts   int
	streamOpts []state.PutOptions
}

func (s *streamRecordingStateStore) Get(ctx context.Context, chunkID int) (string, error) {
	return s.store.Get(ctx, chunkID)
}

func (s *streamRecordingStateStore) Resolve(ctx context.Context, chunkID int) (state.Chunk, error) {
	return s.store.Resolve(ctx, chunkID)
}

func (s *streamRecordingStateStore) ResolveBytes(ctx context.Context, chunkID int) (state.Chunk, error) {
	return s.store.ResolveBytes(ctx, chunkID)
}

func (s *streamRecordingStateStore) Put(ctx context.Context, text string, opts state.PutOptions) (state.ChunkRef, error) {
	s.textPuts++
	return s.store.Put(ctx, text, opts)
}

func (s *streamRecordingStateStore) PutBytesStream(ctx context.Context, payloadSize int, opts state.PutOptions, write func(stdio.Writer) error) (state.ChunkRef, error) {
	s.streamPuts++
	s.streamOpts = append(s.streamOpts, opts)
	writer := &streamRecordingWriter{data: make([]byte, 0, payloadSize)}
	if err := write(writer); err != nil {
		return state.ChunkRef{}, err
	}
	if len(writer.data) != payloadSize {
		return state.ChunkRef{}, core.NewError("stream payload size mismatch")
	}
	return s.store.PutBytes(ctx, writer.data, opts)
}

type streamRecordingWriter struct {
	data []byte
}

func (w *streamRecordingWriter) Write(data []byte) (int, error) {
	w.data = append(w.data, data...)
	return len(data), nil
}

type failingStreamStateStore struct{}

func (s *failingStreamStateStore) Put(context.Context, string, state.PutOptions) (state.ChunkRef, error) {
	return state.ChunkRef{}, core.NewError("unexpected text write")
}

func (s *failingStreamStateStore) PutBytesStream(ctx context.Context, payloadSize int, opts state.PutOptions, write func(stdio.Writer) error) (state.ChunkRef, error) {
	err := write(failingStreamWriter{})
	if err == nil {
		err = core.NewError("expected writer failure")
	}
	return state.ChunkRef{}, err
}

type failingStreamWriter struct{}

func (failingStreamWriter) Write([]byte) (int, error) {
	return 0, core.NewError("stream writer failed")
}

func kvSnapshotBlocksTestSnapshot() *Snapshot {
	return &Snapshot{
		Version:       SnapshotVersion,
		Architecture:  "gemma4_text",
		Tokens:        []int32{1, 2, 3, 4},
		Generated:     []int32{4},
		TokenOffset:   4,
		NumLayers:     1,
		NumHeads:      1,
		SeqLen:        4,
		HeadDim:       2,
		NumQueryHeads: 1,
		LogitShape:    []int32{1, 1, 3},
		Logits:        []float32{0.1, 0.2, 0.7},
		Layers: []LayerSnapshot{{
			Layer:      0,
			CacheIndex: 0,
			Heads: []HeadSnapshot{{
				Key:   []float32{10, 11, 12, 13, 14, 15, 16, 17},
				Value: []float32{20, 21, 22, 23, 24, 25, 26, 27},
			}},
		}},
	}
}

func stateTokenOnlyTestSnapshot(tokens []int32, tokenOffset, headDim int) *Snapshot {
	key := make([]float32, len(tokens)*headDim)
	value := make([]float32, len(tokens)*headDim)
	for i := range key {
		key[i] = float32(i + tokenOffset)
		value[i] = float32(i + tokenOffset + 100)
	}
	return &Snapshot{
		Version:       SnapshotVersion,
		Architecture:  "gemma4_text",
		Tokens:        append([]int32(nil), tokens...),
		TokenOffset:   tokenOffset,
		NumLayers:     1,
		NumHeads:      1,
		SeqLen:        len(tokens),
		HeadDim:       headDim,
		NumQueryHeads: 1,
		Layers: []LayerSnapshot{{
			Layer:      0,
			CacheIndex: 0,
			Heads: []HeadSnapshot{{
				Key:   key,
				Value: value,
			}},
		}},
	}
}
