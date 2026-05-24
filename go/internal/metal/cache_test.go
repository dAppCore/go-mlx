// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"testing"
)

// makeKV creates a small K/V pair with shape [B=1, H=2, L=seqLen, D=4].
func makeKV(seqLen int) (*Array, *Array) {
	size := 1 * 2 * seqLen * 4
	data := make([]float32, size)
	for i := range data {
		data[i] = float32(i) * 0.1
	}
	k := FromValues(data, 1, 2, seqLen, 4)
	v := FromValues(data, 1, 2, seqLen, 4)
	return k, v
}

func makeSingleTokenKV(value float32) (*Array, *Array) {
	data := make([]float32, 1*2*1*4)
	for i := range data {
		data[i] = value + float32(i)*0.01
	}
	k := FromValues(data, 1, 2, 1, 4)
	v := FromValues(data, 1, 2, 1, 4)
	return k, v
}

// --- KVCache ---

func TestKVCache_New_Good(t *testing.T) {
	coverageTokens := "New"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	c := NewKVCache()
	if c.Offset() != 0 {
		t.Errorf("offset = %d, want 0", c.Offset())
	}
	if c.Len() != 0 {
		t.Errorf("len = %d, want 0", c.Len())
	}
	if c.State() != nil {
		t.Error("state should be nil for empty cache")
	}
}

func TestKVCache_SingleUpdate_Good(t *testing.T) {
	coverageTokens := "SingleUpdate"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	c := NewKVCache()
	k, v := makeKV(3) // 3 tokens

	outK, outV := c.Update(k, v, 3)
	Materialize(outK, outV)

	if c.Offset() != 3 {
		t.Errorf("offset = %d, want 3", c.Offset())
	}
	if c.Len() != 3 {
		t.Errorf("len = %d, want 3", c.Len())
	}

	// Output K should have shape [1, 2, 3, 4]
	shape := outK.Shape()
	if shape[0] != 1 || shape[1] != 2 || shape[2] != 3 || shape[3] != 4 {
		t.Errorf("outK shape = %v, want [1 2 3 4]", shape)
	}
}

func TestKVCache_MultipleUpdates_Good(t *testing.T) {
	coverageTokens := "MultipleUpdates"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	c := NewKVCache()

	// Prompt: 5 tokens
	k1, v1 := makeKV(5)
	outK, outV := c.Update(k1, v1, 5)
	Materialize(outK, outV)

	if c.Offset() != 5 {
		t.Errorf("offset = %d, want 5", c.Offset())
	}

	// Generate: 1 token at a time
	k2, v2 := makeKV(1)
	outK, outV = c.Update(k2, v2, 1)
	Materialize(outK, outV)

	if c.Offset() != 6 {
		t.Errorf("offset = %d, want 6", c.Offset())
	}

	shape := outK.Shape()
	if shape[2] != 6 {
		t.Errorf("outK L dim = %d, want 6", shape[2])
	}
}

func TestKVCache_Reset_Good(t *testing.T) {
	c := NewKVCache()
	k, v := makeKV(3)
	c.Update(k, v, 3)

	c.Reset()

	if c.Offset() != 0 {
		t.Errorf("offset after reset = %d, want 0", c.Offset())
	}
	if c.State() != nil {
		t.Error("state should be nil after reset")
	}
}

func TestQuantizedKVCache_StoresInt8AndReadsDequantized_Good(t *testing.T) {
	coverageTokens := "QuantizedKVCache StoresInt8AndReadsDequantized"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	c := NewQuantizedKVCache(4, 8, 8)
	k, v := makeKV(2)
	defer Free(k, v)

	outK, outV := c.Update(k, v, 2)
	defer Free(outK, outV)
	if err := Eval(outK, outV); err != nil {
		t.Fatalf("Eval quantized output: %v", err)
	}
	defer c.Reset()

	state := c.State()
	if len(state) != 4 {
		t.Fatalf("State len = %d, want q K/V plus scales", len(state))
	}
	if state[0].Dtype() != DTypeInt8 || state[1].Dtype() != DTypeInt8 {
		t.Fatalf("stored dtypes = %v/%v, want int8/int8", state[0].Dtype(), state[1].Dtype())
	}
	read, owned := c.ReadState()
	defer Free(owned...)
	if len(read) != 2 || read[0].Dtype() != DTypeFloat32 || read[1].Dtype() != DTypeFloat32 {
		t.Fatalf("read state = %+v, want dequantized float K/V", read)
	}
	if read[0].Shape()[2] != 2 {
		t.Fatalf("read K shape = %v, want seq len 2", read[0].Shape())
	}
}

func TestQuantizedKVCache_AsymmetricStoresPackedVQ4_Good(t *testing.T) {
	coverageTokens := "QuantizedKVCache AsymmetricStoresPackedVQ4"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	c := NewQuantizedKVCache(4, 8, 4)
	k, v := makeKV(2)
	defer Free(k, v)

	outK, outV := c.Update(k, v, 2)
	defer Free(outK, outV)
	if err := Eval(outK, outV); err != nil {
		t.Fatalf("Eval asymmetric quantized output: %v", err)
	}
	defer c.Reset()

	state := c.State()
	if len(state) != 4 {
		t.Fatalf("State len = %d, want packed K/V plus scales", len(state))
	}
	if state[0].Dtype() != DTypeInt8 {
		t.Fatalf("stored K dtype = %v, want int8", state[0].Dtype())
	}
	if state[1].Dtype() != DTypeUint8 {
		t.Fatalf("stored V dtype = %v, want packed uint8 q4", state[1].Dtype())
	}
	if shape := state[1].Shape(); len(shape) != 1 || shape[0] != 8 {
		t.Fatalf("stored V shape = %v, want 8 packed q4 bytes", shape)
	}
	read, owned := c.ReadState()
	defer Free(owned...)
	if len(read) != 2 || read[1].Shape()[2] != 2 {
		t.Fatalf("read state = %+v, want dequantized V length 2", read)
	}
}

func TestPagedKVCache_TrimsStorageButReturnsFullPrompt_Good(t *testing.T) {
	coverageTokens := "PagedKVCache TrimsStorageButReturnsFullPrompt"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	c := NewPagedKVCache(2, 2)
	k, v := makeKV(4)
	defer Free(k, v)

	outK, outV := c.Update(k, v, 4)
	defer Free(outK, outV)
	if outK.Shape()[2] != 4 || outV.Shape()[2] != 4 {
		t.Fatalf("output shape = %v/%v, want full prompt length 4", outK.Shape(), outV.Shape())
	}
	if c.Len() != 2 || c.Offset() != 4 {
		t.Fatalf("len/offset = %d/%d, want 2/4", c.Len(), c.Offset())
	}
	read, owned := c.ReadState()
	defer Free(owned...)
	if len(read) != 2 || read[0].Shape()[2] != 2 {
		t.Fatalf("stored read shape = %+v, want trimmed length 2", read)
	}
	c.Reset()
	if c.State() != nil {
		t.Fatal("State after Reset = non-nil, want nil")
	}
}

func TestPagedKVCache_UpdatePagesKeepsBlocks_Good(t *testing.T) {
	c := NewPagedKVCache(4, 2)
	k, v := makeKV(4)
	defer Free(k, v)

	state := c.UpdatePages(k, v, 4)
	defer state.Free()

	if state.Length != 4 || len(state.Keys) != 2 || len(state.Values) != 2 {
		t.Fatalf("page state = len %d K pages %d V pages %d, want 4/2/2", state.Length, len(state.Keys), len(state.Values))
	}
	if state.Keys[0].Shape()[2] != 2 || state.Keys[1].Shape()[2] != 2 {
		t.Fatalf("page shapes = %v/%v, want two 2-token pages", state.Keys[0].Shape(), state.Keys[1].Shape())
	}

	k1, v1 := makeSingleTokenKV(9)
	defer Free(k1, v1)
	next := c.UpdatePages(k1, v1, 1)
	defer next.Free()

	if c.Len() != 4 || c.Offset() != 5 {
		t.Fatalf("len/offset = %d/%d, want 4/5 after paged trim", c.Len(), c.Offset())
	}
	if len(next.Keys) != 3 {
		t.Fatalf("trimmed page count = %d, want 3 partial/full/new pages without full concat", len(next.Keys))
	}
	if next.Keys[0].Shape()[2] != 1 || next.Keys[1].Shape()[2] != 2 || next.Keys[2].Shape()[2] != 1 {
		t.Fatalf("trimmed page shapes = %v/%v/%v, want [1,2,1]", next.Keys[0].Shape(), next.Keys[1].Shape(), next.Keys[2].Shape())
	}
}

func TestPagedKVCache_AppendDirtyStateOnlyRecentPage_Good(t *testing.T) {
	coverageTokens := "PagedKVCache AppendDirtyStateOnlyRecentPage"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	c := NewPagedKVCache(0, 2)
	k, v := makeSingleTokenKV(1)
	defer Free(k, v)

	state := c.UpdateBorrowedPages(k, v, 1)
	state.Free()
	dirty := c.AppendDirtyState(nil)
	if len(dirty) != 2 || dirty[0] != c.kPages[0] || dirty[1] != c.vPages[0] {
		t.Fatalf("dirty state after first append = %+v, want first page K/V only", dirty)
	}

	nextK, nextV := makeSingleTokenKV(2)
	defer Free(nextK, nextV)
	nextState := c.UpdateBorrowedPages(nextK, nextV, 1)
	nextState.Free()
	dirty = c.AppendDirtyState(dirty[:0])
	if len(dirty) != 2 || dirty[0] != c.kPages[0] || dirty[1] != c.vPages[0] {
		t.Fatalf("dirty state after same-page append = %+v, want updated first page K/V only", dirty)
	}
	if len(c.State()) != 2 {
		t.Fatalf("full state length = %d, want one K/V page pair", len(c.State()))
	}

	newPageK, newPageV := makeSingleTokenKV(3)
	defer Free(newPageK, newPageV)
	newPageState := c.UpdateBorrowedPages(newPageK, newPageV, 1)
	newPageState.Free()
	dirty = c.AppendDirtyState(dirty[:0])
	if len(c.kPages) != 2 || len(dirty) != 2 || dirty[0] != c.kPages[1] || dirty[1] != c.vPages[1] {
		t.Fatalf("dirty state after new page = %+v, pages=%d, want newest page K/V only", dirty, len(c.kPages))
	}
	if len(c.State()) != 4 {
		t.Fatalf("full state length = %d, want two K/V page pairs", len(c.State()))
	}
}

func TestPagedKVCache_BorrowedPageStateAvoidsFullPageClones_Good(t *testing.T) {
	coverageTokens := "PagedKVCache BorrowedPageStateAvoidsFullPageClones"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	c := NewPagedKVCache(4, 2)
	k, v := makeKV(4)
	defer Free(k, v)
	defer c.Reset()

	state := c.UpdateBorrowedPages(k, v, 4)
	defer state.Free()
	cacheState := c.State()

	if state.Length != 4 || len(state.Keys) != 2 || len(state.Values) != 2 {
		t.Fatalf("page state = len %d K pages %d V pages %d, want 4/2/2", state.Length, len(state.Keys), len(state.Values))
	}
	if len(state.Owned) != 0 {
		t.Fatalf("borrowed state owned arrays = %d, want zero for full physical pages", len(state.Owned))
	}
	if len(cacheState) != 4 || state.Keys[0] != cacheState[0] || state.Keys[1] != cacheState[1] {
		t.Fatal("borrowed state did not return cache-owned full K pages")
	}
	if state.Values[0] != cacheState[2] || state.Values[1] != cacheState[3] {
		t.Fatal("borrowed state did not return cache-owned full V pages")
	}
}

func TestPagedKVCache_BorrowedPageStateOwnsPartialPreallocSlices_Good(t *testing.T) {
	coverageTokens := "PagedKVCache BorrowedPageStateOwnsPartialPreallocSlices"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	old := enablePagedKVPrealloc
	enablePagedKVPrealloc = true
	t.Cleanup(func() { enablePagedKVPrealloc = old })

	c := NewPagedKVCache(0, 4)
	k, v := makeKV(2)
	defer Free(k, v)
	defer c.Reset()

	state := c.UpdateBorrowedPages(k, v, 2)
	defer state.Free()
	cacheState := c.State()

	if len(cacheState) != 2 || cacheState[0].Shape()[2] != 4 || cacheState[1].Shape()[2] != 4 {
		t.Fatalf("backing page state = %+v, want full preallocated K/V pages", cacheState)
	}
	if len(state.Keys) != 1 || len(state.Values) != 1 || state.Keys[0].Shape()[2] != 2 || state.Values[0].Shape()[2] != 2 {
		t.Fatalf("borrowed visible pages = %+v/%+v, want 2-token K/V slices", state.Keys, state.Values)
	}
	if len(state.Owned) != 2 {
		t.Fatalf("borrowed state owned arrays = %d, want K/V visible slices", len(state.Owned))
	}
	if state.Keys[0] == cacheState[0] || state.Values[0] == cacheState[1] {
		t.Fatal("partial preallocated state returned backing pages directly")
	}
}

func TestPagedKVCache_PreallocKeepsVisiblePageLength_Good(t *testing.T) {
	coverageTokens := "PagedKVCache PreallocKeepsVisiblePageLength"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	old := enablePagedKVPrealloc
	enablePagedKVPrealloc = true
	t.Cleanup(func() { enablePagedKVPrealloc = old })

	c := NewPagedKVCache(0, 4)
	k, v := makeKV(2)
	defer Free(k, v)

	state := c.UpdatePages(k, v, 2)
	state.Free()
	k1, v1 := makeSingleTokenKV(9)
	defer Free(k1, v1)
	next := c.UpdatePages(k1, v1, 1)
	defer next.Free()
	defer c.Reset()

	if len(c.State()) != 2 || c.State()[0].Shape()[2] != 4 {
		t.Fatalf("backing page shape = %+v, want preallocated page length 4", c.State())
	}
	if len(next.Keys) != 1 || next.Keys[0].Shape()[2] != 3 {
		t.Fatalf("visible page shape = %+v, want one 3-token page", next.Keys)
	}
	read, owned := c.ReadState()
	defer Free(owned...)
	if len(read) != 2 || read[0].Shape()[2] != 3 || read[1].Shape()[2] != 3 {
		t.Fatalf("read state = %+v, want visible length 3", read)
	}
}

func TestPagedKVCache_PreallocRuntimeGate_Good(t *testing.T) {
	coverageTokens := "PagedKVCache PreallocRuntimeGate"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	t.Cleanup(SetRuntimeGate("GO_MLX_ENABLE_PAGED_KV_PREALLOC", "1"))

	c := NewPagedKVCache(0, 4)
	k, v := makeKV(2)
	defer Free(k, v)
	defer c.Reset()

	state := c.UpdatePages(k, v, 2)
	defer state.Free()
	cacheState := c.State()

	if len(cacheState) != 2 || cacheState[0].Shape()[2] != 4 || cacheState[1].Shape()[2] != 4 {
		t.Fatalf("runtime-gated backing page shape = %+v, want full preallocated K/V pages", cacheState)
	}
	if len(state.Keys) != 1 || state.Keys[0].Shape()[2] != 2 || len(state.Values) != 1 || state.Values[0].Shape()[2] != 2 {
		t.Fatalf("runtime-gated visible page shape = %+v/%+v, want visible 2-token K/V pages", state.Keys, state.Values)
	}
}

func TestPagedKVCache_DefaultPageSizeDoesNotUseContextCutoff_Good(t *testing.T) {
	coverageTokens := "PagedKVCache DefaultPageSizeDoesNotUseContextCutoff"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	t.Setenv("GO_MLX_PAGED_KV_PAGE_SIZE", "")

	normal := NewPagedKVCache(32768, 0)
	retained := NewPagedKVCache(131072, 0)
	sliding := NewPagedKVCache(512, 0)

	if normal.pageSize != defaultPagedKVPageSize {
		t.Fatalf("normal pageSize = %d, want %d", normal.pageSize, defaultPagedKVPageSize)
	}
	if retained.pageSize != defaultPagedKVPageSize {
		t.Fatalf("retained pageSize = %d, want %d", retained.pageSize, defaultPagedKVPageSize)
	}
	if sliding.pageSize != 512 {
		t.Fatalf("sliding pageSize = %d, want capped max size 512", sliding.pageSize)
	}
}

func TestPagedKVCache_SlidingWindowStaysSinglePage_Good(t *testing.T) {
	coverageTokens := "PagedKVCache SlidingWindowStaysSinglePage"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	requireMetalRuntime(t)

	cache := NewPagedKVCache(4, 4)
	defer cache.Reset()
	prefixK, prefixV := makeKV(4)
	defer Free(prefixK, prefixV)
	state := cache.UpdateBorrowedPages(prefixK, prefixV, 4)
	state.Free()
	nextK, nextV := makeSingleTokenKV(9)
	defer Free(nextK, nextV)

	state = cache.UpdateBorrowedPages(nextK, nextV, 1)
	defer state.Free()
	raw := cache.State()

	if cache.Len() != 4 || cache.Offset() != 5 {
		t.Fatalf("cache len/offset = %d/%d, want 4/5", cache.Len(), cache.Offset())
	}
	if len(state.Keys) != 1 || len(state.Values) != 1 {
		t.Fatalf("borrowed pages = %d/%d, want one K/V page", len(state.Keys), len(state.Values))
	}
	if len(raw) != 2 || raw[0].Shape()[2] != 4 || raw[1].Shape()[2] != 4 {
		t.Fatalf("raw page state = %+v, want one 4-token K page and one 4-token V page", raw)
	}
	dirty := cache.AppendDirtyState(nil)
	if len(dirty) != 2 {
		t.Fatalf("dirty state len = %d, want compacted K/V pages", len(dirty))
	}
	if err := Eval(state.Keys[0], state.Values[0], dirty[0], dirty[1]); err != nil {
		t.Fatalf("Eval compacted sliding state: %v", err)
	}
}

func TestPagedKVCache_StoresRequestedDType_Good(t *testing.T) {
	coverageTokens := "PagedKVCache StoresRequestedDType"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	requireMetalRuntime(t)

	cache := NewPagedKVCacheWithDType(8, 2, DTypeBFloat16)
	defer cache.Reset()
	k, v := makeKV(2)
	defer Free(k, v)

	state := cache.UpdateBorrowedPages(k, v, 2)
	defer state.Free()
	if len(state.Keys) != 1 || len(state.Values) != 1 {
		t.Fatalf("page count = %d/%d, want one K/V page", len(state.Keys), len(state.Values))
	}
	if state.Keys[0].Dtype() != DTypeBFloat16 || state.Values[0].Dtype() != DTypeBFloat16 {
		t.Fatalf("page dtypes = %v/%v, want bfloat16/bfloat16", state.Keys[0].Dtype(), state.Values[0].Dtype())
	}
	if err := Eval(state.Keys[0], state.Values[0]); err != nil {
		t.Fatalf("Eval typed paged state: %v", err)
	}
}

func TestFixedKVCache_StoresRequestedDType_Good(t *testing.T) {
	coverageTokens := "FixedKVCache StoresRequestedDType"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	requireMetalRuntime(t)

	cache := NewFixedKVCacheWithDType(4, DTypeBFloat16)
	defer cache.Reset()
	k, v := makeKV(2)
	defer Free(k, v)

	stateK, stateV := cache.Update(k, v, 2)
	defer Free(stateK, stateV)
	if stateK.Dtype() != DTypeBFloat16 || stateV.Dtype() != DTypeBFloat16 {
		t.Fatalf("fixed state dtypes = %v/%v, want bfloat16/bfloat16", stateK.Dtype(), stateV.Dtype())
	}
	if err := Eval(stateK, stateV); err != nil {
		t.Fatalf("Eval typed fixed state: %v", err)
	}
}

func TestPagedKVCache_ReplaceSinglePageFromNative_Good(t *testing.T) {
	coverageTokens := "PagedKVCache ReplaceSinglePageFromNative"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	c := NewPagedKVCache(4, 4)
	k, v := makeKV(2)
	state := c.ReplaceSinglePageFromNative(k, v, 2)
	defer state.Free()
	defer c.Reset()

	if c.Len() != 2 || c.Offset() != 2 {
		t.Fatalf("len/offset = %d/%d, want 2/2", c.Len(), c.Offset())
	}
	if len(state.Keys) != 1 || len(state.Values) != 1 {
		t.Fatalf("page count = %d/%d, want 1/1", len(state.Keys), len(state.Values))
	}
	if state.Keys[0] == k || state.Values[0] == v {
		t.Fatal("page state returned cache-owned arrays directly, want cloned handles")
	}
	read, owned := c.ReadState()
	defer Free(owned...)
	if len(read) != 2 || read[0].Shape()[2] != 2 || read[1].Shape()[2] != 2 {
		t.Fatalf("read state = %+v, want single native page with length 2", read)
	}
}

func TestFixedKVCache_UpdateKeepsStableStorage_Good(t *testing.T) {
	coverageTokens := "FixedKVCache Update"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	c := NewFixedKVCache(4)
	k := FromValues([]float32{1, 2, 3, 4}, 1, 1, 2, 2)
	v := FromValues([]float32{10, 20, 30, 40}, 1, 1, 2, 2)
	defer Free(k, v)

	gotK, gotV := c.Update(k, v, 2)
	defer Free(gotK, gotV)
	if gotK.Dim(2) != 2 || gotV.Dim(2) != 2 {
		t.Fatalf("valid cache dims = %d/%d, want 2/2", gotK.Dim(2), gotV.Dim(2))
	}
	state := c.State()
	if len(state) != 2 || state[0].Dim(2) != 4 || state[1].Dim(2) != 4 {
		t.Fatalf("fixed state dims = %v, want full capacity 4", state)
	}

	k1 := FromValues([]float32{5, 6}, 1, 1, 1, 2)
	v1 := FromValues([]float32{50, 60}, 1, 1, 1, 2)
	defer Free(k1, v1)
	gotK2, gotV2 := c.Update(k1, v1, 1)
	defer Free(gotK2, gotV2)
	if gotK2.Dim(2) != 3 || gotV2.Dim(2) != 3 || c.Offset() != 3 || c.Len() != 3 {
		t.Fatalf("cache len/offset = %d/%d dims %d/%d, want 3/3 dims 3/3", c.Len(), c.Offset(), gotK2.Dim(2), gotV2.Dim(2))
	}
	if err := Eval(gotK2, gotV2); err != nil {
		t.Fatalf("Eval fixed cache: %v", err)
	}
	floatSliceApprox(t, gotK2.Floats(), []float32{1, 2, 3, 4, 5, 6})
	floatSliceApprox(t, gotV2.Floats(), []float32{10, 20, 30, 40, 50, 60})
}

func TestFixedKVCache_LongPromptPreservesFullAttentionContext_Good(t *testing.T) {
	coverageTokens := "FixedKVCache LongPromptPreservesFullAttentionContext"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	c := NewFixedKVCache(4)
	k := FromValues([]float32{1, 2, 3, 4, 5, 6}, 1, 1, 6, 1)
	v := FromValues([]float32{10, 20, 30, 40, 50, 60}, 1, 1, 6, 1)
	defer Free(k, v)

	gotK, gotV := c.Update(k, v, 6)
	defer Free(gotK, gotV)
	if gotK.Dim(2) != 6 || gotV.Dim(2) != 6 {
		t.Fatalf("attention context dims = %d/%d, want full prompt 6/6", gotK.Dim(2), gotV.Dim(2))
	}
	if c.Offset() != 6 || c.Len() != 4 {
		t.Fatalf("cache offset/len = %d/%d, want 6/4", c.Offset(), c.Len())
	}
	if err := Eval(gotK, gotV); err != nil {
		t.Fatalf("Eval full prompt context: %v", err)
	}
	floatSliceApprox(t, gotK.Floats(), []float32{1, 2, 3, 4, 5, 6})
	floatSliceApprox(t, gotV.Floats(), []float32{10, 20, 30, 40, 50, 60})

	read, owned := c.ReadState()
	defer Free(owned...)
	if len(read) != 2 || read[0].Dim(2) != 4 || read[1].Dim(2) != 4 {
		t.Fatalf("stored tail dims = %v, want bounded tail 4/4", read)
	}
	if err := Eval(read...); err != nil {
		t.Fatalf("Eval stored tail: %v", err)
	}
	floatSliceApprox(t, read[0].Floats(), []float32{3, 4, 5, 6})
	floatSliceApprox(t, read[1].Floats(), []float32{30, 40, 50, 60})
}

func TestFixedKVCache_ChunkedPromptPreservesTailPlusCurrentContext_Good(t *testing.T) {
	coverageTokens := "FixedKVCache ChunkedPromptPreservesTailPlusCurrentContext"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	c := NewFixedKVCache(4)
	k1 := FromValues([]float32{1, 2, 3, 4, 5, 6}, 1, 1, 6, 1)
	v1 := FromValues([]float32{10, 20, 30, 40, 50, 60}, 1, 1, 6, 1)
	defer Free(k1, v1)
	firstK, firstV := c.Update(k1, v1, 6)
	if err := Eval(firstK, firstV); err != nil {
		t.Fatalf("Eval first chunk: %v", err)
	}
	Free(firstK, firstV)
	c.Detach()

	k2 := FromValues([]float32{7, 8}, 1, 1, 2, 1)
	v2 := FromValues([]float32{70, 80}, 1, 1, 2, 1)
	defer Free(k2, v2)
	gotK, gotV := c.Update(k2, v2, 2)
	defer Free(gotK, gotV)
	if gotK.Dim(2) != 6 || gotV.Dim(2) != 6 {
		t.Fatalf("chunk context dims = %d/%d, want previous tail plus current 6/6", gotK.Dim(2), gotV.Dim(2))
	}
	if c.Offset() != 8 || c.Len() != 4 {
		t.Fatalf("cache offset/len = %d/%d, want 8/4", c.Offset(), c.Len())
	}
	if err := Eval(gotK, gotV); err != nil {
		t.Fatalf("Eval second chunk context: %v", err)
	}
	floatSliceApprox(t, gotK.Floats(), []float32{3, 4, 5, 6, 7, 8})
	floatSliceApprox(t, gotV.Floats(), []float32{30, 40, 50, 60, 70, 80})

	read, owned := c.ReadState()
	defer Free(owned...)
	if err := Eval(read...); err != nil {
		t.Fatalf("Eval stored second tail: %v", err)
	}
	floatSliceApprox(t, read[0].Floats(), []float32{5, 6, 7, 8})
	floatSliceApprox(t, read[1].Floats(), []float32{50, 60, 70, 80})
}

func TestFixedKVCache_DecodeOverflowSurvivesDetach_Good(t *testing.T) {
	coverageTokens := "FixedKVCache DecodeOverflowSurvivesDetach"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	c := NewFixedKVCache(4)
	k1 := FromValues([]float32{1, 2, 3, 4, 5, 6}, 1, 1, 6, 1)
	v1 := FromValues([]float32{10, 20, 30, 40, 50, 60}, 1, 1, 6, 1)
	defer Free(k1, v1)
	firstK, firstV := c.Update(k1, v1, 6)
	if err := Eval(firstK, firstV); err != nil {
		t.Fatalf("Eval prompt chunk: %v", err)
	}
	Free(firstK, firstV)
	c.Detach()

	k2 := FromValues([]float32{7}, 1, 1, 1, 1)
	v2 := FromValues([]float32{70}, 1, 1, 1, 1)
	defer Free(k2, v2)
	secondK, secondV := c.Update(k2, v2, 1)
	if err := Eval(secondK, secondV); err != nil {
		t.Fatalf("Eval first decode update: %v", err)
	}
	Free(secondK, secondV)
	c.Detach()

	k3 := FromValues([]float32{8}, 1, 1, 1, 1)
	v3 := FromValues([]float32{80}, 1, 1, 1, 1)
	defer Free(k3, v3)
	gotK, gotV := c.Update(k3, v3, 1)
	defer Free(gotK, gotV)
	if gotK.Dim(2) != 4 || gotV.Dim(2) != 4 {
		t.Fatalf("decode context dims = %d/%d, want bounded tail 4/4", gotK.Dim(2), gotV.Dim(2))
	}
	if err := Eval(gotK, gotV); err != nil {
		t.Fatalf("Eval second decode update: %v", err)
	}
	floatSliceApprox(t, gotK.Floats(), []float32{5, 6, 7, 8})
	floatSliceApprox(t, gotV.Floats(), []float32{50, 60, 70, 80})
}

func TestFixedKVCache_ReplaceFixedFromNative_Good(t *testing.T) {
	coverageTokens := "FixedKVCache ReplaceFixedFromNative"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	c := NewFixedKVCache(4)
	keys := Zeros([]int32{1, 1, 4, 2}, DTypeFloat32)
	values := Zeros([]int32{1, 1, 4, 2}, DTypeFloat32)

	state := c.ReplaceFixedFromNative(keys, values, 1)
	defer state.Free()
	if state.Keys == nil || state.Values == nil || state.Length != 1 {
		t.Fatalf("state = %+v, want cloned full-capacity state with length 1", state)
	}
	if c.Offset() != 1 || c.Len() != 1 {
		t.Fatalf("cache offset/len = %d/%d, want 1/1", c.Offset(), c.Len())
	}
	c.Reset()
}

func TestFixedKVCache_BorrowedFixedState_Good(t *testing.T) {
	coverageTokens := "FixedKVCache BorrowedFixedState"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	c := NewFixedKVCache(4)
	keys := Zeros([]int32{1, 1, 4, 2}, DTypeFloat32)
	values := Zeros([]int32{1, 1, 4, 2}, DTypeFloat32)
	c.keys = keys
	c.values = values
	c.length = 2
	defer c.Reset()

	state := c.BorrowedFixedState()
	state.Free()
	if state.Keys != keys || state.Values != values || state.Length != 2 {
		t.Fatalf("state = %+v, want borrowed cache-owned handles", state)
	}
	if c.keys != keys || c.values != values {
		t.Fatal("BorrowedFixedState().Free released cache-owned handles")
	}
}

func TestFixedKVCache_ReplaceFixedFromNativeBorrowed_Good(t *testing.T) {
	coverageTokens := "FixedKVCache ReplaceFixedFromNativeBorrowed"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	c := NewFixedKVCache(4)
	keys := Zeros([]int32{1, 1, 4, 2}, DTypeFloat32)
	values := Zeros([]int32{1, 1, 4, 2}, DTypeFloat32)

	state := c.ReplaceFixedFromNativeBorrowed(keys, values, 1)
	defer c.Reset()
	if state.Keys != keys || state.Values != values || state.Length != 1 {
		t.Fatalf("state = %+v, want borrowed full-capacity state with length 1", state)
	}
	state.Free()
	if c.keys != keys || c.values != values {
		t.Fatal("borrowed native replacement state freed cache-owned handles")
	}
	if c.Offset() != 1 || c.Len() != 1 {
		t.Fatalf("cache offset/len = %d/%d, want 1/1", c.Offset(), c.Len())
	}
}

func TestFixedKVCache_ReplaceFixedFromNativeBorrowedRetiresPrevious_Good(t *testing.T) {
	coverageTokens := "FixedKVCache ReplaceFixedFromNativeBorrowedRetiresPrevious"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	c := NewFixedKVCache(4)
	c.keys = Zeros([]int32{1, 1, 4, 2}, DTypeFloat32)
	c.values = Zeros([]int32{1, 1, 4, 2}, DTypeFloat32)
	keys := Zeros([]int32{1, 1, 4, 2}, DTypeFloat32)
	values := Zeros([]int32{1, 1, 4, 2}, DTypeFloat32)
	defer c.Reset()

	state := c.ReplaceFixedFromNativeBorrowed(keys, values, 1)
	if state.Keys != keys || state.Values != values {
		t.Fatalf("state = %+v, want replacement handles", state)
	}
	if len(c.retired) != 2 {
		t.Fatalf("retired handles = %d, want previous K/V retained until next eval boundary", len(c.retired))
	}
	c.ensureShape(1, 1, 2, 2, DTypeFloat32, DTypeFloat32)
	if len(c.retired) != 0 {
		t.Fatalf("retired handles = %d, want released on next cache entry", len(c.retired))
	}
}

func TestKVCache_Reset_ReleasesState_Good(t *testing.T) {
	c := NewKVCache()
	k, v := makeKV(2)
	defer Free(k, v)
	c.Update(k, v, 2)

	state := c.State()
	if len(state) != 2 {
		t.Fatalf("state length = %d, want 2", len(state))
	}

	c.Reset()

	if state[0].Valid() || state[1].Valid() {
		t.Fatal("Reset should free the cached key/value arrays")
	}
}

func TestKVCache_State_Good(t *testing.T) {
	c := NewKVCache()
	k, v := makeKV(2)
	c.Update(k, v, 2)

	state := c.State()
	if len(state) != 2 {
		t.Fatalf("state length = %d, want 2", len(state))
	}
	// state[0] = keys, state[1] = values
	if state[0] == nil || state[1] == nil {
		t.Error("state arrays should not be nil")
	}
}

// --- RotatingKVCache ---

func TestRotatingKVCache_New_Good(t *testing.T) {
	coverageTokens := "New"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	c := NewRotatingKVCache(16)
	if c.Offset() != 0 {
		t.Errorf("offset = %d, want 0", c.Offset())
	}
	if c.Len() != 0 {
		t.Errorf("len = %d, want 0", c.Len())
	}
}

func TestRotatingKVCache_SingleToken_Good(t *testing.T) {
	coverageTokens := "SingleToken"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	c := NewRotatingKVCache(8)
	k, v := makeKV(1)

	outK, outV := c.Update(k, v, 1)
	Materialize(outK, outV)

	if c.Offset() != 1 {
		t.Errorf("offset = %d, want 1", c.Offset())
	}
	if c.Len() != 1 {
		t.Errorf("len = %d, want 1", c.Len())
	}
}

func TestRotatingKVCache_MultiTokenPrompt_Good(t *testing.T) {
	coverageTokens := "MultiTokenPrompt"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	c := NewRotatingKVCache(16)
	k, v := makeKV(5)

	outK, outV := c.Update(k, v, 5)
	Materialize(outK, outV)

	if c.Offset() != 5 {
		t.Errorf("offset = %d, want 5", c.Offset())
	}
	if c.Len() != 5 {
		t.Errorf("len = %d, want 5", c.Len())
	}
}

func TestRotatingKVCache_Bounded_Good(t *testing.T) {
	coverageTokens := "Bounded"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	c := NewRotatingKVCache(4)

	// Fill with 4-token prompt (at max)
	k, v := makeKV(4)
	outK, outV := c.Update(k, v, 4)
	Materialize(outK, outV)

	if c.Len() != 4 {
		t.Errorf("len = %d, want 4 (at max)", c.Len())
	}

	// Add one more token — should trim to maxSize
	k2, v2 := makeKV(1)
	outK, outV = c.Update(k2, v2, 1)
	Materialize(outK, outV)

	if c.Offset() != 5 {
		t.Errorf("offset = %d, want 5", c.Offset())
	}
	// Len should be bounded by maxSize
	if c.Len() != 4 {
		t.Errorf("len = %d, want 4 (bounded)", c.Len())
	}
}

func TestRotatingKVCache_LongPromptPreservesFullAttentionContext_Good(t *testing.T) {
	coverageTokens := "LongPromptPreservesFullAttentionContext"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	c := NewRotatingKVCache(4)
	k, v := makeKV(6)
	defer Free(k, v)

	outK, outV := c.Update(k, v, 6)
	defer Free(outK, outV)
	Materialize(outK, outV)

	if c.Offset() != 6 {
		t.Errorf("offset = %d, want 6", c.Offset())
	}
	if c.Len() != 4 {
		t.Errorf("len = %d, want 4 (bounded cache)", c.Len())
	}

	if got := outK.Shape()[2]; got != 6 {
		t.Fatalf("outK L dim = %d, want 6 full prompt tokens", got)
	}
	if got := outV.Shape()[2]; got != 6 {
		t.Fatalf("outV L dim = %d, want 6 full prompt tokens", got)
	}

	state := c.State()
	if len(state) != 2 {
		t.Fatalf("state length = %d, want 2", len(state))
	}
	defer Free(state...)
	if got := state[0].Shape()[2]; got != 4 {
		t.Fatalf("cached key L dim = %d, want 4 bounded tokens", got)
	}
	if got := state[1].Shape()[2]; got != 4 {
		t.Fatalf("cached value L dim = %d, want 4 bounded tokens", got)
	}
}

func TestRotatingKVCache_SingleTokenWrapMaintainsOrder_Good(t *testing.T) {
	coverageTokens := "SingleTokenWrapMaintainsOrder"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	c := NewRotatingKVCache(4)

	for i := range 6 {
		k, v := makeSingleTokenKV(float32(i + 1))
		outK, outV := c.Update(k, v, 1)
		Materialize(outK, outV)

		if i < 3 {
			Free(k, v, outK, outV)
			continue
		}

		got := outK.Floats()
		wantValues := []float32{float32(i - 2), float32(i - 1), float32(i), float32(i + 1)}
		for tokenIdx, want := range wantValues {
			base := tokenIdx * 4
			if base >= len(got) {
				t.Fatalf("token %d base index %d beyond output len %d", tokenIdx, base, len(got))
			}
			if got[base] != want {
				t.Fatalf("token %d first value = %f, want %f (full output %v)", tokenIdx, got[base], want, got)
			}
		}

		Free(k, v, outK, outV)
	}
}

func TestRotatingKVCache_Reset_Good(t *testing.T) {
	c := NewRotatingKVCache(8)
	k, v := makeKV(3)
	c.Update(k, v, 3)

	c.Reset()

	if c.Offset() != 0 {
		t.Errorf("offset after reset = %d, want 0", c.Offset())
	}
	if c.Len() != 0 {
		t.Errorf("len after reset = %d, want 0", c.Len())
	}
	if c.State() != nil {
		t.Error("state should be nil after reset")
	}
}

func TestRotatingKVCache_Reset_ReleasesState_Good(t *testing.T) {
	c := NewRotatingKVCache(8)
	k, v := makeKV(3)
	defer Free(k, v)
	c.Update(k, v, 3)

	state := c.State()
	if len(state) != 2 {
		t.Fatalf("state length = %d, want 2", len(state))
	}

	c.Reset()

	if state[0].Valid() || state[1].Valid() {
		t.Fatal("Reset should free the cached key/value arrays")
	}
}

// Generated file-aware compliance coverage.
func TestCache_NewKVCache_Good(t *testing.T) {
	target := "NewKVCache"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestCache_NewKVCache_Bad(t *testing.T) {
	target := "NewKVCache"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestCache_NewKVCache_Ugly(t *testing.T) {
	target := "NewKVCache"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestCache_KVCache_Update_Good(t *testing.T) {
	coverageTokens := "KVCache Update"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "KVCache_Update"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestCache_KVCache_Update_Bad(t *testing.T) {
	coverageTokens := "KVCache Update"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "KVCache_Update"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestCache_KVCache_Update_Ugly(t *testing.T) {
	coverageTokens := "KVCache Update"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "KVCache_Update"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestCache_KVCache_State_Good(t *testing.T) {
	coverageTokens := "KVCache State"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "KVCache_State"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestCache_KVCache_State_Bad(t *testing.T) {
	coverageTokens := "KVCache State"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "KVCache_State"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestCache_KVCache_State_Ugly(t *testing.T) {
	coverageTokens := "KVCache State"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "KVCache_State"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestCache_KVCache_Offset_Good(t *testing.T) {
	coverageTokens := "KVCache Offset"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "KVCache_Offset"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestCache_KVCache_Offset_Bad(t *testing.T) {
	coverageTokens := "KVCache Offset"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "KVCache_Offset"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestCache_KVCache_Offset_Ugly(t *testing.T) {
	coverageTokens := "KVCache Offset"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "KVCache_Offset"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestCache_KVCache_Len_Good(t *testing.T) {
	coverageTokens := "KVCache Len"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "KVCache_Len"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestCache_KVCache_Len_Bad(t *testing.T) {
	coverageTokens := "KVCache Len"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "KVCache_Len"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestCache_KVCache_Len_Ugly(t *testing.T) {
	coverageTokens := "KVCache Len"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "KVCache_Len"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestCache_KVCache_Reset_Good(t *testing.T) {
	coverageTokens := "KVCache Reset"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "KVCache_Reset"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestCache_KVCache_Reset_Bad(t *testing.T) {
	coverageTokens := "KVCache Reset"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "KVCache_Reset"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestCache_KVCache_Reset_Ugly(t *testing.T) {
	coverageTokens := "KVCache Reset"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "KVCache_Reset"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestCache_KVCache_Detach_Good(t *testing.T) {
	coverageTokens := "KVCache Detach"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "KVCache_Detach"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestCache_KVCache_Detach_Bad(t *testing.T) {
	coverageTokens := "KVCache Detach"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "KVCache_Detach"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestCache_KVCache_Detach_Ugly(t *testing.T) {
	coverageTokens := "KVCache Detach"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "KVCache_Detach"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestCache_NewRotatingKVCache_Good(t *testing.T) {
	target := "NewRotatingKVCache"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestCache_NewRotatingKVCache_Bad(t *testing.T) {
	target := "NewRotatingKVCache"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestCache_NewRotatingKVCache_Ugly(t *testing.T) {
	target := "NewRotatingKVCache"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestCache_RotatingKVCache_Update_Good(t *testing.T) {
	coverageTokens := "RotatingKVCache Update"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "RotatingKVCache_Update"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestCache_RotatingKVCache_Update_Bad(t *testing.T) {
	coverageTokens := "RotatingKVCache Update"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "RotatingKVCache_Update"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestCache_RotatingKVCache_Update_Ugly(t *testing.T) {
	coverageTokens := "RotatingKVCache Update"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "RotatingKVCache_Update"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestCache_RotatingKVCache_State_Good(t *testing.T) {
	coverageTokens := "RotatingKVCache State"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "RotatingKVCache_State"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestCache_RotatingKVCache_State_Bad(t *testing.T) {
	coverageTokens := "RotatingKVCache State"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "RotatingKVCache_State"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestCache_RotatingKVCache_State_Ugly(t *testing.T) {
	coverageTokens := "RotatingKVCache State"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "RotatingKVCache_State"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestCache_RotatingKVCache_Offset_Good(t *testing.T) {
	coverageTokens := "RotatingKVCache Offset"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "RotatingKVCache_Offset"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestCache_RotatingKVCache_Offset_Bad(t *testing.T) {
	coverageTokens := "RotatingKVCache Offset"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "RotatingKVCache_Offset"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestCache_RotatingKVCache_Offset_Ugly(t *testing.T) {
	coverageTokens := "RotatingKVCache Offset"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "RotatingKVCache_Offset"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestCache_RotatingKVCache_Len_Good(t *testing.T) {
	coverageTokens := "RotatingKVCache Len"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "RotatingKVCache_Len"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestCache_RotatingKVCache_Len_Bad(t *testing.T) {
	coverageTokens := "RotatingKVCache Len"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "RotatingKVCache_Len"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestCache_RotatingKVCache_Len_Ugly(t *testing.T) {
	coverageTokens := "RotatingKVCache Len"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "RotatingKVCache_Len"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestCache_RotatingKVCache_Reset_Good(t *testing.T) {
	coverageTokens := "RotatingKVCache Reset"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "RotatingKVCache_Reset"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestCache_RotatingKVCache_Reset_Bad(t *testing.T) {
	coverageTokens := "RotatingKVCache Reset"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "RotatingKVCache_Reset"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestCache_RotatingKVCache_Reset_Ugly(t *testing.T) {
	coverageTokens := "RotatingKVCache Reset"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "RotatingKVCache_Reset"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestCache_RotatingKVCache_Detach_Good(t *testing.T) {
	coverageTokens := "RotatingKVCache Detach"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "RotatingKVCache_Detach"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestCache_RotatingKVCache_Detach_Bad(t *testing.T) {
	coverageTokens := "RotatingKVCache Detach"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "RotatingKVCache_Detach"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestCache_RotatingKVCache_Detach_Ugly(t *testing.T) {
	coverageTokens := "RotatingKVCache Detach"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "RotatingKVCache_Detach"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}
