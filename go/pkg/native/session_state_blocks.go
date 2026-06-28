// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"unsafe"

	core "dappco.re/go"
)

// SessionStateLayerBlock is one layer's K/V cache bytes for a contiguous token
// range. KeyBytes and ValueBytes are views into the session's resident Metal
// buffers when produced by StateBlockSource or RangeStateBlocks; callers must
// consume or copy them before mutating/closing the source session.
type SessionStateLayerBlock struct {
	Layer      int
	KVHeads    int
	HeadDim    int
	RowBytes   int
	KeyBytes   []byte
	ValueBytes []byte
}

// SessionStateBlock is a contiguous token range from the native session state.
type SessionStateBlock struct {
	Index      int
	TokenStart int
	TokenCount int
	Layers     []SessionStateLayerBlock
}

// SessionStateBlockSource streams native session state blocks without first
// assembling a monolithic SerializeState blob.
type SessionStateBlockSource struct {
	Position           int
	CachedIDs          []int32
	CachedPromptIDs    []int32
	CachedPromptHidden []byte
	CachedPromptLogits []byte
	RetainedHidden     []byte
	RetainedLogits     []byte
	BlockCount         int
	Load               func(int) (SessionStateBlock, error)
	blockSize          int
	firstBlockIndex    int
	totalBlockCount    int
	views              []sessionStateLayerView
}

type sessionStateLayerView struct {
	layer      int
	kvHeads    int
	headDim    int
	rowBytes   int
	keyBytes   []byte
	valueBytes []byte
}

// StateBlockSource returns a block loader over the current resident K/V cache.
// K/V payload slices returned by Load are zero-copy views into this session.
func (s *ArchSession) StateBlockSource(blockSize int) (SessionStateBlockSource, error) {
	return s.StateBlockSourceFrom(0, blockSize)
}

// StateBlockSourceFrom is StateBlockSource with metal-style trusted-prefix
// sleep: full blocks ending at or before startToken are skipped, but yielded
// block indexes remain absolute in the original block grid.
func (s *ArchSession) StateBlockSourceFrom(startToken, blockSize int) (SessionStateBlockSource, error) {
	blockCount, firstBlock, totalBlocks, views, err := s.stateBlockPlan(startToken, blockSize)
	if err != nil {
		return SessionStateBlockSource{}, err
	}
	retainedLogits := s.retainedLogits
	if len(retainedLogits) == 0 && len(s.retainedHidden) == s.arch.Hidden*bf16Size {
		var err error
		retainedLogits, err = s.BoundaryLogits()
		if err != nil {
			return SessionStateBlockSource{}, err
		}
	}
	source := SessionStateBlockSource{
		Position:           s.pos,
		CachedIDs:          append([]int32(nil), s.cachedIDs...),
		CachedPromptIDs:    append([]int32(nil), s.cachedPromptIDs...),
		CachedPromptHidden: append([]byte(nil), s.cachedPromptHidden...),
		CachedPromptLogits: append([]byte(nil), s.cachedPromptLogits...),
		RetainedHidden:     append([]byte(nil), s.retainedHidden...),
		RetainedLogits:     append([]byte(nil), retainedLogits...),
		BlockCount:         blockCount,
		blockSize:          blockSize,
		firstBlockIndex:    firstBlock,
		totalBlockCount:    totalBlocks,
		views:              views,
	}
	source.Load = func(index int) (SessionStateBlock, error) {
		return loadStateBlock(firstBlock+index, blockSize, totalBlocks, source.Position, views)
	}
	return source, nil
}

// RangeStateBlocks visits native session-state blocks in order. It is the
// native analogue of metal's ranged K/V capture, but it stays CGO-free and uses
// ArchSession's resident buffers directly. The yielded block and its layer
// descriptors are only valid until the callback returns.
func (s *ArchSession) RangeStateBlocks(blockSize int, yield func(SessionStateBlock) (bool, error)) error {
	return s.RangeStateBlocksFrom(0, blockSize, yield)
}

// RangeStateBlocksFrom visits native session-state blocks after startToken.
func (s *ArchSession) RangeStateBlocksFrom(startToken, blockSize int, yield func(SessionStateBlock) (bool, error)) error {
	if yield == nil {
		return core.NewError("native.RangeStateBlocks: nil yield")
	}
	blockCount, firstBlock, totalBlocks, views, err := s.stateBlockPlan(startToken, blockSize)
	if err != nil {
		return err
	}
	layers := s.stateBlockLayerScratch(len(views))
	for i := 0; i < blockCount; i++ {
		block, err := fillStateBlock(firstBlock+i, blockSize, totalBlocks, s.pos, views, layers)
		if err != nil {
			return err
		}
		ok, err := yield(block)
		if err != nil || !ok {
			return err
		}
	}
	return nil
}

// RestoreStateBlocks restores a session from streamed native state blocks. It
// copies only the current block's K/V range into resident buffers and restores
// the small prompt/retained metadata needed for GenerateFromCache and prefix
// reuse.
func (s *ArchSession) RestoreStateBlocks(source SessionStateBlockSource) error {
	if s == nil {
		return core.NewError("native.RestoreStateBlocks: nil session")
	}
	if source.Position < 0 || source.Position > s.maxLen {
		return core.NewError("native.RestoreStateBlocks: position outside maxLen")
	}
	if len(source.CachedIDs) > source.Position {
		return core.NewError("native.RestoreStateBlocks: cached ids exceed position")
	}
	if source.BlockCount < 0 {
		return core.NewError("native.RestoreStateBlocks: negative block count")
	}
	if source.BlockCount > 0 && source.Load == nil {
		return core.NewError("native.RestoreStateBlocks: nil block loader")
	}
	if source.Position == 0 && source.BlockCount != 0 {
		return core.NewError("native.RestoreStateBlocks: zero-position source has blocks")
	}
	trustedPrefix := source.trustedPrefixTokens()
	if source.Position > 0 && source.BlockCount == 0 && trustedPrefix != source.Position {
		return core.NewError("native.RestoreStateBlocks: non-empty source has no blocks")
	}
	if trustedPrefix > 0 {
		if err := s.validateStateBlockTrustedPrefix(source, trustedPrefix); err != nil {
			return err
		}
	}
	if source.BlockCount == 0 {
		return s.restoreStateBlockMetadata(source)
	}
	targetViews, err := s.stateLayerViews()
	if err != nil {
		return err
	}
	ownerCount := len(targetViews)
	sourceLayers := s.stateBlockLayerScratch(ownerCount)
	expectedStart := trustedPrefix
	expectedIndex := source.firstBlockIndex
	for i := 0; i < source.BlockCount; i++ {
		block, err := source.loadInto(i, sourceLayers)
		if err != nil {
			return err
		}
		if err := restoreStateBlock(expectedIndex+i, expectedStart, source.Position, ownerCount, targetViews, block); err != nil {
			return err
		}
		expectedStart += block.TokenCount
	}
	if expectedStart != source.Position {
		return core.NewError("native.RestoreStateBlocks: block coverage does not match position")
	}
	if err := s.restoreStateBlockMetadata(source); err != nil {
		return err
	}
	return nil
}

func (source SessionStateBlockSource) trustedPrefixTokens() int {
	if source.blockSize <= 0 || source.firstBlockIndex <= 0 {
		return 0
	}
	prefix := source.firstBlockIndex * source.blockSize
	if prefix > source.Position {
		return source.Position
	}
	return prefix
}

func (s *ArchSession) validateStateBlockTrustedPrefix(source SessionStateBlockSource, trustedPrefix int) error {
	if trustedPrefix < 0 || trustedPrefix > source.Position {
		return core.NewError("native.RestoreStateBlocks: trusted prefix outside position")
	}
	if s.pos < trustedPrefix {
		return core.NewError("native.RestoreStateBlocks: trusted prefix not resident")
	}
	if len(source.CachedIDs) < trustedPrefix {
		return core.NewError("native.RestoreStateBlocks: trusted prefix source ids missing")
	}
	if len(s.cachedIDs) < trustedPrefix {
		return core.NewError("native.RestoreStateBlocks: trusted prefix resident ids missing")
	}
	for i := 0; i < trustedPrefix; i++ {
		if s.cachedIDs[i] != source.CachedIDs[i] {
			return core.NewError("native.RestoreStateBlocks: trusted prefix ids mismatch")
		}
	}
	return nil
}

func (source SessionStateBlockSource) loadInto(index int, layers []SessionStateLayerBlock) (SessionStateBlock, error) {
	if len(source.views) > 0 && source.blockSize > 0 {
		return fillStateBlock(source.firstBlockIndex+index, source.blockSize, source.totalBlockCount, source.Position, source.views, layers)
	}
	return source.Load(index)
}

func loadStateBlock(index, blockSize, blockCount, position int, views []sessionStateLayerView) (SessionStateBlock, error) {
	layers := make([]SessionStateLayerBlock, len(views))
	return fillStateBlock(index, blockSize, blockCount, position, views, layers)
}

func fillStateBlock(index, blockSize, blockCount, position int, views []sessionStateLayerView, layers []SessionStateLayerBlock) (SessionStateBlock, error) {
	if index < 0 || index >= blockCount {
		return SessionStateBlock{}, core.NewError("native.StateBlockSource.Load: block index out of range")
	}
	start := index * blockSize
	if start >= position {
		return SessionStateBlock{}, core.NewError("native.StateBlockSource.Load: block start outside position")
	}
	end := start + blockSize
	if end > position {
		end = position
	}
	tokenCount := end - start
	if len(layers) != len(views) {
		return SessionStateBlock{}, core.NewError("native.StateBlockSource.Load: layer descriptor size mismatch")
	}
	for i, view := range views {
		off := start * view.rowBytes
		n := tokenCount * view.rowBytes
		layers[i] = SessionStateLayerBlock{
			Layer:      view.layer,
			KVHeads:    view.kvHeads,
			HeadDim:    view.headDim,
			RowBytes:   view.rowBytes,
			KeyBytes:   view.keyBytes[off : off+n],
			ValueBytes: view.valueBytes[off : off+n],
		}
	}
	return SessionStateBlock{Index: index, TokenStart: start, TokenCount: tokenCount, Layers: layers}, nil
}

func restoreStateBlock(index, expectedStart, position, ownerCount int, targetViews []sessionStateLayerView, block SessionStateBlock) error {
	if block.Index != index {
		return core.NewError("native.RestoreStateBlocks: block index mismatch")
	}
	if block.TokenStart != expectedStart {
		return core.NewError("native.RestoreStateBlocks: block token start mismatch")
	}
	if block.TokenCount <= 0 {
		return core.NewError("native.RestoreStateBlocks: empty block")
	}
	if block.TokenStart+block.TokenCount > position {
		return core.NewError("native.RestoreStateBlocks: block exceeds position")
	}
	if len(block.Layers) != ownerCount {
		return core.NewError("native.RestoreStateBlocks: block layer count mismatch")
	}
	var seenStack [128]bool
	seen := seenStack[:]
	if len(targetViews) > len(seenStack) {
		seen = make([]bool, len(targetViews))
	} else {
		seen = seen[:len(targetViews)]
	}
	for _, layer := range block.Layers {
		viewIndex := -1
		for i, view := range targetViews {
			if view.layer == layer.Layer {
				viewIndex = i
				break
			}
		}
		if viewIndex < 0 {
			return core.NewError("native.RestoreStateBlocks: invalid block layer")
		}
		if seen[viewIndex] {
			return core.NewError("native.RestoreStateBlocks: duplicate block layer")
		}
		seen[viewIndex] = true
		view := targetViews[viewIndex]
		if layer.KVHeads > 0 && layer.KVHeads != view.kvHeads {
			return core.NewError("native.RestoreStateBlocks: kv-head count mismatch")
		}
		if layer.HeadDim > 0 && layer.HeadDim != view.headDim {
			return core.NewError("native.RestoreStateBlocks: head-dim mismatch")
		}
		if layer.RowBytes != view.rowBytes {
			return core.NewError("native.RestoreStateBlocks: row-byte mismatch")
		}
		n := block.TokenCount * view.rowBytes
		if len(layer.KeyBytes) != n || len(layer.ValueBytes) != n {
			return core.NewError("native.RestoreStateBlocks: block payload size mismatch")
		}
		off := block.TokenStart * view.rowBytes
		copy(view.keyBytes[off:off+n], layer.KeyBytes)
		copy(view.valueBytes[off:off+n], layer.ValueBytes)
	}
	return nil
}

func (s *ArchSession) restoreStateBlockMetadata(source SessionStateBlockSource) error {
	if len(source.CachedPromptHidden) > 0 && len(source.CachedPromptHidden) != s.arch.Hidden*bf16Size {
		return core.NewError("native.RestoreStateBlocks: prompt hidden size mismatch")
	}
	if len(source.CachedPromptLogits) > 0 && len(source.CachedPromptLogits) != s.arch.Vocab*bf16Size {
		return core.NewError("native.RestoreStateBlocks: prompt logits size mismatch")
	}
	if len(source.RetainedHidden) > 0 && len(source.RetainedHidden) != s.arch.Hidden*bf16Size {
		return core.NewError("native.RestoreStateBlocks: retained hidden size mismatch")
	}
	if len(source.RetainedLogits) > 0 && len(source.RetainedLogits) != s.arch.Vocab*bf16Size {
		return core.NewError("native.RestoreStateBlocks: retained logits size mismatch")
	}
	s.pos = source.Position
	s.cachedIDs = append(s.cachedIDs[:0], source.CachedIDs...)
	s.cachedPromptIDs = append(s.cachedPromptIDs[:0], source.CachedPromptIDs...)
	s.cachedPromptHidden = append(s.cachedPromptHidden[:0], source.CachedPromptHidden...)
	s.cachedPromptLogits = append(s.cachedPromptLogits[:0], source.CachedPromptLogits...)
	if len(source.RetainedHidden) == 0 {
		s.resetRetainedHidden()
	} else {
		s.rememberRetainedHidden(source.RetainedHidden)
	}
	if len(source.RetainedLogits) == 0 {
		s.resetRetainedLogits()
	} else {
		s.rememberRetainedLogits(source.RetainedLogits)
	}
	return nil
}

func (s *ArchSession) stateLayerViews() ([]sessionStateLayerView, error) {
	ownerCount := s.ownedStateCacheLayers()
	icb := s.state.icb != nil
	if len(s.stateBlockViews) == ownerCount && s.stateBlockViewsICB == icb {
		return s.stateBlockViews, nil
	}
	views := s.stateBlockViews
	if cap(views) < ownerCount {
		views = make([]sessionStateLayerView, 0, ownerCount)
	} else {
		views = views[:0]
	}
	for li, spec := range s.state.specs {
		if !spec.OwnsCache() {
			continue
		}
		k, _, kPtr, vPtr, err := s.snapshotCacheViews(li)
		if err != nil {
			return nil, err
		}
		cacheBytes := int(k.Length())
		rowBytes, err := s.stateCacheRowBytes(cacheBytes)
		if err != nil {
			return nil, err
		}
		views = append(views, sessionStateLayerView{
			layer:      li,
			kvHeads:    kvHeadsOf(spec, s.arch.KVHeads),
			headDim:    headDimOf(spec, s.arch.HeadDim),
			rowBytes:   rowBytes,
			keyBytes:   unsafe.Slice(kPtr, cacheBytes),
			valueBytes: unsafe.Slice(vPtr, cacheBytes),
		})
	}
	s.stateBlockViews = views
	s.stateBlockViewsICB = icb
	return s.stateBlockViews, nil
}

func (s *ArchSession) stateBlockPlan(startToken, blockSize int) (int, int, int, []sessionStateLayerView, error) {
	if s == nil {
		return 0, 0, 0, nil, core.NewError("native.StateBlockSource: nil session")
	}
	if blockSize <= 0 {
		return 0, 0, 0, nil, core.NewError("native.StateBlockSource: block size must be > 0")
	}
	if startToken < 0 {
		return 0, 0, 0, nil, core.NewError("native.StateBlockSource: start token must be >= 0")
	}
	if s.pos < 0 || s.pos > s.maxLen {
		return 0, 0, 0, nil, core.NewError("native.StateBlockSource: position outside maxLen")
	}
	totalBlocks := 0
	if s.pos > 0 {
		totalBlocks = (s.pos + blockSize - 1) / blockSize
	}
	firstBlock := 0
	for firstBlock < totalBlocks {
		end := (firstBlock + 1) * blockSize
		if end > s.pos {
			end = s.pos
		}
		if end > startToken {
			break
		}
		firstBlock++
	}
	views, err := s.stateLayerViews()
	if err != nil {
		return 0, 0, 0, nil, err
	}
	return totalBlocks - firstBlock, firstBlock, totalBlocks, views, nil
}

func (s *ArchSession) ownedStateCacheLayers() int {
	n := 0
	for _, spec := range s.state.specs {
		if spec.OwnsCache() {
			n++
		}
	}
	return n
}

func (s *ArchSession) stateBlockLayerScratch(n int) []SessionStateLayerBlock {
	if cap(s.stateBlockLayers) < n {
		s.stateBlockLayers = make([]SessionStateLayerBlock, n)
	} else {
		s.stateBlockLayers = s.stateBlockLayers[:n]
	}
	return s.stateBlockLayers
}

func (s *ArchSession) stateCacheRowBytes(cacheBytes int) (int, error) {
	if s.maxLen <= 0 {
		return 0, core.NewError("native.sessionStateBlocks: maxLen must be > 0")
	}
	if cacheBytes%s.maxLen != 0 {
		return 0, core.NewError("native.sessionStateBlocks: cache length is not row-aligned")
	}
	return cacheBytes / s.maxLen, nil
}
