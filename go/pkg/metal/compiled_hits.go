// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import "sync/atomic"

// compiledLayerHitsReader is registered by model families that implement the
// whole-layer compiled decode (gemma4 today), so neutral session metrics can
// report compiled coverage without importing the family package (AX-8).
var compiledLayerHitsReader atomic.Pointer[func() uint64]

// RegisterCompiledLayerHitsReader installs the family's hit-counter reader.
//
//	metal.RegisterCompiledLayerHitsReader(CompiledLayerDecodeHits) // gemma4 init
func RegisterCompiledLayerHitsReader(fn func() uint64) {
	if fn == nil {
		return
	}
	compiledLayerHitsReader.Store(&fn)
}

func readCompiledLayerHits() uint64 {
	if fn := compiledLayerHitsReader.Load(); fn != nil {
		return (*fn)()
	}
	return 0
}
