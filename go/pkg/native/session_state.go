// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"encoding/binary"
	"unsafe"

	core "dappco.re/go"
)

// session_state.go is native conversation continuity (12-14): the metal serve path keeps a multi-turn
// conversation alive with EnableConversationContinuity + a host KV store; the no-cgo path needs the same
// without cgo. SerializeState captures the resident KV cache + position into a portable blob so a session
// can be saved to disk and resumed across process restarts; RestoreState loads it into a fresh session of
// the same shape. The restored session decodes byte-identically to the one that was saved — proven in
// session_state_test.go. Single-goroutine (the ArchSession contract).

const sessionStateMagic = 0x4c544e53 // "LTNS" — Lethean native session

// SerializeState returns a portable snapshot of the session: its position and every owned layer's KV
// cache bytes. ICB-replay sessions are unsupported (their caches live in the recorded replay, not the
// layer bufs) — serialize a session built on the non-ICB bf16 path.
func (s *ArchSession) SerializeState() ([]byte, error) {
	if s.state.icb != nil {
		return nil, core.NewError("native.SerializeState: ICB-replay sessions unsupported; use the non-ICB bf16 path")
	}
	hdr := make([]byte, 12)
	binary.LittleEndian.PutUint32(hdr[0:], sessionStateMagic)
	binary.LittleEndian.PutUint32(hdr[4:], uint32(s.pos))
	binary.LittleEndian.PutUint32(hdr[8:], uint32(len(s.state.specs)))
	out := hdr
	for li := range s.state.specs {
		if !s.state.specs[li].OwnsCache() {
			continue // shared-KV layers reference an owner's cache; only owners carry bytes
		}
		k, v := s.state.lb[li].kCache, s.state.lb[li].vCache
		n := int(k.Length())
		lenBuf := make([]byte, 4)
		binary.LittleEndian.PutUint32(lenBuf, uint32(n))
		out = append(out, lenBuf...)
		out = append(out, unsafe.Slice((*byte)(k.Contents()), n)...)
		out = append(out, unsafe.Slice((*byte)(v.Contents()), n)...)
	}
	return out, nil
}

// RestoreState loads a SerializeState snapshot into this session, overwriting its resident KV cache and
// position. The session must have the same architecture (layer count + cache sizes) as the one saved.
// After restore, decoding continues exactly as if the saved session had never stopped.
func (s *ArchSession) RestoreState(data []byte) error {
	if s.state.icb != nil {
		return core.NewError("native.RestoreState: ICB-replay sessions unsupported")
	}
	if len(data) < 12 || binary.LittleEndian.Uint32(data[0:]) != sessionStateMagic {
		return core.NewError("native.RestoreState: not a native session snapshot")
	}
	pos := int(binary.LittleEndian.Uint32(data[4:]))
	nL := int(binary.LittleEndian.Uint32(data[8:]))
	if nL != len(s.state.specs) {
		return core.NewError("native.RestoreState: layer count mismatch (snapshot vs session)")
	}
	off := 12
	for li := range s.state.specs {
		if !s.state.specs[li].OwnsCache() {
			continue
		}
		if off+4 > len(data) {
			return core.NewError("native.RestoreState: truncated snapshot")
		}
		n := int(binary.LittleEndian.Uint32(data[off:]))
		off += 4
		k, v := s.state.lb[li].kCache, s.state.lb[li].vCache
		if int(k.Length()) != n {
			return core.NewError("native.RestoreState: cache size mismatch (snapshot vs session)")
		}
		if off+2*n > len(data) {
			return core.NewError("native.RestoreState: truncated snapshot")
		}
		copy(unsafe.Slice((*byte)(k.Contents()), n), data[off:off+n])
		off += n
		copy(unsafe.Slice((*byte)(v.Contents()), n), data[off:off+n])
		off += n
	}
	s.pos = pos
	return nil
}
