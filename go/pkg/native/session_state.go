// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"encoding/binary"
	"unsafe"

	core "dappco.re/go"
	"github.com/tmc/apple/metal"
)

// session_state.go is native conversation continuity (12-14): the metal serve path keeps a multi-turn
// conversation alive with EnableConversationContinuity + a host KV store; the no-cgo path needs the same
// without cgo. SerializeState captures the resident KV cache + position into a portable blob so a session
// can be saved to disk and resumed across process restarts; RestoreState loads it into a fresh session of
// the same shape. The restored session decodes byte-identically to the one that was saved — proven in
// session_state_test.go. Single-goroutine (the ArchSession contract).

const sessionStateMagic = 0x4c544e53       // "LTNS" — Lethean native session
const sessionPromptEntryMagic = 0x4c544e50 // "LTNP" — Lethean native prompt-cache entry
const sessionRetainedHiddenMagic = 0x4c544e52

// SerializeState returns a portable snapshot of the session: its position and every owned layer's KV
// cache bytes. Regular sessions snapshot the resident layer buffers; ICB-replay sessions snapshot the
// replay-owned linear K/V buffers, keeping the on-disk format unchanged.
func (s *ArchSession) SerializeState() ([]byte, error) {
	total := 12 + 4 + 4*len(s.cachedIDs)
	promptEntryBytes := s.serializedPromptEntryBytes()
	total += promptEntryBytes
	retainedHiddenBytes := s.serializedRetainedHiddenBytes()
	total += retainedHiddenBytes
	var lengthStack [128]int
	lengths := lengthStack[:]
	if len(s.state.specs) > len(lengthStack) {
		lengths = make([]int, len(s.state.specs))
	} else {
		lengths = lengths[:len(s.state.specs)]
	}
	for li := range s.state.specs {
		if !s.state.specs[li].OwnsCache() {
			continue
		}
		k, _, err := s.snapshotCacheBuffers(li)
		if err != nil {
			return nil, err
		}
		n := int(k.Length())
		lengths[li] = n
		total += 4 + 2*n
	}
	out := make([]byte, total)
	binary.LittleEndian.PutUint32(out[0:], sessionStateMagic)
	binary.LittleEndian.PutUint32(out[4:], uint32(s.pos))
	binary.LittleEndian.PutUint32(out[8:], uint32(len(s.state.specs)))
	off := 12
	for li := range s.state.specs {
		if !s.state.specs[li].OwnsCache() {
			continue // shared-KV layers reference an owner's cache; only owners carry bytes
		}
		k, v, err := s.snapshotCacheBuffers(li)
		if err != nil {
			return nil, err
		}
		n := lengths[li]
		binary.LittleEndian.PutUint32(out[off:], uint32(n))
		off += 4
		copy(out[off:off+n], unsafe.Slice((*byte)(k.Contents()), n))
		off += n
		copy(out[off:off+n], unsafe.Slice((*byte)(v.Contents()), n))
		off += n
	}
	binary.LittleEndian.PutUint32(out[off:], uint32(len(s.cachedIDs)))
	off += 4
	for i, id := range s.cachedIDs {
		binary.LittleEndian.PutUint32(out[off+4*i:], uint32(id))
	}
	off += 4 * len(s.cachedIDs)
	if promptEntryBytes > 0 {
		off = s.appendPromptEntrySnapshot(out, off)
	}
	if retainedHiddenBytes > 0 {
		off = s.appendRetainedHiddenSnapshot(out, off)
	}
	return out, nil
}

// RestoreState loads a SerializeState snapshot into this session, overwriting its resident KV cache and
// position. The session must have the same architecture (layer count + cache sizes) as the one saved.
// After restore, decoding continues exactly as if the saved session had never stopped.
func (s *ArchSession) RestoreState(data []byte) error {
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
		k, v, err := s.snapshotCacheBuffers(li)
		if err != nil {
			return err
		}
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
	s.cachedIDs = nil
	s.clearCachedPromptHidden()
	s.resetRetainedHidden()
	if off == len(data) {
		return nil
	}
	if off+4 > len(data) {
		return core.NewError("native.RestoreState: truncated prompt-cache metadata")
	}
	nIDs := int(binary.LittleEndian.Uint32(data[off:]))
	off += 4
	if off+4*nIDs > len(data) {
		return core.NewError("native.RestoreState: truncated prompt-cache metadata")
	}
	if nIDs > 0 {
		s.cachedIDs = make([]int32, nIDs)
		for i := range s.cachedIDs {
			s.cachedIDs[i] = int32(binary.LittleEndian.Uint32(data[off:]))
			off += 4
		}
	}
	for off < len(data) {
		if off+4 > len(data) {
			return core.NewError("native.RestoreState: truncated prompt-cache entry metadata")
		}
		magic := binary.LittleEndian.Uint32(data[off:])
		off += 4
		var err error
		switch magic {
		case sessionPromptEntryMagic:
			off, err = s.restorePromptEntrySnapshot(data, off)
		case sessionRetainedHiddenMagic:
			off, err = s.restoreRetainedHiddenSnapshot(data, off)
		default:
			return core.NewError("native.RestoreState: trailing prompt-cache metadata")
		}
		if err != nil {
			return err
		}
	}
	return nil
}

func (s *ArchSession) serializedPromptEntryBytes() int {
	if s == nil || len(s.cachedPromptIDs) == 0 {
		return 0
	}
	if len(s.cachedPromptHidden) != s.arch.Hidden*bf16Size || len(s.cachedPromptLogits) != s.arch.Vocab*bf16Size {
		return 0
	}
	return 4 + 4 + 4*len(s.cachedPromptIDs) + 4 + len(s.cachedPromptHidden) + 4 + len(s.cachedPromptLogits)
}

func (s *ArchSession) appendPromptEntrySnapshot(out []byte, off int) int {
	binary.LittleEndian.PutUint32(out[off:], sessionPromptEntryMagic)
	off += 4
	binary.LittleEndian.PutUint32(out[off:], uint32(len(s.cachedPromptIDs)))
	off += 4
	for _, id := range s.cachedPromptIDs {
		binary.LittleEndian.PutUint32(out[off:], uint32(id))
		off += 4
	}
	binary.LittleEndian.PutUint32(out[off:], uint32(len(s.cachedPromptHidden)))
	off += 4
	copy(out[off:off+len(s.cachedPromptHidden)], s.cachedPromptHidden)
	off += len(s.cachedPromptHidden)
	binary.LittleEndian.PutUint32(out[off:], uint32(len(s.cachedPromptLogits)))
	off += 4
	copy(out[off:off+len(s.cachedPromptLogits)], s.cachedPromptLogits)
	off += len(s.cachedPromptLogits)
	return off
}

func (s *ArchSession) serializedRetainedHiddenBytes() int {
	if s == nil || len(s.retainedHidden) != s.arch.Hidden*bf16Size {
		return 0
	}
	return 4 + 4 + len(s.retainedHidden)
}

func (s *ArchSession) appendRetainedHiddenSnapshot(out []byte, off int) int {
	binary.LittleEndian.PutUint32(out[off:], sessionRetainedHiddenMagic)
	off += 4
	binary.LittleEndian.PutUint32(out[off:], uint32(len(s.retainedHidden)))
	off += 4
	copy(out[off:off+len(s.retainedHidden)], s.retainedHidden)
	off += len(s.retainedHidden)
	return off
}

func (s *ArchSession) restorePromptEntrySnapshot(data []byte, off int) (int, error) {
	if off+4 > len(data) {
		return off, core.NewError("native.RestoreState: truncated prompt-cache entry metadata")
	}
	nIDs := int(binary.LittleEndian.Uint32(data[off:]))
	off += 4
	if nIDs <= 0 {
		return off, core.NewError("native.RestoreState: empty prompt-cache entry")
	}
	if off+4*nIDs > len(data) {
		return off, core.NewError("native.RestoreState: truncated prompt-cache entry ids")
	}
	ids := make([]int32, nIDs)
	for i := range ids {
		ids[i] = int32(binary.LittleEndian.Uint32(data[off:]))
		off += 4
	}
	hidden, next, err := readPromptEntryBytes(data, off, s.arch.Hidden*bf16Size, "hidden")
	if err != nil {
		return off, err
	}
	logits, next, err := readPromptEntryBytes(data, next, s.arch.Vocab*bf16Size, "logits")
	if err != nil {
		return off, err
	}
	s.cachedPromptIDs = ids
	s.cachedPromptHidden = hidden
	s.cachedPromptLogits = logits
	return next, nil
}

func (s *ArchSession) restoreRetainedHiddenSnapshot(data []byte, off int) (int, error) {
	hidden, next, err := readPromptEntryBytes(data, off, s.arch.Hidden*bf16Size, "retained hidden")
	if err != nil {
		return off, err
	}
	s.rememberRetainedHidden(hidden)
	return next, nil
}

func readPromptEntryBytes(data []byte, off, want int, label string) ([]byte, int, error) {
	if off+4 > len(data) {
		return nil, off, core.NewError("native.RestoreState: truncated prompt-cache entry " + label)
	}
	n := int(binary.LittleEndian.Uint32(data[off:]))
	off += 4
	if n != want {
		return nil, off, core.NewError("native.RestoreState: prompt-cache entry " + label + " size mismatch")
	}
	if off+n > len(data) {
		return nil, off, core.NewError("native.RestoreState: truncated prompt-cache entry " + label)
	}
	buf := make([]byte, n)
	copy(buf, data[off:off+n])
	return buf, off + n, nil
}

func (s *ArchSession) snapshotCacheBuffers(li int) (metal.MTLBuffer, metal.MTLBuffer, error) {
	if s.state.icb != nil {
		if li >= len(s.state.icb.kCaches) || li >= len(s.state.icb.vCaches) {
			return nil, nil, core.NewError("native.sessionState: ICB cache index out of range")
		}
		k, v := s.state.icb.kCaches[li], s.state.icb.vCaches[li]
		if k == nil || v == nil {
			return nil, nil, core.NewError("native.sessionState: missing ICB cache buffer")
		}
		return k, v, nil
	}
	if li >= len(s.state.lb) {
		return nil, nil, core.NewError("native.sessionState: cache index out of range")
	}
	k, v := s.state.lb[li].kCache, s.state.lb[li].vCache
	if k == nil || v == nil {
		return nil, nil, core.NewError("native.sessionState: missing cache buffer")
	}
	return k, v, nil
}
