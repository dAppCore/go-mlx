// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

// SharedKV is the Gemma 4 runtime's per-layer K/V hand-off between fused kernels
// and the architecture forward pass. It carries only metal-owned arrays + page
// state + scalars (no model architecture type), so it lives in package metal where
// both the kernels (producers) and the model package (consumer) can reference it.
type SharedKV struct {
	Keys     *Array
	Values   *Array
	Pages    PagedKVState
	Offset   int
	Fixed    bool
	Borrowed bool
}

// HasState reports whether the shared K/V carries usable contiguous tensors or
// pages.
func (kv SharedKV) HasState() bool {
	return (kv.Keys != nil && kv.Keys.Valid() && kv.Values != nil && kv.Values.Valid()) || kv.HasPages()
}

// HasPages reports whether the shared K/V carries a complete paged state.
func (kv SharedKV) HasPages() bool {
	if len(kv.Pages.Keys) == 0 || len(kv.Pages.Keys) != len(kv.Pages.Values) {
		return false
	}
	for i := range kv.Pages.Keys {
		if kv.Pages.Keys[i] == nil || !kv.Pages.Keys[i].Valid() || kv.Pages.Values[i] == nil || !kv.Pages.Values[i].Valid() {
			return false
		}
	}
	return true
}

// Free releases the shared K/V handles. Borrowed states leave their (cache-owned)
// contiguous tensors alone and only free page state.
func (kv SharedKV) Free() {
	if !kv.Borrowed {
		Free(kv.Keys, kv.Values)
	}
	kv.Pages.Free()
}

// Clone deep-copies the shared K/V state.
func (kv SharedKV) Clone() SharedKV {
	out := SharedKV{
		Offset: kv.Offset,
		Fixed:  kv.Fixed,
	}
	if kv.Keys != nil && kv.Keys.Valid() {
		out.Keys = kv.Keys.Clone()
	}
	if kv.Values != nil && kv.Values.Valid() {
		out.Values = kv.Values.Clone()
	}
	out.Pages = clonePagedKVState(kv.Pages)
	return out
}

// MoveSharedKV transfers ownership of the shared K/V out of *kv, leaving it zeroed.
func MoveSharedKV(kv *SharedKV) SharedKV {
	if kv == nil {
		return SharedKV{}
	}
	out := *kv
	*kv = SharedKV{}
	return out
}

func clonePagedKVState(state PagedKVState) PagedKVState {
	out := PagedKVState{Length: state.Length}
	if len(state.Keys) == 0 || len(state.Keys) != len(state.Values) {
		return out
	}
	out.Keys = make([]*Array, len(state.Keys))
	out.Values = make([]*Array, len(state.Values))
	out.Owned = make([]*Array, 0, len(state.Keys)+len(state.Values))
	for i := range state.Keys {
		if state.Keys[i] != nil && state.Keys[i].Valid() {
			out.Keys[i] = state.Keys[i].Clone()
			out.Owned = append(out.Owned, out.Keys[i])
		}
		if state.Values[i] != nil && state.Values[i].Valid() {
			out.Values[i] = state.Values[i].Clone()
			out.Owned = append(out.Owned, out.Values[i])
		}
	}
	return out
}
