// SPDX-Licence-Identifier: EUPL-1.2

package mlx

const (
	rootMinInt32 = -1 << 31
	rootMaxInt32 = 1<<31 - 1
)

func normalizeRootInt32Arg(kind string, value any) int32 {
	switch v := value.(type) {
	case int:
		return rootInt64ToInt32(kind, int64(v))
	case int8:
		return int32(v)
	case int16:
		return int32(v)
	case int32:
		return v
	case int64:
		return rootInt64ToInt32(kind, v)
	case uint:
		return rootUint64ToInt32(kind, uint64(v))
	case uint8:
		return int32(v)
	case uint16:
		return int32(v)
	case uint32:
		return rootUint64ToInt32(kind, uint64(v))
	case uint64:
		return rootUint64ToInt32(kind, v)
	default:
		panic("mlx: " + kind + " must be an int-compatible value")
	}
}

func rootInt64ToInt32(kind string, value int64) int32 {
	if value < rootMinInt32 || value > rootMaxInt32 {
		panic("mlx: " + kind + " is out of int32 range")
	}
	return int32(value)
}

func rootUint64ToInt32(kind string, value uint64) int32 {
	if value > rootMaxInt32 {
		panic("mlx: " + kind + " is out of int32 range")
	}
	return int32(value)
}

func normalizeRootIntArg(kind string, value any) int {
	return int(normalizeRootInt32Arg(kind, value))
}

func normalizeRootShapeArgs(shape []any) []int32 {
	if len(shape) == 1 {
		// Typed-slice fast paths skip per-element interface boxing through
		// normalizeRootInt32Arg, which would re-wrap each value in `any`.
		switch dims := shape[0].(type) {
		case []int:
			out := make([]int32, len(dims))
			for i, dim := range dims {
				out[i] = rootInt64ToInt32("shape", int64(dim))
			}
			return out
		case []int32:
			// Skip the defensive clone — the sole caller (Reshape) spreads
			// the result via `...` into metal.Reshape, which copies the
			// values into a C buffer and never retains the slice header.
			// Eliding the clone saves the only allocation in this path and
			// converts it from O(n) memcpy + alloc to O(1) pointer return.
			// Behavioural contract: callers may not mutate the input slice
			// expecting isolation from the returned slice.
			return dims
		case []int64:
			out := make([]int32, len(dims))
			for i, dim := range dims {
				out[i] = rootInt64ToInt32("shape", dim)
			}
			return out
		case []uint:
			out := make([]int32, len(dims))
			for i, dim := range dims {
				out[i] = rootUint64ToInt32("shape", uint64(dim))
			}
			return out
		case []uint32:
			out := make([]int32, len(dims))
			for i, dim := range dims {
				out[i] = rootUint64ToInt32("shape", uint64(dim))
			}
			return out
		case []uint64:
			out := make([]int32, len(dims))
			for i, dim := range dims {
				out[i] = rootUint64ToInt32("shape", dim)
			}
			return out
		}
	}

	// Inline the type switch on the variadic walk — normalizeRootInt32Arg
	// is over the inliner budget (10-case switch), so the per-element
	// function call costs a kind-string push + return jump on every dim.
	// For a 4D shape that's 4 saved calls per Reshape, and Reshape fires
	// per-token during generation. The int / int32 / int64 cases are the
	// only ones the per-token Reshape path actually hits — keep them at
	// the top of the switch; everything else falls through to the shared
	// helper to keep the binary size bounded.
	out := make([]int32, len(shape))
	for i, dim := range shape {
		switch v := dim.(type) {
		case int:
			out[i] = rootInt64ToInt32("shape", int64(v))
		case int32:
			out[i] = v
		case int64:
			out[i] = rootInt64ToInt32("shape", v)
		default:
			out[i] = normalizeRootInt32Arg("shape", dim)
		}
	}
	return out
}
