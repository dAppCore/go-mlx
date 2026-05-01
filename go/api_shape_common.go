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
		switch dims := shape[0].(type) {
		case []int:
			out := make([]int32, len(dims))
			for i, dim := range dims {
				out[i] = normalizeRootInt32Arg("shape", dim)
			}
			return out
		case []int32:
			return append([]int32(nil), dims...)
		case []int64:
			out := make([]int32, len(dims))
			for i, dim := range dims {
				out[i] = normalizeRootInt32Arg("shape", dim)
			}
			return out
		case []uint:
			out := make([]int32, len(dims))
			for i, dim := range dims {
				out[i] = normalizeRootInt32Arg("shape", dim)
			}
			return out
		case []uint32:
			out := make([]int32, len(dims))
			for i, dim := range dims {
				out[i] = normalizeRootInt32Arg("shape", dim)
			}
			return out
		case []uint64:
			out := make([]int32, len(dims))
			for i, dim := range dims {
				out[i] = normalizeRootInt32Arg("shape", dim)
			}
			return out
		}
	}

	out := make([]int32, len(shape))
	for i, dim := range shape {
		out[i] = normalizeRootInt32Arg("shape", dim)
	}
	return out
}
