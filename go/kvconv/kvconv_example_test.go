// SPDX-Licence-Identifier: EUPL-1.2

package kvconv_test

import (
	"fmt"

	"dappco.re/go/inference/kv"
	"dappco.re/go/mlx/kvconv"
	"dappco.re/go/mlx/pkg/metal"
)

// ExampleRootKVHeadDType shows the metal dtype -> root tag mapping. The tag is
// blank when the head carries no raw bytes.
func ExampleRootKVHeadDType() {
	fmt.Println(kvconv.RootKVHeadDType(metal.DTypeBFloat16, []byte{0, 0}))
	fmt.Println(kvconv.RootKVHeadDType(metal.DTypeFloat32, nil))
	// Output:
	// bfloat16
	//
}

// ExampleMetalKVHeadDType shows the root tag -> metal dtype mapping, including
// the safetensors short-alias form. A missing tensor (no bytes) yields the
// zero dtype.
func ExampleMetalKVHeadDType() {
	raw := []byte{0, 0}
	fmt.Println(kvconv.MetalKVHeadDType("float16", raw) == metal.DTypeFloat16)
	fmt.Println(kvconv.MetalKVHeadDType("BF16", raw) == metal.DTypeBFloat16)
	fmt.Println(kvconv.MetalKVHeadDType("float32", nil) == 0)
	// Output:
	// true
	// true
	// true
}

// ExampleToMetalKVSnapshotCaptureOptions shows the capture-option fields
// carried verbatim across the boundary.
func ExampleToMetalKVSnapshotCaptureOptions() {
	out := kvconv.ToMetalKVSnapshotCaptureOptions(kv.CaptureOptions{RawKVOnly: true, BlockStartToken: 8})
	fmt.Println(out.RawKVOnly, out.BlockStartToken)
	// Output: true 8
}

// ExampleToRootKVSnapshot converts a metal snapshot into the root surface and
// reports the layer count and the resolved head dtype tag.
func ExampleToRootKVSnapshot() {
	src := &metal.KVSnapshot{
		Version:   4,
		NumLayers: 1,
		Layers: []metal.KVLayerSnapshot{{
			Layer:     0,
			CacheMode: metal.KVCacheModeFP16,
			Heads: []metal.KVHeadSnapshot{{
				Key:      []float32{1, 2},
				KeyDType: metal.DTypeFloat16,
				KeyBytes: []byte{0, 0},
			}},
		}},
	}
	root := kvconv.ToRootKVSnapshot(src)
	fmt.Println(len(root.Layers), root.Layers[0].CacheMode, root.Layers[0].Heads[0].KeyDType)
	// Output: 1 fp16 float16
}

// ExampleToMetalKVSnapshot converts a root snapshot back to the metal surface
// and reports the layer count and the resolved cache mode.
func ExampleToMetalKVSnapshot() {
	root := &kv.Snapshot{
		Version:   4,
		NumLayers: 1,
		Layers: []kv.LayerSnapshot{{
			Layer:     0,
			CacheMode: "q8",
			Heads: []kv.HeadSnapshot{{
				Key:      []float32{1, 2},
				KeyDType: "float32",
				KeyBytes: []byte{0, 0},
			}},
		}},
	}
	out := kvconv.ToMetalKVSnapshot(root)
	fmt.Println(len(out.Layers), out.Layers[0].CacheMode == metal.KVCacheModeQ8)
	// Output: 1 true
}
