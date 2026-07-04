// SPDX-Licence-Identifier: EUPL-1.2

package hf

import (
	"fmt"
)

// ExampleInferJANG shows JANG metadata inference from a model's id, tags and
// filenames. A "jangtq" token (here in the tag list) selects the fixed JANGTQ
// profile; the group size falls back to 64 when no quantization block declares
// one. The filename is only a needle — quant width comes from the profile, not
// the file name.
func ExampleInferJANG() {
	info := InferJANG(ModelMetadata{
		ID:   "dealignai/MiniMax-M2-JANGTQ",
		Tags: []string{"mlx", "jang", "jangtq"},
		Files: []ModelFile{
			{Name: "model-00001-of-00061.safetensors"},
			{Name: "jangtq_runtime.safetensors"},
		},
	})
	fmt.Println(info.Profile, info.WeightFormat, info.BitsDefault, info.GroupSize)
	// Output: JANGTQ mxtq 2 64
}

// ExampleInferJANG_filenameNeedle shows that the JANGTQ profile is selected
// from a weight *filename* alone — neither the id nor the tags carry a needle
// here. A "jangtq" filename is the strongest signal and pins the JANGTQ
// profile (2-bit MXTQ, group size 64) just as a tag would.
func ExampleInferJANG_filenameNeedle() {
	info := InferJANG(ModelMetadata{
		ID: "acme/MiniMax-M2",
		Files: []ModelFile{
			{Name: "model-00001-of-00061.safetensors"},
			{Name: "jangtq_runtime.safetensors"},
		},
	})
	fmt.Println(info.Profile, info.WeightFormat, info.BitsDefault, info.GroupSize)
	// Output: JANGTQ mxtq 2 64
}

// ExampleInferJANG_noNeedle shows the negative result: a model with no JANG
// needle in its id, tags or filenames is not a JANG model, so InferJANG
// returns nil. Callers treat nil as "ordinary (non-JANG) weights".
func ExampleInferJANG_noNeedle() {
	info := InferJANG(ModelMetadata{
		ID:    "Qwen/Qwen3-0.6B",
		Tags:  []string{"mlx", "text-generation"},
		Files: []ModelFile{{Name: "model.safetensors"}},
	})
	fmt.Println(info == nil)
	// Output: true
}
