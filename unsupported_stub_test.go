//go:build !(darwin && arm64) || nomlx

package mlx

import (
	"testing"

	"dappco.re/go/inference"
)

func TestUnsupportedBuildAPISurface_Compile(t *testing.T) {
	_, _ = LoadModel("/tmp/model", WithContextLength(128), WithQuantization(4), WithDevice("cpu"))
	_, _ = LoadTokenizer("/tmp/tokenizer.json")
	_, _ = LoadModelFromMedium(nil, "models/example", WithMedium(nil))
	_, _ = ReadGGUFInfo("/tmp/model.gguf")
	_ = DiscoverModels("/tmp/models")

	model := &Model{}
	_, _ = model.Generate("hello", WithMaxTokens(8), WithTemperature(0.7), WithTopK(10), WithTopP(0.9), WithMinP(0.05))
	_, _ = model.Chat([]Message{{Role: "user", Content: "hi"}}, WithMaxTokens(8))
	for range model.GenerateStream("hello") {
	}
	for range model.ChatStream([]Message{{Role: "user", Content: "hi"}}) {
	}
	_, _ = model.Classify([]string{"hello"}, WithLogits())
	_, _ = model.BatchGenerate([]string{"hello"})
	_ = model.Err()
	_ = model.Metrics()
	_ = model.ModelType()
	_ = model.Info()
	_, _ = model.InspectAttention("hello")
	_ = model.Tokenizer()
	_ = model.Close()

	tok := &Tokenizer{}
	_, _ = tok.Encode("hello")
	_, _ = tok.Decode([]int32{1, 2, 3})
	_, _ = tok.TokenID("hello")
	_ = tok.IDToken(1)
	_ = tok.BOS()
	_ = tok.EOS()

	arr := FromValues([]int32{1, 2, 3, 4}, 2, 2)
	_ = arr.Valid()
	_ = arr.Shape()
	_ = arr.NumDims()
	_ = arr.Dim(0)
	_ = arr.Dims()
	_ = arr.Dtype()
	_ = arr.Int()
	_ = arr.Float()
	_ = arr.Bool()
	arr.SetFloat64(1)
	_ = arr.Ints()
	_ = arr.DataInt32()
	_ = arr.Floats()
	for range arr.Iter() {
	}
	arr.Set(&Array{})
	_ = arr.Clone()

	_ = MatMul(arr, arr)
	_ = Add(arr, arr)
	_ = Mul(arr, arr)
	_ = Softmax(arr)
	_ = Slice(arr, 0, 1, 0)
	_ = Reshape(arr, 1, 4)
	_, _, _ = VJP(func(xs []*Array) []*Array { return xs }, []*Array{arr}, []*Array{arr})
	_, _, _ = JVP(func(xs []*Array) []*Array { return xs }, []*Array{arr}, []*Array{arr})
	_ = Zeros([]int32{1, 4}, DTypeFloat32)
	Materialize(arr)
	Free(arr)

	lora := NewLoRA(model, &LoRAConfig{
		Rank:         8,
		Alpha:        16,
		Scale:        2,
		TargetKeys:   []string{"q_proj", "v_proj"},
		TargetLayers: []string{"q_proj", "v_proj"},
		Lambda:       0.01,
		DType:        DTypeBFloat16,
	})
	_ = model.MergeLoRA(lora)
	_ = DefaultLoRAConfig()
	_ = DefaultAdamWConfig()

	grad := ValueAndGrad(func(xs []*Array) []*Array { return xs }, 0)
	_, _, _ = grad.Apply(arr)
	grad.Free()

	opt := NewAdamW(&AdamWConfig{LearningRate: 1e-4})
	_ = opt.Step([]*Array{arr}, []*Array{arr})
	opt.Reset()

	_ = CrossEntropyLoss(arr, arr)
	_ = MaskedCrossEntropyLoss(arr, arr, arr)
	_ = Checkpoint(func(xs []*Array) []*Array { return xs })([]*Array{arr})

	adapter := &LoRAAdapter{}
	_ = adapter.TotalParams()
	_ = adapter.SortedNames()
	_ = adapter.AllTrainableParams()
	adapter.SetAllParams([]*Array{arr, arr})
	_ = adapter.Step(Batch{Tokens: [][]int{{1, 2}}, Length: []int{2}}, [][]int{{1, 2}}, opt)
	_ = adapter.Save("/tmp/adapter.safetensors")
	adapter.Merge()

	var infAdapter inference.Adapter
	var infTrainable inference.TrainableModel
	_ = ConcreteAdapter(infAdapter)
	_ = TrainingModel(infTrainable)

	streamAdapter := NewInferenceAdapter(nil, "mlx")
	_ = streamAdapter.Name()
	_ = streamAdapter.Available()
	_ = streamAdapter.Model()
	_, _ = streamAdapter.Generate(nil, "hello", GenOpts{MaxTokens: 8, Temp: 0.1})
	_ = streamAdapter.GenerateStream(nil, "hello", GenOpts{}, func(string) error { return nil })
	_, _ = streamAdapter.Chat(nil, []Message{{Role: "user", Content: "hi"}}, GenOpts{})
	_ = streamAdapter.ChatStream(nil, []Message{{Role: "user", Content: "hi"}}, GenOpts{}, func(string) error { return nil })
	_, _ = NewMLXBackend("/tmp/model")

	compute := DefaultCompute()
	_ = compute.Available()
	_ = compute.DeviceInfo()
	_ = ErrComputeUnavailable
	_ = ErrComputeClosed
	_ = ErrComputeInvalidState
	_ = ErrComputeInvalidDescriptor
	_ = ErrComputeUnsupportedPixelFormat
	_ = ErrComputeInvalidBuffer
	_ = ErrComputeBufferSizeMismatch
	_ = ErrComputeInvalidAllocation
	_ = ErrComputeMissingKernelBuffer
	_ = ErrComputeInvalidKernelArgs
	_ = ErrComputeInvalidScalar
	_ = ErrComputeUnknownKernel
	_ = ErrComputeInternal
	_ = (&ComputeError{Kind: ComputeErrorUnknownKernel}).Error()
	_ = FrameMetrics{}
	_, _ = NewSession(
		WithSessionLabel("stub"),
		WithVerboseKernels(true),
		WithResetPeakMemory(true),
	)
	computeDesc := PixelBufferDesc{
		Width:  1,
		Height: 1,
		Stride: 1,
		Format: PixelIndexed8,
	}
	_ = computeDesc.Validate()
	_ = computeDesc.SizeBytes()
	_ = PixelRGBA8.BytesPerPixel()
	_ = PixelBGRA8.BytesPerPixel()
	_ = PixelRGB565.BytesPerPixel()
	_ = PixelXRGB8888.BytesPerPixel()
	_ = PixelIndexed8.BytesPerPixel()
	_ = KernelArgs{
		Inputs:  map[string]Buffer{},
		Outputs: map[string]Buffer{},
		Scalars: map[string]float64{},
	}
	_ = KernelNearestScale
	_ = KernelBilinearScale
	_ = KernelIntegerScale
	_ = KernelRGB565ToRGBA8
	_ = KernelRGBA8ToBGRA8
	_ = KernelBGRA8ToRGBA8
	_ = KernelXRGB8888ToRGBA8
	_ = KernelPaletteExpandRGBA
	_ = KernelScanlineFilter
	_ = KernelCRTFilter
	_ = KernelSoftenFilter
	_ = KernelSharpenFilter
}
