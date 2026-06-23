// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"testing"

	"dappco.re/go/mlx/pkg/model"
)

func BenchmarkNativeTokenModelEmbed(b *testing.B) {
	g, arch := gemma4BF16Fixture(b, 64, 1, 1, 64, 128, 32, 1)
	tm, err := NewBF16TokenModel(g, arch, 4)
	if err != nil {
		b.Fatal(err)
	}
	b.SetBytes(int64(arch.Hidden * bf16Size))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := tm.Embed(int32(i % arch.Vocab)); err != nil {
			b.Fatal(err)
		}
	}
}

type nativeTokenModelNoDirectGenerate struct {
	*NativeTokenModel
}

func (m nativeTokenModelNoDirectGenerate) OpenSession() (model.DecodeStepper, error) {
	sess, err := m.NativeTokenModel.OpenSession()
	if err != nil {
		return nil, err
	}
	return noDirectGenerateStepper{sess: sess}, nil
}

type noDirectGenerateStepper struct {
	sess model.DecodeStepper
}

func (s noDirectGenerateStepper) Step(emb []byte) ([]byte, error) {
	return s.sess.Step(emb)
}

func (s noDirectGenerateStepper) StepWithID(id int32, emb []byte) ([]byte, error) {
	if stepID, ok := s.sess.(interface {
		StepWithID(int32, []byte) ([]byte, error)
	}); ok {
		return stepID.StepWithID(id, emb)
	}
	return s.sess.Step(emb)
}

func (s noDirectGenerateStepper) Close() error {
	if c, ok := s.sess.(interface{ Close() error }); ok {
		return c.Close()
	}
	return nil
}

func BenchmarkNativeTokenModelGenerateStepwiseHead(b *testing.B) {
	requireNativeRuntime(b)

	g, arch := gemma4BF16Fixture(b, 128, 2, 1, 64, 256, 32768, 2)
	tm, err := NewBF16TokenModel(g, arch, 16)
	if err != nil {
		b.Fatal(err)
	}
	wrapped := nativeTokenModelNoDirectGenerate{NativeTokenModel: tm}
	prompt := []int32{1, 5, 3, 9}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := model.Generate(wrapped, prompt, 6, -1); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkNativeTokenModelGenerateDirectSession(b *testing.B) {
	requireNativeRuntime(b)

	g, arch := gemma4BF16Fixture(b, 128, 2, 1, 64, 256, 32768, 2)
	tm, err := NewBF16TokenModel(g, arch, 16)
	if err != nil {
		b.Fatal(err)
	}
	prompt := []int32{1, 5, 3, 9}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := model.Generate(tm, prompt, 6, -1); err != nil {
			b.Fatal(err)
		}
	}
}
