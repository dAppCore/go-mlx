// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package main

import "testing"

func BenchmarkNativeAudioPromptEmbeddingsFeatureRows(b *testing.B) {
	model := audioPromptEmbeddingCommandModel{
		placeholderID:  77,
		embeddingBytes: 2,
		rows: map[int32][]byte{
			10: {0x10, 0x11},
			11: {0x12, 0x13},
			12: {0x14, 0x15},
			77: {0x00, 0x00},
		},
	}
	ids := []int32{10, 77, 11, 77, 12}
	features := []byte{0xa1, 0xa2, 0xb1, 0xb2}

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if _, err := nativeAudioPromptEmbeddings(&model, ids, features); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkNativeVisionPromptEmbeddingsFeatureRows(b *testing.B) {
	model := visionPromptEmbeddingCommandModel{
		imageID:        77,
		videoID:        88,
		embeddingBytes: 2,
		rows: map[int32][]byte{
			10: {0x10, 0x11},
			11: {0x12, 0x13},
			12: {0x14, 0x15},
			77: {0x00, 0x00},
			88: {0x00, 0x00},
		},
	}
	ids := []int32{10, 77, 11, 88, 12}
	imageFeatures := []byte{0xa1, 0xa2}
	videoFeatures := []byte{0xb1, 0xb2}

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if _, err := nativeVisionPromptEmbeddings(&model, ids, imageFeatures, videoFeatures); err != nil {
			b.Fatal(err)
		}
	}
}
