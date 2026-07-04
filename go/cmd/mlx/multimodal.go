// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"time"

	core "dappco.re/go"
)

// multimodalDecodeResult carries the shared verb decode-loop outcome.
type multimodalDecodeResult struct {
	Generated  []int32
	PrefillDur time.Duration
	DecodeDur  time.Duration
}

// countTokenID reports how many times id occurs in ids.
func countTokenID(ids []int32, id int32) int {
	n := 0
	for _, v := range ids {
		if v == id {
			n++
		}
	}
	return n
}

type nativePromptEmbedInto interface {
	EmbeddingBytes() int
	EmbedInto([]byte, int32) ([]byte, error)
}

func nativePromptEmbeddingRows(source any, ids []int32, embed func(int32) ([]byte, error), emptyErr string) ([][]byte, error) {
	if fast, ok := source.(nativePromptEmbedInto); ok {
		rowBytes := fast.EmbeddingBytes()
		if rowBytes > 0 {
			rows := make([][]byte, len(ids))
			storage := make([]byte, len(ids)*rowBytes)
			for i, id := range ids {
				row := storage[i*rowBytes : (i+1)*rowBytes]
				emb, err := fast.EmbedInto(row, id)
				if err != nil {
					return nil, err
				}
				if len(emb) != rowBytes {
					return nil, core.NewError(emptyErr)
				}
				rows[i] = emb
			}
			return rows, nil
		}
	}
	rows := make([][]byte, len(ids))
	for i, id := range ids {
		emb, err := embed(id)
		if err != nil {
			return nil, err
		}
		if len(emb) == 0 {
			return nil, core.NewError(emptyErr)
		}
		rows[i] = emb
	}
	return rows, nil
}
