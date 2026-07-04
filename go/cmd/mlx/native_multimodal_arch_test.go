// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package main

import (
	"errors"
	"os"
	"testing"
)

func TestSharedMultimodalHelperIsNativeOnly_Good(t *testing.T) {
	data, err := os.ReadFile("multimodal.go")
	if errors.Is(err, os.ErrNotExist) {
		return
	}
	if err != nil {
		t.Fatalf("read multimodal.go: %v", err)
	}
	for _, forbidden := range []string{
		`"dappco.re/go/mlx/pkg/metal"`,
		`"dappco.re/go/mlx/pkg/metal/model/gemma4"`,
	} {
		if containsString(data, forbidden) {
			t.Fatalf("shared multimodal helper still imports %s; move shared audio/vision helpers onto the native path", forbidden)
		}
	}
}

func containsString(data []byte, needle string) bool {
	n := len(needle)
	if n == 0 {
		return true
	}
	if len(data) < n {
		return false
	}
	for i := 0; i <= len(data)-n; i++ {
		if string(data[i:i+n]) == needle {
			return true
		}
	}
	return false
}
