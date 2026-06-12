// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"context"
	"testing"
)

func TestChatMessagesCarryImages_Good(t *testing.T) {
	if chatMessagesCarryImages([]ChatMessage{{Role: "user", Content: "hi"}}) {
		t.Fatal("text-only messages must not read as image-bearing")
	}
	if chatMessagesCarryImages([]ChatMessage{{Role: "user", Images: [][]byte{nil, {}}}}) {
		t.Fatal("empty image slots must not read as image-bearing")
	}
	if !chatMessagesCarryImages([]ChatMessage{{Role: "user", Images: [][]byte{[]byte("png")}}}) {
		t.Fatal("a non-empty image must read as image-bearing")
	}
}

// A text-only checkpoint refuses an image turn loudly — never silently
// dropping the image and answering the text alone (that would fake a
// vision answer).
func TestVisionChat_RejectsTextOnlyModel_Bad(t *testing.T) {
	var nilModel *Model
	if nilModel.AcceptsImages() {
		t.Fatal("nil model must not accept images")
	}
	m := &Model{model: &fakePagedModel{numLayers: 1, pageSize: 8}}
	if m.AcceptsImages() {
		t.Fatal("a text-only internal model must not accept images")
	}

	messages := []ChatMessage{{Role: "user", Content: "what is this?", Images: [][]byte{[]byte("png")}}}
	for range m.chatVision(context.Background(), messages, GenerateConfig{}) {
		t.Fatal("vision chat on a text-only model must emit no tokens")
	}
	err := m.Err()
	if err == nil {
		t.Fatal("vision chat on a text-only model must record an error")
	}
	if got := err.Error(); !containsForTest(got, "does not accept image input") {
		t.Fatalf("error = %q, want the capability refusal", got)
	}
}

func containsForTest(haystack, needle string) bool {
	for i := 0; i+len(needle) <= len(haystack); i++ {
		if haystack[i:i+len(needle)] == needle {
			return true
		}
	}
	return false
}
