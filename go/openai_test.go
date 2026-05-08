// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import "testing"

func TestOpenAI_NewOpenAIResolver_Good_UsesMetalBackend(t *testing.T) {
	resolver := NewOpenAIResolver("/models/qwen3")
	if resolver == nil {
		t.Fatal("NewOpenAIResolver() returned nil")
	}
	if resolver.BackendName != "metal" {
		t.Fatalf("BackendName = %q, want metal", resolver.BackendName)
	}
	if resolver.ModelPath != "/models/qwen3" {
		t.Fatalf("ModelPath = %q", resolver.ModelPath)
	}
}

func TestOpenAI_NewOpenAIHandler_Good_ReturnsHTTPHandler(t *testing.T) {
	handler := NewOpenAIHandler("/models/qwen3")
	if handler == nil {
		t.Fatal("NewOpenAIHandler() returned nil")
	}
}
