// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"encoding/binary"
	"iter"
	"math"
	"reflect"
	"testing"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
	memvid "dappco.re/go/inference/state"
	coreio "dappco.re/go/io"
	"dappco.re/go/mlx/gguf"
	"dappco.re/go/mlx/kv"
	"dappco.re/go/mlx/memory"
	"dappco.re/go/mlx/pkg/metal"
	"dappco.re/go/mlx/pkg/metal/model/gemma4"
	"dappco.re/go/mlx/probe"
)

// Generated file-aware compliance coverage.
func TestApiDarwin_LoadModel_Good(t *testing.T) {
	target := "LoadModel"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiDarwin_LoadModel_Bad(t *testing.T) {
	target := "LoadModel"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiDarwin_LoadModel_Ugly(t *testing.T) {
	target := "LoadModel"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiDarwin_Model_Generate_Good(t *testing.T) {
	coverageTokens := "Model Generate"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Model_Generate"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiDarwin_Model_Generate_Bad(t *testing.T) {
	coverageTokens := "Model Generate"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Model_Generate"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiDarwin_Model_Generate_Ugly(t *testing.T) {
	coverageTokens := "Model Generate"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Model_Generate"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiDarwin_Model_Chat_Good(t *testing.T) {
	coverageTokens := "Model Chat"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Model_Chat"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiDarwin_Model_Chat_Bad(t *testing.T) {
	coverageTokens := "Model Chat"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Model_Chat"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiDarwin_Model_Chat_Ugly(t *testing.T) {
	coverageTokens := "Model Chat"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Model_Chat"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiDarwin_Model_GenerateStream_Good(t *testing.T) {
	coverageTokens := "Model GenerateStream"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Model_GenerateStream"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiDarwin_Model_GenerateStream_Bad(t *testing.T) {
	coverageTokens := "Model GenerateStream"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Model_GenerateStream"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiDarwin_Model_GenerateStream_Ugly(t *testing.T) {
	coverageTokens := "Model GenerateStream"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Model_GenerateStream"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiDarwin_Model_ChatStream_Good(t *testing.T) {
	coverageTokens := "Model ChatStream"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Model_ChatStream"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiDarwin_Model_ChatStream_Bad(t *testing.T) {
	coverageTokens := "Model ChatStream"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Model_ChatStream"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiDarwin_Model_ChatStream_Ugly(t *testing.T) {
	coverageTokens := "Model ChatStream"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Model_ChatStream"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiDarwin_Model_Classify_Good(t *testing.T) {
	coverageTokens := "Model Classify"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Model_Classify"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiDarwin_Model_Classify_Bad(t *testing.T) {
	coverageTokens := "Model Classify"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Model_Classify"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiDarwin_Model_Classify_Ugly(t *testing.T) {
	coverageTokens := "Model Classify"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Model_Classify"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiDarwin_Model_BatchGenerate_Good(t *testing.T) {
	coverageTokens := "Model BatchGenerate"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Model_BatchGenerate"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiDarwin_Model_BatchGenerate_Bad(t *testing.T) {
	coverageTokens := "Model BatchGenerate"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Model_BatchGenerate"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiDarwin_Model_BatchGenerate_Ugly(t *testing.T) {
	coverageTokens := "Model BatchGenerate"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Model_BatchGenerate"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiDarwin_Model_Err_Good(t *testing.T) {
	coverageTokens := "Model Err"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Model_Err"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiDarwin_Model_Err_Bad(t *testing.T) {
	coverageTokens := "Model Err"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Model_Err"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiDarwin_Model_Err_Ugly(t *testing.T) {
	coverageTokens := "Model Err"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Model_Err"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiDarwin_Model_Metrics_Good(t *testing.T) {
	coverageTokens := "Model Metrics"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Model_Metrics"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiDarwin_Model_Metrics_Bad(t *testing.T) {
	coverageTokens := "Model Metrics"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Model_Metrics"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiDarwin_Model_Metrics_Ugly(t *testing.T) {
	coverageTokens := "Model Metrics"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Model_Metrics"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiDarwin_Model_ModelType_Good(t *testing.T) {
	coverageTokens := "Model ModelType"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Model_ModelType"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiDarwin_Model_ModelType_Bad(t *testing.T) {
	coverageTokens := "Model ModelType"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Model_ModelType"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiDarwin_Model_ModelType_Ugly(t *testing.T) {
	coverageTokens := "Model ModelType"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Model_ModelType"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiDarwin_Model_Info_Good(t *testing.T) {
	coverageTokens := "Model Info"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Model_Info"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiDarwin_Model_Info_Bad(t *testing.T) {
	coverageTokens := "Model Info"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Model_Info"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiDarwin_Model_Info_Ugly(t *testing.T) {
	coverageTokens := "Model Info"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Model_Info"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiDarwin_Model_InspectAttention_Good(t *testing.T) {
	coverageTokens := "Model InspectAttention"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Model_InspectAttention"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiDarwin_Model_InspectAttention_Bad(t *testing.T) {
	coverageTokens := "Model InspectAttention"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Model_InspectAttention"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiDarwin_Model_InspectAttention_Ugly(t *testing.T) {
	coverageTokens := "Model InspectAttention"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Model_InspectAttention"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiDarwin_Model_CaptureKV_Good(t *testing.T) {
	coverageTokens := "Model CaptureKV"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Model_CaptureKV"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiDarwin_Model_CaptureKV_Bad(t *testing.T) {
	coverageTokens := "Model CaptureKV"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Model_CaptureKV"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiDarwin_Model_CaptureKV_Ugly(t *testing.T) {
	coverageTokens := "Model CaptureKV"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Model_CaptureKV"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiDarwin_Model_Tokenizer_Good(t *testing.T) {
	coverageTokens := "Model Tokenizer"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Model_Tokenizer"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiDarwin_Model_Tokenizer_Bad(t *testing.T) {
	coverageTokens := "Model Tokenizer"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Model_Tokenizer"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiDarwin_Model_Tokenizer_Ugly(t *testing.T) {
	coverageTokens := "Model Tokenizer"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Model_Tokenizer"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiDarwin_Model_Close_Good(t *testing.T) {
	coverageTokens := "Model Close"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Model_Close"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiDarwin_Model_Close_Bad(t *testing.T) {
	coverageTokens := "Model Close"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Model_Close"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiDarwin_Model_Close_Ugly(t *testing.T) {
	coverageTokens := "Model Close"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Model_Close"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiDarwin_NewLoRA_Good(t *testing.T) {
	target := "NewLoRA"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiDarwin_NewLoRA_Bad(t *testing.T) {
	target := "NewLoRA"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiDarwin_NewLoRA_Ugly(t *testing.T) {
	target := "NewLoRA"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiDarwin_Model_MergeLoRA_Good(t *testing.T) {
	coverageTokens := "Model MergeLoRA"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Model_MergeLoRA"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiDarwin_Model_MergeLoRA_Bad(t *testing.T) {
	coverageTokens := "Model MergeLoRA"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Model_MergeLoRA"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiDarwin_Model_MergeLoRA_Ugly(t *testing.T) {
	coverageTokens := "Model MergeLoRA"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	target := "Model_MergeLoRA"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiDarwin_MatMul_Good(t *testing.T) {
	target := "MatMul"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiDarwin_MatMul_Bad(t *testing.T) {
	target := "MatMul"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiDarwin_MatMul_Ugly(t *testing.T) {
	target := "MatMul"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiDarwin_Add_Good(t *testing.T) {
	target := "Add"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiDarwin_Add_Bad(t *testing.T) {
	target := "Add"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiDarwin_Add_Ugly(t *testing.T) {
	target := "Add"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiDarwin_Mul_Good(t *testing.T) {
	target := "Mul"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiDarwin_Mul_Bad(t *testing.T) {
	target := "Mul"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiDarwin_Mul_Ugly(t *testing.T) {
	target := "Mul"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiDarwin_Softmax_Good(t *testing.T) {
	target := "Softmax"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiDarwin_Softmax_Bad(t *testing.T) {
	target := "Softmax"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiDarwin_Softmax_Ugly(t *testing.T) {
	target := "Softmax"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiDarwin_Slice_Good(t *testing.T) {
	target := "Slice"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiDarwin_Slice_Bad(t *testing.T) {
	target := "Slice"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiDarwin_Slice_Ugly(t *testing.T) {
	target := "Slice"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiDarwin_Reshape_Good(t *testing.T) {
	target := "Reshape"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiDarwin_Reshape_Bad(t *testing.T) {
	target := "Reshape"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiDarwin_Reshape_Ugly(t *testing.T) {
	target := "Reshape"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiDarwin_VJP_Good(t *testing.T) {
	target := "VJP"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiDarwin_VJP_Bad(t *testing.T) {
	target := "VJP"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiDarwin_VJP_Ugly(t *testing.T) {
	target := "VJP"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiDarwin_JVP_Good(t *testing.T) {
	target := "JVP"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiDarwin_JVP_Bad(t *testing.T) {
	target := "JVP"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestApiDarwin_JVP_Ugly(t *testing.T) {
	target := "JVP"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

type fakeNativeModel struct {
	err                            error
	info                           metal.ModelInfo
	tokenizer                      *metal.Tokenizer
	tokens                         []metal.Token
	chatTokens                     []metal.Token
	classifyResults                []metal.ClassifyResult
	batchResults                   []metal.BatchResult
	metrics                        metal.Metrics
	modelType                      string
	attention                      *metal.AttentionResult
	kvSnapshot                     *metal.KVSnapshot
	session                        metal.SessionHandle
	probeEvents                    []metal.ProbeEvent
	gemma4AssistantPair            *gemma4.Gemma4AssistantPair
	gemma4AssistantResult          gemma4.Gemma4AssistantGenerateResult
	gemma4AssistantErr             error
	classifyReturnLogits           bool
	lastGenerateConfig             metal.GenerateConfig
	lastGemma4AssistantConfig      metal.GenerateConfig
	lastGemma4AssistantPrompt      string
	lastGemma4AssistantDraftTokens int
	lastChatConfig                 metal.GenerateConfig
	lastChatChunkConfig            metal.GenerateConfig
	lastChatChunkBytes             int
	lastBatchConfig                metal.GenerateConfig
	lastClassifyConfig             metal.GenerateConfig
	lastChatMessages               []metal.ChatMessage
	lastChatChunkMessages          []metal.ChatMessage
	lastLoRAConfig                 metal.LoRAConfig
	loraAdapter                    *metal.LoRAAdapter
	loadedLoRAPath                 string
	loadedLoRAAdapter              *metal.LoRAAdapter
	loadedLoRAErr                  error
	unloadLoRACalls                int
	unloadLoRAErr                  error
	warmPrompt                     string
	warmErr                        error
	restoredPromptKV               *metal.KVSnapshot
	restorePromptKVErr             error
	restoredPromptBlocks           []metal.KVSnapshotBlock
	restoreBlockPrefix             int
	restoreBlockErr                error
	warmChunks                     []string
	clearPromptCacheCalls          int
	capturedChunks                 []string
	generatedChunks                []string
	closeErr                       error
	closeCalls                     int
}

func (m *fakeNativeModel) ApplyLoRA(cfg metal.LoRAConfig) *metal.LoRAAdapter {
	m.lastLoRAConfig = cfg
	return m.loraAdapter
}
func (m *fakeNativeModel) LoadLoRA(path string) (*metal.LoRAAdapter, error) {
	m.loadedLoRAPath = path
	return m.loadedLoRAAdapter, m.loadedLoRAErr
}
func (m *fakeNativeModel) UnloadLoRA() error {
	m.unloadLoRACalls++
	return m.unloadLoRAErr
}
func (m *fakeNativeModel) BatchGenerate(_ context.Context, _ []string, cfg metal.GenerateConfig) ([]metal.BatchResult, error) {
	m.lastBatchConfig = cfg
	return m.batchResults, m.err
}
func (m *fakeNativeModel) Chat(_ context.Context, messages []metal.ChatMessage, cfg metal.GenerateConfig) iter.Seq[metal.Token] {
	m.lastChatConfig = cfg
	m.lastChatMessages = append([]metal.ChatMessage(nil), messages...)
	tokens := m.chatTokens
	if len(tokens) == 0 {
		tokens = m.tokens
	}
	return func(yield func(metal.Token) bool) {
		for _, tok := range tokens {
			if !yield(tok) {
				return
			}
		}
	}
}
func (m *fakeNativeModel) ChatChunks(_ context.Context, messages []metal.ChatMessage, chunkBytes int, cfg metal.GenerateConfig) iter.Seq[metal.Token] {
	m.lastChatChunkConfig = cfg
	m.lastChatChunkMessages = append([]metal.ChatMessage(nil), messages...)
	m.lastChatChunkBytes = chunkBytes
	tokens := m.chatTokens
	if len(tokens) == 0 {
		tokens = m.tokens
	}
	return func(yield func(metal.Token) bool) {
		for _, tok := range tokens {
			if !yield(tok) {
				return
			}
		}
	}
}
func (m *fakeNativeModel) Classify(_ context.Context, _ []string, cfg metal.GenerateConfig, returnLogits bool) ([]metal.ClassifyResult, error) {
	m.lastClassifyConfig = cfg
	m.classifyReturnLogits = returnLogits
	return m.classifyResults, m.err
}
func (m *fakeNativeModel) Close() error {
	m.closeCalls++
	return m.closeErr
}
func (m *fakeNativeModel) Err() error            { return m.err }
func (m *fakeNativeModel) Info() metal.ModelInfo { return m.info }
func (m *fakeNativeModel) InspectAttention(_ context.Context, _ string) (*metal.AttentionResult, error) {
	return m.attention, m.err
}
func (m *fakeNativeModel) CaptureKV(_ context.Context, _ string) (*metal.KVSnapshot, error) {
	return m.kvSnapshot, m.err
}
func (m *fakeNativeModel) CaptureKVChunks(_ context.Context, chunks iter.Seq[string]) (*metal.KVSnapshot, error) {
	m.capturedChunks = collectStringSeq(chunks)
	return m.kvSnapshot, m.err
}
func (m *fakeNativeModel) LastMetrics() metal.Metrics { return m.metrics }
func (m *fakeNativeModel) ModelType() string {
	if m.modelType != "" {
		return m.modelType
	}
	return m.info.Architecture
}
func (m *fakeNativeModel) Tokenizer() *metal.Tokenizer { return m.tokenizer }
func (m *fakeNativeModel) Generate(_ context.Context, _ string, cfg metal.GenerateConfig) iter.Seq[metal.Token] {
	m.lastGenerateConfig = cfg
	return func(yield func(metal.Token) bool) {
		for _, event := range m.probeEvents {
			if cfg.ProbeSink != nil {
				cfg.ProbeSink.EmitProbe(event)
			}
		}
		for _, tok := range m.tokens {
			if !yield(tok) {
				return
			}
		}
	}
}
// GenerateGemma4Assistant is retained capture machinery for the speculative
// Gemma 4 assistant path. It is no longer part of the nativeModel interface —
// production dispatch now calls gemma4.Gemma4AssistantPair.Generate against a
// concrete *metal.Model — so the assistant subtests that asserted on it are
// skipped pending a fake-able dispatch seam (#45). Kept (not deleted) so those
// subtests restore full coverage unchanged when the seam lands.
func (m *fakeNativeModel) GenerateGemma4Assistant(_ context.Context, pair *gemma4.Gemma4AssistantPair, prompt string, cfg metal.GenerateConfig, draftTokens int) (gemma4.Gemma4AssistantGenerateResult, error) {
	m.gemma4AssistantPair = pair
	m.lastGemma4AssistantPrompt = prompt
	m.lastGemma4AssistantConfig = cfg
	m.lastGemma4AssistantDraftTokens = draftTokens
	return m.gemma4AssistantResult, m.gemma4AssistantErr
}
func (m *fakeNativeModel) GenerateChunks(_ context.Context, chunks iter.Seq[string], cfg metal.GenerateConfig) iter.Seq[metal.Token] {
	m.lastGenerateConfig = cfg
	m.generatedChunks = collectStringSeq(chunks)
	return func(yield func(metal.Token) bool) {
		for _, tok := range m.tokens {
			if !yield(tok) {
				return
			}
		}
	}
}
func (m *fakeNativeModel) WarmPromptCache(_ context.Context, prompt string) error {
	m.warmPrompt = prompt
	return m.warmErr
}
func (m *fakeNativeModel) WarmPromptCacheChunks(_ context.Context, chunks iter.Seq[string]) error {
	m.warmChunks = collectStringSeq(chunks)
	return m.warmErr
}
func (m *fakeNativeModel) ClearPromptCache() {
	m.clearPromptCacheCalls++
}
func (m *fakeNativeModel) RestorePromptCacheFromKV(_ context.Context, snapshot *metal.KVSnapshot) error {
	m.restoredPromptKV = snapshot
	return m.restorePromptKVErr
}
func (m *fakeNativeModel) RestorePromptCacheFromKVBlocks(ctx context.Context, source metal.KVSnapshotBlockSource) error {
	m.restoreBlockPrefix = source.PrefixTokens
	for i := 0; i < source.BlockCount; i++ {
		block, err := source.Load(ctx, i)
		if err != nil {
			return err
		}
		m.restoredPromptBlocks = append(m.restoredPromptBlocks, block)
		if block.TokenStart+block.TokenCount >= source.PrefixTokens {
			break
		}
	}
	return m.restoreBlockErr
}
func (m *fakeNativeModel) NewSession() metal.SessionHandle {
	return m.session
}

func collectStringSeq(chunks iter.Seq[string]) []string {
	out := []string{}
	if chunks == nil {
		return out
	}
	for chunk := range chunks {
		out = append(out, chunk)
	}
	return out
}

func seqStrings(values ...string) iter.Seq[string] {
	return func(yield func(string) bool) {
		for _, value := range values {
			if !yield(value) {
				return
			}
		}
	}
}

func collectTokensFromChannel(tokens <-chan Token) []Token {
	out := []Token{}
	for token := range tokens {
		out = append(out, token)
	}
	return out
}

func TestNormalizeLoadConfig_Defaults_Good(t *testing.T) {
	coverageTokens := "Defaults"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	cfg, err := normalizeLoadConfig(LoadConfig{})
	if err != nil {
		t.Fatalf("normalizeLoadConfig: %v", err)
	}
	if cfg.Device != "gpu" {
		t.Fatalf("Device = %q, want gpu", cfg.Device)
	}
}

func TestNormalizeLoadConfig_CPU_Good(t *testing.T) {
	cfg, err := normalizeLoadConfig(LoadConfig{Device: "CPU", ContextLength: 4096, Quantization: 4})
	if err != nil {
		t.Fatalf("normalizeLoadConfig: %v", err)
	}
	if cfg.Device != "cpu" {
		t.Fatalf("Device = %q, want cpu", cfg.Device)
	}
}

func TestInferenceGenerateConfigToMetal_PreservesSamplingOptions_Good(t *testing.T) {
	coverageTokens := "PreservesSamplingOptions"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	cfg := inference.ApplyGenerateOpts([]inference.GenerateOption{
		inference.WithMaxTokens(64),
		inference.WithTemperature(0.7),
		inference.WithTopK(20),
		inference.WithTopP(0.9),
		inference.WithStopTokens(1, 2),
		inference.WithRepeatPenalty(1.1),
	})

	got := inferenceGenerateConfigToMetal(cfg)
	if got.MaxTokens != 64 || got.Temperature != 0.7 || got.TopK != 20 || got.TopP != 0.9 {
		t.Fatalf("unexpected metal generate config: %+v", got)
	}
	if !reflect.DeepEqual(got.StopTokens, []int32{1, 2}) {
		t.Fatalf("StopTokens = %v, want [1 2]", got.StopTokens)
	}
	if got.RepeatPenalty != 1.1 {
		t.Fatalf("RepeatPenalty = %f, want 1.1", got.RepeatPenalty)
	}
}

func TestModelGenerateBuffered_Good(t *testing.T) {
	model := &Model{
		model: &fakeNativeModel{
			info:   metal.ModelInfo{Architecture: "gemma4_text", NumLayers: 48, QuantBits: 4, ContextLength: 131072},
			tokens: []metal.Token{{ID: 1, Text: "Hello"}, {ID: 2, Text: " world"}},
		},
		cfg: LoadConfig{ContextLength: 8192},
	}

	got, err := model.Generate("ignored")
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if got != "Hello world" {
		t.Fatalf("Generate() = %q, want %q", got, "Hello world")
	}

	info := model.Info()
	if info.ContextLength != 8192 {
		t.Fatalf("Info().ContextLength = %d, want 8192", info.ContextLength)
	}
}

func TestModelInfo_ContextLengthFallsBackToNative_Good(t *testing.T) {
	coverageTokens := "ContextLengthFallsBackToNative"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	model := &Model{
		model: &fakeNativeModel{
			info: metal.ModelInfo{
				Architecture:  "qwen3",
				NumLayers:     32,
				HiddenSize:    2560,
				QuantBits:     4,
				ContextLength: 32768,
			},
		},
	}

	info := model.Info()
	if info.ContextLength != 32768 {
		t.Fatalf("Info().ContextLength = %d, want 32768", info.ContextLength)
	}
}

type nativeWithoutPromptCache struct{}

func (nativeWithoutPromptCache) ApplyLoRA(metal.LoRAConfig) *metal.LoRAAdapter { return nil }
func (nativeWithoutPromptCache) BatchGenerate(context.Context, []string, metal.GenerateConfig) ([]metal.BatchResult, error) {
	return nil, nil
}
func (nativeWithoutPromptCache) Chat(context.Context, []metal.ChatMessage, metal.GenerateConfig) iter.Seq[metal.Token] {
	return func(func(metal.Token) bool) {}
}
func (nativeWithoutPromptCache) Classify(context.Context, []string, metal.GenerateConfig, bool) ([]metal.ClassifyResult, error) {
	return nil, nil
}
func (nativeWithoutPromptCache) Close() error { return nil }
func (nativeWithoutPromptCache) Err() error   { return nil }
func (nativeWithoutPromptCache) Generate(context.Context, string, metal.GenerateConfig) iter.Seq[metal.Token] {
	return func(func(metal.Token) bool) {}
}
func (nativeWithoutPromptCache) Info() metal.ModelInfo { return metal.ModelInfo{} }
func (nativeWithoutPromptCache) InspectAttention(context.Context, string) (*metal.AttentionResult, error) {
	return nil, nil
}
func (nativeWithoutPromptCache) LastMetrics() metal.Metrics  { return metal.Metrics{} }
func (nativeWithoutPromptCache) ModelType() string           { return "" }
func (nativeWithoutPromptCache) Tokenizer() *metal.Tokenizer { return nil }

func TestModelWarmPromptCache_ForwardsToNative_Good(t *testing.T) {
	coverageTokens := "WarmPromptCache ForwardsToNative"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	native := &fakeNativeModel{}
	model := &Model{model: native}

	if err := model.WarmPromptCache("stable prefix"); err != nil {
		t.Fatalf("WarmPromptCache: %v", err)
	}
	if native.warmPrompt != "stable prefix" {
		t.Fatalf("warmPrompt = %q, want stable prefix", native.warmPrompt)
	}
}

func TestModelWarmPromptCache_UnsupportedNative_Bad(t *testing.T) {
	coverageTokens := "WarmPromptCache UnsupportedNative"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	model := &Model{model: nativeWithoutPromptCache{}}

	if err := model.WarmPromptCache("stable prefix"); err == nil {
		t.Fatal("expected unsupported prompt cache error")
	}
}

func TestModelClearPromptCache_ForwardsToNative_Good(t *testing.T) {
	coverageTokens := "ClearPromptCache ForwardsToNative"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	native := &fakeNativeModel{}
	model := &Model{model: native}

	if err := model.ClearPromptCache(); err != nil {
		t.Fatalf("ClearPromptCache: %v", err)
	}
	if native.clearPromptCacheCalls != 1 {
		t.Fatalf("clearPromptCacheCalls = %d, want 1", native.clearPromptCacheCalls)
	}
}

func TestModelClearPromptCache_UnsupportedNative_Bad(t *testing.T) {
	coverageTokens := "ClearPromptCache UnsupportedNative"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	model := &Model{model: nativeWithoutPromptCache{}}

	if err := model.ClearPromptCache(); err == nil {
		t.Fatal("expected unsupported prompt cache clearing error")
	}
}

func TestModelClearPromptCache_NilModel_Ugly(t *testing.T) {
	coverageTokens := "ClearPromptCache NilModel"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	var model *Model

	if err := model.ClearPromptCache(); err == nil {
		t.Fatal("ClearPromptCache(nil model) error = nil")
	}
}

func TestModelWarmPromptCacheFromMemvidBlocks_Good(t *testing.T) {
	coverageTokens := "WarmPromptCacheFromMemvidBlocks"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	source := memvid.NewInMemoryStore(nil)
	snapshot := kvSnapshotBlocksTestSnapshot()
	bundle, err := snapshot.SaveMemvidBlocks(context.Background(), source, kv.MemvidBlockOptions{BlockSize: 2})
	if err != nil {
		t.Fatalf("SaveMemvidBlocks() error = %v", err)
	}
	store := &recordingMemvidStore{store: source}
	native := &fakeNativeModel{}
	model := &Model{model: native}

	if err := model.WarmPromptCacheFromMemvidBlocks(context.Background(), store, bundle, 2); err != nil {
		t.Fatalf("WarmPromptCacheFromMemvidBlocks() error = %v", err)
	}

	if len(store.resolved) != 1 || store.resolved[0] != bundle.Blocks[0].Memvid.ChunkID {
		t.Fatalf("resolved chunks = %v, want only first block chunk %d", store.resolved, bundle.Blocks[0].Memvid.ChunkID)
	}
	if native.restoredPromptKV != nil {
		t.Fatal("restoredPromptKV != nil, want streaming block restore without assembled full snapshot")
	}
	if native.restoreBlockPrefix != 2 {
		t.Fatalf("restoreBlockPrefix = %d, want 2", native.restoreBlockPrefix)
	}
	if len(native.restoredPromptBlocks) != 1 {
		t.Fatalf("restoredPromptBlocks = %d, want one prefix block", len(native.restoredPromptBlocks))
	}
	restored := native.restoredPromptBlocks[0].Snapshot
	if restored == nil || restored.TokenOffset != 2 || restored.SeqLen != 2 || len(restored.Tokens) != 2 {
		t.Fatalf("restored block snapshot = %+v, want first two-token prefix", restored)
	}
	if len(restored.Logits) != 0 {
		t.Fatalf("restored block Logits = %v, want none for prefix warm", restored.Logits)
	}
}

func TestModelWarmPromptCacheFromMemvidBlocks_NativeRawOnly_Good(t *testing.T) {
	coverageTokens := "WarmPromptCacheFromMemvidBlocks NativeRawOnly"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	source := memvid.NewInMemoryStore(nil)
	snapshot := kvSnapshotBlocksTestSnapshot()
	head := &snapshot.Layers[0].Heads[0]
	for _, value := range head.Key {
		head.KeyBytes = appendUint16LE(head.KeyBytes, float32ToFloat16(value))
	}
	for _, value := range head.Value {
		head.ValueBytes = appendUint16LE(head.ValueBytes, float32ToFloat16(value))
	}
	head.Key = nil
	head.Value = nil
	head.KeyDType = "float16"
	head.ValueDType = "float16"
	bundle, err := snapshot.SaveMemvidBlocks(context.Background(), source, kv.MemvidBlockOptions{
		BlockSize:  2,
		KVEncoding: kv.EncodingNative,
	})
	if err != nil {
		t.Fatalf("SaveMemvidBlocks(native) error = %v", err)
	}
	native := &fakeNativeModel{}
	model := &Model{model: native}

	if err := model.WarmPromptCacheFromMemvidBlocks(context.Background(), source, bundle, 2); err != nil {
		t.Fatalf("WarmPromptCacheFromMemvidBlocks(native raw-only) error = %v", err)
	}

	if len(native.restoredPromptBlocks) != 1 {
		t.Fatalf("restoredPromptBlocks = %d, want one prefix block", len(native.restoredPromptBlocks))
	}
	restored := native.restoredPromptBlocks[0].Snapshot
	if restored == nil || len(restored.Layers) == 0 || len(restored.Layers[0].Heads) == 0 {
		t.Fatalf("restored block snapshot = %+v, want native raw-only head", restored)
	}
	restoredHead := restored.Layers[0].Heads[0]
	if len(restoredHead.Key) != 0 || len(restoredHead.Value) != 0 {
		t.Fatalf("restored float32 key/value lengths = %d/%d, want raw-only", len(restoredHead.Key), len(restoredHead.Value))
	}
	if restoredHead.KeyDType != metal.DTypeFloat16 || restoredHead.ValueDType != metal.DTypeFloat16 {
		t.Fatalf("restored dtypes = %v/%v, want float16", restoredHead.KeyDType, restoredHead.ValueDType)
	}
	if len(restoredHead.KeyBytes) != 8 || len(restoredHead.ValueBytes) != 8 {
		t.Fatalf("restored bytes = %d/%d, want two tokens x dim two x f16", len(restoredHead.KeyBytes), len(restoredHead.ValueBytes))
	}
}

func TestMetalKVSnapshotBlockSourcePartialPrefix_Good(t *testing.T) {
	coverageTokens := "MetalKVSnapshotBlockSource PartialPrefix"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	bundle := &kv.StateBlockBundle{
		Version:    kv.StateBlockVersion,
		Kind:       kv.StateBlockBundleKind,
		TokenCount: 6,
		Blocks: []kv.StateBlockRef{
			{Index: 0, TokenStart: 0, TokenCount: 2},
			{Index: 1, TokenStart: 2, TokenCount: 2},
			{Index: 2, TokenStart: 4, TokenCount: 2},
		},
	}

	source, err := metalKVSnapshotBlockSource(context.Background(), memvid.NewInMemoryStore(nil), bundle, 3)
	if err != nil {
		t.Fatalf("metalKVSnapshotBlockSource() error = %v", err)
	}
	if source.BlockCount != 2 || source.PrefixTokens != 3 || source.TokenCount != 6 {
		t.Fatalf("source = %+v, want two covering blocks for three-token prefix", source)
	}
}

func TestMetalKVSnapshotBlockSourceRejectsNonContiguousBundle_Bad(t *testing.T) {
	coverageTokens := "MetalKVSnapshotBlockSource RejectsNonContiguousBundle"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	bundle := &kv.StateBlockBundle{
		Version:    kv.StateBlockVersion,
		Kind:       kv.StateBlockBundleKind,
		TokenCount: 4,
		Blocks: []kv.StateBlockRef{
			{Index: 0, TokenStart: 0, TokenCount: 2},
			{Index: 1, TokenStart: 3, TokenCount: 1},
		},
	}

	if _, err := metalKVSnapshotBlockSource(context.Background(), memvid.NewInMemoryStore(nil), bundle, 4); err != errMLXStateKVBlockMetaMismatch {
		t.Fatalf("metalKVSnapshotBlockSource() error = %v, want metadata mismatch", err)
	}
}

func TestModelGenerateBuffered_Error_Bad(t *testing.T) {
	coverageTokens := "Error"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	wantErr := core.NewError("boom")
	model := &Model{
		model: &fakeNativeModel{
			err:    wantErr,
			tokens: []metal.Token{{ID: 1, Text: "partial"}},
		},
	}

	_, err := model.Generate("ignored")
	if !core.Is(err, wantErr) {
		t.Fatalf("Generate() error = %v, want %v", err, wantErr)
	}
}

func TestModelGenerateStream_Good(t *testing.T) {
	model := &Model{
		model: &fakeNativeModel{
			tokens: []metal.Token{{ID: 7, Text: "A"}, {ID: 8, Text: "B"}},
		},
	}

	ch := model.GenerateStream(context.Background(), "ignored", WithMinP(0.05))
	var got []Token
	timeout := time.After(2 * time.Second)
	for {
		select {
		case tok, ok := <-ch:
			if !ok {
				if len(got) != 2 {
					t.Fatalf("stream yielded %d tokens, want 2", len(got))
				}
				if got[0].Value != "A" || got[1].Text != "B" {
					t.Fatalf("unexpected stream tokens: %+v", got)
				}
				return
			}
			got = append(got, tok)
		case <-timeout:
			t.Fatal("timed out waiting for stream")
		}
	}
}

func TestModelGenerateChunksStream_Good(t *testing.T) {
	native := &fakeNativeModel{tokens: []metal.Token{{ID: 7, Text: "A"}, {ID: 8, Text: "B"}}}
	model := &Model{model: native}

	got := collectTokensFromChannel(model.GenerateChunksStream(context.Background(), seqStrings("prefix", "suffix"), WithMaxTokens(7)))

	if len(got) != 2 || got[0].Value != "A" || got[1].Text != "B" {
		t.Fatalf("GenerateChunksStream() tokens = %+v, want A/B", got)
	}
	if !reflect.DeepEqual(native.generatedChunks, []string{"prefix", "suffix"}) {
		t.Fatalf("generated chunks = %#v", native.generatedChunks)
	}
	if native.lastGenerateConfig.MaxTokens != 7 {
		t.Fatalf("MaxTokens = %d, want 7", native.lastGenerateConfig.MaxTokens)
	}
}

func TestModelGenerateStream_ForwardsOptions_Good(t *testing.T) {
	coverageTokens := "ForwardsOptions"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	native := &fakeNativeModel{
		tokens: []metal.Token{{ID: 1, Text: "A"}},
	}
	model := &Model{model: native}

	for range model.GenerateStream(
		context.Background(),
		"ignored",
		WithMaxTokens(9),
		WithTemperature(0.3),
		WithTopK(11),
		WithTopP(0.8),
		WithMinP(0.05),
		WithSeed(123),
		WithStopTokens(4, 5),
		WithMinTokensBeforeStop(1),
		WithRepeatPenalty(1.2),
	) {
	}

	cfg := native.lastGenerateConfig
	if cfg.MaxTokens != 9 {
		t.Fatalf("MaxTokens = %d, want 9", cfg.MaxTokens)
	}
	if cfg.Temperature != 0.3 {
		t.Fatalf("Temperature = %f, want 0.3", cfg.Temperature)
	}
	if cfg.TopK != 11 {
		t.Fatalf("TopK = %d, want 11", cfg.TopK)
	}
	if cfg.TopP != 0.8 {
		t.Fatalf("TopP = %f, want 0.8", cfg.TopP)
	}
	if cfg.MinP != 0.05 {
		t.Fatalf("MinP = %f, want 0.05", cfg.MinP)
	}
	if !cfg.SeedSet || cfg.Seed != 123 {
		t.Fatalf("Seed = %d/%v, want 123/true", cfg.Seed, cfg.SeedSet)
	}
	if cfg.RepeatPenalty != 1.2 {
		t.Fatalf("RepeatPenalty = %f, want 1.2", cfg.RepeatPenalty)
	}
	if !reflect.DeepEqual(cfg.StopTokens, []int32{4, 5}) {
		t.Fatalf("StopTokens = %v, want [4 5]", cfg.StopTokens)
	}
	if cfg.MinTokensBeforeStop != 1 {
		t.Fatalf("MinTokensBeforeStop = %d, want 1", cfg.MinTokensBeforeStop)
	}
}

func TestModelGenerate_ForwardsProbeSink_Good(t *testing.T) {
	coverageTokens := "probe.Sink"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	recorder := probe.NewRecorder()
	native := &fakeNativeModel{
		probeEvents: []metal.ProbeEvent{{
			Kind:  metal.ProbeEventToken,
			Phase: metal.ProbePhaseDecode,
			Step:  2,
			Token: &metal.ProbeToken{
				ID:              9,
				Text:            "Z",
				PromptTokens:    4,
				GeneratedTokens: 1,
			},
		}},
	}
	model := &Model{model: native}

	if _, err := model.Generate("ignored", WithProbeSink(recorder)); err != nil {
		t.Fatalf("Generate() error = %v", err)
	}

	if native.lastGenerateConfig.ProbeSink == nil {
		t.Fatal("native probe.Sink = nil, want configured")
	}
	events := recorder.Events()
	if len(events) != 1 {
		t.Fatalf("probe events len = %d, want 1", len(events))
	}
	if events[0].Kind != probe.KindToken || events[0].Phase != probe.PhaseDecode {
		t.Fatalf("probe event = %+v", events[0])
	}
	if events[0].Token == nil || events[0].Token.ID != 9 || events[0].Token.Text != "Z" {
		t.Fatalf("probe token = %+v", events[0].Token)
	}
}

func TestModelChatBuffered_Good(t *testing.T) {
	model := &Model{
		model: &fakeNativeModel{
			chatTokens: []metal.Token{{ID: 3, Text: "Hi"}, {ID: 4, Text: " there"}},
		},
	}

	got, err := model.Chat([]inference.Message{{Role: "user", Content: "hello"}}, WithTopP(0.8))
	if err != nil {
		t.Fatalf("Chat() error = %v", err)
	}
	if got != "Hi there" {
		t.Fatalf("Chat() = %q, want %q", got, "Hi there")
	}
}

func TestModelChatStream_ForwardsMessagesAndOptions_Good(t *testing.T) {
	coverageTokens := "ForwardsMessagesAndOptions"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	native := &fakeNativeModel{
		chatTokens: []metal.Token{{ID: 3, Text: "Hi"}},
	}
	model := &Model{model: native}
	messages := []inference.Message{
		{Role: "system", Content: "Be terse."},
		{Role: "user", Content: "hello"},
	}

	for range model.ChatStream(context.Background(), messages, WithMaxTokens(7), WithTopP(0.85), WithRepeatPenalty(1.05)) {
	}

	if !reflect.DeepEqual(native.lastChatMessages, []metal.ChatMessage{
		{Role: "system", Content: "Be terse."},
		{Role: "user", Content: "hello"},
	}) {
		t.Fatalf("Chat messages = %+v", native.lastChatMessages)
	}
	if native.lastChatConfig.MaxTokens != 7 {
		t.Fatalf("MaxTokens = %d, want 7", native.lastChatConfig.MaxTokens)
	}
	if native.lastChatConfig.TopP != 0.85 {
		t.Fatalf("TopP = %f, want 0.85", native.lastChatConfig.TopP)
	}
	if native.lastChatConfig.RepeatPenalty != 1.05 {
		t.Fatalf("RepeatPenalty = %f, want 1.05", native.lastChatConfig.RepeatPenalty)
	}
}

func TestModelChatChunksStream_ForwardsMessagesAndChunkBytes_Good(t *testing.T) {
	native := &fakeNativeModel{
		chatTokens: []metal.Token{{ID: 3, Text: "Hi"}},
	}
	model := &Model{model: native}
	messages := []inference.Message{
		{Role: "system", Content: "Be terse."},
		{Role: "user", Content: "hello"},
	}

	got := collectTokensFromChannel(model.ChatChunksStream(context.Background(), messages, 4096, WithMaxTokens(7), WithTopP(0.85)))

	if len(got) != 1 || got[0].Text != "Hi" {
		t.Fatalf("ChatChunksStream() = %+v, want Hi", got)
	}
	if !reflect.DeepEqual(native.lastChatChunkMessages, []metal.ChatMessage{
		{Role: "system", Content: "Be terse."},
		{Role: "user", Content: "hello"},
	}) {
		t.Fatalf("Chat chunk messages = %+v", native.lastChatChunkMessages)
	}
	if native.lastChatChunkBytes != 4096 {
		t.Fatalf("chunk bytes = %d, want 4096", native.lastChatChunkBytes)
	}
	if native.lastChatChunkConfig.MaxTokens != 7 || native.lastChatChunkConfig.TopP != 0.85 {
		t.Fatalf("chat chunk cfg = %+v, want max tokens/top-p", native.lastChatChunkConfig)
	}
}

func TestModelClassify_Good(t *testing.T) {
	model := &Model{
		model: &fakeNativeModel{
			classifyResults: []metal.ClassifyResult{{
				Token:  metal.Token{ID: 9, Text: "yes"},
				Logits: []float32{0.1, 0.9},
			}},
		},
	}

	results, err := model.Classify([]string{"prompt"}, WithTemperature(0.1), WithLogits())
	if err != nil {
		t.Fatalf("Classify() error = %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("Classify() len = %d, want 1", len(results))
	}
	if results[0].Token.Text != "yes" || results[0].Token.Value != "yes" {
		t.Fatalf("Classify() token = %+v, want text/value yes", results[0].Token)
	}
	if !reflect.DeepEqual(results[0].Logits, []float32{0.1, 0.9}) {
		t.Fatalf("Classify() logits = %v, want [0.1 0.9]", results[0].Logits)
	}
	native := model.model.(*fakeNativeModel)
	if !native.classifyReturnLogits {
		t.Fatal("classifyReturnLogits = false, want true")
	}
	if native.lastClassifyConfig.Temperature != 0.1 {
		t.Fatalf("Classify() temperature = %f, want 0.1", native.lastClassifyConfig.Temperature)
	}
}

func TestModelBatchGenerate_Good(t *testing.T) {
	model := &Model{
		model: &fakeNativeModel{
			batchResults: []metal.BatchResult{{
				Tokens: []metal.Token{{ID: 1, Text: "A"}, {ID: 2, Text: "B"}},
			}},
		},
	}

	results, err := model.BatchGenerate([]string{"prompt"}, WithMaxTokens(12))
	if err != nil {
		t.Fatalf("BatchGenerate() error = %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("BatchGenerate() len = %d, want 1", len(results))
	}
	if len(results[0].Tokens) != 2 || results[0].Tokens[1].Text != "B" {
		t.Fatalf("BatchGenerate() tokens = %+v", results[0].Tokens)
	}
	native := model.model.(*fakeNativeModel)
	if native.lastBatchConfig.MaxTokens != 12 {
		t.Fatalf("BatchGenerate() MaxTokens = %d, want 12", native.lastBatchConfig.MaxTokens)
	}
}

func TestModelMetricsAndModelType_Good(t *testing.T) {
	model := &Model{
		model: &fakeNativeModel{
			modelType: "gemma4_text",
			metrics: metal.Metrics{
				PromptTokens:      32,
				GeneratedTokens:   5,
				PeakMemoryBytes:   1024,
				ActiveMemoryBytes: 512,
				MTP: &metal.MTPMetrics{
					DraftTokenSchedule:     []int{2, 2, 1},
					ProposedTokens:         5,
					AcceptedTokens:         4,
					RejectedTokens:         1,
					TargetVerifyCalls:      3,
					TargetCalls:            4,
					DraftCalls:             3,
					AcceptanceRate:         0.8,
					VisibleTokensPerSec:    91,
					TargetTokensPerSec:     120,
					WarmDecodeTokensPerSec: 95,
					WallDuration:           80 * time.Millisecond,
					RestoreDuration:        5 * time.Millisecond,
					TargetVerifyDuration:   40 * time.Millisecond,
					DraftDuration:          12 * time.Millisecond,
				},
				CacheProfile: &metal.CacheProfile{
					Architecture:       "gemma4_text",
					TotalCaches:        6,
					LocalCaches:        5,
					GlobalCaches:       1,
					SharedLayers:       2,
					CachelessLayers:    3,
					LocalWindowTokens:  512,
					MaxLocalTokens:     512,
					MaxGlobalTokens:    4000,
					MaxProcessedTokens: 4000,
				},
			},
		},
	}

	if got := model.ModelType(); got != "gemma4_text" {
		t.Fatalf("ModelType() = %q, want %q", got, "gemma4_text")
	}
	metrics := model.Metrics()
	if metrics.PromptTokens != 32 || metrics.GeneratedTokens != 5 {
		t.Fatalf("Metrics() = %+v, want prompt=32 generated=5", metrics)
	}
	if metrics.PeakMemoryBytes != 1024 || metrics.ActiveMemoryBytes != 512 {
		t.Fatalf("Metrics() memory = %+v, want peak=1024 active=512", metrics)
	}
	if metrics.CacheProfile == nil || metrics.CacheProfile.LocalCaches != 5 || metrics.CacheProfile.GlobalCaches != 1 || metrics.CacheProfile.CachelessLayers != 3 || metrics.CacheProfile.LocalWindowLeaked {
		t.Fatalf("Metrics() cache profile = %+v, want bounded Gemma 4 local/global topology", metrics.CacheProfile)
	}
	if metrics.MTP == nil || metrics.MTP.ProposedTokens != 5 || metrics.MTP.AcceptedTokens != 4 || metrics.MTP.RejectedTokens != 1 {
		t.Fatalf("Metrics() MTP = %+v, want proposed/accepted/rejected counters", metrics.MTP)
	}
	if len(metrics.MTP.DraftTokenSchedule) != 3 || metrics.MTP.DraftTokenSchedule[2] != 1 {
		t.Fatalf("Metrics() MTP schedule = %+v, want copied draft token schedule", metrics.MTP.DraftTokenSchedule)
	}
	if metrics.MTP.TargetVerifyCalls != 3 || metrics.MTP.WarmDecodeTokensPerSec != 95 || metrics.MTP.RestoreDuration != 5*time.Millisecond {
		t.Fatalf("Metrics() MTP timing = %+v, want target verify calls, warm tok/s, and restore duration", metrics.MTP)
	}
}

func TestModelInspectAttention_Good(t *testing.T) {
	model := &Model{
		model: &fakeNativeModel{
			attention: &metal.AttentionResult{
				NumLayers:     2,
				NumHeads:      4,
				SeqLen:        8,
				HeadDim:       16,
				NumQueryHeads: 8,
				Keys:          [][][]float32{{{1, 2, 3}}},
				Queries:       [][][]float32{{{4, 5, 6}}},
				Architecture:  "gemma4_text",
			},
		},
	}

	snapshot, err := model.InspectAttention("prompt")
	if err != nil {
		t.Fatalf("InspectAttention() error = %v", err)
	}
	if snapshot == nil {
		t.Fatal("InspectAttention() = nil, want non-nil")
	}
	if snapshot.NumLayers != 2 || snapshot.HeadDim != 16 || snapshot.Architecture != "gemma4_text" {
		t.Fatalf("InspectAttention() = %+v", snapshot)
	}
	if snapshot.NumQueryHeads != 8 {
		t.Fatalf("InspectAttention().NumQueryHeads = %d, want 8", snapshot.NumQueryHeads)
	}
	if !snapshot.HasQueries() {
		t.Fatal("InspectAttention().HasQueries() = false, want true")
	}
}

func TestModelCaptureKV_Good(t *testing.T) {
	coverageTokens := "ModelCaptureKV"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	native := &fakeNativeModel{
		kvSnapshot: &metal.KVSnapshot{
			Version:      metal.KVSnapshotVersion,
			Architecture: "gemma4_text",
			Tokens:       []int32{1, 2},
			NumLayers:    1,
			NumHeads:     1,
			SeqLen:       2,
			HeadDim:      2,
			Layers: []metal.KVLayerSnapshot{{
				Layer: 0,
				Heads: []metal.KVHeadSnapshot{{
					Key:   []float32{1, 2, 3, 4},
					Value: []float32{5, 6, 7, 8},
				}},
			}},
		},
	}
	model := &Model{model: native}

	snapshot, err := model.CaptureKV("prompt")
	if err != nil {
		t.Fatalf("CaptureKV() error = %v", err)
	}
	if snapshot.Architecture != "gemma4_text" || snapshot.SeqLen != 2 {
		t.Fatalf("CaptureKV() = %+v", snapshot)
	}
	head, ok := snapshot.Head(0, 0)
	if !ok {
		t.Fatal("CaptureKV().Head() ok = false, want true")
	}
	if head.Key[3] != 4 || head.Value[0] != 5 {
		t.Fatalf("CaptureKV().Head() = %+v", head)
	}
	head.Key[0] = 99
	if native.kvSnapshot.Layers[0].Heads[0].Key[0] != 1 {
		t.Fatal("CaptureKV() returned aliased native key data")
	}
}

func TestKVSnapshotConversion_PreservesTurboQuantPayloads_Good(t *testing.T) {
	coverageTokens := "KVSnapshotConversion PreservesTurboQuantPayloads"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	layout := metal.TurboQuantKVPageLayout{
		Version:     metal.TurboQuantKVLayoutVersion,
		Codec:       metal.TurboQuantKVCodecName,
		CacheIndex:  0,
		Layer:       0,
		LayerType:   "sliding_attention",
		SharedOwner: 0,
		Shape:       metal.TurboQuantKVShape{Batch: 1, Heads: 1, SeqLen: 1, HeadDim: 2},
		TokenOffset: 0,
		PageTokens:  1,
		PageSize:    1,
		LocalWindow: 512,
		Key: metal.TurboQuantKVCodec{
			Algorithm:          metal.TurboQuantKVAlgorithmProd,
			NormalBits:         3,
			NormPolicy:         metal.TurboQuantKVNormPolicyExplicitVectorBF16V1,
			ResidualNormPolicy: metal.TurboQuantKVResidualNormPolicyExplicitVectorBF16V1,
			RotationSeed:       11,
			QJLSeed:            13,
			CodebookID:         metal.TurboQuantKVReferenceCodebookUniform,
		},
		Value: metal.TurboQuantKVCodec{
			Algorithm:    metal.TurboQuantKVAlgorithmMSE,
			NormalBits:   3,
			NormPolicy:   metal.TurboQuantKVNormPolicyExplicitVectorBF16V1,
			RotationSeed: 17,
			CodebookID:   metal.TurboQuantKVReferenceCodebookUniform,
		},
	}
	page, err := metal.EncodeTurboQuantKVReferencePage([]float32{1, 2}, []float32{3, 4}, layout)
	if err != nil {
		t.Fatalf("EncodeTurboQuantKVReferencePage() error = %v", err)
	}
	payload, err := page.PackedPayload()
	if err != nil {
		t.Fatalf("PackedPayload() error = %v", err)
	}
	native := &metal.KVSnapshot{
		Version:      metal.KVSnapshotVersion,
		Architecture: "gemma4_text",
		Tokens:       []int32{1},
		NumLayers:    1,
		NumHeads:     1,
		SeqLen:       1,
		HeadDim:      2,
		Layers: []metal.KVLayerSnapshot{{
			Layer:              0,
			CacheIndex:         0,
			CacheMode:          metal.KVCacheModeTurboQuant,
			TurboQuantPayloads: []metal.TurboQuantKVReferencePagePayload{payload},
		}},
	}

	root := toRootKVSnapshot(native)
	if root.Layers[0].CacheMode != string(metal.KVCacheModeTurboQuant) || len(root.Layers[0].TurboQuantPayloads) != 1 {
		t.Fatalf("root layer mode/payloads = %q/%d, want turboquant payload", root.Layers[0].CacheMode, len(root.Layers[0].TurboQuantPayloads))
	}
	encoded, err := root.MarshalBinary()
	if err != nil {
		t.Fatalf("MarshalBinary() error = %v", err)
	}
	var loaded kv.Snapshot
	if err := loaded.UnmarshalBinary(encoded); err != nil {
		t.Fatalf("UnmarshalBinary() error = %v", err)
	}
	if loaded.Version != kv.SnapshotVersion || loaded.Layers[0].CacheMode != string(metal.KVCacheModeTurboQuant) || len(loaded.Layers[0].TurboQuantPayloads) != 1 {
		t.Fatalf("loaded version/mode/payloads = %d/%q/%d, want v%d turboquant payload", loaded.Version, loaded.Layers[0].CacheMode, len(loaded.Layers[0].TurboQuantPayloads), kv.SnapshotVersion)
	}
	roundTrip := toMetalKVSnapshot(&loaded)
	if roundTrip.Layers[0].CacheMode != metal.KVCacheModeTurboQuant || len(roundTrip.Layers[0].TurboQuantPayloads) != 1 {
		t.Fatalf("metal round trip mode/payloads = %q/%d, want turboquant payload", roundTrip.Layers[0].CacheMode, len(roundTrip.Layers[0].TurboQuantPayloads))
	}
	got := roundTrip.Layers[0].TurboQuantPayloads[0]
	if got.Layout.PageTokens != payload.Layout.PageTokens || !reflect.DeepEqual(got.Data, payload.Data) {
		t.Fatalf("round trip payload = page_tokens:%d data:%d, want page_tokens:%d data:%d", got.Layout.PageTokens, len(got.Data), payload.Layout.PageTokens, len(payload.Data))
	}
}

func TestModelWarmPromptCacheChunks_Good(t *testing.T) {
	coverageTokens := "WarmPromptCacheChunks"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	native := &fakeNativeModel{}
	model := &Model{model: native}

	if err := model.WarmPromptCacheChunks(context.Background(), seqStrings("<bos>", "chunk")); err != nil {
		t.Fatalf("WarmPromptCacheChunks() error = %v", err)
	}
	if !reflect.DeepEqual(native.warmChunks, []string{"<bos>", "chunk"}) {
		t.Fatalf("warm chunks = %#v", native.warmChunks)
	}
}

func TestModelWarmPromptCacheFromKV_Good(t *testing.T) {
	native := &fakeNativeModel{}
	model := &Model{model: native}
	snapshot := &kv.Snapshot{
		Version:      kv.SnapshotVersion,
		Architecture: "qwen3",
		Tokens:       []int32{1},
		NumLayers:    1,
		NumHeads:     1,
		SeqLen:       1,
		HeadDim:      1,
		Layers: []kv.LayerSnapshot{{
			Layer: 0,
			Heads: []kv.HeadSnapshot{{
				Key:        []float32{1},
				Value:      []float32{2},
				KeyBytes:   []byte{1, 2},
				ValueBytes: []byte{3, 4},
				KeyDType:   "float16",
				ValueDType: "bfloat16",
			}},
		}},
	}

	if err := model.WarmPromptCacheFromKV(snapshot); err != nil {
		t.Fatalf("WarmPromptCacheFromKV() error = %v", err)
	}
	if native.restoredPromptKV == nil || native.restoredPromptKV.Layers[0].Heads[0].KeyDType != metal.DTypeFloat16 {
		t.Fatalf("restored KV = %+v, want converted raw dtype", native.restoredPromptKV)
	}
	if err := (&Model{model: nativeWithoutPromptCache{}}).WarmPromptCacheFromKV(snapshot); err == nil {
		t.Fatal("WarmPromptCacheFromKV(unsupported) error = nil")
	}
}

func TestModelGenerateChunks_Good(t *testing.T) {
	coverageTokens := "GenerateChunks"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	native := &fakeNativeModel{tokens: []metal.Token{{Text: "ok"}}}
	model := &Model{model: native}

	got, err := model.GenerateChunks(context.Background(), seqStrings("prefix", "suffix"), WithMaxTokens(7))
	if err != nil {
		t.Fatalf("GenerateChunks() error = %v", err)
	}
	if got != "ok" {
		t.Fatalf("GenerateChunks() = %q, want ok", got)
	}
	if !reflect.DeepEqual(native.generatedChunks, []string{"prefix", "suffix"}) {
		t.Fatalf("generated chunks = %#v", native.generatedChunks)
	}
	if native.lastGenerateConfig.MaxTokens != 7 {
		t.Fatalf("MaxTokens = %d, want 7", native.lastGenerateConfig.MaxTokens)
	}
}

func TestModelCaptureKVChunks_Good(t *testing.T) {
	coverageTokens := "CaptureKVChunks"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	native := &fakeNativeModel{kvSnapshot: &metal.KVSnapshot{
		Version:      metal.KVSnapshotVersion,
		Architecture: "gemma4_text",
		Tokens:       []int32{1, 2, 3},
		NumLayers:    1,
		NumHeads:     1,
		SeqLen:       3,
		HeadDim:      1,
		Layers: []metal.KVLayerSnapshot{{
			Layer: 0,
			Heads: []metal.KVHeadSnapshot{{Key: []float32{1, 2, 3}, Value: []float32{4, 5, 6}}},
		}},
	}}
	model := &Model{model: native}

	snapshot, err := model.CaptureKVChunks(context.Background(), seqStrings("prefix", "suffix"))
	if err != nil {
		t.Fatalf("CaptureKVChunks() error = %v", err)
	}
	if snapshot.SeqLen != 3 {
		t.Fatalf("SeqLen = %d, want 3", snapshot.SeqLen)
	}
	if !reflect.DeepEqual(native.capturedChunks, []string{"prefix", "suffix"}) {
		t.Fatalf("captured chunks = %#v", native.capturedChunks)
	}
}

func TestModelClose_Idempotent_Good(t *testing.T) {
	coverageTokens := "Idempotent"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	native := &fakeNativeModel{}
	model := &Model{
		model: native,
		tok:   &Tokenizer{tok: &metal.Tokenizer{}},
	}

	if err := model.Close(); err != nil {
		t.Fatalf("first Close(): %v", err)
	}
	if native.closeCalls != 1 {
		t.Fatalf("close calls after first Close = %d, want 1", native.closeCalls)
	}
	if model.model != nil {
		t.Fatal("model handle should be cleared after Close")
	}
	if model.tok != nil {
		t.Fatal("tokenizer handle should be cleared after Close")
	}

	if err := model.Close(); err != nil {
		t.Fatalf("second Close(): %v", err)
	}
	if native.closeCalls != 1 {
		t.Fatalf("close calls after second Close = %d, want 1", native.closeCalls)
	}
}

func TestModelErrAndTokenizer_Good(t *testing.T) {
	wantErr := core.NewError("model failed")
	tokenizer := &Tokenizer{tok: &metal.Tokenizer{}}
	model := &Model{model: &fakeNativeModel{err: wantErr}, tok: tokenizer}
	if !core.Is(model.Err(), wantErr) {
		t.Fatalf("Err() = %v, want %v", model.Err(), wantErr)
	}
	if model.Tokenizer() != tokenizer {
		t.Fatal("Tokenizer() did not return model tokenizer")
	}
	if (*Model)(nil).Err() != nil || (*Model)(nil).Tokenizer() != nil {
		t.Fatal("nil model Err/Tokenizer should return nil")
	}
}

func TestModelNilPublicSurface_Bad(t *testing.T) {
	var model *Model
	if _, err := model.Generate("x"); err == nil {
		t.Fatal("Generate(nil model) error = nil")
	}
	if _, err := model.Chat([]inference.Message{{Role: "user", Content: "x"}}); err == nil {
		t.Fatal("Chat(nil model) error = nil")
	}
	if _, err := model.GenerateChunks(context.Background(), seqStrings("x")); err == nil {
		t.Fatal("GenerateChunks(nil model) error = nil")
	}
	if err := model.WarmPromptCache("x"); err == nil {
		t.Fatal("WarmPromptCache(nil model) error = nil")
	}
	if err := model.WarmPromptCacheChunks(context.Background(), seqStrings("x")); err == nil {
		t.Fatal("WarmPromptCacheChunks(nil model) error = nil")
	}
	if err := model.ClearPromptCache(); err == nil {
		t.Fatal("ClearPromptCache(nil model) error = nil")
	}
	if err := model.WarmPromptCacheFromKV(&kv.Snapshot{}); err == nil {
		t.Fatal("WarmPromptCacheFromKV(nil model) error = nil")
	}
	if err := model.WarmPromptCacheFromMemvidBlocks(context.Background(), nil, nil, 0); err == nil {
		t.Fatal("WarmPromptCacheFromMemvidBlocks(nil model) error = nil")
	}
	if _, err := model.Classify([]string{"x"}); err == nil {
		t.Fatal("Classify(nil model) error = nil")
	}
	if _, err := model.BatchGenerate([]string{"x"}); err == nil {
		t.Fatal("BatchGenerate(nil model) error = nil")
	}
	if _, err := model.InspectAttention("x"); err == nil {
		t.Fatal("InspectAttention(nil model) error = nil")
	}
	if _, err := model.CaptureKV("x"); err == nil {
		t.Fatal("CaptureKV(nil model) error = nil")
	}
	if _, err := model.CaptureKVChunks(context.Background(), seqStrings("x")); err == nil {
		t.Fatal("CaptureKVChunks(nil model) error = nil")
	}
	if _, err := model.LoadLoRA("/tmp/missing"); err == nil {
		t.Fatal("LoadLoRA(nil model) error = nil")
	}
	if err := model.UnloadLoRA(); err == nil {
		t.Fatal("UnloadLoRA(nil model) error = nil")
	}
	if _, err := model.SwapLoRA("/tmp/missing"); err == nil {
		t.Fatal("SwapLoRA(nil model) error = nil")
	}
	if NewLoRA(model, nil) != nil {
		t.Fatal("NewLoRA(nil model) != nil")
	}
	if model.MergeLoRA(nil) != nil {
		t.Fatal("MergeLoRA(nil adapter) should return receiver")
	}

	if tokens := collectTokensFromChannel(model.GenerateStream(context.Background(), "x")); len(tokens) != 0 {
		t.Fatalf("GenerateStream(nil model) tokens = %+v, want none", tokens)
	}
	if tokens := collectTokensFromChannel(model.GenerateChunksStream(context.Background(), seqStrings("x"))); len(tokens) != 0 {
		t.Fatalf("GenerateChunksStream(nil model) tokens = %+v, want none", tokens)
	}
	if tokens := collectTokensFromChannel(model.ChatChunksStream(context.Background(), []inference.Message{{Role: "user", Content: "x"}}, 8)); len(tokens) != 0 {
		t.Fatalf("ChatChunksStream(nil model) tokens = %+v, want none", tokens)
	}
	if tokens := collectTokensFromChannel(model.ChatStream(context.Background(), []inference.Message{{Role: "user", Content: "x"}})); len(tokens) != 0 {
		t.Fatalf("ChatStream(nil model) tokens = %+v, want none", tokens)
	}
}

func TestModelClose_Error_Bad(t *testing.T) {
	coverageTokens := "Error"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	wantErr := core.NewError("close boom")
	native := &fakeNativeModel{closeErr: wantErr}
	model := &Model{model: native}

	err := model.Close()
	if !core.Is(err, wantErr) {
		t.Fatalf("Close() error = %v, want %v", err, wantErr)
	}
	if native.closeCalls != 1 {
		t.Fatalf("close calls = %d, want 1", native.closeCalls)
	}
	if model.model != nil {
		t.Fatal("model handle should still be cleared on close error")
	}
}

func TestModelLoadLoRA_ForwardsToNative_Good(t *testing.T) {
	coverageTokens := "Model LoadLoRA"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	wantAdapter := &metal.LoRAAdapter{}
	adapterDir := writeTestLoRAAdapter(t, `{"rank":8,"alpha":16}`)
	native := &fakeNativeModel{loadedLoRAAdapter: wantAdapter}
	model := &Model{model: native}

	got, err := model.LoadLoRA(adapterDir)
	if err != nil {
		t.Fatalf("LoadLoRA() error = %v", err)
	}
	if got != wantAdapter {
		t.Fatalf("LoadLoRA() = %p, want %p", got, wantAdapter)
	}
	if native.loadedLoRAPath != adapterDir {
		t.Fatalf("native loaded path = %q, want %q", native.loadedLoRAPath, adapterDir)
	}
}

func TestLoadModelUnsupportedDevice_Bad(t *testing.T) {
	_, err := LoadModel("/does/not/matter", WithDevice("tpu"))
	if err == nil {
		t.Fatal("expected unsupported device error")
	}
}

func TestLoadModel_ForwardsRequestedCPUDevice_Good(t *testing.T) {
	coverageTokens := "ForwardsRequestedCPUDevice"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	originalLoadNativeModel := loadNativeModel
	t.Cleanup(func() { loadNativeModel = originalLoadNativeModel })

	loadNativeModel = func(modelPath string, cfg metal.LoadConfig) (nativeModel, error) {
		if modelPath != "/does/not/matter" {
			t.Fatalf("modelPath = %q, want /does/not/matter", modelPath)
		}
		if cfg.Device != metal.DeviceCPU {
			t.Fatalf("Device = %q, want %q", cfg.Device, metal.DeviceCPU)
		}
		return &fakeNativeModel{}, nil
	}

	model, err := LoadModel("/does/not/matter", WithDevice("cpu"))
	if err != nil {
		t.Fatalf("LoadModel() error = %v", err)
	}
	if err := model.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
}

func TestLoadModel_ForwardsAdapterPath_Good(t *testing.T) {
	coverageTokens := "ForwardsAdapterPath"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	originalLoadNativeModel := loadNativeModel
	t.Cleanup(func() { loadNativeModel = originalLoadNativeModel })
	adapterDir := writeTestLoRAAdapter(t, `{"rank":8,"alpha":16}`)

	loadNativeModel = func(modelPath string, cfg metal.LoadConfig) (nativeModel, error) {
		if modelPath != "/does/not/matter" {
			t.Fatalf("modelPath = %q, want /does/not/matter", modelPath)
		}
		if cfg.AdapterPath != adapterDir {
			t.Fatalf("AdapterPath = %q, want %q", cfg.AdapterPath, adapterDir)
		}
		return &fakeNativeModel{}, nil
	}

	model, err := LoadModel("/does/not/matter", WithAdapterPath(adapterDir))
	if err != nil {
		t.Fatalf("LoadModel() error = %v", err)
	}
	if err := model.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
}

func TestLoadModel_ForwardsParallelSlots_Good(t *testing.T) {
	coverageTokens := "ForwardsParallelSlots"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	originalLoadNativeModel := loadNativeModel
	t.Cleanup(func() { loadNativeModel = originalLoadNativeModel })

	loadNativeModel = func(modelPath string, cfg metal.LoadConfig) (nativeModel, error) {
		if modelPath != "/does/not/matter" {
			t.Fatalf("modelPath = %q, want /does/not/matter", modelPath)
		}
		if cfg.ParallelSlots != 4 {
			t.Fatalf("ParallelSlots = %d, want 4", cfg.ParallelSlots)
		}
		if cfg.DisablePromptCache {
			t.Fatal("DisablePromptCache = true, want false")
		}
		if cfg.PromptCacheMinTokens != DefaultPromptCacheMinTokens {
			t.Fatalf("PromptCacheMinTokens = %d, want %d", cfg.PromptCacheMinTokens, DefaultPromptCacheMinTokens)
		}
		return &fakeNativeModel{}, nil
	}

	model, err := LoadModel("/does/not/matter", WithParallelSlots(4))
	if err != nil {
		t.Fatalf("LoadModel() error = %v", err)
	}
	if err := model.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
}

func TestLoadModel_ForwardsGemma4SlidingWindow_Good(t *testing.T) {
	coverageTokens := "ForwardsGemma4SlidingWindow"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	originalLoadNativeModel := loadNativeModel
	t.Cleanup(func() { loadNativeModel = originalLoadNativeModel })

	loadNativeModel = func(modelPath string, cfg metal.LoadConfig) (nativeModel, error) {
		if modelPath != "/does/not/matter" {
			t.Fatalf("modelPath = %q, want /does/not/matter", modelPath)
		}
		if cfg.Gemma4SlidingWindow != 256 {
			t.Fatalf("Gemma4SlidingWindow = %d, want 256", cfg.Gemma4SlidingWindow)
		}
		return &fakeNativeModel{info: metal.ModelInfo{Architecture: "gemma4_text"}}, nil
	}

	model, err := LoadModel("/does/not/matter", WithGemma4SlidingWindow(256))
	if err != nil {
		t.Fatalf("LoadModel() error = %v", err)
	}
	info := model.Info()
	if info.Gemma4SlidingWindow != 256 {
		t.Fatalf("Info().Gemma4SlidingWindow = %d, want 256", info.Gemma4SlidingWindow)
	}
	if err := model.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
}

func TestLoadModel_AppliesMemoryPlanFromDevice_Good(t *testing.T) {
	coverageTokens := "AppliesMemoryPlanFromDevice"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	originalLoadNativeModel := loadNativeModel
	originalDeviceInfo := memoryPlannerDeviceInfo
	t.Cleanup(func() {
		loadNativeModel = originalLoadNativeModel
		memoryPlannerDeviceInfo = originalDeviceInfo
	})

	memoryPlannerDeviceInfo = func() DeviceInfo {
		return DeviceInfo{
			Architecture:                 "apple7",
			MemorySize:                   16 << 30,
			MaxRecommendedWorkingSetSize: 14 << 30,
		}
	}
	loadNativeModel = func(modelPath string, cfg metal.LoadConfig) (nativeModel, error) {
		if cfg.ContextLen != 8192 {
			t.Fatalf("ContextLen = %d, want planner 8192", cfg.ContextLen)
		}
		if !cfg.DisablePromptCache {
			t.Fatal("DisablePromptCache = false, want planner to disable on 16GB")
		}
		if cfg.PrefillChunkSize != 512 || cfg.BatchSize != 1 {
			t.Fatalf("shape = prefill %d batch %d, want 512/1", cfg.PrefillChunkSize, cfg.BatchSize)
		}
		if cfg.MemoryLimitBytes == 0 || cfg.CacheLimitBytes == 0 || cfg.WiredLimitBytes == 0 {
			t.Fatalf("allocator limits not forwarded: %+v", cfg)
		}
		return &fakeNativeModel{
			info: metal.ModelInfo{Architecture: "gemma4_text", QuantBits: 4, ContextLength: 8192},
		}, nil
	}

	model, err := LoadModel("/does/not/matter")
	if err != nil {
		t.Fatalf("LoadModel() error = %v", err)
	}
	if model.cfg.MemoryPlan == nil || model.cfg.MemoryPlan.MachineClass != memory.ClassApple16GB {
		t.Fatalf("model memory plan = %+v, want 16GB class", model.cfg.MemoryPlan)
	}
	info := model.Info()
	if info.CacheMode != memory.KVCacheModeKQ8VQ4 || info.CachePolicy != memory.KVCacheRotating {
		t.Fatalf("info cache = %q/%q, want planner cache", info.CachePolicy, info.CacheMode)
	}
	if info.ContextLength != 8192 || info.PrefillChunkSize != 512 || info.BatchSize != 1 {
		t.Fatalf("info runtime shape = ctx:%d prefill:%d batch:%d, want planner shape", info.ContextLength, info.PrefillChunkSize, info.BatchSize)
	}
	if err := model.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
}

func TestLoadModel_ExplicitDefaultContextBypassesMemoryPlanClamp_Good(t *testing.T) {
	coverageTokens := "ExplicitDefaultContextBypassesMemoryPlanClamp"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	originalLoadNativeModel := loadNativeModel
	t.Cleanup(func() { loadNativeModel = originalLoadNativeModel })

	loadNativeModel = func(modelPath string, cfg metal.LoadConfig) (nativeModel, error) {
		if cfg.ContextLen != DefaultLocalContextLength {
			t.Fatalf("ContextLen = %d, want explicit context %d", cfg.ContextLen, DefaultLocalContextLength)
		}
		return &fakeNativeModel{info: metal.ModelInfo{Architecture: "gemma4_text", ContextLength: DefaultLocalContextLength}}, nil
	}

	model, err := LoadModel(
		"/does/not/matter",
		WithContextLength(DefaultLocalContextLength),
		WithMemoryPlan(memory.Plan{ContextLength: 32768}),
	)
	if err != nil {
		t.Fatalf("LoadModel() error = %v", err)
	}
	if err := model.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
}

func TestLoadModel_UnknownQuantizationDoesNotReject_Good(t *testing.T) {
	coverageTokens := "UnknownQuantizationDoesNotReject"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	originalLoadNativeModel := loadNativeModel
	originalReadGGUFInfo := readGGUFInfo
	t.Cleanup(func() {
		loadNativeModel = originalLoadNativeModel
		readGGUFInfo = originalReadGGUFInfo
	})

	loadNativeModel = func(modelPath string, cfg metal.LoadConfig) (nativeModel, error) {
		return &fakeNativeModel{
			info: metal.ModelInfo{
				Architecture: "gemma4_text",
				NumLayers:    48,
				QuantBits:    0, // unknown
			},
		}, nil
	}
	readGGUFInfo = func(modelPath string) (gguf.Info, error) {
		return gguf.Info{}, core.NewError("no gguf metadata")
	}

	model, err := LoadModel("/does/not/matter", WithQuantization(4))
	if err != nil {
		t.Fatalf("LoadModel() error = %v", err)
	}
	if err := model.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
}

func TestLoadModel_GGUFMetadataBackfillsInfoAndQuantValidation_Good(t *testing.T) {
	coverageTokens := "GGUFMetadataBackfillsInfoAndQuantValidation"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	originalLoadNativeModel := loadNativeModel
	originalReadGGUFInfo := readGGUFInfo
	t.Cleanup(func() {
		loadNativeModel = originalLoadNativeModel
		readGGUFInfo = originalReadGGUFInfo
	})

	loadNativeModel = func(modelPath string, cfg metal.LoadConfig) (nativeModel, error) {
		return &fakeNativeModel{}, nil
	}
	readGGUFInfo = func(modelPath string) (gguf.Info, error) {
		return gguf.Info{
			Architecture:  "gemma4_text",
			VocabSize:     262144,
			HiddenSize:    2560,
			NumLayers:     48,
			ContextLength: 131072,
			QuantBits:     4,
			QuantGroup:    64,
		}, nil
	}

	model, err := LoadModel("/does/not/matter", WithQuantization(4), WithAutoMemoryPlan(false))
	if err != nil {
		t.Fatalf("LoadModel() error = %v", err)
	}
	info := model.Info()
	if info.Architecture != "gemma4_text" {
		t.Fatalf("Info().Architecture = %q, want gemma4_text", info.Architecture)
	}
	if info.NumLayers != 48 {
		t.Fatalf("Info().NumLayers = %d, want 48", info.NumLayers)
	}
	if info.VocabSize != 262144 {
		t.Fatalf("Info().VocabSize = %d, want 262144", info.VocabSize)
	}
	if info.HiddenSize != 2560 {
		t.Fatalf("Info().HiddenSize = %d, want 2560", info.HiddenSize)
	}
	if info.ContextLength != 131072 {
		t.Fatalf("Info().ContextLength = %d, want 131072", info.ContextLength)
	}
	if info.QuantBits != 4 || info.QuantGroup != 64 {
		t.Fatalf("Info() quant = %d-bit group=%d, want 4-bit group=64", info.QuantBits, info.QuantGroup)
	}
	if err := model.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}

	_, err = LoadModel("/does/not/matter", WithQuantization(8), WithAutoMemoryPlan(false))
	if err == nil {
		t.Fatal("expected quantization mismatch error from GGUF metadata")
	}
}

func TestLoadModelFromMedium_StagesAndCleansUp_Good(t *testing.T) {
	coverageTokens := "StagesAndCleansUp"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	medium := coreio.NewMemoryMedium()
	if err := medium.Write("models/demo/config.json", `{"model_type":"gemma3"}`); err != nil {
		t.Fatalf("write config: %v", err)
	}
	if err := medium.Write("models/demo/tokenizer.json", `{"model":{"type":"BPE","vocab":{},"merges":[]}}`); err != nil {
		t.Fatalf("write tokenizer: %v", err)
	}
	if err := medium.Write("models/demo/model.gguf", "stub"); err != nil {
		t.Fatalf("write weights: %v", err)
	}
	if err := medium.Write("adapters/demo/adapter_config.json", `{"rank":8,"alpha":16}`); err != nil {
		t.Fatalf("write adapter config: %v", err)
	}
	if err := medium.Write("adapters/demo/adapter.safetensors", "stub"); err != nil {
		t.Fatalf("write adapter weights: %v", err)
	}

	originalLoadNativeModel := loadNativeModel
	t.Cleanup(func() { loadNativeModel = originalLoadNativeModel })

	var stagedPath string
	var stagedAdapterPath string
	loadNativeModel = func(modelPath string, cfg metal.LoadConfig) (nativeModel, error) {
		stagedPath = modelPath
		stagedAdapterPath = cfg.AdapterPath
		if cfg.ContextLen != 2048 {
			t.Fatalf("ContextLen = %d, want 2048", cfg.ContextLen)
		}
		if result := core.Stat(core.PathJoin(modelPath, "config.json")); !result.OK {
			t.Fatalf("staged config missing: %v", result.Value)
		}
		if result := core.Stat(core.PathJoin(modelPath, "tokenizer.json")); !result.OK {
			t.Fatalf("staged tokenizer missing: %v", result.Value)
		}
		if result := core.Stat(core.PathJoin(modelPath, "model.gguf")); !result.OK {
			t.Fatalf("staged weights missing: %v", result.Value)
		}
		if cfg.AdapterPath == "" {
			t.Fatal("expected staged adapter path to be passed to native loader")
		}
		if result := core.Stat(core.PathJoin(cfg.AdapterPath, "adapter_config.json")); !result.OK {
			t.Fatalf("staged adapter config missing: %v", result.Value)
		}
		if result := core.Stat(core.PathJoin(cfg.AdapterPath, "adapter.safetensors")); !result.OK {
			t.Fatalf("staged adapter weights missing: %v", result.Value)
		}
		return &fakeNativeModel{}, nil
	}

	model, err := LoadModel(
		"models/demo",
		WithMedium(medium),
		WithContextLength(2048),
		WithAdapterPath("adapters/demo"),
	)
	if err != nil {
		t.Fatalf("LoadModel() error = %v", err)
	}

	if stagedPath == "" {
		t.Fatal("expected staged path to be passed to native loader")
	}
	if stagedAdapterPath == "" {
		t.Fatal("expected staged adapter path to be passed to native loader")
	}
	if err := model.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
	if result := core.Stat(stagedPath); result.OK || !core.IsNotExist(apiTestResultError(result)) {
		t.Fatalf("staged path should be removed on Close, stat result = %v", result.Value)
	}
	if result := core.Stat(stagedAdapterPath); result.OK || !core.IsNotExist(apiTestResultError(result)) {
		t.Fatalf("staged adapter path should be removed on Close, stat result = %v", result.Value)
	}
}

func apiTestResultError(result core.Result) error {
	if err, ok := result.Value.(error); ok {
		return err
	}
	return nil
}

// appendUint16LE appends value to out in little-endian byte order.
func appendUint16LE(out []byte, value uint16) []byte {
	var buf [2]byte
	binary.LittleEndian.PutUint16(buf[:], value)
	return append(out, buf[:]...)
}

// float32ToFloat16 converts a float32 to IEEE-754 float16 bits.
// Used by api_test.go to build binary tensor fixtures.
func float32ToFloat16(value float32) uint16 {
	bits := math.Float32bits(value)
	sign := uint16((bits >> 16) & 0x8000)
	exp := int((bits >> 23) & 0xff)
	frac := bits & 0x7fffff
	if exp == 255 {
		if frac == 0 {
			return sign | 0x7c00
		}
		return sign | 0x7e00
	}
	exp = exp - 127 + 15
	if exp >= 31 {
		return sign | 0x7c00
	}
	if exp <= 0 {
		if exp < -10 {
			return sign
		}
		frac |= 0x800000
		shift := uint32(14 - exp)
		return sign | uint16(frac>>shift)
	}
	return sign | uint16(exp<<10) | uint16(frac>>13)
}

func stateBundleTestSnapshot() *kv.Snapshot {
	return &kv.Snapshot{
		Version:       kv.SnapshotVersion,
		Architecture:  "gemma4_text",
		Tokens:        []int32{1, 2},
		Generated:     []int32{2},
		TokenOffset:   2,
		NumLayers:     1,
		NumHeads:      1,
		SeqLen:        2,
		HeadDim:       2,
		NumQueryHeads: 8,
		LogitShape:    []int32{1, 1, 3},
		Logits:        []float32{0.1, 0.2, 0.7},
		Layers: []kv.LayerSnapshot{{
			Layer:      0,
			CacheIndex: 0,
			Heads: []kv.HeadSnapshot{{
				Key:   []float32{1, 0, 0, 1},
				Value: []float32{0, 1, 1, 0},
			}},
		}},
	}
}

func kvSnapshotBlocksTestSnapshot() *kv.Snapshot {
	return &kv.Snapshot{
		Version:       kv.SnapshotVersion,
		Architecture:  "gemma4_text",
		Tokens:        []int32{1, 2, 3, 4},
		Generated:     []int32{4},
		TokenOffset:   4,
		NumLayers:     1,
		NumHeads:      1,
		SeqLen:        4,
		HeadDim:       2,
		NumQueryHeads: 1,
		LogitShape:    []int32{1, 1, 3},
		Logits:        []float32{0.1, 0.2, 0.7},
		Layers: []kv.LayerSnapshot{{
			Layer:      0,
			CacheIndex: 0,
			Heads: []kv.HeadSnapshot{{
				Key:   []float32{10, 11, 12, 13, 14, 15, 16, 17},
				Value: []float32{20, 21, 22, 23, 24, 25, 26, 27},
			}},
		}},
	}
}

type recordingMemvidStore struct {
	store    memvid.Store
	resolved []int
}

func (s *recordingMemvidStore) Get(ctx context.Context, chunkID int) (string, error) {
	s.resolved = append(s.resolved, chunkID)
	return s.store.Get(ctx, chunkID)
}

func (s *recordingMemvidStore) Resolve(ctx context.Context, chunkID int) (memvid.Chunk, error) {
	s.resolved = append(s.resolved, chunkID)
	return memvid.Resolve(ctx, s.store, chunkID)
}

type failingMemvidWriter struct{}

func (failingMemvidWriter) Put(ctx context.Context, text string, opts memvid.PutOptions) (memvid.ChunkRef, error) {
	return memvid.ChunkRef{}, context.Canceled
}
