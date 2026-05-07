// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && !nomlx

package mlx

import (
	"context"
	"testing"
)

func TestModelTrainSFT_NilModel_Bad(t *testing.T) {
	coverageTokens := "Model TrainSFT"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	var model *Model
	_, err := model.TrainSFT(context.Background(), NewSFTSliceDataset([]SFTSample{{Text: "x"}}), SFTConfig{})
	if err == nil {
		t.Fatal("expected nil model error")
	}
}
