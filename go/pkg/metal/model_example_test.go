// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import core "dappco.re/go"

func ExampleInternalModel() {
	var model InternalModel = exampleTrainingInternal()
	adapter := model.ApplyLoRA(LoRAConfig{Rank: 4, Scale: 3, TargetLayers: []string{"q_proj"}})

	core.Println(model.ModelType(), model.NumLayers(), adapter.Config.TargetKeys, adapter.Config.Alpha)
	// Output: gemma4_text 3 [q_proj] 12
}
