// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	core "dappco.re/go"
	"dappco.re/go/inference/quant/codebook"
	"dappco.re/go/inference/quant/jang"
	"dappco.re/go/mlx/gguf"
	mp "dappco.re/go/mlx/pack"
	"dappco.re/go/mlx/quant/autoround"
)

func inspectModelPackJANG(pack *mp.ModelPack, root string, dir *modelPackDirIndex) {
	if !dir.has("jang_config.json") {
		return
	}
	info, err := jang.ReadConfig(root)
	if err != nil {
		pack.AddIssue(mp.ModelPackIssueWarning, mp.ModelPackIssueQuantizationMismatch, "jang_config.json could not be parsed: "+err.Error(), core.PathJoin(root, "jang_config.json"))
		return
	}
	if info == nil {
		return
	}
	pack.JANG = info
	pack.PackedQuantization = jang.ClonePackedProfile(info.Packed)
	if info.SourceArchitecture != "" && pack.Architecture == "" {
		pack.Architecture = info.SourceArchitecture
	}
	if info.BitsDefault > 0 {
		pack.QuantBits = info.BitsDefault
	}
	if info.GroupSize > 0 {
		pack.QuantGroup = info.GroupSize
	}
	if info.Packed != nil {
		pack.QuantType = info.Packed.Type
	}
	pack.QuantFamily = "jang"
	pack.Quantization = &gguf.QuantizationInfo{
		Type:      pack.QuantType,
		Family:    pack.QuantFamily,
		Bits:      pack.QuantBits,
		GroupSize: pack.QuantGroup,
		Mixed:     true,
	}
}

func inspectModelPackAutoRound(pack *mp.ModelPack, root string, dir *modelPackDirIndex) {
	if !dir.has(autoround.PackConfigFileAutoRound) && !dir.has(autoround.PackConfigFileQuantization) {
		return
	}
	info, err := autoround.ReadPackInfo(root)
	if err != nil {
		pack.AddIssue(mp.ModelPackIssueError, mp.ModelPackIssueUnsupportedAutoRound, "AutoRound quantization config could not be parsed: "+err.Error(), autoRoundConfigIssuePath(root, dir))
		return
	}
	if info == nil {
		return
	}
	pack.AutoRound = autoround.ClonePackInfo(info)
	pack.QuantBits = firstPositive(info.Bits, pack.QuantBits)
	pack.QuantGroup = firstPositive(info.GroupSize, pack.QuantGroup)
	pack.QuantType = firstNonEmpty(string(info.Scheme), string(info.ExportFormat), "auto-round")
	pack.QuantFamily = autoround.QuantFamilyAutoRound
	pack.Quantization = autoround.ClonePackInfo(info)
	if info.NativeFormat() && pack.Format == mp.ModelPackFormatSafetensors {
		if !info.NativeTensorMap() {
			pack.AddIssue(mp.ModelPackIssueError, mp.ModelPackIssueUnsupportedAutoRound, "AutoRound native safetensors pack metadata is recognised, but no native tensor map was provided", info.Path)
			return
		}
		if err := autoround.ValidateSafetensorsTensorMap(*info, pack.WeightFiles); err != nil {
			pack.AddIssue(mp.ModelPackIssueError, mp.ModelPackIssueUnsupportedAutoRound, "AutoRound native tensor map could not be validated: "+err.Error(), info.Path)
			return
		}
	}
}

func autoRoundConfigIssuePath(root string, dir *modelPackDirIndex) string {
	if dir != nil && dir.has(autoround.PackConfigFileAutoRound) {
		return core.PathJoin(root, autoround.PackConfigFileAutoRound)
	}
	return core.PathJoin(root, autoround.PackConfigFileQuantization)
}

func inspectModelPackCodebook(pack *mp.ModelPack, root string, dir *modelPackDirIndex) {
	if !dir.has("codebook_config.json") {
		return
	}
	profile, err := codebook.ReadProfile(root)
	if err != nil {
		pack.AddIssue(mp.ModelPackIssueError, mp.ModelPackIssueUnsupportedCodebook, "codebook_config.json could not be parsed: "+err.Error(), core.PathJoin(root, "codebook_config.json"))
		return
	}
	if profile == nil {
		return
	}
	pack.Codebook = codebook.CloneProfile(profile)
	pack.QuantType = codebook.FormatVQ
	pack.QuantFamily = codebook.Type
	pack.QuantBits = firstPositive(pack.QuantBits, profile.IndexBits)
	pack.Quantization = &gguf.QuantizationInfo{
		Type:   pack.QuantType,
		Family: pack.QuantFamily,
		Bits:   pack.QuantBits,
		Mixed:  true,
	}
	pack.AddIssue(mp.ModelPackIssueError, mp.ModelPackIssueUnsupportedCodebook, "codebook/VQ tensor matvec is available, but full codebook-quantized model loading is not implemented yet", core.PathJoin(root, "codebook_config.json"))
}

func cloneGGUFQuantizationInfo(info gguf.QuantizationInfo) *gguf.QuantizationInfo {
	if info.Type == "" && info.Family == "" && info.Bits == 0 && len(info.TensorTypes) == 0 {
		return nil
	}
	cloned := info
	cloned.TensorTypes = core.SliceClone(info.TensorTypes)
	return &cloned
}
