// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	core "dappco.re/go"
	mp "dappco.re/go/mlx/pack"
)

// OfficialGemma4E2BPairReport validates the official Google E2B target and
// MTP assistant as a pair. It is intentionally metadata-only so callers can
// prove the attachment contract before running a heavyweight model load.
type OfficialGemma4E2BPairReport struct {
	Target        OfficialGemma4E2BSnapshotReport `json:"target"`
	Assistant     OfficialGemma4E2BSnapshotReport `json:"assistant"`
	PairOK        bool                            `json:"pair_ok"`
	TargetPath    string                          `json:"target_path"`
	AssistantPath string                          `json:"assistant_path"`

	SameVocabSize                  bool `json:"same_vocab_size"`
	SameContextLength              bool `json:"same_context_length"`
	AssistantBackboneMatchesTarget bool `json:"assistant_backbone_matches_target"`
	AssistantAttachable            bool `json:"assistant_attachable"`

	TargetHiddenSize                  int  `json:"target_hidden_size,omitempty"`
	AssistantBackboneHiddenSize       int  `json:"assistant_backbone_hidden_size,omitempty"`
	AssistantOrderedEmbeddings        bool `json:"assistant_ordered_embeddings"`
	AssistantNumCentroids             int  `json:"assistant_num_centroids,omitempty"`
	AssistantCentroidIntermediateTopK int  `json:"assistant_centroid_intermediate_top_k,omitempty"`

	Error string `json:"error,omitempty"`
}

// InspectOfficialGemma4E2BPairSnapshots verifies the default official target
// and assistant locks, then checks the assistant attachment metadata.
func InspectOfficialGemma4E2BPairSnapshots(targetDir, assistantDir string, opts ...mp.ModelPackOption) (OfficialGemma4E2BPairReport, error) {
	return InspectOfficialGemma4E2BPairLocalSnapshots(targetDir, assistantDir, OfficialGemma4E2BTargetLock(), OfficialGemma4E2BAssistantLock(), opts...)
}

// InspectOfficialGemma4E2BPairLocalSnapshots verifies the supplied target and
// assistant locks, then checks that the assistant can attach to the target
// hidden-state/KV contract.
func InspectOfficialGemma4E2BPairLocalSnapshots(targetDir, assistantDir string, targetLock, assistantLock OfficialGemma4E2BLock, opts ...mp.ModelPackOption) (OfficialGemma4E2BPairReport, error) {
	report := OfficialGemma4E2BPairReport{
		TargetPath:    targetDir,
		AssistantPath: assistantDir,
	}

	target, err := InspectOfficialGemma4E2BLocalSnapshot(targetDir, targetLock, opts...)
	report.Target = target
	if err != nil {
		return officialGemma4PairReportError(report, core.E("mlx: official Gemma 4 E2B pair", "target preflight", err))
	}

	assistant, err := InspectOfficialGemma4E2BLocalSnapshot(assistantDir, assistantLock, opts...)
	report.Assistant = assistant
	if err != nil {
		return officialGemma4PairReportError(report, core.E("mlx: official Gemma 4 E2B pair", "assistant preflight", err))
	}

	summary, err := readOfficialGemma4AssistantSummary(assistantDir)
	if err != nil {
		return officialGemma4PairReportError(report, err)
	}

	report.TargetHiddenSize = target.Pack.HiddenSize
	report.AssistantBackboneHiddenSize = summary.BackboneHiddenSize
	report.AssistantOrderedEmbeddings = summary.UseOrderedEmbeddings
	report.AssistantNumCentroids = summary.NumCentroids
	report.AssistantCentroidIntermediateTopK = summary.CentroidIntermediateTopK
	report.SameVocabSize = target.Pack.VocabSize > 0 && target.Pack.VocabSize == assistant.Pack.VocabSize
	report.SameContextLength = target.Pack.ContextLength > 0 && target.Pack.ContextLength == assistant.Pack.ContextLength
	report.AssistantBackboneMatchesTarget = target.Pack.HiddenSize > 0 && summary.BackboneHiddenSize == target.Pack.HiddenSize
	report.AssistantAttachable = assistant.ArchitectureOK &&
		assistant.Pack.Architecture == "gemma4_assistant" &&
		report.SameVocabSize &&
		report.SameContextLength &&
		report.AssistantBackboneMatchesTarget &&
		report.AssistantOrderedEmbeddings &&
		report.AssistantNumCentroids > 0 &&
		report.AssistantCentroidIntermediateTopK > 0
	report.PairOK = target.Verified &&
		assistant.Verified &&
		target.ArchitectureOK &&
		assistant.ArchitectureOK &&
		target.Pack.NativeLoadable &&
		report.AssistantAttachable
	if !report.PairOK {
		return officialGemma4PairReportError(report, core.NewError("mlx: official Gemma 4 E2B target+assistant pair metadata is incompatible"))
	}
	return report, nil
}

type officialGemma4AssistantSummary struct {
	BackboneHiddenSize       int  `json:"backbone_hidden_size"`
	NumCentroids             int  `json:"num_centroids"`
	CentroidIntermediateTopK int  `json:"centroid_intermediate_top_k"`
	UseOrderedEmbeddings     bool `json:"use_ordered_embeddings"`
}

func readOfficialGemma4AssistantSummary(assistantDir string) (officialGemma4AssistantSummary, error) {
	read := core.ReadFile(core.PathJoin(assistantDir, "config.json"))
	if !read.OK {
		return officialGemma4AssistantSummary{}, core.E("mlx: official Gemma 4 E2B pair", "read assistant config", officialGemma4ResultError(read))
	}
	var summary officialGemma4AssistantSummary
	if result := core.JSONUnmarshal(read.Value.([]byte), &summary); !result.OK {
		return officialGemma4AssistantSummary{}, core.E("mlx: official Gemma 4 E2B pair", "parse assistant config", officialGemma4ResultError(result))
	}
	return summary, nil
}

func officialGemma4PairReportError(report OfficialGemma4E2BPairReport, err error) (OfficialGemma4E2BPairReport, error) {
	if err != nil {
		report.PairOK = false
		report.Error = err.Error()
	}
	return report, err
}
