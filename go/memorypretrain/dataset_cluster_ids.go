// SPDX-Licence-Identifier: EUPL-1.2

package memorypretrain

import (
	"context"
	"strings"

	core "dappco.re/go"
)

const (
	// ClusterIDTaskSchema matches upstream schema-style ICL tasks.
	ClusterIDTaskSchema = "schema"
	// ClusterIDTaskMultipleChoice matches upstream multiple-choice ICL tasks.
	ClusterIDTaskMultipleChoice = "multiple_choice"
	// ClusterIDTaskGenerationTaskWithAnswers matches upstream generation tasks.
	ClusterIDTaskGenerationTaskWithAnswers = "generation_task_with_answers"
	// ClusterIDTaskLanguageModeling matches upstream language-modelling tasks.
	ClusterIDTaskLanguageModeling = "language_modeling"
)

// ClusterIDJSONLConfig controls native JSONL enrichment with hierarchical
// memory cluster IDs.
type ClusterIDJSONLConfig struct {
	TaskType        string `json:"task_type,omitempty"`
	ClusterCounts   []int  `json:"cluster_counts,omitempty"`
	TextField       string `json:"text_field,omitempty"`
	ContextKey      string `json:"context_key,omitempty"`
	ContinuationKey string `json:"continuation_key,omitempty"`
	ChoicesKey      string `json:"choices_key,omitempty"`
	QueryKey        string `json:"query_key,omitempty"`
}

// ClusterIDJSONLReport summarises a JSONL cluster-ID enrichment pass.
type ClusterIDJSONLReport struct {
	Rows        int `json:"rows"`
	LearnedRows int `json:"learned_rows,omitempty"`
	GenericRows int `json:"generic_rows,omitempty"`
	SkippedRows int `json:"skipped_rows,omitempty"`
}

// AddClusterIDsToJSONLFile reads inputPath, writes outputPath, and adds
// cluster_ids to each JSONL row using learned routing or generic fallback.
func AddClusterIDsToJSONLFile(ctx context.Context, inputPath string, outputPath string, embedder Embedder, router *Bank, cfg ClusterIDJSONLConfig) (ClusterIDJSONLReport, error) {
	if inputPath == "" {
		return ClusterIDJSONLReport{}, core.NewError("memorypretrain: input JSONL path is required")
	}
	if outputPath == "" {
		return ClusterIDJSONLReport{}, core.NewError("memorypretrain: output JSONL path is required")
	}
	read := core.ReadFile(inputPath)
	if !read.OK {
		return ClusterIDJSONLReport{}, memoryPretrainResultError(read)
	}
	out, report, err := AddClusterIDsToJSONL(ctx, core.AsString(read.Value.([]byte)), embedder, router, cfg)
	if err != nil {
		return report, err
	}
	dir := core.PathDir(outputPath)
	if dir != "" && dir != "." {
		if result := core.MkdirAll(dir, 0o755); !result.OK {
			return report, memoryPretrainResultError(result)
		}
	}
	if result := core.WriteFile(outputPath, []byte(out), 0o644); !result.OK {
		return report, memoryPretrainResultError(result)
	}
	return report, nil
}

// AddClusterIDsToJSONL adds cluster_ids to each JSONL row. If router is nil it
// uses the upstream generic-memory fallback from cfg.ClusterCounts; otherwise it
// embeds each row's memory text and routes through the learned clustering bank.
func AddClusterIDsToJSONL(ctx context.Context, raw string, embedder Embedder, router *Bank, cfg ClusterIDJSONLConfig) (string, ClusterIDJSONLReport, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if core.Trim(raw) == "" {
		return "", ClusterIDJSONLReport{}, core.NewError("memorypretrain: JSONL input is empty")
	}
	cfg = normaliseClusterIDJSONLConfig(cfg)
	if router != nil && embedder == nil {
		return "", ClusterIDJSONLReport{}, core.NewError("memorypretrain: embedder is required for learned cluster routing")
	}
	var genericIDs []int
	var err error
	if router == nil {
		genericIDs, err = GenericClusterIDs(cfg.ClusterCounts)
		if err != nil {
			return "", ClusterIDJSONLReport{}, err
		}
	}
	lines := core.Split(raw, "\n")
	report := ClusterIDJSONLReport{}
	// Stream encoded rows straight into one Builder. Each row is written as
	// "<encoded>\n", which reproduces the previous Join("\n", rows...) +
	// Concat(joined, "\n") output byte-for-byte without the intermediate
	// []string slice or the two whole-corpus string copies.
	var out strings.Builder
	out.Grow(len(raw) + len(lines)*16)
	wrote := 0
	// Reuse one map across rows. clear() empties it while keeping its capacity,
	// so json.Unmarshal decodes into the existing buckets instead of allocating
	// a fresh map header per row. The clear is required for correctness: the JSON
	// decoder adds/overwrites keys but never removes stale ones, so a previous
	// row's keys would otherwise leak. Marshalled key sets are unchanged, and
	// json.Marshal sorts keys, so the output stays byte-identical.
	var row map[string]any
	// Two reused int buffers thread through the learned path: routeScratch backs
	// router.ClusterIDsInto, padScratch backs padClusterIDsWithGenericFallbackInto.
	// Each is fully overwritten per row before it is marshalled, and the row map is
	// discarded (clear) before the next iteration, so reusing the backing produces
	// the same cluster IDs and byte-identical output. The generic path leaves both
	// nil and reuses the shared genericIDs as before.
	var routeScratch, padScratch []int
	for index, line := range lines {
		if err := ctx.Err(); err != nil {
			return "", report, err
		}
		line = core.Trim(line)
		if line == "" {
			continue
		}
		report.Rows++
		clear(row)
		if result := core.JSONUnmarshalString(line, &row); !result.OK {
			return "", report, core.Errorf("memorypretrain: parse JSONL record %d: %w", index+1, result.Value.(error))
		}
		memoryText := clusterIDJSONLMemoryText(row, cfg)
		if memoryText == "" {
			return "", report, core.Errorf("memorypretrain: JSONL record %d has no memory text", index+1)
		}
		clusterIDs := genericIDs
		if router != nil {
			embedding, err := embedder.Embed(ctx, memoryText)
			if err != nil {
				return "", report, core.Errorf("memorypretrain: embed JSONL record %d: %v", index+1, err)
			}
			routeScratch, err = router.ClusterIDsInto(routeScratch, embedding)
			if err != nil {
				return "", report, core.Errorf("memorypretrain: route JSONL record %d: %v", index+1, err)
			}
			padScratch, err = padClusterIDsWithGenericFallbackInto(padScratch, routeScratch, cfg.ClusterCounts)
			if err != nil {
				return "", report, core.Errorf("memorypretrain: route JSONL record %d: %v", index+1, err)
			}
			clusterIDs = padScratch
			report.LearnedRows++
		} else {
			report.GenericRows++
		}
		// clusterIDs is a slice we own here: the learned path threads the reused
		// padScratch buffer (fully overwritten this row), and the generic path's
		// genericIDs is never mutated. The map value is only read by the marshaller
		// in this iteration and the row is discarded (clear) before the next, so
		// storing the reused buffer directly is safe and avoids a per-row copy.
		row["cluster_ids"] = clusterIDs
		encoded := core.JSONMarshalString(row)
		if encoded == "" {
			return "", report, core.Errorf("memorypretrain: marshal JSONL record %d", index+1)
		}
		out.WriteString(encoded)
		out.WriteByte('\n')
		wrote++
	}
	if wrote == 0 {
		return "", report, core.NewError("memorypretrain: JSONL input produced no rows")
	}
	return out.String(), report, nil
}

func normaliseClusterIDJSONLConfig(cfg ClusterIDJSONLConfig) ClusterIDJSONLConfig {
	if cfg.TaskType == "" {
		cfg.TaskType = ClusterIDTaskLanguageModeling
	}
	if cfg.TextField == "" {
		cfg.TextField = "text"
	}
	if cfg.ContextKey == "" {
		cfg.ContextKey = "context"
	}
	if cfg.ContinuationKey == "" {
		cfg.ContinuationKey = "continuation"
	}
	if cfg.ChoicesKey == "" {
		cfg.ChoicesKey = "context_options"
	}
	if cfg.QueryKey == "" {
		cfg.QueryKey = "query"
	}
	return cfg
}

func clusterIDJSONLMemoryText(row map[string]any, cfg ClusterIDJSONLConfig) string {
	switch cfg.TaskType {
	case ClusterIDTaskSchema:
		common := commonStringPair(stringListField(row, cfg.ChoicesKey))
		return core.Trim(core.Concat(common, " ", stringField(row, cfg.ContinuationKey)))
	case ClusterIDTaskMultipleChoice:
		if query := stringField(row, cfg.QueryKey); query != "" {
			return query
		}
		return firstClusterIDJSONLString(row, cfg.ContextKey, cfg.TextField)
	case ClusterIDTaskGenerationTaskWithAnswers, ClusterIDTaskLanguageModeling:
		return firstClusterIDJSONLString(row, cfg.ContextKey, cfg.TextField)
	default:
		return firstClusterIDJSONLString(row, cfg.ContextKey, cfg.TextField)
	}
}

func firstClusterIDJSONLString(row map[string]any, keys ...string) string {
	for _, key := range keys {
		if value := stringField(row, key); value != "" {
			return value
		}
	}
	return ""
}

func stringField(row map[string]any, key string) string {
	if row == nil || key == "" {
		return ""
	}
	value, ok := row[key]
	if !ok {
		return ""
	}
	switch typed := value.(type) {
	case string:
		return core.Trim(typed)
	case []any:
		if len(typed) == 0 {
			return ""
		}
		if first, ok := typed[0].(string); ok {
			return core.Trim(first)
		}
	}
	return ""
}

func stringListField(row map[string]any, key string) []string {
	value, ok := row[key]
	if !ok {
		return nil
	}
	switch typed := value.(type) {
	case []any:
		out := make([]string, 0, len(typed))
		for _, item := range typed {
			if text, ok := item.(string); ok && core.Trim(text) != "" {
				out = append(out, core.Trim(text))
			}
		}
		return out
	case []string:
		return append([]string(nil), typed...)
	case string:
		if typed = core.Trim(typed); typed != "" {
			return []string{typed}
		}
	}
	return nil
}

func commonStringPair(values []string) string {
	if len(values) < 2 {
		if len(values) == 1 {
			return values[0]
		}
		return ""
	}
	left := values[0]
	right := values[1]
	bestStart := 0
	bestLen := 0
	for i := 0; i < len(left); i++ {
		for j := 0; j < len(right); j++ {
			length := 0
			for i+length < len(left) && j+length < len(right) && left[i+length] == right[j+length] {
				length++
			}
			if length > bestLen {
				bestStart = i
				bestLen = length
			}
		}
	}
	if bestLen < 5 {
		return ""
	}
	return core.Trim(left[bestStart : bestStart+bestLen])
}
