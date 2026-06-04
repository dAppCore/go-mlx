// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"flag"
	"hash/fnv"
	"io"
	"math"

	core "dappco.re/go"
	"dappco.re/go/mlx/memorypretrain"
)

type memoryPretrainBuildReport struct {
	Version   int                                             `json:"version"`
	Kind      string                                          `json:"kind"`
	NoPython  bool                                            `json:"no_python"`
	Embedding string                                          `json:"embedding"`
	Report    *memorypretrain.MemoryPretrainingArtifactReport `json:"report,omitempty"`
}

func runMemoryPretrainBuildCommand(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("memory-pretrain-build", flag.ContinueOnError)
	fs.SetOutput(stderr)
	jsonOut := fs.Bool("json", false, "write JSON report")
	corpusPath := fs.String("corpus", "", "input corpus JSONL with id, text, and optional string meta")
	routerPath := fs.String("router", "", "output hierarchical router bank JSON")
	ffnMemoryPath := fs.String("ffn-memory", "", "output FFN memory bank JSON")
	hiddenSize := fs.Int("hidden-size", 0, "anchor hidden size / embedding dimension")
	layers := fs.Int("layers", 0, "number of transformer layers to allocate FFN memory for")
	levels := fs.String("levels", "1,2,3,4", "comma-separated memory level names")
	tokens := fs.String("tokens", "8,16,32,64", "comma-separated FFN memory token counts per level")
	branching := fs.Int("branching", 8, "hierarchical KMeans branching factor")
	depth := fs.Int("depth", 3, "hierarchical KMeans max depth")
	minClusterSize := fs.Int("min-cluster-size", 8, "minimum cluster size before splitting")
	kmeansIters := fs.Int("kmeans-iters", 16, "KMeans iterations per split")
	clusterInput := fs.String("cluster-input", "", "optional task JSONL to enrich with cluster_ids")
	clusterOutput := fs.String("cluster-output", "", "output JSONL for -cluster-input")
	taskType := fs.String("task-type", memorypretrain.ClusterIDTaskLanguageModeling, "cluster task type: language_modeling, multiple_choice, generation_task_with_answers, or schema")
	fs.Usage = func() {
		name := cliCommandName("memory-pretrain-build")
		core.WriteString(stderr, core.Sprintf("Usage: %s [flags]\n", name))
		core.WriteString(stderr, "Build native hierarchical-memory pretraining artifacts from corpus JSONL.\n")
		fs.VisitAll(func(f *flag.Flag) {
			if f.DefValue == "" {
				core.WriteString(stderr, core.Sprintf("  -%s\n\t%s\n", f.Name, f.Usage))
				return
			}
			core.WriteString(stderr, core.Sprintf("  -%s\n\t%s (default %q)\n", f.Name, f.Usage, f.DefValue))
		})
	}
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		core.Print(stderr, "%s memory-pretrain-build: expected no positional arguments", cliName())
		return 2
	}
	levelNames := parseMemoryPretrainCSV(*levels)
	tokenCounts, err := parseMemoryPretrainInts(*tokens)
	if err != nil {
		core.Print(stderr, "%s memory-pretrain-build: %v", cliName(), err)
		return 2
	}
	cfg := memorypretrain.MemoryPretrainingArtifactConfig{
		CorpusPath:          core.Trim(*corpusPath),
		RouterPath:          core.Trim(*routerPath),
		FFNMemoryPath:       core.Trim(*ffnMemoryPath),
		Build:               memorypretrain.BuildConfig{BranchingFactor: *branching, MaxDepth: *depth, MinClusterSize: *minClusterSize, KMeansIters: *kmeansIters},
		FFNMemory:           memorypretrain.FFNMemoryConfig{HiddenSize: *hiddenSize, Layers: *layers, MemoryLevels: levelNames, FFNMemoryTokens: tokenCounts},
		ClusterIDInputPath:  core.Trim(*clusterInput),
		ClusterIDOutputPath: core.Trim(*clusterOutput),
		ClusterIDJSONL:      memorypretrain.ClusterIDJSONLConfig{TaskType: core.Trim(*taskType)},
	}
	artifacts, err := memorypretrain.BuildMemoryPretrainingArtifactsFromFiles(ctx, memoryPretrainTextHashEmbedder(*hiddenSize), cfg)
	if err != nil {
		core.Print(stderr, "%s memory-pretrain-build: %v", cliName(), err)
		return 2
	}
	report := memoryPretrainBuildReport{
		Version:   1,
		Kind:      "memory-pretraining-artifacts",
		NoPython:  true,
		Embedding: "text-hash",
		Report:    artifacts.Report,
	}
	if *jsonOut {
		data := core.JSONMarshalIndent(report, "", "  ")
		if !data.OK {
			core.Print(stderr, "%s memory-pretrain-build: marshal report failed", cliName())
			return 1
		}
		core.WriteString(stdout, core.AsString(data.Value.([]byte)))
		core.WriteString(stdout, "\n")
		return 0
	}
	core.WriteString(stdout, "memory pretraining artifacts\n")
	if report.Report != nil {
		core.WriteString(stdout, core.Sprintf("  corpus: %d records\n", report.Report.CorpusRecords))
		core.WriteString(stdout, core.Sprintf("  router: %d nodes -> %s\n", report.Report.RouterNodes, report.Report.RouterPath))
		core.WriteString(stdout, core.Sprintf("  ffn memory: %d layers -> %s\n", report.Report.FFNMemoryLayers, report.Report.FFNMemoryPath))
	}
	return 0
}

func parseMemoryPretrainCSV(raw string) []string {
	parts := core.Split(raw, ",")
	out := make([]string, 0, len(parts))
	for _, part := range parts {
		part = core.Trim(part)
		if part != "" {
			out = append(out, part)
		}
	}
	return out
}

func parseMemoryPretrainInts(raw string) ([]int, error) {
	parts := parseMemoryPretrainCSV(raw)
	out := make([]int, 0, len(parts))
	for _, part := range parts {
		result := core.Atoi(part)
		if !result.OK {
			return nil, core.Errorf("invalid integer %q", part)
		}
		out = append(out, result.Value.(int))
	}
	return out, nil
}

func memoryPretrainTextHashEmbedder(dim int) memorypretrain.Embedder {
	return memorypretrain.EmbedFunc(func(ctx context.Context, text string) ([]float32, error) {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if dim <= 0 {
			return nil, core.NewError("memorypretrain: text-hash embedding dimension must be positive")
		}
		out := make([]float32, dim)
		for _, token := range core.Split(text, " ") {
			token = core.Trim(token)
			if token == "" {
				continue
			}
			for i := range out {
				h := fnv.New32a()
				_, _ = h.Write([]byte(token))
				_, _ = h.Write([]byte{byte(i), byte(i >> 8)})
				bucket := int(h.Sum32()%2001) - 1000
				out[i] += float32(bucket) / 1000
			}
		}
		var norm float64
		for _, value := range out {
			norm += float64(value * value)
		}
		if norm == 0 {
			out[0] = 1
			return out, nil
		}
		scale := float32(1 / math.Sqrt(norm))
		for i := range out {
			out[i] *= scale
		}
		return out, nil
	})
}
