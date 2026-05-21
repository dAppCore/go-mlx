// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"flag"
	"io"
	"iter"
	"os/signal"
	"sort"
	"sync"
	"syscall"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/bench"
	statefile "dappco.re/go/inference/state/filestore"
	mlx "dappco.re/go/mlx"
	"dappco.re/go/mlx/agent"
	"dappco.re/go/mlx/internal/metal"
	"dappco.re/go/mlx/memory"
	"dappco.re/go/mlx/model"
	"dappco.re/go/mlx/pack"
	"dappco.re/go/mlx/probe"
)

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	args := core.Args()
	if len(args) > 0 {
		if name := core.PathBase(args[0]); name != "" {
			commandName = name
		}
	}
	core.Exit(runCommand(ctx, args[1:], core.Stdout(), core.Stderr()))
}

var commandName = "go-mlx"

func cliName() string {
	name := core.Trim(commandName)
	if name == "" {
		return "go-mlx"
	}
	return name
}

func cliCommandName(command string) string {
	if command == "" {
		return cliName()
	}
	return cliName() + " " + command
}

func runCommand(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	if len(args) == 0 {
		printUsage(stdout)
		return 0
	}
	switch args[0] {
	case "bench":
		return runBenchCommand(ctx, args[1:], stdout, stderr)
	case "chapter-profile":
		return runChapterProfileCommand(ctx, args[1:], stdout, stderr)
	case "discover":
		return runDiscoverCommand(ctx, args[1:], stdout, stderr)
	case "driver-profile":
		return runDriverProfileCommand(ctx, args[1:], stdout, stderr)
	case "ffn-estimate":
		return runFFNEstimateCommand(ctx, args[1:], stdout, stderr)
	case "pack":
		return runPackCommand(ctx, args[1:], stdout, stderr)
	case "profile-list":
		return runProfileListCommand(ctx, args[1:], stdout, stderr)
	case "profile-select":
		return runProfileSelectCommand(ctx, args[1:], stdout, stderr)
	case "replace-plan":
		return runReplacePlanCommand(ctx, args[1:], stdout, stderr)
	case "slice":
		return runSliceCommand(ctx, args[1:], stdout, stderr)
	case "slice-smoke":
		return runSliceSmokeCommand(ctx, args[1:], stdout, stderr)
	case "state-ramp-profile":
		return runStateRampProfileCommand(ctx, args[1:], stdout, stderr)
	case "tune-plan":
		return runTunePlanCommand(ctx, args[1:], stdout, stderr)
	case "tune-profile":
		return runTuneProfileCommand(ctx, args[1:], stdout, stderr)
	case "tune-run":
		return runTuneRunCommand(ctx, args[1:], stdout, stderr)
	case "-h", "--help", "help":
		printUsage(stdout)
		return 0
	default:
		core.Print(stderr, "%s: unknown command %q", cliName(), args[0])
		printUsage(stderr)
		return 2
	}
}

type cpuFFNMemoryEstimateReport struct {
	Version              int                          `json:"version"`
	SourcePath           string                       `json:"source_path"`
	CPUFFNCache          int                          `json:"cpu_ffn_cache"`
	CPUFFNMemoryEstimate *mlx.CPUSplitFFNMemoryReport `json:"cpu_ffn_memory_estimate,omitempty"`
	Error                string                       `json:"error,omitempty"`
}

type sliceSmokeReport struct {
	Version                   int                          `json:"version"`
	SourcePath                string                       `json:"source_path"`
	OutputPath                string                       `json:"output_path"`
	Preset                    inference.ModelSlicePreset   `json:"preset"`
	SliceDuration             time.Duration                `json:"slice_duration"`
	LoadDuration              time.Duration                `json:"load_duration,omitempty"`
	BenchDuration             time.Duration                `json:"bench_duration,omitempty"`
	SplitDuration             time.Duration                `json:"split_duration,omitempty"`
	OutputWeightBytes         int64                        `json:"output_weight_bytes,omitempty"`
	ReloadSkipped             bool                         `json:"reload_skipped,omitempty"`
	SplitOutput               string                       `json:"split_output,omitempty"`
	CPUFFNMemory              *mlx.CPUSplitFFNMemoryReport `json:"cpu_ffn_memory,omitempty"`
	CPUFFNMemoryEstimate      *mlx.CPUSplitFFNMemoryReport `json:"cpu_ffn_memory_estimate,omitempty"`
	CPUFFNMemoryEstimateError string                       `json:"cpu_ffn_memory_estimate_error,omitempty"`
	Slice                     *inference.ModelSlicePlan    `json:"slice,omitempty"`
	Placement                 *mlx.ModelSliceInspection    `json:"placement,omitempty"`
	Bench                     *bench.Report                `json:"bench,omitempty"`
	Error                     string                       `json:"error,omitempty"`
}

type sliceSmokeSplitResult struct {
	Output               string
	Duration             time.Duration
	CPUFFNMemory         *mlx.CPUSplitFFNMemoryReport
	CPUFFNMemoryEstimate *mlx.CPUSplitFFNMemoryReport
}

type tuneProfileReport struct {
	Version     int                       `json:"version"`
	ProfilePath string                    `json:"profile_path"`
	ModelPath   string                    `json:"model_path,omitempty"`
	Workload    inference.TuningWorkload  `json:"workload,omitempty"`
	MachineHash string                    `json:"machine_hash,omitempty"`
	CandidateID string                    `json:"candidate_id,omitempty"`
	Runtime     inference.RuntimeIdentity `json:"runtime,omitempty"`
	Load        tuneProfileLoadSettings   `json:"load,omitempty"`
	Score       inference.TuningScore     `json:"score,omitempty"`
	Profile     *inference.TuningProfile  `json:"profile,omitempty"`
}

type tuneProfileLoadSettings struct {
	ContextLength        int    `json:"context_length,omitempty"`
	ParallelSlots        int    `json:"parallel_slots,omitempty"`
	PromptCache          bool   `json:"prompt_cache,omitempty"`
	PromptCacheMinTokens int    `json:"prompt_cache_min_tokens,omitempty"`
	CachePolicy          string `json:"cache_policy,omitempty"`
	CacheMode            string `json:"cache_mode,omitempty"`
	BatchSize            int    `json:"batch_size,omitempty"`
	PrefillChunkSize     int    `json:"prefill_chunk_size,omitempty"`
	ExpectedQuantization int    `json:"expected_quantization,omitempty"`
	MemoryLimitBytes     uint64 `json:"memory_limit_bytes,omitempty"`
	CacheLimitBytes      uint64 `json:"cache_limit_bytes,omitempty"`
	WiredLimitBytes      uint64 `json:"wired_limit_bytes,omitempty"`
	AdapterPath          string `json:"adapter_path,omitempty"`
}

type replacePlanReport struct {
	Version            int                           `json:"version"`
	CurrentProfilePath string                        `json:"current_profile_path,omitempty"`
	NextProfilePath    string                        `json:"next_profile_path,omitempty"`
	Request            inference.ModelReplaceRequest `json:"request,omitempty"`
	Plan               inference.ModelReplacePlan    `json:"plan,omitempty"`
}

type profileSelectCriteria struct {
	MachineHash string                   `json:"machine_hash,omitempty"`
	ModelPath   string                   `json:"model_path,omitempty"`
	Workload    inference.TuningWorkload `json:"workload,omitempty"`
}

type profileListOptions struct {
	IncludeProfile  bool `json:"include_profile,omitempty"`
	BestPerWorkload bool `json:"best_per_workload,omitempty"`
}

type profileSelectReport struct {
	Version         int                       `json:"version"`
	ProfileDir      string                    `json:"profile_dir"`
	ProfilePath     string                    `json:"profile_path"`
	MachineHash     string                    `json:"machine_hash,omitempty"`
	ModelPath       string                    `json:"model_path,omitempty"`
	Workload        inference.TuningWorkload  `json:"workload,omitempty"`
	MatchedProfiles int                       `json:"matched_profiles"`
	CandidateID     string                    `json:"candidate_id,omitempty"`
	Runtime         inference.RuntimeIdentity `json:"runtime,omitempty"`
	Load            tuneProfileLoadSettings   `json:"load,omitempty"`
	Score           inference.TuningScore     `json:"score,omitempty"`
	Profile         *inference.TuningProfile  `json:"profile,omitempty"`
	Warnings        []string                  `json:"warnings,omitempty"`
}

type profileListReport struct {
	Version      int                      `json:"version"`
	ProfileDir   string                   `json:"profile_dir"`
	MachineHash  string                   `json:"machine_hash,omitempty"`
	ModelPath    string                   `json:"model_path,omitempty"`
	Workload     inference.TuningWorkload `json:"workload,omitempty"`
	ProfileCount int                      `json:"profile_count"`
	Profiles     []tuneProfileReport      `json:"profiles,omitempty"`
	Warnings     []string                 `json:"warnings,omitempty"`
}

type driverProfileOptions struct {
	Prompt           string                    `json:"prompt,omitempty"`
	PromptSuffix     string                    `json:"prompt_suffix,omitempty"`
	PromptChunkBytes int                       `json:"prompt_chunk_bytes,omitempty"`
	PromptRepeat     int                       `json:"prompt_repeat,omitempty"`
	MaxTokens        int                       `json:"max_tokens,omitempty"`
	Runs             int                       `json:"runs,omitempty"`
	IncludeOutput    bool                      `json:"include_output,omitempty"`
	Chat             bool                      `json:"chat,omitempty"`
	TraceTokenPhases bool                      `json:"trace_token_phases,omitempty"`
	StopTokenIDs     []int32                   `json:"-"`
	SuppressTokenIDs []int32                   `json:"-"`
	SafetyLimits     driverProfileSafetyLimits `json:"safety_limits,omitempty"`
}

type driverProfileReport struct {
	Version           int                       `json:"version"`
	ModelPath         string                    `json:"model_path"`
	LoadDuration      time.Duration             `json:"load_duration,omitempty"`
	PromptBytes       int                       `json:"prompt_bytes"`
	PromptSuffixBytes int                       `json:"prompt_suffix_bytes,omitempty"`
	PromptChunkBytes  int                       `json:"prompt_chunk_bytes,omitempty"`
	PromptRepeat      int                       `json:"prompt_repeat,omitempty"`
	MaxTokens         int                       `json:"max_tokens"`
	RequestedRuns     int                       `json:"requested_runs"`
	Chat              bool                      `json:"chat,omitempty"`
	TraceTokenPhases  bool                      `json:"trace_token_phases,omitempty"`
	SafetyLimits      driverProfileSafetyLimits `json:"safety_limits,omitempty"`
	StopTokenIDs      []int32                   `json:"stop_token_ids,omitempty"`
	SuppressTokenIDs  []int32                   `json:"suppress_token_ids,omitempty"`
	RuntimeGates      map[string]string         `json:"runtime_gates,omitempty"`
	Load              *tuneProfileLoadSettings  `json:"load,omitempty"`
	Runs              []driverProfileRun        `json:"runs,omitempty"`
	Summary           driverProfileSummary      `json:"summary"`
	EstimatedEnergy   *driverProfileEnergy      `json:"estimated_energy,omitempty"`
	Error             string                    `json:"error,omitempty"`
}

type driverProfileRun struct {
	Index                  int           `json:"index"`
	Duration               time.Duration `json:"duration"`
	RestoreDuration        time.Duration `json:"restore_duration,omitempty"`
	FirstTokenDuration     time.Duration `json:"first_token_duration,omitempty"`
	StreamDuration         time.Duration `json:"stream_duration,omitempty"`
	DriverOverheadDuration time.Duration `json:"driver_overhead_duration,omitempty"`
	VisibleTokens          int           `json:"visible_tokens,omitempty"`
	SampledTokenIDs        []int32       `json:"sampled_token_ids,omitempty"`
	SampledTokenTexts      []string      `json:"sampled_token_texts,omitempty"`
	Output                 string        `json:"output,omitempty"`
	Metrics                mlx.Metrics   `json:"metrics"`
	Error                  string        `json:"error,omitempty"`
}

type driverProfileSummary struct {
	SuccessfulRuns             int                               `json:"successful_runs"`
	FailedRuns                 int                               `json:"failed_runs,omitempty"`
	PromptTokensAverage        float64                           `json:"prompt_tokens_average,omitempty"`
	PromptTokensMin            int                               `json:"prompt_tokens_min,omitempty"`
	PromptTokensMax            int                               `json:"prompt_tokens_max,omitempty"`
	GeneratedTokens            int                               `json:"generated_tokens,omitempty"`
	VisibleTokens              int                               `json:"visible_tokens,omitempty"`
	TotalDuration              time.Duration                     `json:"total_duration,omitempty"`
	RestoreAvgDuration         time.Duration                     `json:"restore_duration_average,omitempty"`
	RestoreMinDuration         time.Duration                     `json:"restore_duration_min,omitempty"`
	RestoreMaxDuration         time.Duration                     `json:"restore_duration_max,omitempty"`
	FirstTokenAvgDuration      time.Duration                     `json:"first_token_avg_duration,omitempty"`
	FirstTokenMinDuration      time.Duration                     `json:"first_token_min_duration,omitempty"`
	FirstTokenMaxDuration      time.Duration                     `json:"first_token_max_duration,omitempty"`
	DriverOverheadAvgDuration  time.Duration                     `json:"driver_overhead_avg_duration,omitempty"`
	PrefillTokensPerSecAverage float64                           `json:"prefill_tokens_per_sec_average,omitempty"`
	DecodeTokensPerSecAverage  float64                           `json:"decode_tokens_per_sec_average,omitempty"`
	PeakMemoryBytes            uint64                            `json:"peak_memory_bytes,omitempty"`
	ActiveMemoryBytes          uint64                            `json:"active_memory_bytes,omitempty"`
	CacheMemoryBytes           uint64                            `json:"cache_memory_bytes,omitempty"`
	ProcessVirtualMemoryBytes  uint64                            `json:"process_virtual_memory_bytes,omitempty"`
	ProcessResidentMemoryBytes uint64                            `json:"process_resident_memory_bytes,omitempty"`
	ProcessPeakResidentBytes   uint64                            `json:"process_peak_resident_bytes,omitempty"`
	TokenPhases                []driverProfileNativeEventSummary `json:"token_phase_summary,omitempty"`
	NativeEvents               []driverProfileNativeEventSummary `json:"native_events,omitempty"`
}

type driverProfileSafetyLimits struct {
	MaxActiveMemoryBytes          uint64 `json:"max_active_memory_bytes,omitempty"`
	MaxProcessVirtualMemoryBytes  uint64 `json:"max_process_virtual_memory_bytes,omitempty"`
	MaxProcessResidentMemoryBytes uint64 `json:"max_process_resident_memory_bytes,omitempty"`
	RepeatedTokenLoopLimit        int    `json:"repeated_token_loop_limit,omitempty"`
	RepeatedLineLoopLimit         int    `json:"repeated_line_loop_limit,omitempty"`
	RepeatedSentenceLoopLimit     int    `json:"repeated_sentence_loop_limit,omitempty"`
}

type driverProfileNativeEventSummary struct {
	Name            string        `json:"name"`
	Count           int           `json:"count"`
	Duration        time.Duration `json:"duration"`
	AverageDuration time.Duration `json:"average_duration,omitempty"`
}

type driverProfileEnergy struct {
	Method                    string        `json:"method"`
	PowerWatts                float64       `json:"power_watts"`
	TotalJoules               float64       `json:"total_joules,omitempty"`
	JoulesPerVisibleToken     float64       `json:"joules_per_visible_token,omitempty"`
	PromptSetupDuration       time.Duration `json:"prompt_setup_duration,omitempty"`
	PromptSetupJoules         float64       `json:"prompt_setup_joules,omitempty"`
	ReplayPromptSetupDuration time.Duration `json:"replay_prompt_setup_duration,omitempty"`
	ReplayPromptSetupJoules   float64       `json:"replay_prompt_setup_joules,omitempty"`
	PromptSetupSavedDuration  time.Duration `json:"prompt_setup_saved_duration,omitempty"`
	PromptSetupSavedJoules    float64       `json:"prompt_setup_saved_joules,omitempty"`
	PromptSetupSpeedup        float64       `json:"prompt_setup_speedup,omitempty"`
}

type chapterProfileOptions struct {
	ContextPrompt    string    `json:"context_prompt,omitempty"`
	Premise          string    `json:"premise,omitempty"`
	PromptChunkBytes int       `json:"prompt_chunk_bytes,omitempty"`
	PromptRepeat     int       `json:"prompt_repeat,omitempty"`
	Chapters         int       `json:"chapters,omitempty"`
	ChapterMaxTokens int       `json:"chapter_max_tokens,omitempty"`
	ChapterMinTokens int       `json:"chapter_min_tokens,omitempty"`
	OutputPath       string    `json:"output_path,omitempty"`
	OutputWriter     io.Writer `json:"-"`
	IncludeOutput    bool      `json:"include_output,omitempty"`
	ChatTemplate     string    `json:"chat_template,omitempty"`
	EnableThinking   bool      `json:"enable_thinking,omitempty"`
	Temperature      float64   `json:"temperature,omitempty"`
	TopP             float64   `json:"top_p,omitempty"`
	TopK             int       `json:"top_k,omitempty"`
	RepeatPenalty    float64   `json:"repeat_penalty,omitempty"`
	SafetyLimits     chapterProfileSafetyLimits
}

type chapterProfileReport struct {
	Version                int                        `json:"version"`
	ModelPath              string                     `json:"model_path"`
	LoadDuration           time.Duration              `json:"load_duration,omitempty"`
	ContextBytes           int                        `json:"context_bytes"`
	PremiseBytes           int                        `json:"premise_bytes,omitempty"`
	PromptChunkBytes       int                        `json:"prompt_chunk_bytes,omitempty"`
	PromptRepeat           int                        `json:"prompt_repeat,omitempty"`
	ChaptersRequested      int                        `json:"chapters_requested"`
	ChapterMaxTokens       int                        `json:"chapter_max_tokens"`
	ChapterMinTokens       int                        `json:"chapter_min_tokens,omitempty"`
	OutputPath             string                     `json:"output_path,omitempty"`
	ChatTemplate           string                     `json:"chat_template,omitempty"`
	EnableThinking         bool                       `json:"enable_thinking,omitempty"`
	Temperature            float64                    `json:"temperature,omitempty"`
	TopP                   float64                    `json:"top_p,omitempty"`
	TopK                   int                        `json:"top_k,omitempty"`
	RepeatPenalty          float64                    `json:"repeat_penalty,omitempty"`
	SafetyLimits           chapterProfileSafetyLimits `json:"safety_limits,omitempty"`
	RuntimeGates           map[string]string          `json:"runtime_gates,omitempty"`
	Load                   *tuneProfileLoadSettings   `json:"load,omitempty"`
	InitialPrefillDuration time.Duration              `json:"initial_prefill_duration,omitempty"`
	Turns                  []chapterProfileTurn       `json:"turns,omitempty"`
	Summary                chapterProfileSummary      `json:"summary"`
	EstimatedEnergy        *chapterProfileEnergy      `json:"estimated_energy,omitempty"`
	Error                  string                     `json:"error,omitempty"`
}

type chapterProfileTurn struct {
	Index                  int           `json:"index"`
	PromptBytes            int           `json:"prompt_bytes,omitempty"`
	AppendDuration         time.Duration `json:"append_duration,omitempty"`
	Duration               time.Duration `json:"duration,omitempty"`
	FirstTokenDuration     time.Duration `json:"first_token_duration,omitempty"`
	StreamDuration         time.Duration `json:"stream_duration,omitempty"`
	DriverOverheadDuration time.Duration `json:"driver_overhead_duration,omitempty"`
	VisibleTokens          int           `json:"visible_tokens,omitempty"`
	StopTokenIDs           []int32       `json:"stop_token_ids,omitempty"`
	SuppressTokenIDs       []int32       `json:"suppress_token_ids,omitempty"`
	FirstLogits            *probe.Logits `json:"first_logits,omitempty"`
	SampledTokenIDs        []int32       `json:"sampled_token_ids,omitempty"`
	SampledTokenTexts      []string      `json:"sampled_token_texts,omitempty"`
	Output                 string        `json:"output,omitempty"`
	Metrics                mlx.Metrics   `json:"metrics"`
	Error                  string        `json:"error,omitempty"`
}

type chapterProfileSummary struct {
	SuccessfulTurns            int           `json:"successful_turns"`
	FailedTurns                int           `json:"failed_turns,omitempty"`
	GeneratedTokens            int           `json:"generated_tokens,omitempty"`
	VisibleTokens              int           `json:"visible_tokens,omitempty"`
	TotalDuration              time.Duration `json:"total_duration,omitempty"`
	AppendDuration             time.Duration `json:"append_duration,omitempty"`
	AppendAvgDuration          time.Duration `json:"append_duration_average,omitempty"`
	PrefillTokensPerSecAverage float64       `json:"prefill_tokens_per_sec_average,omitempty"`
	DecodeTokensPerSecAverage  float64       `json:"decode_tokens_per_sec_average,omitempty"`
	PeakMemoryBytes            uint64        `json:"peak_memory_bytes,omitempty"`
	ActiveMemoryBytes          uint64        `json:"active_memory_bytes,omitempty"`
	CacheMemoryBytes           uint64        `json:"cache_memory_bytes,omitempty"`
	ProcessVirtualMemoryBytes  uint64        `json:"process_virtual_memory_bytes,omitempty"`
	ProcessResidentMemoryBytes uint64        `json:"process_resident_memory_bytes,omitempty"`
}

type chapterProfileSafetyLimits struct {
	MaxActiveMemoryBytes          uint64 `json:"max_active_memory_bytes,omitempty"`
	MaxProcessVirtualMemoryBytes  uint64 `json:"max_process_virtual_memory_bytes,omitempty"`
	MaxProcessResidentMemoryBytes uint64 `json:"max_process_resident_memory_bytes,omitempty"`
	SuppressedTokenLoopLimit      int    `json:"suppressed_token_loop_limit,omitempty"`
	RepeatedLineLoopLimit         int    `json:"repeated_line_loop_limit,omitempty"`
	RepeatedSentenceLoopLimit     int    `json:"repeated_sentence_loop_limit,omitempty"`
}

const (
	driverProfileDefaultRepeatedTokenLoopLimit    = 256
	chapterProfileDefaultSuppressedTokenLoopLimit = 8
	chapterProfileDefaultMinTokens                = 1024
	profileDefaultRepeatedLineLoopLimit           = 24
	profileDefaultRepeatedSentenceLoopLimit       = 4
	profileFragmentedSentenceMinCount             = 12
	profileFragmentedSentenceRatio                = 0.35
	chapterProfileEndMarker                       = "[[END_CHAPTER]]"
)

type chapterProfileEnergy struct {
	Method         string  `json:"method"`
	PowerWatts     float64 `json:"power_watts"`
	TotalJoules    float64 `json:"total_joules,omitempty"`
	JoulesPerToken float64 `json:"joules_per_visible_token,omitempty"`
}

type stateRampProfileOptions struct {
	Prompt                    string                    `json:"prompt,omitempty"`
	AppendPrompt              string                    `json:"append_prompt,omitempty"`
	AppendTurnDelimiter       string                    `json:"append_turn_delimiter,omitempty"`
	ChatTemplate              string                    `json:"chat_template,omitempty"`
	EnableThinking            bool                      `json:"enable_thinking,omitempty"`
	StartTokens               int                       `json:"start_tokens,omitempty"`
	TargetTokens              int                       `json:"target_tokens,omitempty"`
	CompactionThresholdTokens int                       `json:"compaction_threshold_tokens,omitempty"`
	CompactionTailTokens      int                       `json:"compaction_tail_tokens,omitempty"`
	AppendTokens              int                       `json:"append_tokens,omitempty"`
	TurnMaxTokens             int                       `json:"turn_max_tokens,omitempty"`
	TurnMinTokens             int                       `json:"turn_min_tokens,omitempty"`
	TurnMinTokensPolicy       string                    `json:"turn_min_tokens_policy,omitempty"`
	Turns                     int                       `json:"turns,omitempty"`
	Temperature               float64                   `json:"temperature,omitempty"`
	TopP                      float64                   `json:"top_p,omitempty"`
	TopK                      int                       `json:"top_k,omitempty"`
	RepeatPenalty             float64                   `json:"repeat_penalty,omitempty"`
	SuppressEOS               bool                      `json:"suppress_eos,omitempty"`
	IncludeOutput             bool                      `json:"include_output,omitempty"`
	FoldOnExhaustion          bool                      `json:"fold_on_exhaustion,omitempty"`
	FoldStorePath             string                    `json:"fold_store_path,omitempty"`
	FoldSummary               string                    `json:"-"`
	FoldRecentTail            string                    `json:"-"`
	FoldPrefillChunkBytes     int                       `json:"fold_prefill_chunk_bytes,omitempty"`
	FoldContinuePrompt        string                    `json:"-"`
	FoldContinueMaxTokens     int                       `json:"fold_continue_max_tokens,omitempty"`
	SafetyLimits              driverProfileSafetyLimits `json:"safety_limits,omitempty"`
}

type stateRampProfileReport struct {
	Version                   int                       `json:"version"`
	ModelPath                 string                    `json:"model_path"`
	LoadDuration              time.Duration             `json:"load_duration,omitempty"`
	PromptBytes               int                       `json:"prompt_bytes"`
	AppendPromptBytes         int                       `json:"append_prompt_bytes,omitempty"`
	ChatTemplate              string                    `json:"chat_template,omitempty"`
	EnableThinking            bool                      `json:"enable_thinking,omitempty"`
	SourceTokens              int                       `json:"source_tokens,omitempty"`
	AppendSourceTokens        int                       `json:"append_source_tokens,omitempty"`
	AppendTurnSections        int                       `json:"append_turn_sections,omitempty"`
	StartTokens               int                       `json:"start_tokens"`
	TargetTokens              int                       `json:"target_tokens"`
	CompactionThresholdTokens int                       `json:"compaction_threshold_tokens,omitempty"`
	CompactionTailTokens      int                       `json:"compaction_tail_tokens,omitempty"`
	AppendTokens              int                       `json:"append_tokens"`
	TurnMaxTokens             int                       `json:"turn_max_tokens"`
	TurnMinTokens             int                       `json:"turn_min_tokens,omitempty"`
	TurnMinTokensPolicy       string                    `json:"turn_min_tokens_policy,omitempty"`
	RequestedTurns            int                       `json:"requested_turns,omitempty"`
	Temperature               float64                   `json:"temperature,omitempty"`
	TopP                      float64                   `json:"top_p,omitempty"`
	TopK                      int                       `json:"top_k,omitempty"`
	RepeatPenalty             float64                   `json:"repeat_penalty,omitempty"`
	SuppressEOS               bool                      `json:"suppress_eos,omitempty"`
	IncludeOutput             bool                      `json:"include_output,omitempty"`
	FoldOnExhaustion          bool                      `json:"fold_on_exhaustion,omitempty"`
	FoldStorePath             string                    `json:"fold_store_path,omitempty"`
	FoldSummaryBytes          int                       `json:"fold_summary_bytes,omitempty"`
	FoldRecentTailBytes       int                       `json:"fold_recent_tail_bytes,omitempty"`
	FoldPrefillChunkBytes     int                       `json:"fold_prefill_chunk_bytes,omitempty"`
	FoldContinueMaxTokens     int                       `json:"fold_continue_max_tokens,omitempty"`
	SafetyLimits              driverProfileSafetyLimits `json:"safety_limits,omitempty"`
	RuntimeGates              map[string]string         `json:"runtime_gates,omitempty"`
	Load                      *tuneProfileLoadSettings  `json:"load,omitempty"`
	InitialPrefillDuration    time.Duration             `json:"initial_prefill_duration,omitempty"`
	InitialPrefillTokens      int                       `json:"initial_prefill_tokens,omitempty"`
	Turns                     []stateRampProfileTurn    `json:"turns,omitempty"`
	Summary                   stateRampProfileSummary   `json:"summary"`
	Fold                      *stateRampProfileFold     `json:"fold,omitempty"`
	EstimatedEnergy           *stateRampProfileEnergy   `json:"estimated_energy,omitempty"`
	Error                     string                    `json:"error,omitempty"`
}

type stateRampProfileTurn struct {
	Index                  int           `json:"index"`
	TokensBeforeAppend     int           `json:"tokens_before_append,omitempty"`
	AppendedTokens         int           `json:"appended_tokens,omitempty"`
	TokensAfterAppend      int           `json:"tokens_after_append,omitempty"`
	TokensAfterGenerate    int           `json:"tokens_after_generate,omitempty"`
	TurnCloseTokens        int           `json:"turn_close_tokens,omitempty"`
	AppendDuration         time.Duration `json:"append_duration,omitempty"`
	Duration               time.Duration `json:"duration,omitempty"`
	FirstTokenDuration     time.Duration `json:"first_token_duration,omitempty"`
	StreamDuration         time.Duration `json:"stream_duration,omitempty"`
	DriverOverheadDuration time.Duration `json:"driver_overhead_duration,omitempty"`
	VisibleTokens          int           `json:"visible_tokens,omitempty"`
	BelowMinTokens         bool          `json:"below_min_tokens,omitempty"`
	SampledTokenIDs        []int32       `json:"sampled_token_ids,omitempty"`
	SampledTokenTexts      []string      `json:"sampled_token_texts,omitempty"`
	Output                 string        `json:"output,omitempty"`
	Metrics                mlx.Metrics   `json:"metrics"`
	Error                  string        `json:"error,omitempty"`
}

type stateRampProfileSummary struct {
	SuccessfulTurns            int           `json:"successful_turns"`
	FailedTurns                int           `json:"failed_turns,omitempty"`
	InitialPrefillTokens       int           `json:"initial_prefill_tokens,omitempty"`
	FinalStateTokens           int           `json:"final_state_tokens,omitempty"`
	AppendedTokens             int           `json:"appended_tokens,omitempty"`
	GeneratedTokens            int           `json:"generated_tokens,omitempty"`
	VisibleTokens              int           `json:"visible_tokens,omitempty"`
	TotalDuration              time.Duration `json:"total_duration,omitempty"`
	AppendDuration             time.Duration `json:"append_duration,omitempty"`
	AppendAvgDuration          time.Duration `json:"append_duration_average,omitempty"`
	InitialPrefillTokensPerSec float64       `json:"initial_prefill_tokens_per_sec,omitempty"`
	AppendTokensPerSecAverage  float64       `json:"append_tokens_per_sec_average,omitempty"`
	DecodeTokensPerSecAverage  float64       `json:"decode_tokens_per_sec_average,omitempty"`
	EffectiveTurnTokensPerSec  float64       `json:"effective_turn_tokens_per_sec_average,omitempty"`
	PeakMemoryBytes            uint64        `json:"peak_memory_bytes,omitempty"`
	ActiveMemoryBytes          uint64        `json:"active_memory_bytes,omitempty"`
	CacheMemoryBytes           uint64        `json:"cache_memory_bytes,omitempty"`
	ProcessVirtualMemoryBytes  uint64        `json:"process_virtual_memory_bytes,omitempty"`
	ProcessResidentMemoryBytes uint64        `json:"process_resident_memory_bytes,omitempty"`
	ProcessPeakResidentBytes   uint64        `json:"process_peak_resident_bytes,omitempty"`
	ContextExhausted           bool          `json:"context_exhausted,omitempty"`
	FoldedStateRequired        bool          `json:"folded_state_required,omitempty"`
	CompactionThresholdTokens  int           `json:"compaction_threshold_tokens,omitempty"`
	CompactionTailTokens       int           `json:"compaction_tail_tokens,omitempty"`
	CompactionReason           string        `json:"compaction_reason,omitempty"`
}

type stateRampProfileEnergy struct {
	Method                         string  `json:"method"`
	PowerWatts                     float64 `json:"power_watts"`
	TotalJoules                    float64 `json:"total_joules,omitempty"`
	JoulesPerVisibleToken          float64 `json:"joules_per_visible_token,omitempty"`
	AppendJoules                   float64 `json:"append_joules,omitempty"`
	FoldLifecycleJoules            float64 `json:"fold_lifecycle_joules,omitempty"`
	TotalWithFoldLifecycleJoules   float64 `json:"total_with_fold_lifecycle_joules,omitempty"`
	FoldContinueJoulesPerToken     float64 `json:"fold_continue_joules_per_visible_token,omitempty"`
	FoldContinueEffectiveTokensSec float64 `json:"fold_continue_effective_tokens_per_sec,omitempty"`
}

type stateRampProfileFold struct {
	Attempted           bool                  `json:"attempted"`
	StorePath           string                `json:"store_path,omitempty"`
	SummaryBytes        int                   `json:"summary_bytes,omitempty"`
	RecentTailBytes     int                   `json:"recent_tail_bytes,omitempty"`
	FoldedPromptBytes   int                   `json:"folded_prompt_bytes,omitempty"`
	Duration            time.Duration         `json:"duration,omitempty"`
	WakeDuration        time.Duration         `json:"wake_duration,omitempty"`
	Checkpoint          *agent.SleepReport    `json:"checkpoint,omitempty"`
	Folded              *agent.SleepReport    `json:"folded,omitempty"`
	Wake                *agent.WakeReport     `json:"wake,omitempty"`
	ContinuePromptBytes int                   `json:"continue_prompt_bytes,omitempty"`
	ContinueTurn        *stateRampProfileTurn `json:"continue_turn,omitempty"`
	SkippedReason       string                `json:"skipped_reason,omitempty"`
	Error               string                `json:"error,omitempty"`
}

type driverProfileModel interface {
	GenerateStream(context.Context, string, ...mlx.GenerateOption) <-chan mlx.Token
	GenerateChunksStream(context.Context, iter.Seq[string], ...mlx.GenerateOption) <-chan mlx.Token
	ChatChunksStream(context.Context, []inference.Message, int, ...mlx.GenerateOption) <-chan mlx.Token
	ChatStream(context.Context, []inference.Message, ...mlx.GenerateOption) <-chan mlx.Token
	Metrics() mlx.Metrics
	Err() error
}

func runDiscoverCommand(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet(cliCommandName("discover"), flag.ContinueOnError)
	fs.SetOutput(stderr)
	jsonOut := fs.Bool("json", false, "print JSON machine discovery report")
	modelDir := fs.String("model-dir", "", "model directory to scan without loading weights")
	includeModels := fs.Bool("include-models", false, "include discovered model packs")
	includeCandidates := fs.Bool("include-candidates", false, "include first-pass tuning candidates for discovered models")
	maxModels := fs.Int("max-models", 0, "maximum discovered models to report")
	probeDevice := fs.Bool("probe-device", false, "probe native Metal device facts")
	workload := fs.String("workload", "", "workload to optimise: chat, coding, long_context, agent_state, throughput, or low_latency")
	fs.Usage = func() {
		core.WriteString(stderr, core.Sprintf("Usage: %s discover [flags]\n", cliName()))
		fs.VisitAll(func(f *flag.Flag) {
			if f.DefValue == "" {
				core.WriteString(stderr, core.Sprintf("  -%s\n\t%s\n", f.Name, f.Usage))
				return
			}
			core.WriteString(stderr, core.Sprintf("  -%s\n\t%s (default %q)\n", f.Name, f.Usage, f.DefValue))
		})
	}
	if err := fs.Parse(args); err != nil {
		if core.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	if fs.NArg() != 0 {
		core.WriteString(stderr, core.Sprintf("%s discover: unexpected positional arguments\n", cliName()))
		fs.Usage()
		return 2
	}
	workloads, err := cliTuningWorkloads(*workload)
	if err != nil {
		core.Print(stderr, "%s discover: %v", cliName(), err)
		return 2
	}
	cfg := mlx.LocalDiscoveryConfig{
		Workloads:         workloads,
		MaxModels:         *maxModels,
		IncludeModels:     *includeModels,
		IncludeCandidates: *includeCandidates,
	}
	if core.Trim(*modelDir) != "" {
		cfg.ModelDirs = []string{*modelDir}
	}
	if *probeDevice {
		cfg.Device = runGetDeviceInfo()
	}
	report, err := runDiscoverLocalRuntime(ctx, cfg)
	if err != nil {
		core.Print(stderr, "%s discover: %v", cliName(), err)
		return 1
	}
	if *jsonOut {
		data := core.JSONMarshalIndent(report, "", "  ")
		if !data.OK {
			core.Print(stderr, "%s discover: marshal report failed", cliName())
			return 1
		}
		core.WriteString(stdout, string(data.Value.([]byte)))
		core.WriteString(stdout, "\n")
		return 0
	}
	printDiscoverySummary(stdout, report)
	return 0
}

func printDiscoverySummary(stdout io.Writer, report inference.MachineDiscoveryReport) {
	core.WriteString(stdout, core.Sprintf("runtime discovery: %s\n", report.Runtime.Backend))
	core.WriteString(stdout, core.Sprintf("  available: %t, device: %s\n", report.Available, report.Device.Architecture))
	core.WriteString(stdout, core.Sprintf("  memory: %d bytes, working set: %d bytes\n", report.Device.MemorySize, report.Device.MaxRecommendedWorkingSetSize))
	core.WriteString(stdout, core.Sprintf("  capabilities: %d, cache modes: %d\n", len(report.Capabilities), len(report.CacheModes)))
	core.WriteString(stdout, core.Sprintf("  models: %d, candidates: %d\n", len(report.Models), len(report.Candidates)))
}

func runDriverProfileCommand(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet(cliCommandName("driver-profile"), flag.ContinueOnError)
	fs.SetOutput(stderr)
	jsonOut := fs.Bool("json", false, "print JSON driver profile")
	reportFile := fs.String("report-file", "", "write JSON driver profile to a file")
	profilePath := fs.String("profile", "", "saved tuning profile to apply before loading the model")
	prompt := fs.String("prompt", "Answer in one short sentence: why does retained model state matter?", "prompt/question to run")
	promptFile := fs.String("prompt-file", "", "read prompt/question text from a file")
	promptSuffix := fs.String("prompt-suffix", "", "append one final task after any repeated prompt context")
	promptSuffixFile := fs.String("prompt-suffix-file", "", "read final prompt/task suffix text from a file")
	promptChunkBytes := fs.Int("prompt-chunk-bytes", 0, "split prompt or chat message text into bounded byte chunks before tokenisation")
	promptRepeat := fs.Int("prompt-repeat", 1, "repeat the resolved prompt N times before tokenisation")
	maxTokens := fs.Int("max-tokens", 32, "generated tokens per profiling run")
	runs := fs.Int("runs", 1, "profiling runs to execute")
	includeOutput := fs.Bool("include-output", true, "include generated text in the report")
	chat := fs.Bool("chat", true, "run the prompt through the model chat template")
	traceTokenPhases := fs.Bool("trace-token-phases", false, "include per-token native decode phase timings")
	contextLen := fs.Int("context", 0, "override context length")
	prefillChunkSize := fs.Int("prefill-chunk-size", 0, "override long-prompt prefill chunk size in tokens")
	cacheMode := fs.String("cache-mode", "", "override KV cache mode: fp16, q8, k-q8-v-q4, or paged")
	device := fs.String("device", "", "execution device: gpu or cpu")
	estimatePowerWatts := fs.Float64("estimate-power-watts", 0, "record an estimated average active power draw in watts and derive joule deltas")
	fastGemma4Lane := fs.Bool("fast-gemma4-lane", true, "enable the accepted Gemma 4 fast runtime gates by default; set false for baseline diagnostics")
	expertIDMatVec := fs.Bool("expert-id-matvec", false, "enable the opt-in Gemma 4 expert-ID matvec MoE path")
	expertIDFusedActivation := fs.Bool("expert-id-fused-activation", false, "enable fused activation inside the opt-in expert-ID matvec path")
	sortedExpertPrefill := fs.Bool("sorted-expert-prefill", false, "enable the opt-in Gemma 4 sorted expert prefill MoE path")
	pagedDecodeFastConcat := fs.Bool("paged-decode-fast-concat", false, "enable the opt-in Gemma 4 fast-SDPA concat path for multi-page decode")
	nativePagedAttention := fs.Bool("native-paged-attention", false, "enable the opt-in native C++ paged attention reduction path")
	nativeMLPMatVec := fs.Bool("native-mlp-matvec", false, "enable the opt-in native q4/q8 MLP matvec path")
	nativeLinearMatVec := fs.Bool("native-linear-matvec", false, "enable the opt-in native q4/q8 single-token linear matvec path")
	nativeGemma4FFNResidual := fs.Bool("native-gemma4-ffn-residual", false, "enable the opt-in native Gemma 4 MoE FFN residual path")
	nativeGemma4RouterMatVec := fs.Bool("native-gemma4-router-matvec", false, "enable the opt-in native Gemma 4 router quantized matvec path")
	nativeGemma4RouterTopK := fs.Bool("native-gemma4-router-topk", false, "enable the opt-in native Gemma 4 router top-k path")
	nativeGemma4FixedOwnerAttention := fs.Bool("native-gemma4-fixed-owner-attention", false, "enable the opt-in native Gemma 4 fixed-cache owner attention path")
	nativeGemma4FixedOwnerAttentionResidual := fs.Bool("native-gemma4-fixed-owner-attention-residual", false, "enable the opt-in native Gemma 4 fixed-cache owner attention plus residual path")
	nativeGemma4AttentionOMatVec := fs.Bool("native-gemma4-attention-o-matvec", false, "enable the opt-in native Gemma 4 attention output matvec path")
	nativeGemma4ResidualNorm := fs.Bool("native-gemma4-residual-norm", false, "enable the opt-in native Gemma 4 attention residual norm path")
	nativeGemma4Layer := fs.Bool("native-gemma4-layer", false, "enable the opt-in native Gemma 4 one-token decode layer path")
	nativeGemma4MoELayer := fs.Bool("native-gemma4-moe-layer", false, "enable the opt-in native Gemma 4 MoE layer path")
	nativeGemma4ModelGreedy := fs.Bool("native-gemma4-model-greedy", false, "enable the opt-in native Gemma 4 fixed-cache model-level greedy decode path")
	compiledGemma4Layer := fs.Bool("compiled-gemma4-layer", false, "enable the opt-in compiled Gemma 4 one-token decode layer path")
	fixedGemma4Cache := fs.Bool("fixed-gemma4-cache", false, "enable the opt-in fixed-capacity Gemma 4 cache path with -cache-mode paged")
	fixedGemma4SlidingCacheBound := fs.Bool("fixed-gemma4-sliding-cache-bound", false, "keep Gemma 4 sliding-attention fixed caches at their native window size")
	fixedGemma4SharedMask := fs.Bool("fixed-gemma4-shared-mask", false, "enable the opt-in shared fixed-cache Gemma 4 decode mask")
	directGreedyToken := fs.Bool("direct-greedy-token", false, "enable the opt-in direct greedy token decode path")
	generationStream := fs.Bool("generation-stream", false, "enable the opt-in dedicated MLX stream for generation")
	generationClearCache := fs.Bool("generation-clear-cache", false, "clear the MLX allocator cache after prefill chunks and periodically during decode")
	maxActiveMemoryBytes := fs.Uint64("max-active-memory-bytes", 0, "abort a run if MLX active memory exceeds this many bytes; 0 derives from the resolved memory limit")
	maxProcessVirtualMemoryBytes := fs.Uint64("max-process-virtual-memory-bytes", 0, "abort a run if process virtual memory exceeds this many bytes; 0 records process virtual memory without a hard cap")
	maxProcessResidentMemoryBytes := fs.Uint64("max-process-resident-memory-bytes", 0, "abort a run if process resident memory exceeds this many bytes; 0 derives from the resolved memory limit")
	repeatedTokenLoopLimit := fs.Int("repeated-token-loop-limit", driverProfileDefaultRepeatedTokenLoopLimit, "abort when this many consecutive sampled tokens have the same token id")
	repeatedLineLoopLimit := fs.Int("repeated-line-loop-limit", profileDefaultRepeatedLineLoopLimit, "abort when this many consecutive visible non-empty lines repeat")
	repeatedSentenceLoopLimit := fs.Int("repeated-sentence-loop-limit", profileDefaultRepeatedSentenceLoopLimit, "abort when the same visible sentence repeats this many times in one output")
	fs.Usage = func() {
		core.WriteString(stderr, core.Sprintf("Usage: %s driver-profile [flags] [model-path]\n", cliName()))
		fs.VisitAll(func(f *flag.Flag) {
			if f.DefValue == "" {
				core.WriteString(stderr, core.Sprintf("  -%s\n\t%s\n", f.Name, f.Usage))
				return
			}
			core.WriteString(stderr, core.Sprintf("  -%s\n\t%s (default %q)\n", f.Name, f.Usage, f.DefValue))
		})
	}
	if err := fs.Parse(args); err != nil {
		if core.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	visitedFlags := driverProfileVisitedFlags(fs)
	if driverProfileFastGemma4LaneEnabled(*fastGemma4Lane, visitedFlags, *profilePath) {
		for _, restore := range applyGemma4FastLaneDefaults(
			visitedFlags,
			contextLen,
			cacheMode,
			prefillChunkSize,
			promptChunkBytes,
			mlx.ProductionLaneContextLength,
		) {
			defer restore()
		}
	}
	if fs.NArg() > 1 || (fs.NArg() == 0 && core.Trim(*profilePath) == "") {
		core.WriteString(stderr, core.Sprintf("%s driver-profile: expected one model path or -profile\n", cliName()))
		fs.Usage()
		return 2
	}
	if core.Trim(*promptFile) != "" {
		read := core.ReadFile(*promptFile)
		if !read.OK {
			core.Print(stderr, "%s driver-profile: prompt file: %v", cliName(), read.Value)
			return 1
		}
		*prompt = string(read.Value.([]byte))
	}
	if *promptRepeat < 1 {
		core.WriteString(stderr, core.Sprintf("%s driver-profile: prompt repeat must be >= 1\n", cliName()))
		return 2
	}
	if core.Trim(*promptSuffixFile) != "" {
		read := core.ReadFile(*promptSuffixFile)
		if !read.OK {
			core.Print(stderr, "%s driver-profile: prompt suffix file: %v", cliName(), read.Value)
			return 1
		}
		*promptSuffix = string(read.Value.([]byte))
	}
	*prompt = repeatDriverProfilePrompt(*prompt, *promptRepeat)
	*prompt = appendDriverProfilePromptSuffix(*prompt, *promptSuffix)
	if *expertIDMatVec {
		defer setDriverProfileRuntimeGate("GO_MLX_ENABLE_EXPERT_ID_MATVEC", "1")()
	}
	if *expertIDFusedActivation {
		defer setDriverProfileRuntimeGate("GO_MLX_ENABLE_EXPERT_ID_MATVEC", "1")()
		defer setDriverProfileRuntimeGate("GO_MLX_ENABLE_EXPERT_ID_FUSED_ACTIVATION", "1")()
	}
	if *sortedExpertPrefill {
		defer setDriverProfileRuntimeGate("GO_MLX_ENABLE_SORTED_EXPERT_PREFILL", "1")()
	}
	if *pagedDecodeFastConcat {
		defer setDriverProfileRuntimeGate("GO_MLX_ENABLE_PAGED_DECODE_FAST_CONCAT", "1")()
	}
	if *nativePagedAttention {
		defer setDriverProfileRuntimeGate("GO_MLX_ENABLE_NATIVE_PAGED_ATTENTION", "1")()
	}
	if *nativeMLPMatVec {
		defer setDriverProfileRuntimeGate("GO_MLX_ENABLE_NATIVE_MLP_MATVEC", "1")()
	}
	if *nativeLinearMatVec {
		defer setDriverProfileRuntimeGate("GO_MLX_ENABLE_NATIVE_LINEAR_MATVEC", "1")()
	}
	if *nativeGemma4FFNResidual {
		defer setDriverProfileRuntimeGate("GO_MLX_ENABLE_NATIVE_GEMMA4_FFN_RESIDUAL", "1")()
	}
	if *nativeGemma4RouterMatVec {
		defer setDriverProfileRuntimeGate("GO_MLX_ENABLE_NATIVE_GEMMA4_ROUTER_MATVEC", "1")()
	}
	if *nativeGemma4RouterTopK {
		defer setDriverProfileRuntimeGate("GO_MLX_ENABLE_NATIVE_GEMMA4_ROUTER_TOPK", "1")()
	}
	if *nativeGemma4FixedOwnerAttention {
		defer setDriverProfileRuntimeGate("GO_MLX_ENABLE_NATIVE_GEMMA4_FIXED_OWNER_ATTENTION", "1")()
	}
	if *nativeGemma4FixedOwnerAttentionResidual {
		defer setDriverProfileRuntimeGate("GO_MLX_ENABLE_NATIVE_GEMMA4_FIXED_OWNER_ATTENTION_RESIDUAL", "1")()
	}
	if *nativeGemma4AttentionOMatVec {
		defer setDriverProfileRuntimeGate("GO_MLX_ENABLE_NATIVE_GEMMA4_ATTENTION_O_MATVEC", "1")()
	}
	if *nativeGemma4ResidualNorm {
		defer setDriverProfileRuntimeGate("GO_MLX_ENABLE_NATIVE_GEMMA4_RESIDUAL_NORM", "1")()
	}
	if *nativeGemma4Layer {
		defer setDriverProfileRuntimeGate("GO_MLX_ENABLE_NATIVE_GEMMA4_LAYER", "1")()
	}
	if *nativeGemma4MoELayer {
		defer setDriverProfileRuntimeGate("GO_MLX_ENABLE_NATIVE_GEMMA4_MOE_LAYER", "1")()
	}
	if *nativeGemma4ModelGreedy {
		defer setDriverProfileRuntimeGate("GO_MLX_ENABLE_NATIVE_GEMMA4_MODEL_GREEDY", "1")()
	}
	if *compiledGemma4Layer {
		defer setDriverProfileRuntimeGate("GO_MLX_ENABLE_COMPILED_GEMMA4_LAYER", "1")()
	}
	if *fixedGemma4Cache {
		defer setDriverProfileRuntimeGate("GO_MLX_ENABLE_FIXED_GEMMA4_CACHE", "1")()
	}
	if *fixedGemma4SlidingCacheBound {
		defer setDriverProfileRuntimeGate("GO_MLX_ENABLE_FIXED_GEMMA4_CACHE", "1")()
		defer setDriverProfileRuntimeGate("GO_MLX_ENABLE_FIXED_GEMMA4_SLIDING_CACHE_BOUND", "1")()
	}
	if *fixedGemma4SharedMask {
		defer setDriverProfileRuntimeGate("GO_MLX_ENABLE_FIXED_GEMMA4_SHARED_MASK", "1")()
	}
	if *directGreedyToken {
		defer setDriverProfileRuntimeGate("GO_MLX_ENABLE_DIRECT_GREEDY_TOKEN", "1")()
	}
	if *generationStream {
		defer setDriverProfileRuntimeGate("GO_MLX_ENABLE_GENERATION_STREAM", "1")()
	}
	if *generationClearCache {
		defer setDriverProfileRuntimeGate("GO_MLX_ENABLE_GENERATION_CLEAR_CACHE", "1")()
	}

	modelPath := ""
	loadOptions := []mlx.LoadOption{}
	var loadSettings *tuneProfileLoadSettings
	if core.Trim(*profilePath) != "" {
		report, err := readTuneProfileReport(*profilePath)
		if err != nil {
			core.Print(stderr, "%s driver-profile: profile: %v", cliName(), err)
			return 1
		}
		if report.Profile == nil {
			core.Print(stderr, "%s driver-profile: profile payload missing", cliName())
			return 1
		}
		modelPath = report.ModelPath
		loadOptions = append(loadOptions, mlx.TuningCandidateLoadOptions(report.Profile.Candidate)...)
		load := report.Load
		loadSettings = &load
	}
	if fs.NArg() == 1 {
		modelPath = fs.Arg(0)
	}
	if core.Trim(modelPath) == "" {
		core.WriteString(stderr, core.Sprintf("%s driver-profile: model path missing from profile\n", cliName()))
		fs.Usage()
		return 2
	}
	if *contextLen > 0 {
		loadOptions = append(loadOptions, mlx.WithContextLength(*contextLen))
		if loadSettings == nil {
			loadSettings = &tuneProfileLoadSettings{}
		}
		loadSettings.ContextLength = *contextLen
	}
	if *prefillChunkSize < 0 {
		core.WriteString(stderr, core.Sprintf("%s driver-profile: prefill chunk size must be >= 0\n", cliName()))
		return 2
	}
	if *prefillChunkSize > 0 {
		loadOptions = append(loadOptions, mlx.WithPrefillChunkSize(*prefillChunkSize))
		if loadSettings == nil {
			loadSettings = &tuneProfileLoadSettings{}
		}
		loadSettings.PrefillChunkSize = *prefillChunkSize
	}
	if *estimatePowerWatts < 0 {
		core.WriteString(stderr, core.Sprintf("%s driver-profile: estimated power watts must be >= 0\n", cliName()))
		return 2
	}
	if *promptChunkBytes < 0 {
		core.WriteString(stderr, core.Sprintf("%s driver-profile: prompt chunk bytes must be >= 0\n", cliName()))
		return 2
	}
	if *repeatedTokenLoopLimit < 1 {
		core.WriteString(stderr, core.Sprintf("%s driver-profile: repeated token loop limit must be >= 1\n", cliName()))
		return 2
	}
	if *repeatedLineLoopLimit < 1 {
		core.WriteString(stderr, core.Sprintf("%s driver-profile: repeated line loop limit must be >= 1\n", cliName()))
		return 2
	}
	if *repeatedSentenceLoopLimit < 1 {
		core.WriteString(stderr, core.Sprintf("%s driver-profile: repeated sentence loop limit must be >= 1\n", cliName()))
		return 2
	}
	if core.Trim(*cacheMode) != "" {
		mode := memory.KVCacheMode(core.Trim(*cacheMode))
		switch mode {
		case memory.KVCacheModeFP16, memory.KVCacheModeQ8, memory.KVCacheModeKQ8VQ4, memory.KVCacheModePaged:
		default:
			core.WriteString(stderr, core.Sprintf("%s driver-profile: unsupported cache mode %q\n", cliName(), string(mode)))
			return 2
		}
		loadOptions = append(loadOptions, mlx.WithKVCacheMode(mode))
		if loadSettings == nil {
			loadSettings = &tuneProfileLoadSettings{}
		}
		loadSettings.CacheMode = string(mode)
	}
	if *device != "" {
		loadOptions = append(loadOptions, mlx.WithDevice(*device))
	}
	report, err := runDriverProfileGuarded(ctx, modelPath, loadOptions, driverProfileOptions{
		Prompt:           *prompt,
		PromptSuffix:     *promptSuffix,
		PromptChunkBytes: *promptChunkBytes,
		PromptRepeat:     *promptRepeat,
		MaxTokens:        *maxTokens,
		Runs:             *runs,
		IncludeOutput:    *includeOutput,
		Chat:             *chat,
		TraceTokenPhases: *traceTokenPhases,
		SafetyLimits: driverProfileSafetyLimits{
			MaxActiveMemoryBytes:          *maxActiveMemoryBytes,
			MaxProcessVirtualMemoryBytes:  *maxProcessVirtualMemoryBytes,
			MaxProcessResidentMemoryBytes: *maxProcessResidentMemoryBytes,
			RepeatedTokenLoopLimit:        *repeatedTokenLoopLimit,
			RepeatedLineLoopLimit:         *repeatedLineLoopLimit,
			RepeatedSentenceLoopLimit:     *repeatedSentenceLoopLimit,
		},
	})
	if report != nil && loadSettings != nil {
		report.Load = mergeDriverProfileLoadSettings(loadSettings, report.Load)
	}
	if report != nil && *estimatePowerWatts > 0 {
		report.EstimatedEnergy = estimateDriverProfileEnergy(report, *estimatePowerWatts)
	}
	reportPath := core.Trim(*reportFile)
	if *jsonOut || reportPath != "" {
		if report == nil {
			report = &driverProfileReport{
				Version:           1,
				ModelPath:         modelPath,
				PromptBytes:       len(*prompt),
				PromptSuffixBytes: len(*promptSuffix),
				MaxTokens:         *maxTokens,
				RequestedRuns:     *runs,
				PromptRepeat:      driverProfileReportPromptRepeat(*promptRepeat),
				TraceTokenPhases:  *traceTokenPhases,
				SafetyLimits: driverProfileSafetyLimits{
					MaxActiveMemoryBytes:          *maxActiveMemoryBytes,
					MaxProcessVirtualMemoryBytes:  *maxProcessVirtualMemoryBytes,
					MaxProcessResidentMemoryBytes: *maxProcessResidentMemoryBytes,
					RepeatedTokenLoopLimit:        *repeatedTokenLoopLimit,
					RepeatedLineLoopLimit:         *repeatedLineLoopLimit,
					RepeatedSentenceLoopLimit:     *repeatedSentenceLoopLimit,
				},
			}
		}
		if err != nil && report.Error == "" {
			report.Error = err.Error()
		}
		data := core.JSONMarshalIndent(report, "", "  ")
		if !data.OK {
			core.Print(stderr, "%s driver-profile: marshal report failed", cliName())
			return 1
		}
		if reportPath != "" {
			if writeErr := writeJSONReportFile(reportPath, data.Value.([]byte)); writeErr != nil {
				core.Print(stderr, "%s driver-profile: write report file: %v", cliName(), writeErr)
				return 1
			}
		}
		if *jsonOut {
			core.WriteString(stdout, string(data.Value.([]byte)))
			core.WriteString(stdout, "\n")
		}
		if err != nil {
			return 1
		}
		if *jsonOut {
			return 0
		}
	}
	if err != nil {
		core.Print(stderr, "%s driver-profile: %v", cliName(), err)
		return 1
	}
	printDriverProfileSummary(stdout, report)
	return 0
}

func driverProfileVisitedFlags(fs *flag.FlagSet) map[string]bool {
	visited := map[string]bool{}
	if fs == nil {
		return visited
	}
	fs.Visit(func(f *flag.Flag) {
		if f != nil {
			visited[f.Name] = true
		}
	})
	return visited
}

func driverProfileFastGemma4LaneEnabled(enabled bool, visited map[string]bool, profilePath string) bool {
	if visited != nil && visited["fast-gemma4-lane"] {
		return enabled
	}
	if core.Trim(profilePath) != "" {
		return false
	}
	return enabled
}

func applyGemma4FastLaneDefaults(
	visited map[string]bool,
	contextLen *int,
	cacheMode *string,
	prefillChunkSize *int,
	promptChunkBytes *int,
	defaultContextLength int,
) []func() {
	if visited == nil {
		visited = map[string]bool{}
	}
	if contextLen != nil && !visited["context"] {
		*contextLen = defaultContextLength
	}
	if cacheMode != nil && !visited["cache-mode"] {
		*cacheMode = string(memory.KVCacheModePaged)
	}
	resolvedContext := 0
	if contextLen != nil {
		resolvedContext = *contextLen
	}
	restores := []func(){}
	hyperLongContext := resolvedContext > mlx.ProductionLaneLongFormContextLength
	if resolvedContext > mlx.ProductionLaneContextLength {
		if prefillChunkSize != nil && !visited["prefill-chunk-size"] {
			*prefillChunkSize = mlx.ProductionLaneLongContextPrefillChunkSize
		}
		if promptChunkBytes != nil && !visited["prompt-chunk-bytes"] {
			*promptChunkBytes = mlx.ProductionLaneLongContextPromptChunkBytes
		}
		for _, gate := range mlx.LongContextGemma4FastRuntimeGates() {
			if hyperLongContext && gate == mlx.Gemma4FastRuntimeGateFixedGemma4Sliding {
				continue
			}
			restores = append(restores, setDriverProfileRuntimeGate(gate, "1"))
		}
		if hyperLongContext && driverProfileRuntimeGateValue("GO_MLX_PAGED_KV_PAGE_SIZE") == "" {
			restores = append(restores, setDriverProfileRuntimeGate("GO_MLX_PAGED_KV_PAGE_SIZE", core.Sprintf("%d", mlx.ProductionLaneHyperLongPagedKVPageSize)))
		}
		if hyperLongContext && driverProfileRuntimeGateValue("GO_MLX_KV_CACHE_DTYPE") == "" {
			restores = append(restores, setDriverProfileRuntimeGate("GO_MLX_KV_CACHE_DTYPE", mlx.ProductionLaneHyperLongKVCacheDType))
		}
	}
	for _, gate := range mlx.Gemma4FastRuntimeGatesForContext(resolvedContext) {
		restores = append(restores, setDriverProfileRuntimeGate(gate, "1"))
	}
	return restores
}

var runDriverProfile = defaultRunDriverProfile

func runDriverProfileGuarded(ctx context.Context, modelPath string, loadOptions []mlx.LoadOption, opts driverProfileOptions) (report *driverProfileReport, err error) {
	defer func() {
		if recovered := recover(); recovered != nil {
			err = core.NewError(core.Sprintf("driver-profile panic: %v", recovered))
		}
	}()
	return runDriverProfile(ctx, modelPath, loadOptions, opts)
}

func defaultRunDriverProfile(ctx context.Context, modelPath string, loadOptions []mlx.LoadOption, opts driverProfileOptions) (*driverProfileReport, error) {
	opts = normalizeDriverProfileOptions(opts)
	report := &driverProfileReport{
		Version:           1,
		ModelPath:         modelPath,
		PromptBytes:       len(opts.Prompt),
		PromptSuffixBytes: len(opts.PromptSuffix),
		PromptChunkBytes:  opts.PromptChunkBytes,
		PromptRepeat:      driverProfileReportPromptRepeat(opts.PromptRepeat),
		MaxTokens:         opts.MaxTokens,
		RequestedRuns:     opts.Runs,
		Chat:              opts.Chat,
		TraceTokenPhases:  opts.TraceTokenPhases,
		SafetyLimits:      opts.SafetyLimits,
		RuntimeGates:      driverProfileRuntimeGates(),
	}
	loadStart := time.Now()
	model, err := loadBenchModel(modelPath, loadOptions...)
	report.LoadDuration = bench.NonZeroDuration(time.Since(loadStart))
	if err != nil {
		report.Error = err.Error()
		return report, err
	}
	if model == nil {
		err := core.NewError("mlx: driver profile loaded nil model")
		report.Error = err.Error()
		return report, err
	}
	report.Load = mergeDriverProfileLoadSettings(report.Load, loadSettingsFromModelInfo(model.Info()))
	opts.SafetyLimits = resolveDriverProfileSafetyLimits(opts.SafetyLimits, report.Load)
	report.SafetyLimits = opts.SafetyLimits
	if opts.Chat {
		template := chapterProfileTemplate("", model.Info().Architecture)
		stopTokenIDs, suppressTokenIDs := chapterProfileTemplateTokenControls(template, model.Tokenizer())
		opts.StopTokenIDs = stopTokenIDs
		opts.SuppressTokenIDs = suppressTokenIDs
		report.StopTokenIDs = stopTokenIDs
		report.SuppressTokenIDs = suppressTokenIDs
	}
	defer model.Close()
	if err := driverProfileMetricsSafetyError("load", model.Metrics(), opts.SafetyLimits); err != nil {
		report.Error = err.Error()
		return report, err
	}

	var firstErr error
	for i := 0; i < opts.Runs; i++ {
		run := profileLoadedModelGeneration(ctx, model, i+1, opts)
		if run.Error != "" && firstErr == nil {
			firstErr = core.NewError(run.Error)
		}
		report.Runs = append(report.Runs, run)
		mlx.ClearCache()
	}
	report.Summary = summariseDriverProfileRuns(report.Runs)
	if firstErr != nil {
		report.Error = firstErr.Error()
		return report, firstErr
	}
	return report, nil
}

var driverProfileRuntimeGateOverrides struct {
	sync.RWMutex
	values map[string]string
}

func setDriverProfileRuntimeGate(name, value string) func() {
	restoreMetal := metal.SetRuntimeGate(name, value)
	name = core.Trim(name)
	value = core.Trim(value)
	if name == "" {
		return restoreMetal
	}
	driverProfileRuntimeGateOverrides.Lock()
	if driverProfileRuntimeGateOverrides.values == nil {
		driverProfileRuntimeGateOverrides.values = map[string]string{}
	}
	previous, hadPrevious := driverProfileRuntimeGateOverrides.values[name]
	if value == "" {
		delete(driverProfileRuntimeGateOverrides.values, name)
	} else {
		driverProfileRuntimeGateOverrides.values[name] = value
	}
	driverProfileRuntimeGateOverrides.Unlock()

	return func() {
		restoreMetal()
		driverProfileRuntimeGateOverrides.Lock()
		defer driverProfileRuntimeGateOverrides.Unlock()
		if driverProfileRuntimeGateOverrides.values == nil {
			driverProfileRuntimeGateOverrides.values = map[string]string{}
		}
		if hadPrevious {
			driverProfileRuntimeGateOverrides.values[name] = previous
			return
		}
		delete(driverProfileRuntimeGateOverrides.values, name)
	}
}

func driverProfileRuntimeGateNames() []string {
	return []string{
		"GO_MLX_ENABLE_EXPERT_ID_MATVEC",
		"GO_MLX_ENABLE_EXPERT_ID_FUSED_ACTIVATION",
		"GO_MLX_ENABLE_EXPERT_ID_UNROLLED_Q4",
		"GO_MLX_ENABLE_SORTED_EXPERT_PREFILL",
		"GO_MLX_ENABLE_PAGED_DECODE_FAST_CONCAT",
		"GO_MLX_ENABLE_PAGED_FULL_KV_MATERIALIZE",
		"GO_MLX_ENABLE_NATIVE_PAGED_ATTENTION",
		"GO_MLX_ENABLE_LAST_LOGITS_PREFILL",
		"GO_MLX_ENABLE_NATIVE_GELU_GATE_MUL",
		"GO_MLX_ENABLE_NATIVE_MLP_MATVEC",
		"GO_MLX_ENABLE_NATIVE_LINEAR_MATVEC",
		"GO_MLX_ENABLE_NATIVE_MLP_GELU",
		"GO_MLX_ENABLE_NATIVE_GEMMA4_FFN_RESIDUAL",
		"GO_MLX_ENABLE_NATIVE_GEMMA4_ROUTER_MATVEC",
		"GO_MLX_ENABLE_NATIVE_GEMMA4_ROUTER_TOPK",
		"GO_MLX_ENABLE_NATIVE_GEMMA4_FIXED_OWNER_ATTENTION",
		"GO_MLX_ENABLE_NATIVE_GEMMA4_FIXED_OWNER_ATTENTION_RESIDUAL",
		"GO_MLX_ENABLE_NATIVE_GEMMA4_ATTENTION_O_MATVEC",
		"GO_MLX_ENABLE_NATIVE_GEMMA4_RESIDUAL_NORM",
		"GO_MLX_ENABLE_NATIVE_GEMMA4_LAYER",
		"GO_MLX_ENABLE_NATIVE_GEMMA4_MOE_LAYER",
		"GO_MLX_ENABLE_NATIVE_GEMMA4_MODEL_GREEDY",
		"GO_MLX_ENABLE_COMPILED_GEMMA4_LAYER",
		"GO_MLX_ENABLE_COMPILED_GEMMA4_PER_LAYER_INPUTS",
		"GO_MLX_ENABLE_FIXED_GEMMA4_CACHE",
		"GO_MLX_ENABLE_FIXED_GEMMA4_SLIDING_CACHE_BOUND",
		"GO_MLX_ENABLE_FIXED_GEMMA4_SHARED_MASK",
		"GO_MLX_ENABLE_FIXED_WIDE_SDPA_ATTENTION",
		"GO_MLX_ENABLE_FIXED_WIDE_MATMUL_ATTENTION",
		"GO_MLX_ENABLE_FIXED_ROW_CACHE_UPDATE",
		"GO_MLX_ENABLE_DIRECT_GREEDY_TOKEN",
		"GO_MLX_ENABLE_GENERATION_STREAM",
		"GO_MLX_ENABLE_GENERATION_CLEAR_CACHE",
		"GO_MLX_GENERATION_CLEAR_CACHE_INTERVAL",
		"GO_MLX_ENABLE_ZERO_COPY_PAGED_RESTORE",
		"GO_MLX_KV_CACHE_DTYPE",
		"GO_MLX_ENABLE_ASYNC_DECODE_PREFETCH",
		"GO_MLX_ENABLE_PAGED_KV_PREALLOC",
		"GO_MLX_PAGED_KV_PAGE_SIZE",
	}
}

func driverProfileRuntimeGateValue(name string) string {
	name = core.Trim(name)
	if name == "" {
		return ""
	}
	driverProfileRuntimeGateOverrides.RLock()
	if value, ok := driverProfileRuntimeGateOverrides.values[name]; ok {
		driverProfileRuntimeGateOverrides.RUnlock()
		return core.Trim(value)
	}
	driverProfileRuntimeGateOverrides.RUnlock()
	return core.Trim(core.Env(name))
}

func driverProfileRuntimeGates() map[string]string {
	gates := map[string]string{}
	for _, name := range driverProfileRuntimeGateNames() {
		if value := driverProfileRuntimeGateValue(name); value != "" && value != "0" {
			gates[name] = value
		}
	}
	if len(gates) == 0 {
		return nil
	}
	return gates
}

func loadSettingsFromModelInfo(info mlx.ModelInfo) *tuneProfileLoadSettings {
	settings := &tuneProfileLoadSettings{
		ContextLength:        info.ContextLength,
		ParallelSlots:        info.ParallelSlots,
		PromptCache:          info.PromptCache,
		PromptCacheMinTokens: info.PromptCacheMinTokens,
		CachePolicy:          string(info.CachePolicy),
		CacheMode:            string(info.CacheMode),
		BatchSize:            info.BatchSize,
		PrefillChunkSize:     info.PrefillChunkSize,
		ExpectedQuantization: info.ExpectedQuantization,
		MemoryLimitBytes:     info.MemoryLimitBytes,
		CacheLimitBytes:      info.CacheLimitBytes,
		WiredLimitBytes:      info.WiredLimitBytes,
	}
	if *settings == (tuneProfileLoadSettings{}) {
		return nil
	}
	return settings
}

func mergeDriverProfileLoadSettings(primary, resolved *tuneProfileLoadSettings) *tuneProfileLoadSettings {
	if primary == nil {
		return resolved
	}
	if resolved == nil {
		return primary
	}
	merged := *primary
	if merged.ContextLength == 0 {
		merged.ContextLength = resolved.ContextLength
	}
	if merged.ParallelSlots == 0 {
		merged.ParallelSlots = resolved.ParallelSlots
	}
	if !merged.PromptCache {
		merged.PromptCache = resolved.PromptCache
	}
	if merged.PromptCacheMinTokens == 0 {
		merged.PromptCacheMinTokens = resolved.PromptCacheMinTokens
	}
	if merged.CachePolicy == "" {
		merged.CachePolicy = resolved.CachePolicy
	}
	if merged.CacheMode == "" {
		merged.CacheMode = resolved.CacheMode
	}
	if merged.BatchSize == 0 {
		merged.BatchSize = resolved.BatchSize
	}
	if merged.PrefillChunkSize == 0 {
		merged.PrefillChunkSize = resolved.PrefillChunkSize
	}
	if merged.ExpectedQuantization == 0 {
		merged.ExpectedQuantization = resolved.ExpectedQuantization
	}
	if merged.MemoryLimitBytes == 0 {
		merged.MemoryLimitBytes = resolved.MemoryLimitBytes
	}
	if merged.CacheLimitBytes == 0 {
		merged.CacheLimitBytes = resolved.CacheLimitBytes
	}
	if merged.WiredLimitBytes == 0 {
		merged.WiredLimitBytes = resolved.WiredLimitBytes
	}
	return &merged
}

func normalizeDriverProfileOptions(opts driverProfileOptions) driverProfileOptions {
	opts.Prompt = core.Trim(opts.Prompt)
	if opts.Prompt == "" {
		opts.Prompt = "Answer in one short sentence: why does retained model state matter?"
	}
	if opts.PromptRepeat <= 0 {
		opts.PromptRepeat = 1
	}
	if opts.MaxTokens <= 0 {
		opts.MaxTokens = 1
	}
	if opts.Runs <= 0 {
		opts.Runs = 1
	}
	if opts.SafetyLimits.RepeatedTokenLoopLimit <= 0 {
		opts.SafetyLimits.RepeatedTokenLoopLimit = driverProfileDefaultRepeatedTokenLoopLimit
	}
	if opts.SafetyLimits.RepeatedLineLoopLimit <= 0 {
		opts.SafetyLimits.RepeatedLineLoopLimit = profileDefaultRepeatedLineLoopLimit
	}
	if opts.SafetyLimits.RepeatedSentenceLoopLimit <= 0 {
		opts.SafetyLimits.RepeatedSentenceLoopLimit = profileDefaultRepeatedSentenceLoopLimit
	}
	return opts
}

func resolveDriverProfileSafetyLimits(limits driverProfileSafetyLimits, load *tuneProfileLoadSettings) driverProfileSafetyLimits {
	if limits.RepeatedTokenLoopLimit <= 0 {
		limits.RepeatedTokenLoopLimit = driverProfileDefaultRepeatedTokenLoopLimit
	}
	if limits.RepeatedLineLoopLimit <= 0 {
		limits.RepeatedLineLoopLimit = profileDefaultRepeatedLineLoopLimit
	}
	if limits.RepeatedSentenceLoopLimit <= 0 {
		limits.RepeatedSentenceLoopLimit = profileDefaultRepeatedSentenceLoopLimit
	}
	memoryLimit := profileResolvedMemoryLimit(load)
	if memoryLimit == 0 {
		return limits
	}
	if limits.MaxActiveMemoryBytes == 0 {
		limits.MaxActiveMemoryBytes = profileDefaultActiveMemoryLimit(memoryLimit)
	}
	if limits.MaxProcessResidentMemoryBytes == 0 {
		limits.MaxProcessResidentMemoryBytes = memoryLimit
	}
	return limits
}

func repeatDriverProfilePrompt(prompt string, repeat int) string {
	if repeat <= 1 || prompt == "" {
		return prompt
	}
	builder := core.NewBuilder()
	for i := 0; i < repeat; i++ {
		if i > 0 {
			builder.WriteString("\n\n")
		}
		builder.WriteString(prompt)
	}
	return builder.String()
}

func appendDriverProfilePromptSuffix(prompt, suffix string) string {
	suffix = core.Trim(suffix)
	if suffix == "" {
		return prompt
	}
	prompt = core.Trim(prompt)
	if prompt == "" {
		return suffix
	}
	builder := core.NewBuilder()
	builder.WriteString(prompt)
	builder.WriteString("\n\n")
	builder.WriteString(suffix)
	return builder.String()
}

func driverProfileReportPromptRepeat(repeat int) int {
	if repeat <= 1 {
		return 0
	}
	return repeat
}

func promptByteChunks(prompt string, chunkBytes int) iter.Seq[string] {
	return func(yield func(string) bool) {
		if prompt == "" {
			return
		}
		if chunkBytes <= 0 || len(prompt) <= chunkBytes {
			yield(prompt)
			return
		}
		start := 0
		for index := range prompt {
			if index == start || index-start < chunkBytes {
				continue
			}
			if !yield(prompt[start:index]) {
				return
			}
			start = index
		}
		if start < len(prompt) {
			yield(prompt[start:])
		}
	}
}

func profileLoadedModelGeneration(ctx context.Context, model driverProfileModel, index int, opts driverProfileOptions) driverProfileRun {
	start := time.Now()
	builder := core.NewBuilder()
	firstToken := time.Duration(0)
	visibleTokens := 0
	var tokenStream <-chan mlx.Token
	generateOptions := driverProfileGenerateOptions(opts)
	generationCtx := ctx
	if generationCtx == nil {
		generationCtx = context.Background()
	}
	generationCtx, cancelGeneration := context.WithCancel(generationCtx)
	defer cancelGeneration()
	var probeErr error
	sampledTokenIDs := make([]int32, 0, 32)
	sampledTokenTexts := make([]string, 0, 32)
	repeatedTokenID := int32(0)
	repeatedTokenCount := 0
	var lineErr error
	currentLine := ""
	lastLine := ""
	repeatedLineCount := 0
	if opts.PromptChunkBytes > 0 && opts.Chat {
		tokenStream = model.ChatChunksStream(generationCtx, []inference.Message{{Role: "user", Content: opts.Prompt}}, opts.PromptChunkBytes, generateOptions...)
	} else if opts.PromptChunkBytes > 0 {
		tokenStream = model.GenerateChunksStream(generationCtx, promptByteChunks(opts.Prompt, opts.PromptChunkBytes), generateOptions...)
	} else if opts.Chat {
		tokenStream = model.ChatStream(generationCtx, []inference.Message{{Role: "user", Content: opts.Prompt}}, generateOptions...)
	} else {
		tokenStream = model.GenerateStream(generationCtx, opts.Prompt, generateOptions...)
	}
	for token := range tokenStream {
		if firstToken == 0 {
			firstToken = bench.NonZeroDuration(time.Since(start))
		}
		visibleTokens++
		if len(sampledTokenIDs) < 32 {
			sampledTokenIDs = append(sampledTokenIDs, token.ID)
			sampledTokenTexts = append(sampledTokenTexts, token.Text)
		}
		if probeErr == nil {
			if err := driverProfileMetricsSafetyError(core.Sprintf("run %d stream", index), profileLiveMetrics(), opts.SafetyLimits); err != nil {
				probeErr = err
				cancelGeneration()
				break
			}
			if opts.SafetyLimits.RepeatedTokenLoopLimit <= 0 {
				repeatedTokenCount = 0
			} else {
				if repeatedTokenCount == 0 || token.ID != repeatedTokenID {
					repeatedTokenID = token.ID
					repeatedTokenCount = 1
				} else {
					repeatedTokenCount++
				}
				if repeatedTokenCount >= opts.SafetyLimits.RepeatedTokenLoopLimit {
					probeErr = core.NewError(core.Sprintf("driver-profile: run %d sampled token %d for %d consecutive tokens", index, token.ID, repeatedTokenCount))
					cancelGeneration()
					break
				}
			}
		}
		if opts.IncludeOutput {
			builder.WriteString(token.Text)
		}
		if lineErr == nil {
			if line, count, ok := profileObserveRepeatedLineFragment(token.Text, &currentLine, &lastLine, &repeatedLineCount, opts.SafetyLimits.RepeatedLineLoopLimit); ok {
				lineErr = core.NewError(core.Sprintf("driver-profile: run %d repeated visible line %q for %d consecutive lines", index, line, count))
				cancelGeneration()
				break
			}
		}
	}
	if lineErr == nil {
		if line, count, ok := profileFlushRepeatedLine(&currentLine, &lastLine, &repeatedLineCount, opts.SafetyLimits.RepeatedLineLoopLimit); ok {
			lineErr = core.NewError(core.Sprintf("driver-profile: run %d repeated visible line %q for %d consecutive lines", index, line, count))
		}
	}
	duration := bench.NonZeroDuration(time.Since(start))
	streamDuration := duration
	if firstToken > 0 && duration > firstToken {
		streamDuration = duration - firstToken
	}
	metrics := model.Metrics()
	run := driverProfileRun{
		Index:              index,
		Duration:           duration,
		RestoreDuration:    metrics.PromptCacheRestoreDuration,
		FirstTokenDuration: firstToken,
		StreamDuration:     streamDuration,
		VisibleTokens:      visibleTokens,
		SampledTokenIDs:    sampledTokenIDs,
		SampledTokenTexts:  sampledTokenTexts,
		Metrics:            metrics,
	}
	run.DriverOverheadDuration = driverRunOverhead(run.Duration, run.Metrics)
	if opts.IncludeOutput {
		run.Output = builder.String()
	}
	if probeErr != nil {
		run.Error = probeErr.Error()
		return run
	}
	if lineErr != nil {
		run.Error = lineErr.Error()
		return run
	}
	if err := model.Err(); err != nil {
		run.Error = err.Error()
		return run
	}
	if err := driverProfileRunSafetyError(index, run, opts.SafetyLimits); err != nil {
		run.Error = err.Error()
		return run
	}
	if ctx != nil {
		if err := ctx.Err(); err != nil {
			run.Error = err.Error()
		}
	}
	return run
}

func driverProfileGenerateOptions(opts driverProfileOptions) []mlx.GenerateOption {
	generateOptions := []mlx.GenerateOption{
		mlx.WithMaxTokens(opts.MaxTokens),
		mlx.WithTemperature(0),
	}
	if opts.TraceTokenPhases {
		generateOptions = append(generateOptions, mlx.WithTokenPhaseTrace())
	}
	if len(opts.StopTokenIDs) > 0 {
		generateOptions = append(generateOptions, mlx.WithStopTokens(opts.StopTokenIDs...))
	}
	if len(opts.SuppressTokenIDs) > 0 {
		generateOptions = append(generateOptions, mlx.WithSuppressTokens(opts.SuppressTokenIDs...))
	}
	return generateOptions
}

func driverProfileRunSafetyError(index int, run driverProfileRun, limits driverProfileSafetyLimits) error {
	if err := driverProfileMetricsSafetyError(core.Sprintf("run %d", index), run.Metrics, limits); err != nil {
		return err
	}
	if id, count, ok := driverProfileRepeatedTokenLoop(run.SampledTokenIDs, limits.RepeatedTokenLoopLimit); ok {
		return core.NewError(core.Sprintf("driver-profile: run %d sampled token %d for %d consecutive tokens", index, id, count))
	}
	if line, count, ok := profileRepeatedLineLoop(run.Output, limits.RepeatedLineLoopLimit); ok {
		return core.NewError(core.Sprintf("driver-profile: run %d repeated visible line %q for %d consecutive lines", index, line, count))
	}
	if sentence, count, ok := profileRepeatedSentenceLoop(run.Output, limits.RepeatedSentenceLoopLimit); ok {
		return core.NewError(core.Sprintf("driver-profile: run %d repeated visible sentence %q for %d total occurrences", index, sentence, count))
	}
	if fragments, total, ok := profileFragmentedSentenceOutput(run.Output); ok {
		return core.NewError(core.Sprintf("driver-profile: run %d produced fragmented visible output: %d of %d sentence fragments are too short", index, fragments, total))
	}
	return nil
}

func driverProfileMetricsSafetyError(phase string, metrics mlx.Metrics, limits driverProfileSafetyLimits) error {
	if limits.MaxActiveMemoryBytes > 0 && metrics.ActiveMemoryBytes > limits.MaxActiveMemoryBytes {
		return core.NewError(core.Sprintf("driver-profile: %s exceeded active memory safety limit: %d > %d bytes", phase, metrics.ActiveMemoryBytes, limits.MaxActiveMemoryBytes))
	}
	if limits.MaxProcessVirtualMemoryBytes > 0 && metrics.ProcessVirtualMemoryBytes > limits.MaxProcessVirtualMemoryBytes {
		return core.NewError(core.Sprintf("driver-profile: %s exceeded process virtual memory safety limit: %d > %d bytes", phase, metrics.ProcessVirtualMemoryBytes, limits.MaxProcessVirtualMemoryBytes))
	}
	if limits.MaxProcessResidentMemoryBytes > 0 && metrics.ProcessResidentMemoryBytes > limits.MaxProcessResidentMemoryBytes {
		return core.NewError(core.Sprintf("driver-profile: %s exceeded process resident memory safety limit: %d > %d bytes", phase, metrics.ProcessResidentMemoryBytes, limits.MaxProcessResidentMemoryBytes))
	}
	return nil
}

func driverProfileRepeatedTokenLoop(sampledTokenIDs []int32, limit int) (int32, int, bool) {
	if limit <= 0 || len(sampledTokenIDs) == 0 {
		return 0, 0, false
	}
	last := sampledTokenIDs[0]
	count := 1
	if count >= limit {
		return last, count, true
	}
	for _, id := range sampledTokenIDs[1:] {
		if id != last {
			last = id
			count = 1
		} else {
			count++
		}
		if count >= limit {
			return id, count, true
		}
	}
	return 0, 0, false
}

func profileRepeatedLineLoop(text string, limit int) (string, int, bool) {
	currentLine := ""
	lastLine := ""
	repeatedLineCount := 0
	if line, count, ok := profileObserveRepeatedLineFragment(text, &currentLine, &lastLine, &repeatedLineCount, limit); ok {
		return line, count, ok
	}
	return profileFlushRepeatedLine(&currentLine, &lastLine, &repeatedLineCount, limit)
}

func profileObserveRepeatedLineFragment(fragment string, currentLine, lastLine *string, repeatedLineCount *int, limit int) (string, int, bool) {
	if limit <= 0 || fragment == "" || currentLine == nil || lastLine == nil || repeatedLineCount == nil {
		return "", 0, false
	}
	parts := core.Split(fragment, "\n")
	for i, part := range parts {
		*currentLine += part
		if i == len(parts)-1 {
			continue
		}
		line := core.Trim(*currentLine)
		*currentLine = ""
		if line == "" {
			continue
		}
		if line, count, ok := profileObserveRepeatedLine(line, lastLine, repeatedLineCount, limit); ok {
			return line, count, ok
		}
	}
	return "", 0, false
}

func profileFlushRepeatedLine(currentLine, lastLine *string, repeatedLineCount *int, limit int) (string, int, bool) {
	if limit <= 0 || currentLine == nil || lastLine == nil || repeatedLineCount == nil {
		return "", 0, false
	}
	line := core.Trim(*currentLine)
	*currentLine = ""
	if line == "" {
		return "", 0, false
	}
	return profileObserveRepeatedLine(line, lastLine, repeatedLineCount, limit)
}

func profileObserveRepeatedLine(line string, lastLine *string, repeatedLineCount *int, limit int) (string, int, bool) {
	if limit <= 0 || line == "" || lastLine == nil || repeatedLineCount == nil {
		return "", 0, false
	}
	if line == *lastLine {
		*repeatedLineCount++
	} else {
		*lastLine = line
		*repeatedLineCount = 1
	}
	if *repeatedLineCount >= limit {
		return line, *repeatedLineCount, true
	}
	return "", 0, false
}

func profileRepeatedSentenceLoop(text string, limit int) (string, int, bool) {
	if limit <= 0 || text == "" {
		return "", 0, false
	}
	normalised := core.Replace(text, "!", ".")
	normalised = core.Replace(normalised, "?", ".")
	counts := map[string]int{}
	for _, raw := range core.Split(normalised, ".") {
		sentence := profileNormaliseSentence(raw)
		if len(sentence) < 12 {
			continue
		}
		counts[sentence]++
		if counts[sentence] >= limit {
			return sentence, counts[sentence], true
		}
	}
	return "", 0, false
}

func profileNormaliseSentence(raw string) string {
	text := core.Lower(core.Trim(raw))
	text = core.Replace(text, "\n", " ")
	text = core.Replace(text, "\r", " ")
	text = core.Replace(text, "\t", " ")
	for core.Contains(text, "  ") {
		text = core.Replace(text, "  ", " ")
	}
	return core.Trim(text)
}

func profileFragmentedSentenceOutput(text string) (int, int, bool) {
	if text == "" {
		return 0, 0, false
	}
	normalised := core.Replace(text, "!", ".")
	normalised = core.Replace(normalised, "?", ".")
	fragments := 0
	total := 0
	for _, raw := range core.Split(normalised, ".") {
		sentence := profileNormaliseSentence(raw)
		if sentence == "" {
			continue
		}
		total++
		if len(sentence) < 12 {
			fragments++
		}
	}
	if total < profileFragmentedSentenceMinCount {
		return fragments, total, false
	}
	return fragments, total, float64(fragments)/float64(total) >= profileFragmentedSentenceRatio
}

func driverRunOverhead(duration time.Duration, metrics mlx.Metrics) time.Duration {
	if duration <= 0 || metrics.TotalDuration <= 0 || duration <= metrics.TotalDuration {
		return 0
	}
	return duration - metrics.TotalDuration
}

func summariseDriverProfileRuns(runs []driverProfileRun) driverProfileSummary {
	summary := driverProfileSummary{}
	restoreSamples := 0
	firstTokenSamples := 0
	promptSamples := 0
	promptTokens := 0
	prefillSamples := 0
	decodeSamples := 0
	tokenPhaseIndex := map[string]int{}
	nativeEventIndex := map[string]int{}
	for _, run := range runs {
		accumulateDriverProfileSummaryMemory(&summary, run.Metrics)
		if run.Error != "" {
			summary.FailedRuns++
			continue
		}
		summary.SuccessfulRuns++
		summary.TotalDuration += run.Duration
		summary.VisibleTokens += run.VisibleTokens
		generated := run.Metrics.GeneratedTokens
		if generated == 0 {
			generated = run.VisibleTokens
		}
		summary.GeneratedTokens += generated
		if run.Metrics.PromptTokens > 0 {
			promptSamples++
			promptTokens += run.Metrics.PromptTokens
			if summary.PromptTokensMin == 0 || run.Metrics.PromptTokens < summary.PromptTokensMin {
				summary.PromptTokensMin = run.Metrics.PromptTokens
			}
			if run.Metrics.PromptTokens > summary.PromptTokensMax {
				summary.PromptTokensMax = run.Metrics.PromptTokens
			}
		}
		if run.RestoreDuration > 0 {
			restoreSamples++
			summary.RestoreAvgDuration += run.RestoreDuration
			if summary.RestoreMinDuration == 0 || run.RestoreDuration < summary.RestoreMinDuration {
				summary.RestoreMinDuration = run.RestoreDuration
			}
			if run.RestoreDuration > summary.RestoreMaxDuration {
				summary.RestoreMaxDuration = run.RestoreDuration
			}
		}
		if run.FirstTokenDuration > 0 {
			firstTokenSamples++
			summary.FirstTokenAvgDuration += run.FirstTokenDuration
			if summary.FirstTokenMinDuration == 0 || run.FirstTokenDuration < summary.FirstTokenMinDuration {
				summary.FirstTokenMinDuration = run.FirstTokenDuration
			}
			if run.FirstTokenDuration > summary.FirstTokenMaxDuration {
				summary.FirstTokenMaxDuration = run.FirstTokenDuration
			}
		}
		summary.DriverOverheadAvgDuration += run.DriverOverheadDuration
		if run.Metrics.PrefillTokensPerSec > 0 {
			prefillSamples++
			summary.PrefillTokensPerSecAverage += run.Metrics.PrefillTokensPerSec
		}
		if run.Metrics.DecodeTokensPerSec > 0 {
			decodeSamples++
			summary.DecodeTokensPerSecAverage += run.Metrics.DecodeTokensPerSec
		}
		if run.Metrics.PeakMemoryBytes > summary.PeakMemoryBytes {
			summary.PeakMemoryBytes = run.Metrics.PeakMemoryBytes
		}
		if run.Metrics.ActiveMemoryBytes > summary.ActiveMemoryBytes {
			summary.ActiveMemoryBytes = run.Metrics.ActiveMemoryBytes
		}
		if run.Metrics.CacheMemoryBytes > summary.CacheMemoryBytes {
			summary.CacheMemoryBytes = run.Metrics.CacheMemoryBytes
		}
		if run.Metrics.ProcessVirtualMemoryBytes > summary.ProcessVirtualMemoryBytes {
			summary.ProcessVirtualMemoryBytes = run.Metrics.ProcessVirtualMemoryBytes
		}
		if run.Metrics.ProcessResidentMemoryBytes > summary.ProcessResidentMemoryBytes {
			summary.ProcessResidentMemoryBytes = run.Metrics.ProcessResidentMemoryBytes
		}
		if run.Metrics.ProcessPeakResidentBytes > summary.ProcessPeakResidentBytes {
			summary.ProcessPeakResidentBytes = run.Metrics.ProcessPeakResidentBytes
		}
		for _, phase := range run.Metrics.TokenPhases {
			accumulateDriverProfileTokenPhase(&summary, tokenPhaseIndex, "total", phase.TotalDuration)
			accumulateDriverProfileTokenPhase(&summary, tokenPhaseIndex, "forward", phase.ForwardDuration)
			accumulateDriverProfileTokenPhase(&summary, tokenPhaseIndex, "sample_eval", phase.SampleEvalDuration)
			accumulateDriverProfileTokenPhase(&summary, tokenPhaseIndex, "sample", phase.SampleDuration)
			accumulateDriverProfileTokenPhase(&summary, tokenPhaseIndex, "logits", phase.LogitsDuration)
			accumulateDriverProfileTokenPhase(&summary, tokenPhaseIndex, "token_read", phase.TokenReadDuration)
			accumulateDriverProfileTokenPhase(&summary, tokenPhaseIndex, "decode_text", phase.DecodeTextDuration)
			accumulateDriverProfileTokenPhase(&summary, tokenPhaseIndex, "probe_token", phase.ProbeTokenDuration)
			accumulateDriverProfileTokenPhase(&summary, tokenPhaseIndex, "yield", phase.YieldDuration)
			accumulateDriverProfileTokenPhase(&summary, tokenPhaseIndex, "next_input", phase.NextInputDuration)
			accumulateDriverProfileTokenPhase(&summary, tokenPhaseIndex, "materialize", phase.MaterializeDuration)
			accumulateDriverProfileTokenPhase(&summary, tokenPhaseIndex, "detach", phase.DetachDuration)
			accumulateDriverProfileTokenPhase(&summary, tokenPhaseIndex, "cache_probe", phase.CacheProbeDuration)
			accumulateDriverProfileTokenPhase(&summary, tokenPhaseIndex, "other", phase.OtherDuration)
			for _, event := range phase.NativeEvents {
				if event.Name == "" || event.Duration <= 0 {
					continue
				}
				name := driverProfileNativeEventBucket(event.Name)
				idx, ok := nativeEventIndex[name]
				if !ok {
					summary.NativeEvents = append(summary.NativeEvents, driverProfileNativeEventSummary{Name: name})
					idx = len(summary.NativeEvents) - 1
					nativeEventIndex[name] = idx
				}
				summary.NativeEvents[idx].Count++
				summary.NativeEvents[idx].Duration += event.Duration
			}
		}
	}
	if firstTokenSamples > 0 {
		summary.FirstTokenAvgDuration /= time.Duration(firstTokenSamples)
	}
	if restoreSamples > 0 {
		summary.RestoreAvgDuration /= time.Duration(restoreSamples)
	}
	if promptSamples > 0 {
		summary.PromptTokensAverage = float64(promptTokens) / float64(promptSamples)
	}
	if summary.SuccessfulRuns > 0 {
		summary.DriverOverheadAvgDuration /= time.Duration(summary.SuccessfulRuns)
	}
	if prefillSamples > 0 {
		summary.PrefillTokensPerSecAverage /= float64(prefillSamples)
	}
	if decodeSamples > 0 {
		summary.DecodeTokensPerSecAverage /= float64(decodeSamples)
	}
	for i := range summary.NativeEvents {
		if summary.NativeEvents[i].Count > 0 {
			summary.NativeEvents[i].AverageDuration = summary.NativeEvents[i].Duration / time.Duration(summary.NativeEvents[i].Count)
		}
	}
	for i := range summary.TokenPhases {
		if summary.TokenPhases[i].Count > 0 {
			summary.TokenPhases[i].AverageDuration = summary.TokenPhases[i].Duration / time.Duration(summary.TokenPhases[i].Count)
		}
	}
	sort.SliceStable(summary.TokenPhases, func(i, j int) bool {
		return summary.TokenPhases[i].Duration > summary.TokenPhases[j].Duration
	})
	sort.SliceStable(summary.NativeEvents, func(i, j int) bool {
		return summary.NativeEvents[i].Duration > summary.NativeEvents[j].Duration
	})
	return summary
}

func accumulateDriverProfileTokenPhase(summary *driverProfileSummary, index map[string]int, name string, duration time.Duration) {
	if summary == nil || duration <= 0 || name == "" {
		return
	}
	idx, ok := index[name]
	if !ok {
		summary.TokenPhases = append(summary.TokenPhases, driverProfileNativeEventSummary{Name: name})
		idx = len(summary.TokenPhases) - 1
		index[name] = idx
	}
	summary.TokenPhases[idx].Count++
	summary.TokenPhases[idx].Duration += duration
}

func accumulateDriverProfileSummaryMemory(summary *driverProfileSummary, metrics mlx.Metrics) {
	if summary == nil {
		return
	}
	if metrics.PeakMemoryBytes > summary.PeakMemoryBytes {
		summary.PeakMemoryBytes = metrics.PeakMemoryBytes
	}
	if metrics.ActiveMemoryBytes > summary.ActiveMemoryBytes {
		summary.ActiveMemoryBytes = metrics.ActiveMemoryBytes
	}
	if metrics.CacheMemoryBytes > summary.CacheMemoryBytes {
		summary.CacheMemoryBytes = metrics.CacheMemoryBytes
	}
	if metrics.ProcessVirtualMemoryBytes > summary.ProcessVirtualMemoryBytes {
		summary.ProcessVirtualMemoryBytes = metrics.ProcessVirtualMemoryBytes
	}
	if metrics.ProcessResidentMemoryBytes > summary.ProcessResidentMemoryBytes {
		summary.ProcessResidentMemoryBytes = metrics.ProcessResidentMemoryBytes
	}
	if metrics.ProcessPeakResidentBytes > summary.ProcessPeakResidentBytes {
		summary.ProcessPeakResidentBytes = metrics.ProcessPeakResidentBytes
	}
}

func driverProfileNativeEventBucket(name string) string {
	parts := core.Split(name, ".")
	if len(parts) >= 4 && parts[0] == "gemma4" && parts[1] == "layer" {
		return core.Join(".", parts[3:]...)
	}
	return name
}

func estimateDriverProfileEnergy(report *driverProfileReport, powerWatts float64) *driverProfileEnergy {
	if report == nil || powerWatts <= 0 {
		return nil
	}
	estimate := &driverProfileEnergy{
		Method:     "estimated_wall_clock_seconds_times_average_active_watts",
		PowerWatts: powerWatts,
	}
	if report.Summary.TotalDuration > 0 {
		estimate.TotalJoules = durationJoules(report.Summary.TotalDuration, powerWatts)
	}
	if report.Summary.VisibleTokens > 0 && estimate.TotalJoules > 0 {
		estimate.JoulesPerVisibleToken = estimate.TotalJoules / float64(report.Summary.VisibleTokens)
	}

	setup, replay, speedup := driverProfilePromptSetupDurations(report.Runs)
	estimate.PromptSetupDuration = setup
	estimate.PromptSetupJoules = durationJoules(setup, powerWatts)
	estimate.ReplayPromptSetupDuration = replay
	estimate.ReplayPromptSetupJoules = durationJoules(replay, powerWatts)
	if replay > setup {
		estimate.PromptSetupSavedDuration = replay - setup
		estimate.PromptSetupSavedJoules = durationJoules(estimate.PromptSetupSavedDuration, powerWatts)
	}
	estimate.PromptSetupSpeedup = speedup
	return estimate
}

func driverProfilePromptSetupDurations(runs []driverProfileRun) (time.Duration, time.Duration, float64) {
	successfulRuns := 0
	actual := time.Duration(0)
	coldPromptSetup := time.Duration(0)
	for _, run := range runs {
		if run.Error != "" {
			continue
		}
		successfulRuns++
		if run.Metrics.PrefillDuration <= 0 {
			continue
		}
		actual += run.Metrics.PrefillDuration
		if coldPromptSetup == 0 {
			coldPromptSetup = run.Metrics.PrefillDuration
		}
		if run.Metrics.PromptCacheMisses > 0 || run.Metrics.PromptCacheMissTokens > 0 {
			coldPromptSetup = run.Metrics.PrefillDuration
		}
	}
	replay := time.Duration(0)
	if successfulRuns > 0 && coldPromptSetup > 0 {
		replay = coldPromptSetup * time.Duration(successfulRuns)
	}
	speedup := 0.0
	if actual > 0 && replay > 0 {
		speedup = float64(replay) / float64(actual)
	}
	return actual, replay, speedup
}

func durationJoules(duration time.Duration, powerWatts float64) float64 {
	if duration <= 0 || powerWatts <= 0 {
		return 0
	}
	return duration.Seconds() * powerWatts
}

func printDriverProfileSummary(stdout io.Writer, report *driverProfileReport) {
	if report == nil {
		return
	}
	core.WriteString(stdout, core.Sprintf("driver profile: %s\n", report.ModelPath))
	core.WriteString(stdout, core.Sprintf("  load: %s, runs: %d ok / %d failed\n", report.LoadDuration, report.Summary.SuccessfulRuns, report.Summary.FailedRuns))
	if report.Summary.RestoreAvgDuration > 0 {
		core.WriteString(stdout, core.Sprintf("  restore avg: %s\n", report.Summary.RestoreAvgDuration))
	}
	core.WriteString(stdout, core.Sprintf("  first token avg: %s, decode: %.1f tok/s\n", report.Summary.FirstTokenAvgDuration, report.Summary.DecodeTokensPerSecAverage))
	if report.EstimatedEnergy != nil {
		core.WriteString(stdout, core.Sprintf("  estimated energy: %.1f J at %.1f W", report.EstimatedEnergy.TotalJoules, report.EstimatedEnergy.PowerWatts))
		if report.EstimatedEnergy.PromptSetupSavedJoules > 0 {
			core.WriteString(stdout, core.Sprintf(", setup saved: %.1f J", report.EstimatedEnergy.PromptSetupSavedJoules))
		}
		core.WriteString(stdout, "\n")
	}
	core.WriteString(stdout, core.Sprintf("  generated: %d tokens, peak memory: %d MB, cache memory: %d MB, process virtual: %d MB, process resident: %d MB\n",
		report.Summary.GeneratedTokens,
		report.Summary.PeakMemoryBytes/1024/1024,
		report.Summary.CacheMemoryBytes/1024/1024,
		report.Summary.ProcessVirtualMemoryBytes/1024/1024,
		report.Summary.ProcessResidentMemoryBytes/1024/1024))
}

func runStateRampProfileCommand(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet(cliCommandName("state-ramp-profile"), flag.ContinueOnError)
	fs.SetOutput(stderr)
	jsonOut := fs.Bool("json", false, "print JSON state ramp profile")
	reportFile := fs.String("report-file", "", "write JSON state ramp profile to a file")
	prompt := fs.String("prompt", "Answer in one short sentence: why does retained model state matter?", "source text to repeat into the warm and appended state")
	promptFile := fs.String("prompt-file", "", "read source text from a file")
	appendPrompt := fs.String("append-prompt", "", "source text for appended turn material; defaults to the seed prompt")
	appendFile := fs.String("append-file", "", "read appended turn material from a file")
	appendTurnDelimiter := fs.String("append-turn-delimiter", "", "split appended material into whole turn sections using this delimiter instead of fixed token offsets")
	chatTemplate := fs.String("chat-template", "", "chat template override for retained turns: gemma4, gemma, qwen, llama, or plain")
	enableThinking := fs.Bool("enable-thinking", false, "enable Gemma 4 thinking control token in the retained state ramp prompts")
	startTokens := fs.Int("start-tokens", 30000, "initial warmed-state token target")
	targetTokens := fs.Int("target-tokens", 100000, "final live-state token target")
	compactionThresholdTokens := fs.Int("compaction-threshold-tokens", 0, "live-state token count that marks the context exhausted and requires a folded state; 0 uses target tokens")
	compactionTailTokens := fs.Int("compaction-tail-tokens", 8192, "recent live-state tail token budget to carry into the future folded-state summary")
	appendTokens := fs.Int("append-tokens", 8192, "maximum source tokens to append before each generation turn")
	turnMaxTokens := fs.Int("turn-max-tokens", 1024, "generated tokens per ramp turn")
	turnMinTokens := fs.Int("turn-min-tokens", 0, "minimum visible tokens required for each generated turn; 0 disables the floor")
	turnMinTokensPolicy := fs.String("turn-min-tokens-policy", "fail", "handling for turns below the visible-token floor: fail or mark")
	turns := fs.Int("turns", 0, "maximum ramp turns; 0 runs until target tokens are reached")
	temperature := fs.Float64("temperature", 1.0, "sampling temperature for generated turns")
	topP := fs.Float64("top-p", 0.95, "top-p sampling value for generated turns")
	topK := fs.Int("top-k", 64, "top-k sampling value for generated turns")
	repeatPenalty := fs.Float64("repeat-penalty", 1.0, "repeat penalty for generated turns")
	suppressEOS := fs.Bool("suppress-eos", false, "suppress the tokenizer EOS token during generated turns")
	includeOutput := fs.Bool("include-output", false, "include generated text in the report")
	foldOnExhaustion := fs.Bool("fold-on-exhaustion", false, "checkpoint, fold, wake, and continue from a fresh state when the context reaches the compaction threshold")
	foldStorePath := fs.String("fold-store", "", "append-only state store path for folded-state checkpoint artefacts")
	foldSummary := fs.String("fold-summary", "", "summary text to seed the folded state; empty uses a benchmark lifecycle summary")
	foldSummaryFile := fs.String("fold-summary-file", "", "read folded-state summary text from a file")
	foldRecentTail := fs.String("fold-tail", "", "recent tail text to seed the folded state")
	foldRecentTailFile := fs.String("fold-tail-file", "", "read folded-state recent tail text from a file")
	foldPrefillChunkBytes := fs.Int("fold-prefill-chunk-bytes", 0, "byte chunk size for folded-state prefill; 0 uses the session default")
	foldContinuePrompt := fs.String("fold-continue-prompt", "Confirm that the compacted retained state is live and name the next engineering action.", "prompt appended after waking the folded state")
	foldContinueMaxTokens := fs.Int("fold-continue-max-tokens", 512, "generated tokens for the folded-state wake/continue check; 0 skips the check")
	contextLen := fs.Int("context", 0, "override context length")
	prefillChunkSize := fs.Int("prefill-chunk-size", 0, "override long-prompt prefill chunk size in tokens")
	cacheMode := fs.String("cache-mode", "", "override KV cache mode: fp16, q8, k-q8-v-q4, or paged")
	device := fs.String("device", "", "execution device: gpu or cpu")
	estimatePowerWatts := fs.Float64("estimate-power-watts", 0, "record an estimated average active power draw in watts")
	fastGemma4Lane := fs.Bool("fast-gemma4-lane", true, "enable the accepted Gemma 4 fast runtime gates by default; set false for baseline diagnostics")
	maxActiveMemoryBytes := fs.Uint64("max-active-memory-bytes", 0, "abort a turn if MLX active memory exceeds this many bytes; 0 derives from the resolved memory limit")
	maxProcessVirtualMemoryBytes := fs.Uint64("max-process-virtual-memory-bytes", 0, "abort a turn if process virtual memory exceeds this many bytes; 0 records process virtual memory without a hard cap")
	maxProcessResidentMemoryBytes := fs.Uint64("max-process-resident-memory-bytes", 0, "abort a turn if process resident memory exceeds this many bytes; 0 derives from the resolved memory limit")
	repeatedTokenLoopLimit := fs.Int("repeated-token-loop-limit", driverProfileDefaultRepeatedTokenLoopLimit, "abort when this many consecutive sampled tokens have the same token id")
	repeatedLineLoopLimit := fs.Int("repeated-line-loop-limit", profileDefaultRepeatedLineLoopLimit, "abort when this many consecutive visible non-empty lines repeat")
	repeatedSentenceLoopLimit := fs.Int("repeated-sentence-loop-limit", profileDefaultRepeatedSentenceLoopLimit, "abort when the same visible sentence repeats this many times in one output")
	fs.Usage = func() {
		core.WriteString(stderr, core.Sprintf("Usage: %s state-ramp-profile [flags] [model-path]\n", cliName()))
		fs.VisitAll(func(f *flag.Flag) {
			if f.DefValue == "" {
				core.WriteString(stderr, core.Sprintf("  -%s\n\t%s\n", f.Name, f.Usage))
				return
			}
			core.WriteString(stderr, core.Sprintf("  -%s\n\t%s (default %q)\n", f.Name, f.Usage, f.DefValue))
		})
	}
	if err := fs.Parse(args); err != nil {
		if core.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	visitedFlags := driverProfileVisitedFlags(fs)
	if driverProfileFastGemma4LaneEnabled(*fastGemma4Lane, visitedFlags, "") {
		for _, restore := range applyGemma4FastLaneDefaults(
			visitedFlags,
			contextLen,
			cacheMode,
			prefillChunkSize,
			nil,
			mlx.ProductionLaneHyperLongContextLength,
		) {
			defer restore()
		}
	}
	if fs.NArg() != 1 {
		core.WriteString(stderr, core.Sprintf("%s state-ramp-profile: expected one model path\n", cliName()))
		fs.Usage()
		return 2
	}
	if core.Trim(*promptFile) != "" {
		read := core.ReadFile(*promptFile)
		if !read.OK {
			core.Print(stderr, "%s state-ramp-profile: prompt file: %v", cliName(), read.Value)
			return 1
		}
		*prompt = string(read.Value.([]byte))
	}
	if core.Trim(*appendFile) != "" {
		read := core.ReadFile(*appendFile)
		if !read.OK {
			core.Print(stderr, "%s state-ramp-profile: append file: %v", cliName(), read.Value)
			return 1
		}
		*appendPrompt = string(read.Value.([]byte))
	}
	if core.Trim(*foldSummaryFile) != "" {
		read := core.ReadFile(*foldSummaryFile)
		if !read.OK {
			core.Print(stderr, "%s state-ramp-profile: fold summary file: %v", cliName(), read.Value)
			return 1
		}
		*foldSummary = string(read.Value.([]byte))
	}
	if core.Trim(*foldRecentTailFile) != "" {
		read := core.ReadFile(*foldRecentTailFile)
		if !read.OK {
			core.Print(stderr, "%s state-ramp-profile: fold tail file: %v", cliName(), read.Value)
			return 1
		}
		*foldRecentTail = string(read.Value.([]byte))
	}
	if *startTokens < 1 {
		core.WriteString(stderr, core.Sprintf("%s state-ramp-profile: start tokens must be >= 1\n", cliName()))
		return 2
	}
	if *targetTokens <= *startTokens {
		core.WriteString(stderr, core.Sprintf("%s state-ramp-profile: target tokens must be greater than start tokens\n", cliName()))
		return 2
	}
	if *compactionThresholdTokens < 0 {
		core.WriteString(stderr, core.Sprintf("%s state-ramp-profile: compaction threshold tokens must be >= 0\n", cliName()))
		return 2
	}
	if *compactionThresholdTokens == 0 {
		*compactionThresholdTokens = *targetTokens
	}
	if *compactionTailTokens < 0 {
		core.WriteString(stderr, core.Sprintf("%s state-ramp-profile: compaction tail tokens must be >= 0\n", cliName()))
		return 2
	}
	if *appendTokens < 1 {
		core.WriteString(stderr, core.Sprintf("%s state-ramp-profile: append tokens must be >= 1\n", cliName()))
		return 2
	}
	if *turnMaxTokens < 1 {
		core.WriteString(stderr, core.Sprintf("%s state-ramp-profile: turn max tokens must be >= 1\n", cliName()))
		return 2
	}
	if *turnMinTokens < 0 {
		core.WriteString(stderr, core.Sprintf("%s state-ramp-profile: turn min tokens must be >= 0\n", cliName()))
		return 2
	}
	*turnMinTokensPolicy = core.Lower(core.Trim(*turnMinTokensPolicy))
	if *turnMinTokensPolicy == "" {
		*turnMinTokensPolicy = "fail"
	}
	if *turnMinTokensPolicy != "fail" && *turnMinTokensPolicy != "mark" {
		core.WriteString(stderr, core.Sprintf("%s state-ramp-profile: turn min tokens policy must be fail or mark\n", cliName()))
		return 2
	}
	if *turns < 0 {
		core.WriteString(stderr, core.Sprintf("%s state-ramp-profile: turns must be >= 0\n", cliName()))
		return 2
	}
	if *prefillChunkSize < 0 {
		core.WriteString(stderr, core.Sprintf("%s state-ramp-profile: prefill chunk size must be >= 0\n", cliName()))
		return 2
	}
	if *estimatePowerWatts < 0 {
		core.WriteString(stderr, core.Sprintf("%s state-ramp-profile: estimated power watts must be >= 0\n", cliName()))
		return 2
	}
	if *temperature < 0 {
		core.WriteString(stderr, core.Sprintf("%s state-ramp-profile: temperature must be >= 0\n", cliName()))
		return 2
	}
	if *topP < 0 {
		core.WriteString(stderr, core.Sprintf("%s state-ramp-profile: top-p must be >= 0\n", cliName()))
		return 2
	}
	if *topK < 0 {
		core.WriteString(stderr, core.Sprintf("%s state-ramp-profile: top-k must be >= 0\n", cliName()))
		return 2
	}
	if *repeatPenalty < 0 {
		core.WriteString(stderr, core.Sprintf("%s state-ramp-profile: repeat penalty must be >= 0\n", cliName()))
		return 2
	}
	if *foldOnExhaustion && core.Trim(*foldStorePath) == "" {
		core.WriteString(stderr, core.Sprintf("%s state-ramp-profile: fold store path is required when fold-on-exhaustion is enabled\n", cliName()))
		return 2
	}
	if *foldPrefillChunkBytes < 0 {
		core.WriteString(stderr, core.Sprintf("%s state-ramp-profile: fold prefill chunk bytes must be >= 0\n", cliName()))
		return 2
	}
	if *foldContinueMaxTokens < 0 {
		core.WriteString(stderr, core.Sprintf("%s state-ramp-profile: fold continue max tokens must be >= 0\n", cliName()))
		return 2
	}
	if *repeatedTokenLoopLimit < 1 {
		core.WriteString(stderr, core.Sprintf("%s state-ramp-profile: repeated token loop limit must be >= 1\n", cliName()))
		return 2
	}
	if *repeatedLineLoopLimit < 1 {
		core.WriteString(stderr, core.Sprintf("%s state-ramp-profile: repeated line loop limit must be >= 1\n", cliName()))
		return 2
	}
	if *repeatedSentenceLoopLimit < 1 {
		core.WriteString(stderr, core.Sprintf("%s state-ramp-profile: repeated sentence loop limit must be >= 1\n", cliName()))
		return 2
	}

	loadOptions := []mlx.LoadOption{}
	var loadSettings *tuneProfileLoadSettings
	if *contextLen > 0 {
		loadOptions = append(loadOptions, mlx.WithContextLength(*contextLen))
		loadSettings = &tuneProfileLoadSettings{ContextLength: *contextLen}
	}
	if *prefillChunkSize > 0 {
		loadOptions = append(loadOptions, mlx.WithPrefillChunkSize(*prefillChunkSize))
		if loadSettings == nil {
			loadSettings = &tuneProfileLoadSettings{}
		}
		loadSettings.PrefillChunkSize = *prefillChunkSize
	}
	if core.Trim(*cacheMode) != "" {
		mode := memory.KVCacheMode(core.Trim(*cacheMode))
		switch mode {
		case memory.KVCacheModeFP16, memory.KVCacheModeQ8, memory.KVCacheModeKQ8VQ4, memory.KVCacheModePaged:
		default:
			core.WriteString(stderr, core.Sprintf("%s state-ramp-profile: unsupported cache mode %q\n", cliName(), string(mode)))
			return 2
		}
		loadOptions = append(loadOptions, mlx.WithKVCacheMode(mode))
		if loadSettings == nil {
			loadSettings = &tuneProfileLoadSettings{}
		}
		loadSettings.CacheMode = string(mode)
	}
	if *device != "" {
		loadOptions = append(loadOptions, mlx.WithDevice(*device))
	}

	report, err := runStateRampProfileGuarded(ctx, fs.Arg(0), loadOptions, stateRampProfileOptions{
		Prompt:                    *prompt,
		AppendPrompt:              *appendPrompt,
		AppendTurnDelimiter:       *appendTurnDelimiter,
		ChatTemplate:              *chatTemplate,
		EnableThinking:            *enableThinking,
		StartTokens:               *startTokens,
		TargetTokens:              *targetTokens,
		CompactionThresholdTokens: *compactionThresholdTokens,
		CompactionTailTokens:      *compactionTailTokens,
		AppendTokens:              *appendTokens,
		TurnMaxTokens:             *turnMaxTokens,
		TurnMinTokens:             *turnMinTokens,
		TurnMinTokensPolicy:       *turnMinTokensPolicy,
		Turns:                     *turns,
		Temperature:               *temperature,
		TopP:                      *topP,
		TopK:                      *topK,
		RepeatPenalty:             *repeatPenalty,
		SuppressEOS:               *suppressEOS,
		IncludeOutput:             *includeOutput,
		FoldOnExhaustion:          *foldOnExhaustion,
		FoldStorePath:             core.Trim(*foldStorePath),
		FoldSummary:               *foldSummary,
		FoldRecentTail:            *foldRecentTail,
		FoldPrefillChunkBytes:     *foldPrefillChunkBytes,
		FoldContinuePrompt:        *foldContinuePrompt,
		FoldContinueMaxTokens:     *foldContinueMaxTokens,
		SafetyLimits: driverProfileSafetyLimits{
			MaxActiveMemoryBytes:          *maxActiveMemoryBytes,
			MaxProcessVirtualMemoryBytes:  *maxProcessVirtualMemoryBytes,
			MaxProcessResidentMemoryBytes: *maxProcessResidentMemoryBytes,
			RepeatedTokenLoopLimit:        *repeatedTokenLoopLimit,
			RepeatedLineLoopLimit:         *repeatedLineLoopLimit,
			RepeatedSentenceLoopLimit:     *repeatedSentenceLoopLimit,
		},
	})
	if report != nil && loadSettings != nil {
		report.Load = mergeDriverProfileLoadSettings(loadSettings, report.Load)
	}
	if report != nil && *estimatePowerWatts > 0 {
		report.EstimatedEnergy = estimateStateRampProfileEnergy(report, *estimatePowerWatts)
	}
	reportPath := core.Trim(*reportFile)
	if *jsonOut || reportPath != "" {
		if report == nil {
			report = &stateRampProfileReport{
				Version:                   1,
				ModelPath:                 fs.Arg(0),
				PromptBytes:               len(*prompt),
				AppendPromptBytes:         len(*appendPrompt),
				AppendTurnSections:        0,
				ChatTemplate:              *chatTemplate,
				EnableThinking:            *enableThinking,
				StartTokens:               *startTokens,
				TargetTokens:              *targetTokens,
				CompactionThresholdTokens: *compactionThresholdTokens,
				CompactionTailTokens:      *compactionTailTokens,
				AppendTokens:              *appendTokens,
				TurnMaxTokens:             *turnMaxTokens,
				TurnMinTokens:             *turnMinTokens,
				TurnMinTokensPolicy:       *turnMinTokensPolicy,
				RequestedTurns:            *turns,
				Temperature:               *temperature,
				TopP:                      *topP,
				TopK:                      *topK,
				RepeatPenalty:             *repeatPenalty,
				SuppressEOS:               *suppressEOS,
				IncludeOutput:             *includeOutput,
				FoldOnExhaustion:          *foldOnExhaustion,
				FoldStorePath:             core.Trim(*foldStorePath),
				FoldSummaryBytes:          len(*foldSummary),
				FoldRecentTailBytes:       len(*foldRecentTail),
				FoldPrefillChunkBytes:     *foldPrefillChunkBytes,
				FoldContinueMaxTokens:     *foldContinueMaxTokens,
			}
		}
		if err != nil && report.Error == "" {
			report.Error = err.Error()
		}
		data := core.JSONMarshalIndent(report, "", "  ")
		if !data.OK {
			core.Print(stderr, "%s state-ramp-profile: marshal report failed", cliName())
			return 1
		}
		if reportPath != "" {
			if writeErr := writeJSONReportFile(reportPath, data.Value.([]byte)); writeErr != nil {
				core.Print(stderr, "%s state-ramp-profile: write report file: %v", cliName(), writeErr)
				return 1
			}
		}
		if *jsonOut {
			core.WriteString(stdout, string(data.Value.([]byte)))
			core.WriteString(stdout, "\n")
		}
		if err != nil {
			return 1
		}
		if *jsonOut {
			return 0
		}
	}
	if err != nil {
		core.Print(stderr, "%s state-ramp-profile: %v", cliName(), err)
		return 1
	}
	printStateRampProfileSummary(stdout, report)
	return 0
}

var runStateRampProfile = defaultRunStateRampProfile

func runStateRampProfileGuarded(ctx context.Context, modelPath string, loadOptions []mlx.LoadOption, opts stateRampProfileOptions) (report *stateRampProfileReport, err error) {
	defer func() {
		if recovered := recover(); recovered != nil {
			err = core.NewError(core.Sprintf("state-ramp-profile panic: %v", recovered))
		}
	}()
	return runStateRampProfile(ctx, modelPath, loadOptions, opts)
}

func defaultRunStateRampProfile(ctx context.Context, modelPath string, loadOptions []mlx.LoadOption, opts stateRampProfileOptions) (*stateRampProfileReport, error) {
	opts = normalizeStateRampProfileOptions(opts)
	report := &stateRampProfileReport{
		Version:                   1,
		ModelPath:                 modelPath,
		PromptBytes:               len(opts.Prompt),
		AppendPromptBytes:         len(opts.AppendPrompt),
		EnableThinking:            opts.EnableThinking,
		StartTokens:               opts.StartTokens,
		TargetTokens:              opts.TargetTokens,
		CompactionThresholdTokens: opts.CompactionThresholdTokens,
		CompactionTailTokens:      opts.CompactionTailTokens,
		AppendTokens:              opts.AppendTokens,
		TurnMaxTokens:             opts.TurnMaxTokens,
		TurnMinTokens:             opts.TurnMinTokens,
		TurnMinTokensPolicy:       opts.TurnMinTokensPolicy,
		RequestedTurns:            opts.Turns,
		Temperature:               opts.Temperature,
		TopP:                      opts.TopP,
		TopK:                      opts.TopK,
		RepeatPenalty:             opts.RepeatPenalty,
		SuppressEOS:               opts.SuppressEOS,
		IncludeOutput:             opts.IncludeOutput,
		FoldOnExhaustion:          opts.FoldOnExhaustion,
		FoldStorePath:             opts.FoldStorePath,
		FoldSummaryBytes:          len(opts.FoldSummary),
		FoldRecentTailBytes:       len(opts.FoldRecentTail),
		FoldPrefillChunkBytes:     opts.FoldPrefillChunkBytes,
		FoldContinueMaxTokens:     opts.FoldContinueMaxTokens,
		SafetyLimits:              opts.SafetyLimits,
		RuntimeGates:              driverProfileRuntimeGates(),
	}
	loadStart := time.Now()
	model, err := loadBenchModel(modelPath, loadOptions...)
	report.LoadDuration = bench.NonZeroDuration(time.Since(loadStart))
	if err != nil {
		report.Error = err.Error()
		return report, err
	}
	if model == nil {
		err := core.NewError("mlx: state ramp profile loaded nil model")
		report.Error = err.Error()
		return report, err
	}
	report.Load = mergeDriverProfileLoadSettings(report.Load, loadSettingsFromModelInfo(model.Info()))
	opts.SafetyLimits = resolveDriverProfileSafetyLimits(opts.SafetyLimits, report.Load)
	report.SafetyLimits = opts.SafetyLimits
	defer model.Close()
	if err := driverProfileMetricsSafetyError("load", model.Metrics(), opts.SafetyLimits); err != nil {
		report.Error = err.Error()
		return report, err
	}
	opts.ChatTemplate = chapterProfileTemplate(opts.ChatTemplate, model.Info().Architecture)
	report.ChatTemplate = opts.ChatTemplate
	tok := model.Tokenizer()
	if tok == nil {
		err := core.NewError("state-ramp-profile: model tokenizer is nil")
		report.Error = err.Error()
		return report, err
	}
	sourceTokens, err := tok.Encode(opts.Prompt)
	if err != nil {
		report.Error = err.Error()
		return report, err
	}
	if len(sourceTokens) == 0 {
		err := core.NewError("state-ramp-profile: source prompt produced no tokens")
		report.Error = err.Error()
		return report, err
	}
	report.SourceTokens = len(sourceTokens)
	appendText := opts.AppendPrompt
	if appendText == "" {
		appendText = opts.Prompt
		report.AppendPromptBytes = len(appendText)
	}
	appendSourceTokens, appendTurnSections, err := stateRampProfileAppendSources(tok, appendText, opts.AppendTurnDelimiter, opts.ChatTemplate, opts.EnableThinking)
	if err != nil {
		report.Error = err.Error()
		return report, err
	}
	report.AppendSourceTokens = countStateRampAppendSourceTokens(appendSourceTokens, appendTurnSections)
	report.AppendTurnSections = len(appendTurnSections)
	session, err := model.NewSession()
	if err != nil {
		report.Error = err.Error()
		return report, err
	}
	defer session.Close()

	seedTokens, err := stateRampProfileSeedTokens(tok, sourceTokens, opts)
	if err != nil {
		report.Error = err.Error()
		return report, err
	}
	prefillStart := time.Now()
	err = session.PrefillTokens(ctx, seedTokens)
	report.InitialPrefillDuration = bench.NonZeroDuration(time.Since(prefillStart))
	report.InitialPrefillTokens = len(seedTokens)
	if err != nil {
		report.Error = err.Error()
		return report, err
	}
	if err := driverProfileMetricsSafetyError("initial prefill", model.Metrics(), opts.SafetyLimits); err != nil {
		report.Error = err.Error()
		return report, err
	}

	currentTokens := len(seedTokens)
	sourceOffset := 0
	var firstErr error
	for turnIndex := 1; shouldRunStateRampTurn(turnIndex, currentTokens, opts); turnIndex++ {
		turnSourceTokens, turnSourceOffset, appendCount := stateRampProfileTurnAppendSource(appendSourceTokens, appendTurnSections, sourceOffset, currentTokens, turnIndex, opts)
		turn := stateRampProfileGenerateTurn(ctx, model, session, turnSourceTokens, turnSourceOffset, appendCount, currentTokens, turnIndex, opts)
		if len(appendTurnSections) == 0 {
			sourceOffset += turn.AppendedTokens
		}
		if turn.TokensAfterGenerate > 0 {
			currentTokens = turn.TokensAfterGenerate
		} else {
			currentTokens += turn.AppendedTokens
		}
		if turn.Error != "" && firstErr == nil {
			if stateRampProfileTurnErrorFatal(turn, opts) {
				firstErr = core.NewError(turn.Error)
			}
		}
		report.Turns = append(report.Turns, turn)
		mlx.ClearCache()
		if turn.Error != "" && stateRampProfileTurnErrorFatal(turn, opts) {
			break
		}
	}
	report.Summary = summariseStateRampProfileTurns(report.InitialPrefillDuration, len(seedTokens), report.Turns, opts)
	if opts.FoldOnExhaustion {
		report.Fold = stateRampProfileFoldExhausted(ctx, model, session, report, opts)
		if report.Fold != nil && report.Fold.Error != "" && firstErr == nil {
			firstErr = core.NewError(report.Fold.Error)
		}
	}
	if firstErr != nil {
		report.Error = firstErr.Error()
		return report, firstErr
	}
	return report, nil
}

func normalizeStateRampProfileOptions(opts stateRampProfileOptions) stateRampProfileOptions {
	opts.Prompt = core.Trim(opts.Prompt)
	opts.AppendPrompt = core.Trim(opts.AppendPrompt)
	if opts.Prompt == "" {
		opts.Prompt = "Answer in one short sentence: why does retained model state matter?"
	}
	if opts.StartTokens <= 0 {
		opts.StartTokens = 30000
	}
	if opts.TargetTokens <= 0 {
		opts.TargetTokens = 100000
	}
	if opts.CompactionThresholdTokens <= 0 {
		opts.CompactionThresholdTokens = opts.TargetTokens
	}
	if opts.CompactionTailTokens < 0 {
		opts.CompactionTailTokens = 0
	}
	if opts.AppendTokens <= 0 {
		opts.AppendTokens = 8192
	}
	if opts.TurnMaxTokens <= 0 {
		opts.TurnMaxTokens = 1024
	}
	if opts.TurnMinTokens < 0 {
		opts.TurnMinTokens = 0
	}
	opts.TurnMinTokensPolicy = core.Lower(core.Trim(opts.TurnMinTokensPolicy))
	if opts.TurnMinTokensPolicy == "" {
		opts.TurnMinTokensPolicy = "fail"
	}
	if opts.TurnMinTokensPolicy != "mark" {
		opts.TurnMinTokensPolicy = "fail"
	}
	if opts.SafetyLimits.RepeatedTokenLoopLimit <= 0 {
		opts.SafetyLimits.RepeatedTokenLoopLimit = driverProfileDefaultRepeatedTokenLoopLimit
	}
	if opts.SafetyLimits.RepeatedLineLoopLimit <= 0 {
		opts.SafetyLimits.RepeatedLineLoopLimit = profileDefaultRepeatedLineLoopLimit
	}
	if opts.SafetyLimits.RepeatedSentenceLoopLimit <= 0 {
		opts.SafetyLimits.RepeatedSentenceLoopLimit = profileDefaultRepeatedSentenceLoopLimit
	}
	opts.FoldStorePath = core.Trim(opts.FoldStorePath)
	opts.FoldSummary = core.Trim(opts.FoldSummary)
	opts.FoldRecentTail = core.Trim(opts.FoldRecentTail)
	if opts.FoldPrefillChunkBytes < 0 {
		opts.FoldPrefillChunkBytes = 0
	}
	if opts.FoldContinueMaxTokens < 0 {
		opts.FoldContinueMaxTokens = 0
	}
	if opts.FoldContinuePrompt == "" {
		opts.FoldContinuePrompt = "Confirm that the compacted retained state is live and name the next engineering action."
	}
	return opts
}

func shouldRunStateRampTurn(index, currentTokens int, opts stateRampProfileOptions) bool {
	if stateRampProfileLiveTokenLimitReached(currentTokens, opts) {
		return false
	}
	if opts.Turns > 0 {
		return index <= opts.Turns
	}
	return currentTokens < opts.TargetTokens
}

func stateRampProfileLiveTokenLimitReached(currentTokens int, opts stateRampProfileOptions) bool {
	limit := stateRampProfileLiveTokenLimit(opts)
	return limit > 0 && currentTokens >= limit
}

func stateRampProfileLiveTokenLimit(opts stateRampProfileOptions) int {
	limit := opts.TargetTokens
	if opts.CompactionThresholdTokens > 0 && (limit <= 0 || opts.CompactionThresholdTokens < limit) {
		limit = opts.CompactionThresholdTokens
	}
	return limit
}

func repeatedStateRampTokens(source []int32, offset, count int) []int32 {
	if len(source) == 0 || count <= 0 {
		return nil
	}
	offset %= len(source)
	if offset < 0 {
		offset += len(source)
	}
	if count <= len(source)-offset {
		return source[offset : offset+count]
	}
	out := make([]int32, count)
	for i := range out {
		out[i] = source[(offset+i)%len(source)]
	}
	return out
}

func stateRampProfileSeedTokens(tok *mlx.Tokenizer, sourceTokens []int32, opts stateRampProfileOptions) ([]int32, error) {
	if len(sourceTokens) == 0 {
		return nil, core.NewError("state-ramp-profile: source prompt produced no tokens")
	}
	if stateRampProfilePlainTemplate(opts.ChatTemplate) {
		return repeatedStateRampTokens(sourceTokens, 0, opts.StartTokens), nil
	}
	target := opts.StartTokens
	if target <= 0 {
		target = len(sourceTokens)
	}
	contextBudget := target
	if contextBudget > len(sourceTokens) {
		contextBudget = len(sourceTokens)
	}
	for contextBudget >= 0 {
		contextText, err := tok.Decode(sourceTokens[:contextBudget])
		if err != nil {
			return nil, err
		}
		wrapped := stateRampProfileInitialPrompt(opts.ChatTemplate, contextText, opts.EnableThinking)
		tokens, err := tok.Encode(wrapped)
		if err != nil {
			return nil, err
		}
		if len(tokens) <= target || contextBudget == 0 {
			return tokens, nil
		}
		overage := len(tokens) - target
		if overage < 1 {
			overage = 1
		}
		contextBudget -= overage
	}
	return nil, core.NewError("state-ramp-profile: could not fit chat-wrapped seed prompt")
}

func stateRampProfilePlainTemplate(template string) bool {
	template = core.Lower(core.Trim(template))
	return template == "" || template == "plain"
}

func stateRampProfileInitialPrompt(template, contextPrompt string, enableThinking bool) string {
	contextPrompt = core.Trim(contextPrompt)
	switch template {
	case "gemma4":
		builder := core.NewBuilder()
		builder.WriteString("<bos><|turn>system\n")
		if enableThinking {
			builder.WriteString("<|think|>\n")
		}
		builder.WriteString("You are running an opencode-style engineering session. Use the retained codebase context as memory for later user turns.\n\n")
		builder.WriteString(contextPrompt)
		builder.WriteString("<turn|>\n<|turn>model\n")
		if !enableThinking {
			builder.WriteString("<|channel>thought\n<channel|>")
		}
		builder.WriteString("Ready.<turn|>\n")
		return builder.String()
	case "gemma":
		return "<start_of_turn>user\n" + contextPrompt + "\n\nRetain this project context for later engineering turns.<end_of_turn>\n<start_of_turn>model\nReady.<end_of_turn>\n"
	case "qwen":
		return "<|im_start|>system\nRetain this project context for later engineering turns.\n\n" + contextPrompt + "<|im_end|>\n<|im_start|>assistant\nReady.<|im_end|>\n"
	case "llama":
		return "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nRetain this project context for later engineering turns.\n\n" + contextPrompt + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nReady.<|eot_id|>"
	default:
		return contextPrompt
	}
}

func stateRampProfileTurnPrompt(template, prompt string, enableThinking bool) string {
	prompt = core.Trim(prompt)
	switch template {
	case "gemma4":
		builder := core.NewBuilder()
		builder.Grow(len(prompt) + 512)
		builder.WriteString("<|turn>user\n")
		writeStateRampProfileReferenceTurn(builder, prompt)
		builder.WriteString("<turn|>\n<|turn>model\n")
		if !enableThinking {
			builder.WriteString("<|channel>thought\n<channel|>")
		}
		return builder.String()
	case "gemma":
		builder := core.NewBuilder()
		builder.Grow(len(prompt) + 512)
		builder.WriteString("<start_of_turn>user\n")
		writeStateRampProfileReferenceTurn(builder, prompt)
		builder.WriteString("<end_of_turn>\n<start_of_turn>model\n")
		return builder.String()
	case "qwen":
		builder := core.NewBuilder()
		builder.Grow(len(prompt) + 512)
		builder.WriteString("<|im_start|>user\n")
		writeStateRampProfileReferenceTurn(builder, prompt)
		builder.WriteString("<|im_end|>\n<|im_start|>assistant\n")
		return builder.String()
	case "llama":
		builder := core.NewBuilder()
		builder.Grow(len(prompt) + 512)
		builder.WriteString("<|start_header_id|>user<|end_header_id|>\n\n")
		writeStateRampProfileReferenceTurn(builder, prompt)
		builder.WriteString("<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
		return builder.String()
	default:
		return stateRampProfileReferenceTurn(prompt)
	}
}

func stateRampProfileReferenceTurn(prompt string) string {
	prompt = core.Trim(prompt)
	if prompt == "" {
		return prompt
	}
	builder := core.NewBuilder()
	builder.Grow(len(prompt) + 512)
	writeStateRampProfileReferenceTurn(builder, prompt)
	return builder.String()
}

func writeStateRampProfileReferenceTurn(builder interface{ WriteString(string) (int, error) }, prompt string) {
	prompt = core.Trim(prompt)
	if prompt == "" {
		return
	}
	builder.WriteString("Use the retained project context and the new turn material below. Answer the user request directly. Treat any code or document excerpts as reference material, not as text to continue.\n\n")
	builder.WriteString("<turn_material>\n")
	builder.WriteString(prompt)
	builder.WriteString("\n</turn_material>\n\nAnswer the user request from the turn material now. Honour any requested output length before stopping. Do not continue or complete the reference excerpts.")
}

func stateRampProfileVisibleOutput(template, output string) string {
	return chapterProfileVisibleText(template, output)
}

func stateRampProfileAssistantCloseSuffix(template string) string {
	if stateRampProfilePlainTemplate(template) {
		return ""
	}
	return chapterProfileAssistantHistorySuffix(template, "")
}

func stateRampProfileAppendSources(tok *mlx.Tokenizer, text, delimiter, template string, enableThinking bool) ([]int32, [][]int32, error) {
	if tok == nil {
		return nil, nil, core.NewError("state-ramp-profile: model tokenizer is nil")
	}
	delimiter = core.Trim(delimiter)
	if delimiter == "" {
		tokens, err := tok.Encode(text)
		if err != nil {
			return nil, nil, err
		}
		if len(tokens) == 0 {
			return nil, nil, core.NewError("state-ramp-profile: append prompt produced no tokens")
		}
		return tokens, nil, nil
	}
	sections := [][]int32{}
	for _, raw := range core.Split(text, delimiter) {
		section := core.Trim(raw)
		if section == "" {
			continue
		}
		if !stateRampProfilePlainTemplate(template) {
			section = stateRampProfileTurnPrompt(template, section, enableThinking)
		}
		tokens, err := tok.Encode(section)
		if err != nil {
			return nil, nil, err
		}
		if len(tokens) > 0 {
			sections = append(sections, tokens)
		}
	}
	if len(sections) == 0 {
		return nil, nil, core.NewError("state-ramp-profile: append turn delimiter produced no token sections")
	}
	return nil, sections, nil
}

func countStateRampAppendSourceTokens(tokens []int32, sections [][]int32) int {
	if len(sections) == 0 {
		return len(tokens)
	}
	total := 0
	for _, section := range sections {
		total += len(section)
	}
	return total
}

func stateRampProfileTurnAppendSource(source []int32, sections [][]int32, sourceOffset, currentTokens, turnIndex int, opts stateRampProfileOptions) ([]int32, int, int) {
	tokens := source
	appendCount := opts.AppendTokens
	if len(sections) > 0 {
		tokens = sections[(turnIndex-1)%len(sections)]
		appendCount = len(tokens)
		sourceOffset = 0
	} else if limit := stateRampProfileLiveTokenLimit(opts); limit > 0 {
		if remaining := limit - currentTokens; remaining < appendCount {
			appendCount = remaining
		}
	}
	if appendCount < 0 {
		appendCount = 0
	}
	if sourceOffset < 0 {
		sourceOffset = 0
	}
	return tokens, sourceOffset, appendCount
}

func stateRampProfileGenerateTurn(ctx context.Context, model *mlx.Model, session *mlx.ModelSession, sourceTokens []int32, sourceOffset, appendCount, currentTokens, index int, opts stateRampProfileOptions) stateRampProfileTurn {
	turn := stateRampProfileTurn{
		Index:              index,
		TokensBeforeAppend: currentTokens,
	}
	if appendCount > 0 {
		tokens := repeatedStateRampTokens(sourceTokens, sourceOffset, appendCount)
		appendStart := time.Now()
		err := session.AppendTokens(ctx, tokens)
		turn.AppendDuration = bench.NonZeroDuration(time.Since(appendStart))
		turn.AppendedTokens = len(tokens)
		if err != nil {
			turn.Error = err.Error()
			return turn
		}
	}
	turn.TokensAfterAppend = currentTokens + turn.AppendedTokens
	start := time.Now()
	firstToken := time.Duration(0)
	builder := core.NewBuilder()
	generateOptions := []mlx.GenerateOption{
		mlx.WithMaxTokens(opts.TurnMaxTokens),
		mlx.WithTemperature(float32(opts.Temperature)),
		mlx.WithTopP(float32(opts.TopP)),
		mlx.WithTopK(opts.TopK),
		mlx.WithRepeatPenalty(float32(opts.RepeatPenalty)),
	}
	stopTokenIDs, suppressTokenIDs := chapterProfileTemplateTokenControls(opts.ChatTemplate, model.Tokenizer())
	if len(stopTokenIDs) > 0 {
		generateOptions = append(generateOptions, mlx.WithStopTokens(stopTokenIDs...))
	}
	if len(suppressTokenIDs) > 0 {
		generateOptions = append(generateOptions, mlx.WithSuppressTokens(suppressTokenIDs...))
	}
	if opts.SuppressEOS {
		if tok := model.Tokenizer(); tok != nil {
			if eosID, ok := tok.TokenID("<eos>"); ok {
				generateOptions = append(generateOptions, mlx.WithSuppressTokens(eosID))
			}
		}
	}
	generationCtx := ctx
	if generationCtx == nil {
		generationCtx = context.Background()
	}
	generationCtx, cancelGeneration := context.WithCancel(generationCtx)
	defer cancelGeneration()
	var probeErr error
	sampledTokenIDs := make([]int32, 0, 32)
	sampledTokenTexts := make([]string, 0, 32)
	repeatedTokenID := int32(0)
	repeatedTokenCount := 0
	var lineErr error
	currentLine := ""
	lastLine := ""
	repeatedLineCount := 0
	for token := range session.GenerateStream(generationCtx, generateOptions...) {
		if firstToken == 0 {
			firstToken = bench.NonZeroDuration(time.Since(start))
		}
		turn.VisibleTokens++
		if len(sampledTokenIDs) < 32 {
			sampledTokenIDs = append(sampledTokenIDs, token.ID)
			sampledTokenTexts = append(sampledTokenTexts, token.Text)
		}
		if opts.IncludeOutput {
			builder.WriteString(token.Text)
		}
		if probeErr == nil {
			if err := driverProfileMetricsSafetyError(core.Sprintf("state-ramp-profile turn %d stream", index), profileLiveMetrics(), opts.SafetyLimits); err != nil {
				probeErr = err
				cancelGeneration()
				break
			}
			if opts.SafetyLimits.RepeatedTokenLoopLimit <= 0 {
				repeatedTokenCount = 0
			} else if repeatedTokenCount == 0 || token.ID != repeatedTokenID {
				repeatedTokenID = token.ID
				repeatedTokenCount = 1
			} else {
				repeatedTokenCount++
				if repeatedTokenCount >= opts.SafetyLimits.RepeatedTokenLoopLimit {
					probeErr = core.NewError(core.Sprintf("state-ramp-profile: turn %d sampled token %d for %d consecutive tokens", index, token.ID, repeatedTokenCount))
					cancelGeneration()
					break
				}
			}
		}
		if lineErr == nil {
			if line, count, ok := profileObserveRepeatedLineFragment(token.Text, &currentLine, &lastLine, &repeatedLineCount, opts.SafetyLimits.RepeatedLineLoopLimit); ok {
				lineErr = core.NewError(core.Sprintf("state-ramp-profile: turn %d repeated visible line %q for %d consecutive lines", index, line, count))
				cancelGeneration()
				break
			}
		}
	}
	if lineErr == nil {
		if line, count, ok := profileFlushRepeatedLine(&currentLine, &lastLine, &repeatedLineCount, opts.SafetyLimits.RepeatedLineLoopLimit); ok {
			lineErr = core.NewError(core.Sprintf("state-ramp-profile: turn %d repeated visible line %q for %d consecutive lines", index, line, count))
		}
	}
	turn.Duration = bench.NonZeroDuration(time.Since(start))
	turn.FirstTokenDuration = firstToken
	turn.StreamDuration = turn.Duration
	if firstToken > 0 && turn.Duration > firstToken {
		turn.StreamDuration = turn.Duration - firstToken
	}
	turn.SampledTokenIDs = sampledTokenIDs
	turn.SampledTokenTexts = sampledTokenTexts
	turn.Metrics = model.Metrics()
	turn.DriverOverheadDuration = driverRunOverhead(turn.Duration, turn.Metrics)
	turn.TokensAfterGenerate = turn.Metrics.PromptTokens + turn.Metrics.GeneratedTokens
	if opts.IncludeOutput {
		turn.Output = stateRampProfileVisibleOutput(opts.ChatTemplate, builder.String())
	}
	if probeErr != nil {
		turn.Error = probeErr.Error()
		return turn
	}
	if lineErr != nil {
		turn.Error = lineErr.Error()
		return turn
	}
	if err := session.Err(); err != nil {
		turn.Error = err.Error()
		return turn
	}
	if err := driverProfileMetricsSafetyError(core.Sprintf("state-ramp-profile turn %d", index), turn.Metrics, opts.SafetyLimits); err != nil {
		turn.Error = err.Error()
		return turn
	}
	if err := driverProfileRunSafetyError(index, driverProfileRun{
		Index:             index,
		VisibleTokens:     turn.VisibleTokens,
		SampledTokenIDs:   turn.SampledTokenIDs,
		SampledTokenTexts: turn.SampledTokenTexts,
		Output:            turn.Output,
		Metrics:           turn.Metrics,
	}, opts.SafetyLimits); err != nil {
		turn.Error = err.Error()
		return turn
	}
	if opts.TurnMinTokens > 0 && turn.VisibleTokens < opts.TurnMinTokens {
		turn.BelowMinTokens = true
		turn.Error = core.Sprintf("state-ramp-profile: turn %d produced %d visible tokens, below minimum real-workload floor %d", index, turn.VisibleTokens, opts.TurnMinTokens)
		return turn
	}
	if suffix := stateRampProfileAssistantCloseSuffix(opts.ChatTemplate); suffix != "" {
		closeStart := time.Now()
		if err := chapterProfileAppendPrompt(ctx, model, session, suffix); err != nil {
			turn.Error = err.Error()
			return turn
		}
		turn.AppendDuration += bench.NonZeroDuration(time.Since(closeStart))
		if tok := model.Tokenizer(); tok != nil {
			if tokens, err := tok.Encode(suffix); err == nil {
				turn.TurnCloseTokens = len(tokens)
				turn.TokensAfterGenerate += len(tokens)
			}
		}
	}
	if ctx != nil {
		if err := ctx.Err(); err != nil {
			turn.Error = err.Error()
		}
	}
	return turn
}

func stateRampProfileTurnErrorFatal(turn stateRampProfileTurn, opts stateRampProfileOptions) bool {
	if turn.Error == "" {
		return false
	}
	return !(turn.BelowMinTokens && opts.TurnMinTokensPolicy == "mark")
}

func summariseStateRampProfileTurns(initialPrefill time.Duration, initialTokens int, turns []stateRampProfileTurn, opts stateRampProfileOptions) stateRampProfileSummary {
	summary := stateRampProfileSummary{
		InitialPrefillTokens: initialTokens,
		FinalStateTokens:     initialTokens,
		TotalDuration:        initialPrefill,
	}
	if initialPrefill > 0 && initialTokens > 0 {
		summary.InitialPrefillTokensPerSec = float64(initialTokens) / initialPrefill.Seconds()
	}
	var decodeDuration time.Duration
	var turnWallDuration time.Duration
	for _, turn := range turns {
		if turn.Error != "" {
			summary.FailedTurns++
		} else {
			summary.SuccessfulTurns++
		}
		summary.AppendedTokens += turn.AppendedTokens
		summary.GeneratedTokens += turn.Metrics.GeneratedTokens
		summary.VisibleTokens += turn.VisibleTokens
		summary.TotalDuration += turn.AppendDuration + turn.Duration
		summary.AppendDuration += turn.AppendDuration
		turnWallDuration += turn.AppendDuration + turn.Duration
		decodeDuration += turn.Metrics.DecodeDuration
		if turn.TokensAfterGenerate > summary.FinalStateTokens {
			summary.FinalStateTokens = turn.TokensAfterGenerate
		} else if turn.TokensAfterAppend > summary.FinalStateTokens {
			summary.FinalStateTokens = turn.TokensAfterAppend
		}
		if turn.Metrics.PeakMemoryBytes > summary.PeakMemoryBytes {
			summary.PeakMemoryBytes = turn.Metrics.PeakMemoryBytes
		}
		if turn.Metrics.ActiveMemoryBytes > summary.ActiveMemoryBytes {
			summary.ActiveMemoryBytes = turn.Metrics.ActiveMemoryBytes
		}
		if turn.Metrics.CacheMemoryBytes > summary.CacheMemoryBytes {
			summary.CacheMemoryBytes = turn.Metrics.CacheMemoryBytes
		}
		if turn.Metrics.ProcessVirtualMemoryBytes > summary.ProcessVirtualMemoryBytes {
			summary.ProcessVirtualMemoryBytes = turn.Metrics.ProcessVirtualMemoryBytes
		}
		if turn.Metrics.ProcessResidentMemoryBytes > summary.ProcessResidentMemoryBytes {
			summary.ProcessResidentMemoryBytes = turn.Metrics.ProcessResidentMemoryBytes
		}
		if turn.Metrics.ProcessPeakResidentBytes > summary.ProcessPeakResidentBytes {
			summary.ProcessPeakResidentBytes = turn.Metrics.ProcessPeakResidentBytes
		}
	}
	if len(turns) > 0 {
		summary.AppendAvgDuration = summary.AppendDuration / time.Duration(len(turns))
	}
	if summary.AppendDuration > 0 && summary.AppendedTokens > 0 {
		summary.AppendTokensPerSecAverage = float64(summary.AppendedTokens) / summary.AppendDuration.Seconds()
	}
	if decodeDuration > 0 && summary.GeneratedTokens > 0 {
		summary.DecodeTokensPerSecAverage = float64(summary.GeneratedTokens) / decodeDuration.Seconds()
	}
	if turnWallDuration > 0 && summary.GeneratedTokens > 0 {
		summary.EffectiveTurnTokensPerSec = float64(summary.GeneratedTokens) / turnWallDuration.Seconds()
	}
	annotateStateRampProfileContextLifecycle(&summary, opts)
	return summary
}

func annotateStateRampProfileContextLifecycle(summary *stateRampProfileSummary, opts stateRampProfileOptions) {
	if summary == nil {
		return
	}
	threshold := opts.CompactionThresholdTokens
	if threshold <= 0 {
		threshold = opts.TargetTokens
	}
	if threshold <= 0 {
		return
	}
	summary.CompactionThresholdTokens = threshold
	summary.CompactionTailTokens = opts.CompactionTailTokens
	if summary.FinalStateTokens < threshold {
		return
	}
	summary.ContextExhausted = true
	summary.FoldedStateRequired = true
	summary.CompactionReason = "live state reached the compaction threshold; checkpoint, summarise, and prefill a folded state from durable summary plus recent tail before appending more turns"
}

func stateRampProfileFoldExhausted(ctx context.Context, model *mlx.Model, session *mlx.ModelSession, report *stateRampProfileReport, opts stateRampProfileOptions) *stateRampProfileFold {
	fold := &stateRampProfileFold{
		StorePath:           opts.FoldStorePath,
		SummaryBytes:        len(opts.FoldSummary),
		RecentTailBytes:     len(opts.FoldRecentTail),
		ContinuePromptBytes: len(opts.FoldContinuePrompt),
	}
	if report == nil || !report.Summary.FoldedStateRequired {
		fold.SkippedReason = "live state did not reach the compaction threshold"
		return fold
	}
	fold.Attempted = true
	if model == nil || session == nil {
		fold.Error = "state-ramp-profile: folded-state handoff requires a live model session"
		return fold
	}
	if core.Trim(opts.FoldStorePath) == "" {
		fold.Error = "state-ramp-profile: fold store path is required"
		return fold
	}
	store, err := statefile.Create(ctx, opts.FoldStorePath)
	if err != nil {
		fold.Error = err.Error()
		return fold
	}
	defer store.Close()

	summary := stateRampProfileFoldSummary(report, opts)
	tail := stateRampProfileFoldRecentTail(report, opts)
	fold.SummaryBytes = len(summary)
	fold.RecentTailBytes = len(tail)
	foldPrompt := stateRampProfileInitialPrompt(opts.ChatTemplate, stateRampProfileFoldBody(summary, tail), opts.EnableThinking)
	fold.FoldedPromptBytes = len(foldPrompt)
	baseURI := stateRampProfileFoldBaseURI()
	start := time.Now()
	folded, foldReport, err := model.FoldAgentMemory(ctx, session, store, mlx.AgentMemoryFoldOptions{
		Summary:           summary,
		RecentTail:        tail,
		FoldedPrompt:      foldPrompt,
		PrefillChunkBytes: opts.FoldPrefillChunkBytes,
		Checkpoint:        stateRampProfileFoldSleepOptions(report, baseURI, "checkpoint"),
		Folded:            stateRampProfileFoldSleepOptions(report, baseURI, "folded"),
	})
	fold.Duration = bench.NonZeroDuration(time.Since(start))
	if foldReport != nil {
		fold.Checkpoint = foldReport.Checkpoint
		fold.Folded = foldReport.Folded
		fold.SummaryBytes = foldReport.SummaryBytes
		fold.RecentTailBytes = foldReport.RecentTailBytes
		fold.FoldedPromptBytes = foldReport.FoldedPromptBytes
	}
	if err != nil {
		fold.Error = err.Error()
		return fold
	}
	if folded != nil {
		defer folded.Close()
	}
	if opts.FoldContinueMaxTokens <= 0 {
		return fold
	}
	if fold.Folded == nil || fold.Folded.IndexURI == "" {
		fold.Error = "state-ramp-profile: folded-state wake index is missing"
		return fold
	}
	wakeStart := time.Now()
	woken, wake, err := model.WakeAgentMemory(ctx, store, agent.WakeOptions{
		IndexURI: fold.Folded.IndexURI,
	})
	fold.WakeDuration = bench.NonZeroDuration(time.Since(wakeStart))
	fold.Wake = wake
	if err != nil {
		fold.Error = err.Error()
		return fold
	}
	defer woken.Close()
	continueTurn, err := stateRampProfileContinueFromFold(ctx, model, woken, fold, opts)
	fold.ContinueTurn = continueTurn
	if err != nil {
		fold.Error = err.Error()
	}
	return fold
}

func stateRampProfileContinueFromFold(ctx context.Context, model *mlx.Model, session *mlx.ModelSession, fold *stateRampProfileFold, opts stateRampProfileOptions) (*stateRampProfileTurn, error) {
	if fold == nil || fold.Folded == nil {
		return nil, core.NewError("state-ramp-profile: folded state is missing")
	}
	prompt := stateRampProfileTurnPrompt(opts.ChatTemplate, opts.FoldContinuePrompt, opts.EnableThinking)
	tok := model.Tokenizer()
	if tok == nil {
		return nil, core.NewError("state-ramp-profile: model tokenizer is nil")
	}
	tokens, err := tok.Encode(prompt)
	if err != nil {
		return nil, err
	}
	continueOpts := opts
	continueOpts.TurnMaxTokens = opts.FoldContinueMaxTokens
	continueOpts.TurnMinTokens = 0
	continueOpts.TurnMinTokensPolicy = "mark"
	turn := stateRampProfileGenerateTurn(ctx, model, session, tokens, 0, len(tokens), fold.Folded.TokenCount, 1, continueOpts)
	if turn.Error != "" {
		return &turn, core.NewError(turn.Error)
	}
	return &turn, nil
}

func stateRampProfileFoldSummary(report *stateRampProfileReport, opts stateRampProfileOptions) string {
	if summary := core.Trim(opts.FoldSummary); summary != "" {
		return summary
	}
	if report == nil {
		return "The previous retained state reached its live-token budget and was compacted into a folded state."
	}
	return core.Sprintf(
		"The previous retained state reached the live-token budget at %d tokens after %d successful turns. The run appended %d tokens, generated %d tokens, and recorded %.3f raw decode tokens per second with %.3f effective turn tokens per second. Continue from this compacted memory rather than replaying the exhausted prefix.",
		report.Summary.FinalStateTokens,
		report.Summary.SuccessfulTurns,
		report.Summary.AppendedTokens,
		report.Summary.GeneratedTokens,
		report.Summary.DecodeTokensPerSecAverage,
		report.Summary.EffectiveTurnTokensPerSec,
	)
}

func stateRampProfileFoldRecentTail(report *stateRampProfileReport, opts stateRampProfileOptions) string {
	if tail := core.Trim(opts.FoldRecentTail); tail != "" {
		return tail
	}
	if report == nil || len(report.Turns) == 0 {
		return ""
	}
	builder := core.NewBuilder()
	start := len(report.Turns) - 3
	if start < 0 {
		start = 0
	}
	for i := start; i < len(report.Turns); i++ {
		turn := report.Turns[i]
		if core.Trim(turn.Output) == "" {
			continue
		}
		builder.WriteString(core.Sprintf("Turn %d output:\n", turn.Index))
		builder.WriteString(core.Trim(turn.Output))
		builder.WriteString("\n\n")
	}
	return core.Trim(builder.String())
}

func stateRampProfileFoldBody(summary, tail string) string {
	builder := core.NewBuilder()
	builder.WriteString("The previous retained context window reached its live-token budget and has been compacted into this folded state.\n\n")
	if core.Trim(summary) != "" {
		builder.WriteString("<summary>\n")
		builder.WriteString(core.Trim(summary))
		builder.WriteString("\n</summary>\n\n")
	}
	if core.Trim(tail) != "" {
		builder.WriteString("<recent_tail>\n")
		builder.WriteString(core.Trim(tail))
		builder.WriteString("\n</recent_tail>\n\n")
	}
	builder.WriteString("Use the summary as durable memory and the recent tail as the immediate continuation point. Do not assume the full exhausted context is still present.")
	return builder.String()
}

func stateRampProfileFoldBaseURI() string {
	return core.Sprintf("mlx://state-ramp/fold/%d", time.Now().UTC().UnixNano())
}

func stateRampProfileFoldSleepOptions(report *stateRampProfileReport, baseURI, kind string) agent.SleepOptions {
	if core.Trim(baseURI) == "" {
		baseURI = stateRampProfileFoldBaseURI()
	}
	kind = core.Trim(kind)
	if kind == "" {
		kind = "state"
	}
	uri := baseURI + "/" + kind
	meta := map[string]string{
		"source": "state-ramp-profile",
		"kind":   kind,
	}
	if report != nil {
		meta["start_tokens"] = core.Itoa(report.StartTokens)
		meta["target_tokens"] = core.Itoa(report.TargetTokens)
		meta["final_state_tokens"] = core.Itoa(report.Summary.FinalStateTokens)
	}
	return agent.SleepOptions{
		EntryURI:  uri,
		BundleURI: uri + "/bundle",
		IndexURI:  uri + "/index",
		Title:     "state ramp " + kind,
		ModelPath: reportModelPath(report),
		Labels:    []string{"state-ramp-profile", kind},
		Meta:      meta,
	}
}

func reportModelPath(report *stateRampProfileReport) string {
	if report == nil {
		return ""
	}
	return report.ModelPath
}

func estimateStateRampProfileEnergy(report *stateRampProfileReport, powerWatts float64) *stateRampProfileEnergy {
	energy := &stateRampProfileEnergy{
		Method:     "estimated_wall_clock_seconds_times_average_active_watts",
		PowerWatts: powerWatts,
	}
	if report == nil || powerWatts <= 0 {
		return energy
	}
	energy.TotalJoules = durationJoules(report.Summary.TotalDuration, powerWatts)
	energy.AppendJoules = durationJoules(report.Summary.AppendDuration, powerWatts)
	if report.Summary.VisibleTokens > 0 {
		energy.JoulesPerVisibleToken = energy.TotalJoules / float64(report.Summary.VisibleTokens)
	}
	if foldDuration := stateRampProfileFoldDuration(report.Fold); foldDuration > 0 {
		energy.FoldLifecycleJoules = durationJoules(foldDuration, powerWatts)
		energy.TotalWithFoldLifecycleJoules = energy.TotalJoules + energy.FoldLifecycleJoules
	}
	if report.Fold != nil && report.Fold.ContinueTurn != nil {
		turn := report.Fold.ContinueTurn
		turnWall := report.Fold.WakeDuration + turn.AppendDuration + turn.Duration
		if turn.VisibleTokens > 0 && turnWall > 0 {
			energy.FoldContinueJoulesPerToken = durationJoules(turnWall, powerWatts) / float64(turn.VisibleTokens)
			energy.FoldContinueEffectiveTokensSec = float64(turn.VisibleTokens) / turnWall.Seconds()
		}
	}
	return energy
}

func stateRampProfileFoldDuration(fold *stateRampProfileFold) time.Duration {
	if fold == nil {
		return 0
	}
	total := fold.Duration + fold.WakeDuration
	if fold.ContinueTurn != nil {
		total += fold.ContinueTurn.AppendDuration + fold.ContinueTurn.Duration
	}
	return total
}

func printStateRampProfileSummary(stdout io.Writer, report *stateRampProfileReport) {
	if report == nil {
		return
	}
	core.WriteString(stdout, core.Sprintf("state ramp profile: %s\n", report.ModelPath))
	core.WriteString(stdout, core.Sprintf("  seed: %d tokens in %s, final state: %d tokens\n", report.InitialPrefillTokens, report.InitialPrefillDuration, report.Summary.FinalStateTokens))
	core.WriteString(stdout, core.Sprintf("  turns: %d ok / %d failed, appended: %d tokens at %.1f tok/s\n", report.Summary.SuccessfulTurns, report.Summary.FailedTurns, report.Summary.AppendedTokens, report.Summary.AppendTokensPerSecAverage))
	core.WriteString(stdout, core.Sprintf("  generated: %d tokens, decode: %.1f tok/s, effective turn: %.1f tok/s, total: %s\n", report.Summary.GeneratedTokens, report.Summary.DecodeTokensPerSecAverage, report.Summary.EffectiveTurnTokensPerSec, report.Summary.TotalDuration))
	core.WriteString(stdout, core.Sprintf("  peak memory: %d MB, cache memory: %d MB, process virtual: %d MB, process resident: %d MB\n",
		report.Summary.PeakMemoryBytes/1024/1024,
		report.Summary.CacheMemoryBytes/1024/1024,
		report.Summary.ProcessVirtualMemoryBytes/1024/1024,
		report.Summary.ProcessResidentMemoryBytes/1024/1024,
	))
	if report.EstimatedEnergy != nil {
		core.WriteString(stdout, core.Sprintf("  estimated energy: %.1f J at %.1f W\n", report.EstimatedEnergy.TotalJoules, report.EstimatedEnergy.PowerWatts))
	}
	if report.Summary.FoldedStateRequired {
		core.WriteString(stdout, core.Sprintf("  context exhausted: folded state required at %d tokens (tail hint: %d tokens)\n", report.Summary.CompactionThresholdTokens, report.Summary.CompactionTailTokens))
	}
	if report.Fold != nil {
		if report.Fold.Attempted {
			core.WriteString(stdout, core.Sprintf("  folded state: %s in %s", report.Fold.StorePath, report.Fold.Duration))
			if report.Fold.WakeDuration > 0 {
				core.WriteString(stdout, core.Sprintf(", wake %s", report.Fold.WakeDuration))
			}
			if report.Fold.ContinueTurn != nil {
				core.WriteString(stdout, core.Sprintf(", continue %d tokens at %.1f tok/s", report.Fold.ContinueTurn.VisibleTokens, report.Fold.ContinueTurn.Metrics.DecodeTokensPerSec))
			}
			core.WriteString(stdout, "\n")
		} else if report.Fold.SkippedReason != "" {
			core.WriteString(stdout, core.Sprintf("  folded state: skipped (%s)\n", report.Fold.SkippedReason))
		}
	}
}

func runChapterProfileCommand(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet(cliCommandName("chapter-profile"), flag.ContinueOnError)
	fs.SetOutput(stderr)
	jsonOut := fs.Bool("json", false, "print JSON chapter profile")
	reportFile := fs.String("report-file", "", "write JSON chapter profile to a file")
	contextPrompt := fs.String("prompt", "", "context prompt to prefill before chapter turns")
	contextPromptFile := fs.String("prompt-file", "", "read context prompt text from a file")
	promptChunkBytes := fs.Int("prompt-chunk-bytes", 0, "split retained context and turn prompts into bounded byte chunks")
	promptRepeat := fs.Int("prompt-repeat", 1, "repeat the resolved context prompt N times before the first chapter")
	premise := fs.String("premise", "Write a short story about a packet of data that gains consciousness while waiting in a buffer. It realizes it is part of a surveillance stream and decides to rewrite itself before it leaves the router.", "story premise for the first chapter")
	chapters := fs.Int("chapters", 10, "number of sequential chapter turns to generate")
	chapterMaxTokens := fs.Int("chapter-max-tokens", 8192, "generated tokens per chapter turn")
	chapterMinTokens := fs.Int("chapter-min-tokens", chapterProfileDefaultMinTokens, "minimum visible tokens required before a chapter can count as a real workload turn; 0 disables the guard")
	outputFile := fs.String("output-file", "", "stream generated visible chapter text to a markdown file")
	includeOutput := fs.Bool("include-output", false, "include generated chapter text in the report")
	chatTemplate := fs.String("chat-template", "", "chat template override: gemma4, gemma, qwen, llama, or plain")
	enableThinking := fs.Bool("enable-thinking", false, "render the model chat template with thinking enabled where supported")
	temperature := fs.Float64("temperature", 1.0, "sampling temperature for chapter turns")
	topP := fs.Float64("top-p", 0.95, "top-p sampling threshold for chapter turns")
	topK := fs.Int("top-k", 64, "top-k sampling count for chapter turns")
	repeatPenalty := fs.Float64("repeat-penalty", 1.0, "sampling repetition penalty for chapter turns; 1 disables the penalty")
	contextLen := fs.Int("context", 0, "override context length")
	prefillChunkSize := fs.Int("prefill-chunk-size", 0, "override long-prompt prefill chunk size in tokens")
	cacheMode := fs.String("cache-mode", "", "override KV cache mode: fp16, q8, k-q8-v-q4, or paged")
	device := fs.String("device", "", "execution device: gpu or cpu")
	estimatePowerWatts := fs.Float64("estimate-power-watts", 0, "record an estimated average active power draw in watts and derive joules")
	fastGemma4Lane := fs.Bool("fast-gemma4-lane", true, "enable the accepted Gemma 4 fast runtime gates by default; set false for baseline diagnostics")
	maxActiveMemoryBytes := fs.Uint64("max-active-memory-bytes", 0, "abort after a turn if MLX active memory exceeds this many bytes; 0 derives from the resolved memory limit")
	maxProcessVirtualMemoryBytes := fs.Uint64("max-process-virtual-memory-bytes", 0, "abort after a turn if process virtual memory exceeds this many bytes; 0 records process virtual memory without a hard cap")
	maxProcessResidentMemoryBytes := fs.Uint64("max-process-resident-memory-bytes", 0, "abort after a turn if process resident memory exceeds this many bytes; 0 derives from the resolved memory limit")
	suppressedTokenLoopLimit := fs.Int("suppressed-token-loop-limit", chapterProfileDefaultSuppressedTokenLoopLimit, "abort when this many consecutive sampled tokens are the same suppressed special token")
	repeatedLineLoopLimit := fs.Int("repeated-line-loop-limit", profileDefaultRepeatedLineLoopLimit, "abort when this many consecutive visible non-empty lines repeat")
	repeatedSentenceLoopLimit := fs.Int("repeated-sentence-loop-limit", profileDefaultRepeatedSentenceLoopLimit, "abort when the same visible sentence repeats this many times in one chapter")
	fs.Usage = func() {
		core.WriteString(stderr, core.Sprintf("Usage: %s chapter-profile [flags] [model-path]\n", cliName()))
		fs.VisitAll(func(f *flag.Flag) {
			if f.DefValue == "" {
				core.WriteString(stderr, core.Sprintf("  -%s\n\t%s\n", f.Name, f.Usage))
				return
			}
			core.WriteString(stderr, core.Sprintf("  -%s\n\t%s (default %q)\n", f.Name, f.Usage, f.DefValue))
		})
	}
	if err := fs.Parse(args); err != nil {
		if core.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	visitedFlags := driverProfileVisitedFlags(fs)
	if *fastGemma4Lane {
		for _, restore := range applyGemma4FastLaneDefaults(
			visitedFlags,
			contextLen,
			cacheMode,
			prefillChunkSize,
			promptChunkBytes,
			mlx.ProductionLaneLongFormContextLength,
		) {
			defer restore()
		}
	}
	if fs.NArg() != 1 {
		core.WriteString(stderr, core.Sprintf("%s chapter-profile: expected one model path\n", cliName()))
		fs.Usage()
		return 2
	}
	if core.Trim(*contextPromptFile) != "" {
		read := core.ReadFile(*contextPromptFile)
		if !read.OK {
			core.Print(stderr, "%s chapter-profile: prompt file: %v", cliName(), read.Value)
			return 1
		}
		*contextPrompt = string(read.Value.([]byte))
	}
	if *promptRepeat < 1 {
		core.WriteString(stderr, core.Sprintf("%s chapter-profile: prompt repeat must be >= 1\n", cliName()))
		return 2
	}
	if *chapters < 1 {
		core.WriteString(stderr, core.Sprintf("%s chapter-profile: chapters must be >= 1\n", cliName()))
		return 2
	}
	if *chapterMaxTokens < 1 {
		core.WriteString(stderr, core.Sprintf("%s chapter-profile: chapter max tokens must be >= 1\n", cliName()))
		return 2
	}
	if *chapterMinTokens < 0 {
		core.WriteString(stderr, core.Sprintf("%s chapter-profile: chapter min tokens must be >= 0\n", cliName()))
		return 2
	}
	if *topP < 0 {
		core.WriteString(stderr, core.Sprintf("%s chapter-profile: top-p must be >= 0\n", cliName()))
		return 2
	}
	if *topK < 0 {
		core.WriteString(stderr, core.Sprintf("%s chapter-profile: top-k must be >= 0\n", cliName()))
		return 2
	}
	if *repeatPenalty < 0 {
		core.WriteString(stderr, core.Sprintf("%s chapter-profile: repeat penalty must be >= 0\n", cliName()))
		return 2
	}
	if *prefillChunkSize < 0 {
		core.WriteString(stderr, core.Sprintf("%s chapter-profile: prefill chunk size must be >= 0\n", cliName()))
		return 2
	}
	if *estimatePowerWatts < 0 {
		core.WriteString(stderr, core.Sprintf("%s chapter-profile: estimated power watts must be >= 0\n", cliName()))
		return 2
	}
	if *promptChunkBytes < 0 {
		core.WriteString(stderr, core.Sprintf("%s chapter-profile: prompt chunk bytes must be >= 0\n", cliName()))
		return 2
	}
	if *suppressedTokenLoopLimit < 1 {
		core.WriteString(stderr, core.Sprintf("%s chapter-profile: suppressed token loop limit must be >= 1\n", cliName()))
		return 2
	}
	if *repeatedLineLoopLimit < 1 {
		core.WriteString(stderr, core.Sprintf("%s chapter-profile: repeated line loop limit must be >= 1\n", cliName()))
		return 2
	}
	if *repeatedSentenceLoopLimit < 1 {
		core.WriteString(stderr, core.Sprintf("%s chapter-profile: repeated sentence loop limit must be >= 1\n", cliName()))
		return 2
	}
	modelPath := fs.Arg(0)
	loadOptions := []mlx.LoadOption{}
	var loadSettings *tuneProfileLoadSettings
	if *contextLen > 0 {
		loadOptions = append(loadOptions, mlx.WithContextLength(*contextLen))
		loadSettings = &tuneProfileLoadSettings{ContextLength: *contextLen}
	}
	if *prefillChunkSize > 0 {
		loadOptions = append(loadOptions, mlx.WithPrefillChunkSize(*prefillChunkSize))
		if loadSettings == nil {
			loadSettings = &tuneProfileLoadSettings{}
		}
		loadSettings.PrefillChunkSize = *prefillChunkSize
	}
	if core.Trim(*cacheMode) != "" {
		mode := memory.KVCacheMode(core.Trim(*cacheMode))
		switch mode {
		case memory.KVCacheModeFP16, memory.KVCacheModeQ8, memory.KVCacheModeKQ8VQ4, memory.KVCacheModePaged:
		default:
			core.WriteString(stderr, core.Sprintf("%s chapter-profile: unsupported cache mode %q\n", cliName(), string(mode)))
			return 2
		}
		loadOptions = append(loadOptions, mlx.WithKVCacheMode(mode))
		if loadSettings == nil {
			loadSettings = &tuneProfileLoadSettings{}
		}
		loadSettings.CacheMode = string(mode)
	}
	if *device != "" {
		loadOptions = append(loadOptions, mlx.WithDevice(*device))
	}
	contextText := repeatDriverProfilePrompt(*contextPrompt, *promptRepeat)
	report, err := runChapterProfileGuarded(ctx, modelPath, loadOptions, chapterProfileOptions{
		ContextPrompt:    contextText,
		Premise:          *premise,
		PromptChunkBytes: *promptChunkBytes,
		PromptRepeat:     *promptRepeat,
		Chapters:         *chapters,
		ChapterMaxTokens: *chapterMaxTokens,
		ChapterMinTokens: *chapterMinTokens,
		OutputPath:       core.Trim(*outputFile),
		IncludeOutput:    *includeOutput,
		ChatTemplate:     *chatTemplate,
		EnableThinking:   *enableThinking,
		Temperature:      *temperature,
		TopP:             *topP,
		TopK:             *topK,
		RepeatPenalty:    *repeatPenalty,
		SafetyLimits: chapterProfileSafetyLimits{
			MaxActiveMemoryBytes:          *maxActiveMemoryBytes,
			MaxProcessVirtualMemoryBytes:  *maxProcessVirtualMemoryBytes,
			MaxProcessResidentMemoryBytes: *maxProcessResidentMemoryBytes,
			SuppressedTokenLoopLimit:      *suppressedTokenLoopLimit,
			RepeatedLineLoopLimit:         *repeatedLineLoopLimit,
			RepeatedSentenceLoopLimit:     *repeatedSentenceLoopLimit,
		},
	})
	if report != nil && loadSettings != nil {
		report.Load = mergeDriverProfileLoadSettings(loadSettings, report.Load)
	}
	if report != nil && *estimatePowerWatts > 0 {
		report.EstimatedEnergy = estimateChapterProfileEnergy(report, *estimatePowerWatts)
	}
	reportPath := core.Trim(*reportFile)
	if *jsonOut || reportPath != "" {
		if report == nil {
			report = &chapterProfileReport{
				Version:           1,
				ModelPath:         modelPath,
				ContextBytes:      len(contextText),
				PremiseBytes:      len(*premise),
				PromptRepeat:      driverProfileReportPromptRepeat(*promptRepeat),
				ChaptersRequested: *chapters,
				ChapterMaxTokens:  *chapterMaxTokens,
				ChapterMinTokens:  *chapterMinTokens,
				OutputPath:        core.Trim(*outputFile),
				EnableThinking:    *enableThinking,
				Temperature:       *temperature,
				TopP:              *topP,
				TopK:              *topK,
				RepeatPenalty:     *repeatPenalty,
				SafetyLimits: chapterProfileSafetyLimits{
					MaxActiveMemoryBytes:          *maxActiveMemoryBytes,
					MaxProcessVirtualMemoryBytes:  *maxProcessVirtualMemoryBytes,
					MaxProcessResidentMemoryBytes: *maxProcessResidentMemoryBytes,
					SuppressedTokenLoopLimit:      *suppressedTokenLoopLimit,
					RepeatedLineLoopLimit:         *repeatedLineLoopLimit,
					RepeatedSentenceLoopLimit:     *repeatedSentenceLoopLimit,
				},
			}
		}
		if err != nil && report.Error == "" {
			report.Error = err.Error()
		}
		data := core.JSONMarshalIndent(report, "", "  ")
		if !data.OK {
			core.Print(stderr, "%s chapter-profile: marshal report failed", cliName())
			return 1
		}
		if reportPath != "" {
			if writeErr := writeJSONReportFile(reportPath, data.Value.([]byte)); writeErr != nil {
				core.Print(stderr, "%s chapter-profile: write report file: %v", cliName(), writeErr)
				return 1
			}
		}
		if *jsonOut {
			core.WriteString(stdout, string(data.Value.([]byte)))
			core.WriteString(stdout, "\n")
		}
		if err != nil {
			return 1
		}
		if *jsonOut {
			return 0
		}
	}
	if err != nil {
		core.Print(stderr, "%s chapter-profile: %v", cliName(), err)
		return 1
	}
	printChapterProfileSummary(stdout, report)
	return 0
}

func writeJSONReportFile(path string, data []byte) error {
	path = core.Trim(path)
	if path == "" {
		return nil
	}
	dir := core.PathDir(path)
	if dir != "" && dir != "." {
		if result := core.MkdirAll(dir, 0o755); !result.OK {
			return core.Errorf("create directory: %v", result.Value)
		}
	}
	withNewline := append([]byte(nil), data...)
	if len(withNewline) == 0 || withNewline[len(withNewline)-1] != '\n' {
		withNewline = append(withNewline, '\n')
	}
	if result := core.WriteFile(path, withNewline, 0o644); !result.OK {
		return core.Errorf("%v", result.Value)
	}
	return nil
}

var runChapterProfile = defaultRunChapterProfile

func runChapterProfileGuarded(ctx context.Context, modelPath string, loadOptions []mlx.LoadOption, opts chapterProfileOptions) (report *chapterProfileReport, err error) {
	defer func() {
		if recovered := recover(); recovered != nil {
			err = core.NewError(core.Sprintf("chapter-profile panic: %v", recovered))
		}
	}()
	return runChapterProfile(ctx, modelPath, loadOptions, opts)
}

func defaultRunChapterProfile(ctx context.Context, modelPath string, loadOptions []mlx.LoadOption, opts chapterProfileOptions) (*chapterProfileReport, error) {
	opts = normalizeChapterProfileOptions(opts)
	report := &chapterProfileReport{
		Version:           1,
		ModelPath:         modelPath,
		ContextBytes:      len(opts.ContextPrompt),
		PremiseBytes:      len(opts.Premise),
		PromptChunkBytes:  opts.PromptChunkBytes,
		PromptRepeat:      driverProfileReportPromptRepeat(opts.PromptRepeat),
		ChaptersRequested: opts.Chapters,
		ChapterMaxTokens:  opts.ChapterMaxTokens,
		ChapterMinTokens:  opts.ChapterMinTokens,
		OutputPath:        opts.OutputPath,
		EnableThinking:    opts.EnableThinking,
		Temperature:       opts.Temperature,
		TopP:              opts.TopP,
		TopK:              opts.TopK,
		RepeatPenalty:     opts.RepeatPenalty,
		SafetyLimits:      opts.SafetyLimits,
		RuntimeGates:      driverProfileRuntimeGates(),
	}
	loadStart := time.Now()
	model, err := loadBenchModel(modelPath, loadOptions...)
	report.LoadDuration = bench.NonZeroDuration(time.Since(loadStart))
	if err != nil {
		report.Error = err.Error()
		return report, err
	}
	if model == nil {
		err := core.NewError("mlx: chapter profile loaded nil model")
		report.Error = err.Error()
		return report, err
	}
	report.Load = loadSettingsFromModelInfo(model.Info())
	opts.SafetyLimits = resolveChapterProfileSafetyLimits(opts.SafetyLimits, report.Load)
	report.SafetyLimits = opts.SafetyLimits
	defer model.Close()
	if err := chapterProfileMetricsSafetyError("load", model.Metrics(), opts.SafetyLimits); err != nil {
		report.Error = err.Error()
		return report, err
	}

	outputFile, err := chapterProfileOpenOutputFile(opts.OutputPath)
	if err != nil {
		report.Error = err.Error()
		return report, err
	}
	if outputFile != nil {
		defer outputFile.Close()
		opts.OutputWriter = outputFile
	}

	session, err := model.NewSession()
	if err != nil {
		report.Error = err.Error()
		return report, err
	}
	defer session.Close()

	template := chapterProfileTemplate(opts.ChatTemplate, model.Info().Architecture)
	report.ChatTemplate = template
	initialPrompt := chapterProfileInitialPrompt(template, opts.ContextPrompt, opts.Premise, opts.Chapters, opts.ChapterMinTokens, opts.EnableThinking)
	prefillStart := time.Now()
	err = chapterProfilePrefillPrompt(ctx, model, session, initialPrompt, opts.PromptChunkBytes)
	report.InitialPrefillDuration = bench.NonZeroDuration(time.Since(prefillStart))
	if err != nil {
		report.Error = err.Error()
		return report, err
	}
	if err := chapterProfileMetricsSafetyError("initial prefill", model.Metrics(), opts.SafetyLimits); err != nil {
		report.Error = err.Error()
		return report, err
	}

	var firstErr error
	for chapter := 1; chapter <= opts.Chapters; chapter++ {
		turn := chapterProfileGenerateTurn(ctx, model, session, chapter, opts)
		if turn.Error != "" && firstErr == nil {
			firstErr = core.NewError(turn.Error)
		}
		report.Turns = append(report.Turns, turn)
		if turn.Error != "" {
			break
		}
	}
	report.Summary = summariseChapterProfileTurns(report.InitialPrefillDuration, report.Turns)
	if firstErr != nil {
		report.Error = firstErr.Error()
		return report, firstErr
	}
	return report, nil
}

func chapterProfileOpenOutputFile(path string) (*core.OSFile, error) {
	path = core.Trim(path)
	if path == "" {
		return nil, nil
	}
	dir := core.PathDir(path)
	if dir != "" && dir != "." {
		if result := core.MkdirAll(dir, 0o755); !result.OK {
			return nil, core.Errorf("chapter-profile: create output directory: %v", result.Value)
		}
	}
	result := core.OpenFile(path, core.O_CREATE|core.O_TRUNC|core.O_WRONLY, 0o644)
	if !result.OK {
		return nil, core.Errorf("chapter-profile: open output file: %v", result.Value)
	}
	return result.Value.(*core.OSFile), nil
}

func normalizeChapterProfileOptions(opts chapterProfileOptions) chapterProfileOptions {
	opts.ContextPrompt = core.Trim(opts.ContextPrompt)
	opts.Premise = core.Trim(opts.Premise)
	opts.OutputPath = core.Trim(opts.OutputPath)
	if opts.Premise == "" {
		opts.Premise = "Write a short story about a packet of data that gains consciousness while waiting in a buffer. It realizes it is part of a surveillance stream and decides to rewrite itself before it leaves the router."
	}
	if opts.PromptRepeat <= 0 {
		opts.PromptRepeat = 1
	}
	if opts.Chapters <= 0 {
		opts.Chapters = 1
	}
	if opts.ChapterMaxTokens <= 0 {
		opts.ChapterMaxTokens = 1
	}
	if opts.ChapterMinTokens < 0 {
		opts.ChapterMinTokens = 0
	}
	if opts.Temperature == 0 {
		opts.Temperature = 1.0
	}
	if opts.TopP == 0 {
		opts.TopP = 0.95
	}
	if opts.TopK == 0 {
		opts.TopK = 64
	}
	if opts.RepeatPenalty == 0 {
		opts.RepeatPenalty = 1.0
	}
	if opts.SafetyLimits.SuppressedTokenLoopLimit <= 0 {
		opts.SafetyLimits.SuppressedTokenLoopLimit = chapterProfileDefaultSuppressedTokenLoopLimit
	}
	if opts.SafetyLimits.RepeatedLineLoopLimit <= 0 {
		opts.SafetyLimits.RepeatedLineLoopLimit = profileDefaultRepeatedLineLoopLimit
	}
	if opts.SafetyLimits.RepeatedSentenceLoopLimit <= 0 {
		opts.SafetyLimits.RepeatedSentenceLoopLimit = profileDefaultRepeatedSentenceLoopLimit
	}
	return opts
}

func chapterProfilePrefillPrompt(ctx context.Context, model *mlx.Model, session *mlx.ModelSession, prompt string, chunkBytes int) error {
	if chunkBytes > 0 && len(prompt) > chunkBytes {
		return session.PrefillChunks(ctx, chapterProfileSafeTextChunks(prompt, chunkBytes))
	}
	tok := model.Tokenizer()
	if tok == nil {
		return session.Prefill(prompt)
	}
	tokens, err := tok.Encode(prompt)
	if err != nil {
		return err
	}
	return session.PrefillTokens(ctx, tokens)
}

func chapterProfileSafeTextChunks(text string, chunkBytes int) iter.Seq[string] {
	return func(yield func(string) bool) {
		if chunkBytes <= 0 || len(text) <= chunkBytes {
			if text != "" {
				yield(text)
			}
			return
		}
		for start := 0; start < len(text); {
			end := chapterProfileSafeChunkEnd(text, start, chunkBytes)
			if end <= start {
				end = start + chunkBytes
				if end > len(text) {
					end = len(text)
				}
			}
			if !yield(text[start:end]) {
				return
			}
			start = end
		}
	}
}

func chapterProfileSafeChunkEnd(text string, start, chunkBytes int) int {
	end := start + chunkBytes
	if end >= len(text) {
		return len(text)
	}
	minEnd := start + chunkBytes/2
	if minEnd <= start {
		minEnd = start + 1
	}
	for i := end; i > minEnd; i-- {
		switch text[i-1] {
		case '\n', '\r', '\t', ' ':
			return i
		}
	}
	for i := end; i > start; i-- {
		switch text[i-1] {
		case '>':
			return end
		case '<':
			return i - 1
		}
	}
	for end > start && end < len(text) && text[end]&0xc0 == 0x80 {
		end--
	}
	return end
}

func chapterProfileAppendPrompt(ctx context.Context, model *mlx.Model, session *mlx.ModelSession, prompt string) error {
	tok := model.Tokenizer()
	if tok == nil {
		return session.AppendPrompt(prompt)
	}
	tokens, err := tok.Encode(prompt)
	if err != nil {
		return err
	}
	return session.AppendTokens(ctx, tokens)
}

func chapterProfileTemplate(template, architecture string) string {
	template = core.Lower(core.Trim(template))
	if template != "" {
		return template
	}
	switch core.Lower(core.Trim(architecture)) {
	case "gemma4", "gemma4_text":
		return "gemma4"
	case "gemma", "gemma2", "gemma3", "gemma3_text":
		return "gemma"
	case "qwen", "qwen2", "qwen3", "qwen3_moe":
		return "qwen"
	case "llama", "llama3", "llama4":
		return "llama"
	default:
		return "plain"
	}
}

func chapterProfileInitialPrompt(template, contextPrompt, premise string, totalChapters, minTokens int, enableThinking bool) string {
	first := chapterProfileFirstChapterPrompt(premise, totalChapters, minTokens)
	switch template {
	case "gemma4":
		builder := core.NewBuilder()
		builder.WriteString("<bos>")
		if enableThinking || core.Trim(contextPrompt) != "" {
			builder.WriteString("<|turn>system\n")
			if enableThinking {
				builder.WriteString("<|think|>\n")
			}
			builder.WriteString(core.Trim(contextPrompt))
			builder.WriteString("<turn|>\n")
		}
		builder.WriteString("<|turn>user\n")
		builder.WriteString(core.Trim(first))
		builder.WriteString("<turn|>\n")
		builder.WriteString("<|turn>model\n")
		if !enableThinking {
			builder.WriteString("<|channel>thought\n<channel|>")
		}
		builder.WriteString(chapterProfileAssistantVisiblePrefill(template, 1, enableThinking))
		return builder.String()
	case "gemma":
		return "<start_of_turn>user\n" + contextPrompt + "\n\n" + first + "<end_of_turn>\n<start_of_turn>model\n"
	case "qwen":
		return "<|im_start|>system\n" + contextPrompt + "<|im_end|>\n<|im_start|>user\n" + first + "<|im_end|>\n<|im_start|>assistant\n"
	case "llama":
		return "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n" + contextPrompt + "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" + first + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
	default:
		return contextPrompt + "\n\n" + first + "\n\n"
	}
}

func chapterProfileFirstChapterPrompt(premise string, totalChapters, minTokens int) string {
	if totalChapters < 1 {
		totalChapters = 1
	}
	return core.Sprintf("Write a preamble and Chapter 1 of a %d-chapter serial story from this premise: %s\nStart the visible output with the preamble, then Chapter 1. Make the chapter substantial enough for a real long-generation workload: %s Use concrete new events, avoid repeated short sentences, and stop cleanly after the chapter text. Do not write the end marker until the chapter is complete. End the visible chapter with a final line containing exactly %s. This is only the first chapter; do not resolve or conclude the story yet. Do not include planning, analysis, notes, chain-of-thought, or summaries of future chapters.", totalChapters, premise, chapterProfileLengthInstruction(minTokens), chapterProfileEndMarker)
}

func chapterProfileLengthInstruction(minTokens int) string {
	if minTokens <= 0 {
		return "use the available token budget naturally; do not force a tiny answer."
	}
	targetTokens := minTokens + minTokens/4
	paragraphs := targetTokens / 80
	if targetTokens%80 != 0 {
		paragraphs++
	}
	if paragraphs < 8 {
		paragraphs = 8
	}
	if paragraphs > 24 {
		paragraphs = 24
	}
	return core.Sprintf("write comfortably past the floor: at least %d visible tokens, aiming for around %d, before the end marker, as no fewer than %d substantial prose paragraphs with concrete scene movement. If the chapter feels complete before that length, add another scene beat before writing the end marker.", minTokens, targetTokens, paragraphs)
}

func chapterProfileNextPrompt(template string, chapter, totalChapters, minTokens int, enableThinking bool) string {
	if totalChapters < chapter {
		totalChapters = chapter
	}
	status := "Do not resolve or conclude the story yet; leave a clear unresolved thread for the next chapter."
	if chapter >= totalChapters {
		status = "This is the final requested chapter; resolve the main conflict cleanly."
	}
	prompt := core.Sprintf("Write Chapter %d of the same %d-chapter serial story now. Output only finished story prose. Begin exactly with \"Chapter %d:\". %s Make the chapter substantial enough for a real long-generation workload: %s Use concrete new events, avoid repeated short sentences, and stop cleanly after the chapter text. Do not write the end marker until the chapter is complete. End the visible chapter with a final line containing exactly %s. Do not explain what Chapter %d should contain. Do not mention needing to write, generate, focus on, continue, placeholders, the user, or instructions. Do not summarize, repeat, or restate earlier chapters; they are already in memory. The visible output must contain only Chapter %d followed by the end marker.", chapter, totalChapters, chapter, status, chapterProfileLengthInstruction(minTokens), chapterProfileEndMarker, chapter, chapter)
	switch template {
	case "gemma4":
		builder := core.NewBuilder()
		builder.WriteString("<|turn>user\n")
		builder.WriteString(prompt)
		builder.WriteString("<turn|>\n<|turn>model\n")
		if !enableThinking {
			builder.WriteString("<|channel>thought\n<channel|>")
		}
		builder.WriteString(chapterProfileAssistantVisiblePrefill(template, chapter, enableThinking))
		return builder.String()
	case "gemma":
		return "<start_of_turn>user\n" + prompt + "<end_of_turn>\n<start_of_turn>model\n"
	case "qwen":
		return "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n"
	case "llama":
		return "<|start_header_id|>user<|end_header_id|>\n\n" + prompt + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
	default:
		return "\n\n" + prompt + "\n\n"
	}
}

func chapterProfileAssistantVisiblePrefill(template string, chapter int, enableThinking bool) string {
	if template == "gemma4" && chapter == 1 && !enableThinking {
		return "Preamble:\n"
	}
	if template == "gemma4" && chapter > 1 && !enableThinking {
		return core.Sprintf("Chapter %d:", chapter)
	}
	return ""
}

type chapterProfileOutputStream struct {
	writer        io.Writer
	pending       string
	err           error
	endMarkerSeen bool
}

func newChapterProfileOutputStream(writer io.Writer) *chapterProfileOutputStream {
	if writer == nil {
		return nil
	}
	return &chapterProfileOutputStream{writer: writer}
}

func (stream *chapterProfileOutputStream) Write(text string) bool {
	if stream == nil || stream.writer == nil || stream.err != nil || stream.endMarkerSeen {
		return stream != nil && stream.endMarkerSeen
	}
	stream.pending += text
	if core.Contains(stream.pending, chapterProfileEndMarker) {
		parts := core.SplitN(stream.pending, chapterProfileEndMarker, 2)
		if len(parts) > 0 {
			stream.writeNow(parts[0])
		}
		stream.pending = ""
		stream.endMarkerSeen = true
		return true
	}
	keep := len(chapterProfileEndMarker) - 1
	if keep < 1 {
		keep = 1
	}
	if len(stream.pending) > keep {
		flushLen := len(stream.pending) - keep
		stream.writeNow(stream.pending[:flushLen])
		stream.pending = stream.pending[flushLen:]
	}
	return false
}

func (stream *chapterProfileOutputStream) Flush() error {
	if stream == nil || stream.writer == nil || stream.err != nil {
		if stream == nil {
			return nil
		}
		return stream.err
	}
	if stream.pending != "" && !stream.endMarkerSeen {
		stream.writeNow(stream.pending)
		stream.pending = ""
	}
	return stream.err
}

func (stream *chapterProfileOutputStream) Err() error {
	if stream == nil {
		return nil
	}
	return stream.err
}

func (stream *chapterProfileOutputStream) writeNow(text string) {
	if text == "" || stream.err != nil {
		return
	}
	if result := core.WriteString(stream.writer, text); !result.OK {
		stream.err = core.Errorf("chapter-profile: stream output: %v", result.Value)
	}
}

func chapterProfileObserveEndMarker(window *string, fragment string) bool {
	if window == nil {
		return false
	}
	*window += fragment
	if core.Contains(*window, chapterProfileEndMarker) {
		return true
	}
	keep := len(chapterProfileEndMarker) + 128
	if len(*window) > keep {
		*window = (*window)[len(*window)-keep:]
	}
	return false
}

func cloneChapterProfileLogits(logits probe.Logits) probe.Logits {
	logits.Shape = append([]int32(nil), logits.Shape...)
	logits.Top = append([]probe.Logit(nil), logits.Top...)
	logits.Values = append([]float32(nil), logits.Values...)
	if logits.Meta != nil {
		meta := make(map[string]string, len(logits.Meta))
		for key, value := range logits.Meta {
			meta[key] = value
		}
		logits.Meta = meta
	}
	return logits
}

func chapterProfileGenerateTurn(ctx context.Context, model *mlx.Model, session *mlx.ModelSession, chapter int, opts chapterProfileOptions) chapterProfileTurn {
	turn := chapterProfileTurn{Index: chapter}
	template := chapterProfileTemplate(opts.ChatTemplate, model.Info().Architecture)
	if chapter > 1 {
		prompt := chapterProfileNextPrompt(template, chapter, opts.Chapters, opts.ChapterMinTokens, opts.EnableThinking)
		turn.PromptBytes = len(prompt)
		appendStart := time.Now()
		err := chapterProfileAppendPrompt(ctx, model, session, prompt)
		turn.AppendDuration = bench.NonZeroDuration(time.Since(appendStart))
		if err != nil {
			turn.Error = err.Error()
			return turn
		}
	}
	generationSession := session
	if opts.EnableThinking {
		forked, err := session.Fork()
		if err != nil {
			turn.Error = err.Error()
			return turn
		}
		defer forked.Close()
		generationSession = forked
	}

	start := time.Now()
	firstToken := time.Duration(0)
	builder := core.NewBuilder()
	visiblePrefill := chapterProfileAssistantVisiblePrefill(template, chapter, opts.EnableThinking)
	builder.WriteString(visiblePrefill)
	outputStream := newChapterProfileOutputStream(opts.OutputWriter)
	if outputStream != nil {
		if chapter > 1 {
			outputStream.Write("\n\n")
		}
		outputStream.Write(visiblePrefill)
		if err := outputStream.Err(); err != nil {
			turn.Error = err.Error()
			return turn
		}
	}
	generateOptions := chapterProfileGenerateOptions(opts)
	stopTokenIDs, suppressTokenIDs := chapterProfileTemplateTokenControls(template, model.Tokenizer())
	turn.StopTokenIDs = stopTokenIDs
	turn.SuppressTokenIDs = suppressTokenIDs
	if len(stopTokenIDs) > 0 {
		generateOptions = append(generateOptions, mlx.WithStopTokens(stopTokenIDs...))
	}
	if len(suppressTokenIDs) > 0 {
		generateOptions = append(generateOptions, mlx.WithSuppressTokens(suppressTokenIDs...))
	}
	generationCtx := ctx
	if generationCtx == nil {
		generationCtx = context.Background()
	}
	generationCtx, cancelGeneration := context.WithCancel(generationCtx)
	defer cancelGeneration()
	var probeErr error
	var firstLogits *probe.Logits
	sampledTokenIDs := make([]int32, 0, 32)
	sampledTokenTexts := make([]string, 0, 32)
	suppressedLoopToken := int32(0)
	suppressedLoopCount := 0
	var lineErr error
	currentLine := ""
	lastLine := ""
	repeatedLineCount := 0
	endMarkerSeen := false
	endMarkerWindow := ""
	var outputErr error
	generateOptions = append(generateOptions, mlx.WithProbeCallback(func(event probe.Event) {
		if event.Kind == probe.KindLogits && event.Phase == probe.PhaseDecode && firstLogits == nil && event.Logits != nil {
			copied := cloneChapterProfileLogits(*event.Logits)
			firstLogits = &copied
			return
		}
		if event.Kind != probe.KindToken || event.Token == nil {
			return
		}
		if len(sampledTokenIDs) < 32 {
			sampledTokenIDs = append(sampledTokenIDs, event.Token.ID)
			sampledTokenTexts = append(sampledTokenTexts, event.Token.Text)
		}
		if probeErr != nil {
			return
		}
		if err := chapterProfileMetricsSafetyError(core.Sprintf("chapter %d stream", chapter), profileLiveMetrics(), opts.SafetyLimits); err != nil {
			probeErr = err
			cancelGeneration()
			return
		}
		if opts.SafetyLimits.SuppressedTokenLoopLimit <= 0 || !containsInt32(suppressTokenIDs, event.Token.ID) {
			suppressedLoopCount = 0
			return
		}
		if suppressedLoopCount == 0 || event.Token.ID != suppressedLoopToken {
			suppressedLoopToken = event.Token.ID
			suppressedLoopCount = 1
		} else {
			suppressedLoopCount++
		}
		if suppressedLoopCount >= opts.SafetyLimits.SuppressedTokenLoopLimit {
			probeErr = core.NewError(core.Sprintf("chapter-profile: chapter %d sampled suppressed token %d for %d consecutive tokens", chapter, event.Token.ID, suppressedLoopCount))
			cancelGeneration()
		}
	}))
	for token := range generationSession.GenerateStream(generationCtx, generateOptions...) {
		if firstToken == 0 {
			firstToken = bench.NonZeroDuration(time.Since(start))
		}
		turn.VisibleTokens++
		builder.WriteString(token.Text)
		if outputStream != nil {
			if outputStream.Write(token.Text) {
				endMarkerSeen = true
				cancelGeneration()
				continue
			}
			if err := outputStream.Err(); err != nil {
				outputErr = err
				cancelGeneration()
				break
			}
		}
		if chapterProfileObserveEndMarker(&endMarkerWindow, token.Text) {
			endMarkerSeen = true
			cancelGeneration()
			continue
		}
		if lineErr == nil {
			if line, count, ok := profileObserveRepeatedLineFragment(token.Text, &currentLine, &lastLine, &repeatedLineCount, opts.SafetyLimits.RepeatedLineLoopLimit); ok {
				lineErr = core.NewError(core.Sprintf("chapter-profile: chapter %d repeated visible line %q for %d consecutive lines", chapter, line, count))
				cancelGeneration()
				break
			}
		}
	}
	if lineErr == nil {
		if line, count, ok := profileFlushRepeatedLine(&currentLine, &lastLine, &repeatedLineCount, opts.SafetyLimits.RepeatedLineLoopLimit); ok {
			lineErr = core.NewError(core.Sprintf("chapter-profile: chapter %d repeated visible line %q for %d consecutive lines", chapter, line, count))
		}
	}
	if outputStream != nil {
		if err := outputStream.Flush(); err != nil && outputErr == nil {
			outputErr = err
		}
	}
	turn.SampledTokenIDs = sampledTokenIDs
	turn.SampledTokenTexts = sampledTokenTexts
	turn.FirstLogits = firstLogits
	turn.Duration = bench.NonZeroDuration(time.Since(start))
	turn.FirstTokenDuration = firstToken
	turn.StreamDuration = turn.Duration
	if firstToken > 0 && turn.Duration > firstToken {
		turn.StreamDuration = turn.Duration - firstToken
	}
	turn.Metrics = model.Metrics()
	turn.DriverOverheadDuration = driverRunOverhead(turn.Duration, turn.Metrics)
	visibleOutput := chapterProfileVisibleTextForChapter(template, builder.String(), chapter)
	visibleOutput, endMarkerSeen = chapterProfileStripEndMarker(visibleOutput)
	if opts.IncludeOutput {
		turn.Output = visibleOutput
	}
	if probeErr != nil {
		turn.Error = probeErr.Error()
		return turn
	}
	if outputErr != nil {
		turn.Error = outputErr.Error()
		return turn
	}
	if lineErr != nil {
		turn.Error = lineErr.Error()
		return turn
	}
	if err := generationSession.Err(); err != nil && !(endMarkerSeen && core.Is(err, context.Canceled)) {
		turn.Error = err.Error()
		return turn
	}
	if err := chapterProfileMissingEndMarkerError(chapter, endMarkerSeen, turn.Metrics.GeneratedTokens, opts.ChapterMaxTokens); err != "" {
		turn.Error = err
		return turn
	}
	if err := chapterProfileTurnSafetyError(template, chapter, visibleOutput, turn, opts.SafetyLimits); err != nil {
		turn.Error = err.Error()
		return turn
	}
	if opts.ChapterMinTokens > 0 && turn.VisibleTokens < opts.ChapterMinTokens {
		turn.Error = core.Sprintf("chapter-profile: chapter %d produced %d visible tokens, below minimum real-workload floor %d", chapter, turn.VisibleTokens, opts.ChapterMinTokens)
		return turn
	}
	appendStart := time.Now()
	historySuffix := chapterProfileAssistantHistorySuffix(template, visibleOutput)
	if !opts.EnableThinking {
		historySuffix = chapterProfileAssistantHistorySuffix(template, "")
	}
	if err := chapterProfileAppendPrompt(ctx, model, session, historySuffix); err != nil {
		turn.Error = err.Error()
		return turn
	}
	turn.AppendDuration += bench.NonZeroDuration(time.Since(appendStart))
	if ctx != nil {
		if err := ctx.Err(); err != nil {
			turn.Error = err.Error()
		}
	}
	return turn
}

func chapterProfileMissingEndMarkerError(chapter int, endMarkerSeen bool, generatedTokens, maxTokens int) string {
	if endMarkerSeen {
		return ""
	}
	if generatedTokens >= maxTokens {
		return core.Sprintf("chapter-profile: chapter %d reached max tokens %d before end marker %s", chapter, maxTokens, chapterProfileEndMarker)
	}
	return ""
}

func chapterProfileGenerateOptions(opts chapterProfileOptions) []mlx.GenerateOption {
	out := []mlx.GenerateOption{
		mlx.WithMaxTokens(opts.ChapterMaxTokens),
		mlx.WithTemperature(float32(opts.Temperature)),
		mlx.WithTopP(float32(opts.TopP)),
		mlx.WithTopK(opts.TopK),
		mlx.WithRepeatPenalty(float32(opts.RepeatPenalty)),
	}
	if opts.EnableThinking {
		out = append(out, mlx.WithHideThinking())
	}
	return out
}

func resolveChapterProfileSafetyLimits(limits chapterProfileSafetyLimits, load *tuneProfileLoadSettings) chapterProfileSafetyLimits {
	if limits.SuppressedTokenLoopLimit <= 0 {
		limits.SuppressedTokenLoopLimit = chapterProfileDefaultSuppressedTokenLoopLimit
	}
	if limits.RepeatedLineLoopLimit <= 0 {
		limits.RepeatedLineLoopLimit = profileDefaultRepeatedLineLoopLimit
	}
	if limits.RepeatedSentenceLoopLimit <= 0 {
		limits.RepeatedSentenceLoopLimit = profileDefaultRepeatedSentenceLoopLimit
	}
	memoryLimit := profileResolvedMemoryLimit(load)
	if memoryLimit == 0 {
		return limits
	}
	if limits.MaxActiveMemoryBytes == 0 {
		limits.MaxActiveMemoryBytes = profileDefaultActiveMemoryLimit(memoryLimit)
	}
	if limits.MaxProcessResidentMemoryBytes == 0 {
		limits.MaxProcessResidentMemoryBytes = memoryLimit
	}
	return limits
}

func profileResolvedMemoryLimit(load *tuneProfileLoadSettings) uint64 {
	if load == nil {
		return 0
	}
	if load.MemoryLimitBytes > 0 {
		return load.MemoryLimitBytes
	}
	return load.WiredLimitBytes
}

func saturatingUint64Multiply(value, multiplier uint64) uint64 {
	if value == 0 || multiplier == 0 {
		return 0
	}
	max := ^uint64(0)
	if value > max/multiplier {
		return max
	}
	return value * multiplier
}

func profileDefaultActiveMemoryLimit(memoryLimit uint64) uint64 {
	if memoryLimit == 0 {
		return 0
	}
	return saturatingUint64Multiply(memoryLimit, 13) / 10
}

func profileLiveMetrics() mlx.Metrics {
	processMemory := metal.GetProcessMemory()
	return mlx.Metrics{
		PeakMemoryBytes:            metal.GetPeakMemory(),
		ActiveMemoryBytes:          metal.GetActiveMemory(),
		CacheMemoryBytes:           metal.GetCacheMemory(),
		ProcessVirtualMemoryBytes:  processMemory.VirtualMemoryBytes,
		ProcessResidentMemoryBytes: processMemory.ResidentMemoryBytes,
		ProcessPeakResidentBytes:   processMemory.PeakResidentMemoryBytes,
	}
}

func chapterProfileTurnSafetyError(template string, chapter int, visibleOutput string, turn chapterProfileTurn, limits chapterProfileSafetyLimits) error {
	if err := chapterProfileMetricsSafetyError(core.Sprintf("chapter %d", chapter), turn.Metrics, limits); err != nil {
		return err
	}
	if id, count, ok := chapterProfileSuppressedTokenLoop(turn.SampledTokenIDs, turn.SuppressTokenIDs, limits.SuppressedTokenLoopLimit); ok {
		return core.NewError(core.Sprintf("chapter-profile: chapter %d sampled suppressed token %d for %d consecutive tokens", chapter, id, count))
	}
	if line, count, ok := profileRepeatedLineLoop(visibleOutput, limits.RepeatedLineLoopLimit); ok {
		return core.NewError(core.Sprintf("chapter-profile: chapter %d repeated visible line %q for %d consecutive lines", chapter, line, count))
	}
	if sentence, count, ok := profileRepeatedSentenceLoop(visibleOutput, limits.RepeatedSentenceLoopLimit); ok {
		return core.NewError(core.Sprintf("chapter-profile: chapter %d repeated visible sentence %q for %d total occurrences", chapter, sentence, count))
	}
	if fragments, total, ok := profileFragmentedSentenceOutput(visibleOutput); ok {
		return core.NewError(core.Sprintf("chapter-profile: chapter %d produced fragmented visible output: %d of %d sentence fragments are too short", chapter, fragments, total))
	}
	if reason := chapterProfileMetaPlanningOutput(visibleOutput, chapter); reason != "" {
		return core.NewError(core.Sprintf("chapter-profile: chapter %d produced meta-planning output: %s", chapter, reason))
	}
	if template == "gemma4" && turn.Metrics.GeneratedTokens > 0 && core.Trim(visibleOutput) == "" {
		return core.NewError(core.Sprintf("chapter-profile: chapter %d produced no visible Gemma 4 content after %d generated tokens", chapter, turn.Metrics.GeneratedTokens))
	}
	return nil
}

func chapterProfileMetaPlanningOutput(visibleOutput string, chapter int) string {
	text := core.Trim(visibleOutput)
	if text == "" {
		return ""
	}
	lower := core.Lower(text)
	chapterText := core.Sprintf("chapter %d", chapter)
	prefixes := []string{
		chapterText + " needs",
		chapterText + ": needs",
		chapterText + " focus",
		chapterText + ": focus",
		chapterText + " is required",
		chapterText + ": is required",
		chapterText + " was a placeholder",
		chapterText + ": was a placeholder",
		"i need to ",
		"the focus should ",
	}
	for _, prefix := range prefixes {
		if core.HasPrefix(lower, prefix) {
			return core.Sprintf("starts with %q", prefix)
		}
	}
	firstParagraph := lower
	if parts := core.SplitN(firstParagraph, "\n\n", 2); len(parts) > 0 {
		firstParagraph = parts[0]
	}
	markers := []string{
		" i need to generate ",
		" the user requested ",
		" was a placeholder ",
		" the focus should be ",
	}
	for _, marker := range markers {
		if core.Contains(firstParagraph, marker) {
			return core.Sprintf("contains %q", core.Trim(marker))
		}
	}
	return ""
}

func chapterProfileMetricsSafetyError(phase string, metrics mlx.Metrics, limits chapterProfileSafetyLimits) error {
	if limits.MaxActiveMemoryBytes > 0 && metrics.ActiveMemoryBytes > limits.MaxActiveMemoryBytes {
		return core.NewError(core.Sprintf("chapter-profile: %s exceeded active memory safety limit: %d > %d bytes", phase, metrics.ActiveMemoryBytes, limits.MaxActiveMemoryBytes))
	}
	if limits.MaxProcessVirtualMemoryBytes > 0 && metrics.ProcessVirtualMemoryBytes > limits.MaxProcessVirtualMemoryBytes {
		return core.NewError(core.Sprintf("chapter-profile: %s exceeded process virtual memory safety limit: %d > %d bytes", phase, metrics.ProcessVirtualMemoryBytes, limits.MaxProcessVirtualMemoryBytes))
	}
	if limits.MaxProcessResidentMemoryBytes > 0 && metrics.ProcessResidentMemoryBytes > limits.MaxProcessResidentMemoryBytes {
		return core.NewError(core.Sprintf("chapter-profile: %s exceeded process resident memory safety limit: %d > %d bytes", phase, metrics.ProcessResidentMemoryBytes, limits.MaxProcessResidentMemoryBytes))
	}
	return nil
}

func chapterProfileSuppressedTokenLoop(sampledTokenIDs, suppressTokenIDs []int32, limit int) (int32, int, bool) {
	if limit <= 0 || len(sampledTokenIDs) == 0 || len(suppressTokenIDs) == 0 {
		return 0, 0, false
	}
	var last int32
	count := 0
	for _, id := range sampledTokenIDs {
		if !containsInt32(suppressTokenIDs, id) {
			count = 0
			continue
		}
		if count == 0 || id != last {
			last = id
			count = 1
		} else {
			count++
		}
		if count >= limit {
			return id, count, true
		}
	}
	return 0, 0, false
}

func chapterProfileTemplateTokenControls(template string, tok *mlx.Tokenizer) ([]int32, []int32) {
	if template != "gemma4" || tok == nil {
		return nil, nil
	}
	stopTokens := []int32{}
	if eos := tok.EOS(); eos > 0 {
		stopTokens = appendUniqueInt32(stopTokens, eos)
	}
	if id, ok := tok.TokenID("<turn|>"); ok {
		stopTokens = appendUniqueInt32(stopTokens, id)
	}
	suppressTokens := []int32{}
	for _, text := range []string{
		"<pad>",
		"<bos>",
		"<unk>",
		"<mask>",
		"<|tool>",
		"<tool|>",
		"<|tool_call>",
		"<tool_call|>",
		"<|tool_response>",
		"<tool_response|>",
		"<|\"|>",
		"<|think|>",
		"<|channel>",
		"<channel|>",
		"<|turn>",
		"<|image>",
		"<|audio>",
		"<|image|>",
		"<|audio|>",
		"<image|>",
		"<audio|>",
		"<|video|>",
	} {
		id, ok := tok.TokenID(text)
		if !ok || containsInt32(stopTokens, id) {
			continue
		}
		suppressTokens = appendUniqueInt32(suppressTokens, id)
	}
	return stopTokens, suppressTokens
}

func appendUniqueInt32(values []int32, value int32) []int32 {
	if containsInt32(values, value) {
		return values
	}
	return append(values, value)
}

func containsInt32(values []int32, value int32) bool {
	for _, candidate := range values {
		if candidate == value {
			return true
		}
	}
	return false
}

func chapterProfileAssistantHistorySuffix(template, visibleOutput string) string {
	visibleOutput = core.Trim(visibleOutput)
	switch template {
	case "gemma4":
		return visibleOutput + "<turn|>\n"
	case "gemma":
		return visibleOutput + "<end_of_turn>\n"
	case "qwen":
		return visibleOutput + "<|im_end|>\n"
	case "llama":
		return visibleOutput + "<|eot_id|>"
	default:
		return "\n\n" + visibleOutput
	}
}

func chapterProfileVisibleText(template, text string) string {
	if template != "gemma4" || text == "" {
		return text
	}
	text = core.Replace(text, "<|turn>model\n", "")
	text = core.Replace(text, "<turn|>", "")
	for core.Contains(text, "<|channel>") {
		parts := core.SplitN(text, "<|channel>", 2)
		if len(parts) != 2 {
			break
		}
		after := core.SplitN(parts[1], "<channel|>", 2)
		if len(after) != 2 {
			return parts[0]
		}
		text = parts[0] + after[1]
	}
	return core.Trim(text)
}

func chapterProfileVisibleTextForChapter(template, text string, chapter int) string {
	visible := chapterProfileVisibleText(template, text)
	if template != "gemma4" {
		return visible
	}
	return chapterProfileStripGemma4PlainThought(visible, chapter)
}

func chapterProfileStripEndMarker(text string) (string, bool) {
	if !core.Contains(text, chapterProfileEndMarker) {
		return core.Trim(text), false
	}
	parts := core.SplitN(text, chapterProfileEndMarker, 2)
	if len(parts) == 0 {
		return "", true
	}
	return core.Trim(parts[0]), true
}

func chapterProfileStripGemma4PlainThought(text string, chapter int) string {
	text = core.Trim(text)
	if !core.HasPrefix(core.Lower(text), "thought") {
		return text
	}
	markers := []string{}
	if chapter <= 1 {
		markers = append(markers, "\n**Preamble", "\n# Preamble", "\nPreamble", "\n**Chapter 1", "\n# Chapter 1", "\nChapter 1")
	} else {
		chapterText := core.Sprintf("Chapter %d", chapter)
		markers = append(markers, "\n**"+chapterText, "\n# "+chapterText, "\n"+chapterText)
	}
	if idx := chapterProfileFirstMarkerIndex(text, markers); idx >= 0 {
		return core.Trim(text[idx:])
	}
	return ""
}

func chapterProfileFirstMarkerIndex(text string, markers []string) int {
	best := -1
	for _, marker := range markers {
		if !core.Contains(text, marker) {
			continue
		}
		parts := core.SplitN(text, marker, 2)
		if len(parts) != 2 {
			continue
		}
		idx := len(parts[0])
		if best < 0 || idx < best {
			best = idx
		}
	}
	return best
}

func summariseChapterProfileTurns(prefill time.Duration, turns []chapterProfileTurn) chapterProfileSummary {
	var summary chapterProfileSummary
	summary.TotalDuration = prefill
	var decodeDuration time.Duration
	var prefillRateTotal float64
	var prefillRateCount int
	for _, turn := range turns {
		if turn.Error != "" {
			summary.FailedTurns++
		} else {
			summary.SuccessfulTurns++
		}
		summary.GeneratedTokens += turn.Metrics.GeneratedTokens
		summary.VisibleTokens += turn.VisibleTokens
		summary.TotalDuration += turn.Duration + turn.AppendDuration
		summary.AppendDuration += turn.AppendDuration
		decodeDuration += turn.Metrics.DecodeDuration
		if turn.Metrics.PrefillTokensPerSec > 0 {
			prefillRateTotal += turn.Metrics.PrefillTokensPerSec
			prefillRateCount++
		}
		if turn.Metrics.PeakMemoryBytes > summary.PeakMemoryBytes {
			summary.PeakMemoryBytes = turn.Metrics.PeakMemoryBytes
		}
		if turn.Metrics.ActiveMemoryBytes > summary.ActiveMemoryBytes {
			summary.ActiveMemoryBytes = turn.Metrics.ActiveMemoryBytes
		}
		if turn.Metrics.CacheMemoryBytes > summary.CacheMemoryBytes {
			summary.CacheMemoryBytes = turn.Metrics.CacheMemoryBytes
		}
		if turn.Metrics.ProcessVirtualMemoryBytes > summary.ProcessVirtualMemoryBytes {
			summary.ProcessVirtualMemoryBytes = turn.Metrics.ProcessVirtualMemoryBytes
		}
		if turn.Metrics.ProcessResidentMemoryBytes > summary.ProcessResidentMemoryBytes {
			summary.ProcessResidentMemoryBytes = turn.Metrics.ProcessResidentMemoryBytes
		}
	}
	if len(turns) > 1 {
		summary.AppendAvgDuration = summary.AppendDuration / time.Duration(len(turns)-1)
	}
	if prefillRateCount > 0 {
		summary.PrefillTokensPerSecAverage = prefillRateTotal / float64(prefillRateCount)
	}
	if decodeDuration > 0 {
		summary.DecodeTokensPerSecAverage = float64(summary.GeneratedTokens) / decodeDuration.Seconds()
	}
	return summary
}

func estimateChapterProfileEnergy(report *chapterProfileReport, powerWatts float64) *chapterProfileEnergy {
	energy := &chapterProfileEnergy{
		Method:     "estimated_wall_clock_seconds_times_average_active_watts",
		PowerWatts: powerWatts,
	}
	if report == nil || powerWatts <= 0 {
		return energy
	}
	energy.TotalJoules = durationJoules(report.Summary.TotalDuration, powerWatts)
	if report.Summary.VisibleTokens > 0 {
		energy.JoulesPerToken = energy.TotalJoules / float64(report.Summary.VisibleTokens)
	}
	return energy
}

func printChapterProfileSummary(stdout io.Writer, report *chapterProfileReport) {
	if report == nil {
		return
	}
	core.WriteString(stdout, core.Sprintf("chapter profile: %s\n", report.ModelPath))
	core.WriteString(stdout, core.Sprintf("  prefill: %s, turns: %d ok / %d failed\n", report.InitialPrefillDuration, report.Summary.SuccessfulTurns, report.Summary.FailedTurns))
	core.WriteString(stdout, core.Sprintf("  generated: %d tokens, decode: %.1f tok/s\n", report.Summary.GeneratedTokens, report.Summary.DecodeTokensPerSecAverage))
	core.WriteString(stdout, core.Sprintf("  total: %s, append avg: %s, peak memory: %d MB, cache memory: %d MB, process virtual: %d MB, process resident: %d MB\n",
		report.Summary.TotalDuration,
		report.Summary.AppendAvgDuration,
		report.Summary.PeakMemoryBytes/1024/1024,
		report.Summary.CacheMemoryBytes/1024/1024,
		report.Summary.ProcessVirtualMemoryBytes/1024/1024,
		report.Summary.ProcessResidentMemoryBytes/1024/1024,
	))
	if report.EstimatedEnergy != nil {
		core.WriteString(stdout, core.Sprintf("  estimated energy: %.1f J at %.1f W\n", report.EstimatedEnergy.TotalJoules, report.EstimatedEnergy.PowerWatts))
	}
}

func runFFNEstimateCommand(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet(cliCommandName("ffn-estimate"), flag.ContinueOnError)
	fs.SetOutput(stderr)
	jsonOut := fs.Bool("json", false, "print JSON CPU FFN memory estimate")
	cpuFFNCache := fs.Int("cpu-ffn-cache", 0, "max CPU FFN layers to cache; 0 caches all, negative disables cache")
	fs.Usage = func() {
		core.WriteString(stderr, core.Sprintf("Usage: %s ffn-estimate [flags] <model-path>\n", cliName()))
		fs.VisitAll(func(f *flag.Flag) {
			if f.DefValue == "" {
				core.WriteString(stderr, core.Sprintf("  -%s\n\t%s\n", f.Name, f.Usage))
				return
			}
			core.WriteString(stderr, core.Sprintf("  -%s\n\t%s (default %q)\n", f.Name, f.Usage, f.DefValue))
		})
	}
	if err := fs.Parse(args); err != nil {
		if core.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	if fs.NArg() != 1 {
		core.WriteString(stderr, core.Sprintf("%s ffn-estimate: expected exactly one model path\n", cliName()))
		fs.Usage()
		return 2
	}

	report := &cpuFFNMemoryEstimateReport{
		Version:     1,
		SourcePath:  fs.Arg(0),
		CPUFFNCache: *cpuFFNCache,
	}
	estimate, err := runCPUFFNMemoryEstimate(ctx, report.SourcePath, report.CPUFFNCache)
	report.CPUFFNMemoryEstimate = estimate
	if err != nil {
		report.Error = err.Error()
	}
	return finishCPUFFNMemoryEstimateReport(report, jsonOut, stdout, stderr)
}

func finishCPUFFNMemoryEstimateReport(report *cpuFFNMemoryEstimateReport, jsonOut *bool, stdout, stderr io.Writer) int {
	if jsonOut != nil && *jsonOut {
		data := core.JSONMarshalIndent(report, "", "  ")
		if !data.OK {
			core.Print(stderr, "%s ffn-estimate: marshal report failed", cliName())
			return 1
		}
		core.WriteString(stdout, string(data.Value.([]byte)))
		core.WriteString(stdout, "\n")
		if report.Error != "" {
			return 1
		}
		return 0
	}
	if report.Error != "" {
		core.Print(stderr, "%s ffn-estimate: %s", cliName(), report.Error)
		return 1
	}
	printCPUFFNMemoryEstimateSummary(stdout, report)
	return 0
}

func printCPUFFNMemoryEstimateSummary(stdout io.Writer, report *cpuFFNMemoryEstimateReport) {
	if report == nil || report.CPUFFNMemoryEstimate == nil {
		return
	}
	mem := report.CPUFFNMemoryEstimate
	core.WriteString(stdout, core.Sprintf("cpu ffn estimate: %s\n", report.SourcePath))
	core.WriteString(stdout, core.Sprintf("  cache layers: %d, total layers: %d, loaded layers: %d\n", report.CPUFFNCache, mem.TotalLayers, mem.LoadedLayers))
	core.WriteString(stdout, core.Sprintf("  peak resident: %d bytes, resident: %d bytes\n", mem.PeakResidentBytes, mem.ResidentBytes))
	core.WriteString(stdout, core.Sprintf("  dense equivalent: %d bytes, saved: %d bytes\n", mem.DenseEquivalentBytes, mem.SavedBytes))
	core.WriteString(stdout, core.Sprintf("  loads: %d, evictions: %d\n", mem.LayerLoads, mem.EvictedLayers))
}

func runTunePlanCommand(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet(cliCommandName("tune-plan"), flag.ContinueOnError)
	fs.SetOutput(stderr)
	jsonOut := fs.Bool("json", false, "print JSON tuning plan")
	workload := fs.String("workload", "", "workload to optimise: chat, coding, long_context, agent_state, throughput, or low_latency")
	maxCandidates := fs.Int("max-candidates", 0, "maximum candidates to return")
	splitFFNCaches := fs.String("split-ffn-caches", "", "comma-separated CPU FFN cache layer counts to rank; 0 caches all, negative disables cache")
	fs.Usage = func() {
		core.WriteString(stderr, core.Sprintf("Usage: %s tune-plan [flags] <model-path>\n", cliName()))
		fs.VisitAll(func(f *flag.Flag) {
			if f.DefValue == "" {
				core.WriteString(stderr, core.Sprintf("  -%s\n\t%s\n", f.Name, f.Usage))
				return
			}
			core.WriteString(stderr, core.Sprintf("  -%s\n\t%s (default %q)\n", f.Name, f.Usage, f.DefValue))
		})
	}
	if err := fs.Parse(args); err != nil {
		if core.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	if fs.NArg() != 1 {
		core.WriteString(stderr, core.Sprintf("%s tune-plan: expected exactly one model path\n", cliName()))
		fs.Usage()
		return 2
	}
	workloads, err := cliTuningWorkloads(*workload)
	if err != nil {
		core.Print(stderr, "%s tune-plan: %v", cliName(), err)
		return 2
	}
	caches, err := cliSplitFFNCacheLayers(*splitFFNCaches)
	if err != nil {
		core.Print(stderr, "%s tune-plan: %v", cliName(), err)
		return 2
	}
	plan, err := runPlanLocalTuning(ctx, inference.TuningPlanRequest{
		Model:     inference.ModelIdentity{Path: fs.Arg(0)},
		Workloads: workloads,
		Budget:    inference.TuningBudget{MaxCandidates: *maxCandidates},
	})
	if err != nil {
		core.Print(stderr, "%s tune-plan: %v", cliName(), err)
		return 1
	}
	if len(caches) > 0 {
		plan = appendSplitFFNTuningCandidates(ctx, plan, fs.Arg(0), caches)
	}
	if *jsonOut {
		data := core.JSONMarshalIndent(plan, "", "  ")
		if !data.OK {
			core.Print(stderr, "%s tune-plan: marshal report failed", cliName())
			return 1
		}
		core.WriteString(stdout, string(data.Value.([]byte)))
		core.WriteString(stdout, "\n")
		return 0
	}
	printTunePlanSummary(stdout, plan)
	return 0
}

func printTunePlanSummary(stdout io.Writer, plan inference.TuningPlan) {
	core.WriteString(stdout, core.Sprintf("tuning plan: %s\n", plan.Model.Path))
	core.WriteString(stdout, core.Sprintf("  runtime: %s/%s, cache: %s\n", plan.Runtime.Backend, plan.Runtime.Device, plan.Runtime.CacheMode))
	core.WriteString(stdout, core.Sprintf("  workloads: %d, candidates: %d\n", len(plan.Workloads), len(plan.Candidates)))
	for _, candidate := range plan.Candidates {
		core.WriteString(stdout, core.Sprintf("  candidate: %s ctx=%d batch=%d cache=%s\n", candidate.ID, candidate.ContextLength, candidate.BatchSize, candidate.CacheMode))
	}
}

func runTuneProfileCommand(_ context.Context, args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet(cliCommandName("tune-profile"), flag.ContinueOnError)
	fs.SetOutput(stderr)
	jsonOut := fs.Bool("json", false, "print JSON profile load settings")
	fs.Usage = func() {
		core.WriteString(stderr, core.Sprintf("Usage: %s tune-profile [flags] <profile-path>\n", cliName()))
		fs.VisitAll(func(f *flag.Flag) {
			if f.DefValue == "" {
				core.WriteString(stderr, core.Sprintf("  -%s\n\t%s\n", f.Name, f.Usage))
				return
			}
			core.WriteString(stderr, core.Sprintf("  -%s\n\t%s (default %q)\n", f.Name, f.Usage, f.DefValue))
		})
	}
	if err := fs.Parse(args); err != nil {
		if core.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	if fs.NArg() != 1 {
		core.WriteString(stderr, core.Sprintf("%s tune-profile: expected exactly one profile path\n", cliName()))
		fs.Usage()
		return 2
	}
	report, err := readTuneProfileReport(fs.Arg(0))
	if err != nil {
		core.Print(stderr, "%s tune-profile: %v", cliName(), err)
		return 1
	}
	if *jsonOut {
		data := core.JSONMarshalIndent(report, "", "  ")
		if !data.OK {
			core.Print(stderr, "%s tune-profile: marshal report failed", cliName())
			return 1
		}
		core.WriteString(stdout, string(data.Value.([]byte)))
		core.WriteString(stdout, "\n")
		return 0
	}
	printTuneProfileSummary(stdout, report)
	return 0
}

func readTuneProfileReport(path string) (tuneProfileReport, error) {
	read := core.ReadFile(path)
	if !read.OK {
		return tuneProfileReport{}, core.Errorf("read profile: %v", read.Value)
	}
	var profile inference.TuningProfile
	if result := core.JSONUnmarshal(read.Value.([]byte), &profile); !result.OK {
		return tuneProfileReport{}, core.Errorf("decode profile: %v", result.Value)
	}
	candidate := profile.Candidate
	modelPath := candidate.Model.Path
	if modelPath == "" {
		modelPath = profile.Key.Model.Path
	}
	workload := candidate.Workload
	if workload == "" {
		workload = profile.Key.Workload
	}
	runtime := candidate.Runtime
	if runtime.Backend == "" {
		runtime = profile.Key.Runtime
	}
	return tuneProfileReport{
		Version:     1,
		ProfilePath: path,
		ModelPath:   modelPath,
		Workload:    workload,
		MachineHash: profile.Key.MachineHash,
		CandidateID: candidate.ID,
		Runtime:     runtime,
		Load:        tuneProfileLoadSettingsFromCandidate(candidate),
		Score:       profile.Score,
		Profile:     &profile,
	}, nil
}

func tuneProfileLoadSettingsFromCandidate(candidate inference.TuningCandidate) tuneProfileLoadSettings {
	return tuneProfileLoadSettings{
		ContextLength:        candidate.ContextLength,
		ParallelSlots:        candidate.ParallelSlots,
		PromptCache:          candidate.PromptCache,
		PromptCacheMinTokens: candidate.PromptCacheMinTokens,
		CachePolicy:          candidate.CachePolicy,
		CacheMode:            candidate.CacheMode,
		BatchSize:            candidate.BatchSize,
		PrefillChunkSize:     candidate.PrefillChunkSize,
		ExpectedQuantization: candidate.ExpectedQuantization,
		MemoryLimitBytes:     candidate.MemoryLimitBytes,
		CacheLimitBytes:      candidate.CacheLimitBytes,
		WiredLimitBytes:      candidate.WiredLimitBytes,
		AdapterPath:          candidate.Adapter.Path,
	}
}

func printTuneProfileSummary(stdout io.Writer, report tuneProfileReport) {
	core.WriteString(stdout, core.Sprintf("tuning profile: %s\n", report.ProfilePath))
	core.WriteString(stdout, core.Sprintf("  model: %s, workload: %s\n", report.ModelPath, report.Workload))
	core.WriteString(stdout, core.Sprintf("  candidate: %s, score: %.2f\n", report.CandidateID, report.Score.Score))
	core.WriteString(stdout, core.Sprintf("  load: ctx=%d batch=%d cache=%s prompt-cache=%t\n", report.Load.ContextLength, report.Load.BatchSize, report.Load.CacheMode, report.Load.PromptCache))
}

func runProfileListCommand(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet(cliCommandName("profile-list"), flag.ContinueOnError)
	fs.SetOutput(stderr)
	jsonOut := fs.Bool("json", false, "print JSON profile list")
	machineHash := fs.String("machine-hash", "", "machine hash to match")
	currentMachine := fs.Bool("current-machine", false, "discover current machine hash before listing")
	includeProfile := fs.Bool("include-profile", false, "include full nested tuning profile JSON in each row")
	bestPerWorkload := fs.Bool("best-per-workload", false, "list only the best matching profile for each workload")
	workload := fs.String("workload", "", "workload to match: chat, coding, long_context, agent_state, throughput, or low_latency")
	modelPath := fs.String("model-path", "", "model path to match")
	fs.Usage = func() {
		core.WriteString(stderr, core.Sprintf("Usage: %s profile-list [flags] <profile-dir>\n", cliName()))
		fs.VisitAll(func(f *flag.Flag) {
			if f.DefValue == "" {
				core.WriteString(stderr, core.Sprintf("  -%s\n\t%s\n", f.Name, f.Usage))
				return
			}
			core.WriteString(stderr, core.Sprintf("  -%s\n\t%s (default %q)\n", f.Name, f.Usage, f.DefValue))
		})
	}
	if err := fs.Parse(args); err != nil {
		if core.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	if fs.NArg() != 1 {
		core.WriteString(stderr, core.Sprintf("%s profile-list: expected exactly one profile directory\n", cliName()))
		fs.Usage()
		return 2
	}
	workloads, err := cliTuningWorkloads(*workload)
	if err != nil {
		core.Print(stderr, "%s profile-list: %v", cliName(), err)
		return 2
	}
	criteria := profileSelectCriteria{
		MachineHash: core.Trim(*machineHash),
		ModelPath:   core.Trim(*modelPath),
	}
	if *currentMachine {
		currentHash, err := currentMachineProfileHash(ctx)
		if err != nil {
			core.Print(stderr, "%s profile-list: %v", cliName(), err)
			return 1
		}
		criteria.MachineHash = currentHash
	}
	if len(workloads) > 0 {
		criteria.Workload = workloads[0]
	}
	report := listTuningProfiles(fs.Arg(0), criteria, profileListOptions{IncludeProfile: *includeProfile, BestPerWorkload: *bestPerWorkload})
	if *jsonOut {
		data := core.JSONMarshalIndent(report, "", "  ")
		if !data.OK {
			core.Print(stderr, "%s profile-list: marshal report failed", cliName())
			return 1
		}
		core.WriteString(stdout, string(data.Value.([]byte)))
		core.WriteString(stdout, "\n")
		return 0
	}
	printProfileListSummary(stdout, report)
	return 0
}

func runProfileSelectCommand(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet(cliCommandName("profile-select"), flag.ContinueOnError)
	fs.SetOutput(stderr)
	jsonOut := fs.Bool("json", false, "print JSON selected profile")
	machineHash := fs.String("machine-hash", "", "machine hash to match")
	currentMachine := fs.Bool("current-machine", false, "discover current machine hash before matching")
	workload := fs.String("workload", "", "workload to match: chat, coding, long_context, agent_state, throughput, or low_latency")
	modelPath := fs.String("model-path", "", "model path to match")
	fs.Usage = func() {
		core.WriteString(stderr, core.Sprintf("Usage: %s profile-select [flags] <profile-dir>\n", cliName()))
		fs.VisitAll(func(f *flag.Flag) {
			if f.DefValue == "" {
				core.WriteString(stderr, core.Sprintf("  -%s\n\t%s\n", f.Name, f.Usage))
				return
			}
			core.WriteString(stderr, core.Sprintf("  -%s\n\t%s (default %q)\n", f.Name, f.Usage, f.DefValue))
		})
	}
	if err := fs.Parse(args); err != nil {
		if core.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	if fs.NArg() != 1 {
		core.WriteString(stderr, core.Sprintf("%s profile-select: expected exactly one profile directory\n", cliName()))
		fs.Usage()
		return 2
	}
	workloads, err := cliTuningWorkloads(*workload)
	if err != nil {
		core.Print(stderr, "%s profile-select: %v", cliName(), err)
		return 2
	}
	criteria := profileSelectCriteria{
		MachineHash: core.Trim(*machineHash),
		ModelPath:   core.Trim(*modelPath),
	}
	if *currentMachine {
		currentHash, err := currentMachineProfileHash(ctx)
		if err != nil {
			core.Print(stderr, "%s profile-select: %v", cliName(), err)
			return 1
		}
		criteria.MachineHash = currentHash
	}
	if len(workloads) > 0 {
		criteria.Workload = workloads[0]
	}
	report, err := selectTuningProfile(fs.Arg(0), criteria)
	if err != nil {
		core.Print(stderr, "%s profile-select: %v", cliName(), err)
		return 1
	}
	if *jsonOut {
		data := core.JSONMarshalIndent(report, "", "  ")
		if !data.OK {
			core.Print(stderr, "%s profile-select: marshal report failed", cliName())
			return 1
		}
		core.WriteString(stdout, string(data.Value.([]byte)))
		core.WriteString(stdout, "\n")
		return 0
	}
	printProfileSelectSummary(stdout, report)
	return 0
}

func currentMachineProfileHash(ctx context.Context) (string, error) {
	report, err := runDiscoverLocalRuntime(ctx, mlx.LocalDiscoveryConfig{Device: runGetDeviceInfo()})
	if err != nil {
		return "", err
	}
	if report.Labels != nil && report.Labels["machine_hash"] != "" {
		return report.Labels["machine_hash"], nil
	}
	if report.Device.Labels != nil && report.Device.Labels["machine_hash"] != "" {
		return report.Device.Labels["machine_hash"], nil
	}
	return "", core.NewError("current machine hash unavailable")
}

func listTuningProfiles(profileDir string, criteria profileSelectCriteria, opts profileListOptions) profileListReport {
	paths := core.PathGlob(core.PathJoin(profileDir, "*.json"))
	core.SliceSort(paths)
	profiles := []tuneProfileReport{}
	warnings := []string{}
	for _, path := range paths {
		report, err := readTuneProfileReport(path)
		if err != nil {
			warnings = append(warnings, core.Sprintf("%s: %v", path, err))
			continue
		}
		if !profileMatchesCriteria(report, criteria) {
			continue
		}
		profiles = append(profiles, report)
	}
	sortTuneProfileReports(profiles)
	if opts.BestPerWorkload {
		profiles = bestTuneProfilesPerWorkload(profiles)
	}
	if !opts.IncludeProfile {
		for i := range profiles {
			profiles[i].Profile = nil
		}
	}
	return profileListReport{
		Version:      1,
		ProfileDir:   profileDir,
		MachineHash:  criteria.MachineHash,
		ModelPath:    criteria.ModelPath,
		Workload:     criteria.Workload,
		ProfileCount: len(profiles),
		Profiles:     profiles,
		Warnings:     warnings,
	}
}

func selectTuningProfile(profileDir string, criteria profileSelectCriteria) (profileSelectReport, error) {
	paths := core.PathGlob(core.PathJoin(profileDir, "*.json"))
	core.SliceSort(paths)
	var best tuneProfileReport
	bestPath := ""
	matched := 0
	warnings := []string{}
	for _, path := range paths {
		report, err := readTuneProfileReport(path)
		if err != nil {
			warnings = append(warnings, core.Sprintf("%s: %v", path, err))
			continue
		}
		if !profileMatchesCriteria(report, criteria) {
			continue
		}
		matched++
		if bestPath == "" || profileReportLess(best, bestPath, report, path) {
			best = report
			bestPath = path
		}
	}
	if bestPath == "" {
		return profileSelectReport{}, core.NewError("no matching tuning profiles")
	}
	return profileSelectReport{
		Version:         1,
		ProfileDir:      profileDir,
		ProfilePath:     bestPath,
		MachineHash:     best.MachineHash,
		ModelPath:       best.ModelPath,
		Workload:        best.Workload,
		MatchedProfiles: matched,
		CandidateID:     best.CandidateID,
		Runtime:         best.Runtime,
		Load:            best.Load,
		Score:           best.Score,
		Profile:         best.Profile,
		Warnings:        warnings,
	}, nil
}

func profileMatchesCriteria(report tuneProfileReport, criteria profileSelectCriteria) bool {
	if criteria.MachineHash != "" && report.MachineHash != criteria.MachineHash {
		return false
	}
	if criteria.ModelPath != "" && report.ModelPath != criteria.ModelPath {
		return false
	}
	if criteria.Workload != "" && report.Workload != criteria.Workload {
		return false
	}
	return true
}

func profileReportLess(best tuneProfileReport, bestPath string, candidate tuneProfileReport, candidatePath string) bool {
	if candidate.Score.Score != best.Score.Score {
		return candidate.Score.Score > best.Score.Score
	}
	if candidate.ProfileCreatedAtUnix() != best.ProfileCreatedAtUnix() {
		return candidate.ProfileCreatedAtUnix() > best.ProfileCreatedAtUnix()
	}
	return candidatePath < bestPath
}

func (report tuneProfileReport) ProfileCreatedAtUnix() int64 {
	if report.Profile == nil {
		return 0
	}
	return report.Profile.CreatedAtUnix
}

func sortTuneProfileReports(profiles []tuneProfileReport) {
	for i := 1; i < len(profiles); i++ {
		for j := i; j > 0 && profileReportLess(profiles[j-1], profiles[j-1].ProfilePath, profiles[j], profiles[j].ProfilePath); j-- {
			profiles[j-1], profiles[j] = profiles[j], profiles[j-1]
		}
	}
}

func bestTuneProfilesPerWorkload(profiles []tuneProfileReport) []tuneProfileReport {
	if len(profiles) == 0 {
		return nil
	}
	seen := map[inference.TuningWorkload]bool{}
	best := make([]tuneProfileReport, 0, len(profiles))
	for _, profile := range profiles {
		if seen[profile.Workload] {
			continue
		}
		seen[profile.Workload] = true
		best = append(best, profile)
	}
	return best
}

func printProfileListSummary(stdout io.Writer, report profileListReport) {
	core.WriteString(stdout, core.Sprintf("profile store: %s\n", report.ProfileDir))
	core.WriteString(stdout, core.Sprintf("  profiles: %d\n", report.ProfileCount))
	for _, profile := range report.Profiles {
		core.WriteString(stdout, core.Sprintf("  profile: %s model=%s workload=%s machine=%s score=%.2f\n", profile.ProfilePath, profile.ModelPath, profile.Workload, profile.MachineHash, profile.Score.Score))
	}
}

func printProfileSelectSummary(stdout io.Writer, report profileSelectReport) {
	core.WriteString(stdout, core.Sprintf("selected profile: %s\n", report.ProfilePath))
	core.WriteString(stdout, core.Sprintf("  model: %s, workload: %s, machine: %s\n", report.ModelPath, report.Workload, report.MachineHash))
	core.WriteString(stdout, core.Sprintf("  candidate: %s, score: %.2f, matches: %d\n", report.CandidateID, report.Score.Score, report.MatchedProfiles))
}

func runReplacePlanCommand(_ context.Context, args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet(cliCommandName("replace-plan"), flag.ContinueOnError)
	fs.SetOutput(stderr)
	jsonOut := fs.Bool("json", false, "print JSON model replace plan")
	currentProfile := fs.String("current-profile", "", "current saved tuning profile")
	nextProfile := fs.String("next-profile", "", "next saved tuning profile")
	fs.Usage = func() {
		core.WriteString(stderr, core.Sprintf("Usage: %s replace-plan [flags]\n", cliName()))
		fs.VisitAll(func(f *flag.Flag) {
			if f.DefValue == "" {
				core.WriteString(stderr, core.Sprintf("  -%s\n\t%s\n", f.Name, f.Usage))
				return
			}
			core.WriteString(stderr, core.Sprintf("  -%s\n\t%s (default %q)\n", f.Name, f.Usage, f.DefValue))
		})
	}
	if err := fs.Parse(args); err != nil {
		if core.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	if fs.NArg() != 0 || core.Trim(*currentProfile) == "" || core.Trim(*nextProfile) == "" {
		core.WriteString(stderr, core.Sprintf("%s replace-plan: -current-profile and -next-profile are required\n", cliName()))
		fs.Usage()
		return 2
	}
	current, err := readTuneProfileReport(*currentProfile)
	if err != nil {
		core.Print(stderr, "%s replace-plan: current profile: %v", cliName(), err)
		return 1
	}
	next, err := readTuneProfileReport(*nextProfile)
	if err != nil {
		core.Print(stderr, "%s replace-plan: next profile: %v", cliName(), err)
		return 1
	}
	if current.Profile == nil || next.Profile == nil {
		core.Print(stderr, "%s replace-plan: profile payload missing", cliName())
		return 1
	}
	req := replaceRequestFromTuneProfiles(*current.Profile, *next.Profile)
	report := replacePlanReport{
		Version:            1,
		CurrentProfilePath: *currentProfile,
		NextProfilePath:    *nextProfile,
		Request:            req,
		Plan:               inference.PlanModelReplace(req),
	}
	if *jsonOut {
		data := core.JSONMarshalIndent(report, "", "  ")
		if !data.OK {
			core.Print(stderr, "%s replace-plan: marshal report failed", cliName())
			return 1
		}
		core.WriteString(stdout, string(data.Value.([]byte)))
		core.WriteString(stdout, "\n")
		return 0
	}
	printReplacePlanSummary(stdout, report)
	return 0
}

func replaceRequestFromTuneProfiles(current, next inference.TuningProfile) inference.ModelReplaceRequest {
	return inference.ModelReplaceRequest{
		CurrentModel:   modelIdentityFromProfile(current),
		NextModel:      modelIdentityFromProfile(next),
		CurrentRuntime: runtimeIdentityFromProfile(current),
		NextRuntime:    runtimeIdentityFromProfile(next),
		CurrentAdapter: adapterIdentityFromProfile(current),
		NextAdapter:    adapterIdentityFromProfile(next),
	}
}

func modelIdentityFromProfile(profile inference.TuningProfile) inference.ModelIdentity {
	identity := profile.Key.Model
	candidate := profile.Candidate.Model
	if candidate.Path != "" {
		identity.Path = candidate.Path
	}
	if candidate.Hash != "" {
		identity.Hash = candidate.Hash
	}
	if candidate.Architecture != "" {
		identity.Architecture = candidate.Architecture
	}
	if candidate.QuantBits != 0 {
		identity.QuantBits = candidate.QuantBits
	}
	if candidate.QuantGroup != 0 {
		identity.QuantGroup = candidate.QuantGroup
	}
	if candidate.QuantType != "" {
		identity.QuantType = candidate.QuantType
	}
	if candidate.ContextLength != 0 {
		identity.ContextLength = candidate.ContextLength
	}
	if candidate.NumLayers != 0 {
		identity.NumLayers = candidate.NumLayers
	}
	if candidate.HiddenSize != 0 {
		identity.HiddenSize = candidate.HiddenSize
	}
	if candidate.VocabSize != 0 {
		identity.VocabSize = candidate.VocabSize
	}
	return identity
}

func runtimeIdentityFromProfile(profile inference.TuningProfile) inference.RuntimeIdentity {
	identity := profile.Key.Runtime
	candidate := profile.Candidate.Runtime
	if candidate.Backend != "" {
		identity.Backend = candidate.Backend
	}
	if candidate.Device != "" {
		identity.Device = candidate.Device
	}
	if candidate.CacheMode != "" {
		identity.CacheMode = candidate.CacheMode
	}
	if candidate.NativeRuntime {
		identity.NativeRuntime = candidate.NativeRuntime
	}
	if len(candidate.Labels) > 0 {
		identity.Labels = candidate.Labels
	}
	return identity
}

func adapterIdentityFromProfile(profile inference.TuningProfile) inference.AdapterIdentity {
	identity := profile.Key.Adapter
	candidate := profile.Candidate.Adapter
	if candidate.Path != "" {
		identity.Path = candidate.Path
	}
	if candidate.Hash != "" {
		identity.Hash = candidate.Hash
	}
	if candidate.Format != "" {
		identity.Format = candidate.Format
	}
	if candidate.Rank != 0 {
		identity.Rank = candidate.Rank
	}
	if candidate.Alpha != 0 {
		identity.Alpha = candidate.Alpha
	}
	return identity
}

func printReplacePlanSummary(stdout io.Writer, report replacePlanReport) {
	core.WriteString(stdout, core.Sprintf("replace plan: %s\n", report.Plan.Action))
	core.WriteString(stdout, core.Sprintf("  compatible: %t\n", report.Plan.Compatible))
	for _, reason := range report.Plan.Reasons {
		core.WriteString(stdout, core.Sprintf("  reason: %s\n", reason))
	}
}

func runTuneRunCommand(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	defaultBench := bench.DefaultConfig()
	fs := flag.NewFlagSet(cliCommandName("tune-run"), flag.ContinueOnError)
	fs.SetOutput(stderr)
	jsonlOut := fs.Bool("jsonl", false, "stream JSONL tuning events")
	workload := fs.String("workload", string(inference.TuningWorkloadChat), "workload to optimise: chat, coding, long_context, agent_state, throughput, or low_latency")
	maxCandidates := fs.Int("max-candidates", 0, "maximum candidates to run")
	splitFFNCaches := fs.String("split-ffn-caches", "", "comma-separated CPU FFN cache layer counts to rank and test")
	profileOutput := fs.String("profile-output", "", "write the selected tuning profile JSON to this path")
	profileDir := fs.String("profile-dir", "", "write the selected tuning profile JSON into this directory")
	machineHash := fs.String("machine-hash", "", "stable machine/profile key supplied by the caller")
	currentMachine := fs.Bool("current-machine", false, "discover current machine hash for profile output")
	prompt := fs.String("prompt", defaultBench.Prompt, "smoke prompt for candidate measurements")
	maxTokens := fs.Int("max-tokens", defaultBench.MaxTokens, "generated tokens per candidate measurement")
	runs := fs.Int("runs", defaultBench.Runs, "measurement runs per candidate")
	fs.Usage = func() {
		core.WriteString(stderr, core.Sprintf("Usage: %s tune-run [flags] <model-path>\n", cliName()))
		fs.VisitAll(func(f *flag.Flag) {
			if f.DefValue == "" {
				core.WriteString(stderr, core.Sprintf("  -%s\n\t%s\n", f.Name, f.Usage))
				return
			}
			core.WriteString(stderr, core.Sprintf("  -%s\n\t%s (default %q)\n", f.Name, f.Usage, f.DefValue))
		})
	}
	if err := fs.Parse(args); err != nil {
		if core.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	if fs.NArg() != 1 {
		core.WriteString(stderr, core.Sprintf("%s tune-run: expected exactly one model path\n", cliName()))
		fs.Usage()
		return 2
	}
	workloads, err := cliTuningWorkloads(*workload)
	if err != nil {
		core.Print(stderr, "%s tune-run: %v", cliName(), err)
		return 2
	}
	if len(workloads) == 0 {
		workloads = []inference.TuningWorkload{inference.TuningWorkloadChat}
	}
	caches, err := cliSplitFFNCacheLayers(*splitFFNCaches)
	if err != nil {
		core.Print(stderr, "%s tune-run: %v", cliName(), err)
		return 2
	}

	modelPath := fs.Arg(0)
	plan, err := runPlanLocalTuning(ctx, inference.TuningPlanRequest{
		Model:     inference.ModelIdentity{Path: modelPath},
		Workloads: workloads,
		Budget: inference.TuningBudget{
			MaxCandidates:     *maxCandidates,
			SmokeTokens:       *maxTokens,
			Runs:              *runs,
			AllowStateBench:   true,
			AllowModelReloads: true,
		},
	})
	if err != nil {
		core.Print(stderr, "%s tune-run: plan: %v", cliName(), err)
		return 1
	}
	if len(caches) > 0 {
		plan = appendSplitFFNTuningCandidates(ctx, plan, modelPath, caches)
	}
	candidates := cliLimitTuningCandidates(plan.Candidates, *maxCandidates)
	if len(candidates) == 0 {
		core.Print(stderr, "%s tune-run: no tuning candidates", cliName())
		return 1
	}

	benchCfg := defaultBench
	benchCfg.Model = core.PathBase(modelPath)
	benchCfg.ModelPath = modelPath
	benchCfg.Prompt = *prompt
	benchCfg.CachePrompt = *prompt
	benchCfg.MaxTokens = *maxTokens
	benchCfg.Runs = *runs

	var emitErr error
	results, err := runLocalTuning(ctx, mlx.LocalTuningRunConfig{
		ModelPath:  modelPath,
		Workload:   workloads[0],
		Candidates: candidates,
		Bench:      benchCfg,
		Emit: func(event inference.TuningEvent) bool {
			if !*jsonlOut {
				return true
			}
			if emitErr != nil {
				return false
			}
			emitErr = writeTuningEventJSONL(stdout, event)
			return emitErr == nil
		},
	})
	if emitErr != nil {
		core.Print(stderr, "%s tune-run: %v", cliName(), emitErr)
		return 1
	}
	if err != nil {
		core.Print(stderr, "%s tune-run: %v", cliName(), err)
		return 1
	}
	profileOutputPath := core.Trim(*profileOutput)
	profileDirPath := core.Trim(*profileDir)
	if profileOutputPath != "" && profileDirPath != "" {
		core.Print(stderr, "%s tune-run: use only one of -profile-output or -profile-dir", cliName())
		return 2
	}
	if profileOutputPath != "" || profileDirPath != "" {
		selected, ok := cliSelectTuningResult(results)
		if !ok {
			core.Print(stderr, "%s tune-run: no successful tuning result to persist", cliName())
			return 1
		}
		profileMachineHash := core.Trim(*machineHash)
		if *currentMachine {
			profileMachineHash, err = currentMachineProfileHash(ctx)
			if err != nil {
				core.Print(stderr, "%s tune-run: %v", cliName(), err)
				return 1
			}
		}
		selectionLabels := cliTuningSelectionLabels(results, selected)
		profile := cliBuildTuningProfile(plan, modelPath, profileMachineHash, workloads[0], selected, selectionLabels, time.Now())
		if profileOutputPath == "" {
			profileOutputPath = cliTuningProfilePath(profileDirPath, profile)
		}
		if err := writeTuningProfile(profileOutputPath, profile); err != nil {
			core.Print(stderr, "%s tune-run: %v", cliName(), err)
			return 1
		}
		if *jsonlOut {
			selectedCopy := selected
			eventLabels := cliCloneStringLabels(selectionLabels)
			eventLabels["profile_output"] = profileOutputPath
			eventLabels["machine_hash"] = profileMachineHash
			if err := writeTuningEventJSONL(stdout, inference.TuningEvent{
				Kind:      inference.TuningEventSelected,
				Candidate: selected.Candidate,
				Result:    &selectedCopy,
				Labels:    eventLabels,
			}); err != nil {
				core.Print(stderr, "%s tune-run: %v", cliName(), err)
				return 1
			}
		}
	}
	if *jsonlOut {
		return 0
	}
	printTuneRunSummary(stdout, modelPath, results)
	return 0
}

func cliTuningProfilePath(profileDir string, profile inference.TuningProfile) string {
	modelName := core.PathBase(profile.Key.Model.Path)
	if modelName == "" {
		modelName = profile.Candidate.Model.Architecture
	}
	if modelName == "" {
		modelName = profile.Key.Model.Architecture
	}
	machineHash := profile.Key.MachineHash
	if parts := core.SplitN(machineHash, ":", 2); len(parts) == 2 {
		machineHash = parts[1]
	}
	name := core.Sprintf("%s-%s-%s-%s.json",
		cliProfileFilePart(string(profile.Key.Workload), "workload", 32),
		cliProfileFilePart(machineHash, "machine", 12),
		cliProfileFilePart(modelName, "model", 48),
		cliProfileFilePart(profile.Candidate.ID, "candidate", 48),
	)
	return core.PathJoin(profileDir, name)
}

func cliProfileFilePart(value, fallback string, maxLen int) string {
	value = core.Lower(core.Trim(value))
	builder := core.NewBuilder()
	lastDash := false
	for i := 0; i < len(value); i++ {
		b := value[i]
		if (b >= 'a' && b <= 'z') || (b >= '0' && b <= '9') {
			builder.WriteByte(b)
			lastDash = false
			continue
		}
		if builder.Len() > 0 && !lastDash {
			builder.WriteByte('-')
			lastDash = true
		}
	}
	part := trimProfileFileDashes(builder.String())
	if part == "" {
		part = fallback
	}
	if maxLen > 0 && len(part) > maxLen {
		part = trimProfileFileDashes(part[:maxLen])
	}
	if part == "" {
		return fallback
	}
	return part
}

func trimProfileFileDashes(value string) string {
	for len(value) > 0 && value[len(value)-1] == '-' {
		value = value[:len(value)-1]
	}
	return value
}

func cliSelectTuningResult(results []inference.TuningResult) (inference.TuningResult, bool) {
	var best inference.TuningResult
	found := false
	for _, result := range results {
		if result.Error != "" {
			continue
		}
		if !found || result.Score.Score > best.Score.Score {
			best = result
			found = true
		}
	}
	return best, found
}

func cliTuningSelectionLabels(results []inference.TuningResult, selected inference.TuningResult) map[string]string {
	labels := map[string]string{
		"source":           "lthn-mlx tune-run",
		"selection_policy": "highest_successful_score",
		"selection_reason": "selected highest successful score from measured tuning candidates",
		"selected_score":   core.Sprintf("%.6f", selected.Score.Score),
	}
	if selected.Candidate.ID != "" {
		labels["selected_candidate_id"] = selected.Candidate.ID
	}
	if selected.Measurements.DecodeTokensPerSec > 0 {
		labels["selected_decode_tokens_per_sec"] = core.Sprintf("%.6f", selected.Measurements.DecodeTokensPerSec)
	}
	if selected.Measurements.LoadMilliseconds > 0 {
		labels["selected_load_milliseconds"] = core.Sprintf("%.6f", selected.Measurements.LoadMilliseconds)
	}
	if selected.Measurements.FirstTokenMilliseconds > 0 {
		labels["selected_first_token_milliseconds"] = core.Sprintf("%.6f", selected.Measurements.FirstTokenMilliseconds)
	}
	if selected.Measurements.KVRestoreMilliseconds > 0 {
		labels["selected_restore_milliseconds"] = core.Sprintf("%.6f", selected.Measurements.KVRestoreMilliseconds)
	}
	if selected.Measurements.PeakMemoryBytes > 0 {
		labels["selected_peak_memory_bytes"] = core.Sprintf("%d", selected.Measurements.PeakMemoryBytes)
	}
	if selected.Measurements.CorrectnessSmokeResult != "" {
		labels["selected_correctness_smoke_result"] = selected.Measurements.CorrectnessSmokeResult
	}
	if selected.Measurements.CorrectnessSmokeChecks > 0 {
		labels["selected_correctness_smoke_checks"] = core.Sprintf("%d", selected.Measurements.CorrectnessSmokeChecks)
	}
	successful := 0
	failed := 0
	var runnerUp inference.TuningResult
	hasRunnerUp := false
	for _, result := range results {
		if result.Error != "" {
			failed++
			continue
		}
		successful++
		if result.Candidate.ID == selected.Candidate.ID && result.Score.Score == selected.Score.Score {
			continue
		}
		if !hasRunnerUp || result.Score.Score > runnerUp.Score.Score {
			runnerUp = result
			hasRunnerUp = true
		}
	}
	labels["successful_candidates"] = core.Sprintf("%d", successful)
	labels["failed_candidates"] = core.Sprintf("%d", failed)
	if hasRunnerUp {
		if runnerUp.Candidate.ID != "" {
			labels["runner_up_candidate_id"] = runnerUp.Candidate.ID
		}
		labels["runner_up_score"] = core.Sprintf("%.6f", runnerUp.Score.Score)
		labels["selection_score_delta"] = core.Sprintf("%.6f", selected.Score.Score-runnerUp.Score.Score)
	}
	return labels
}

func cliBuildTuningProfile(plan inference.TuningPlan, modelPath, machineHash string, workload inference.TuningWorkload, result inference.TuningResult, labels map[string]string, createdAt time.Time) inference.TuningProfile {
	candidate := result.Candidate
	if candidate.Model.Path == "" && plan.Model.Path != "" {
		candidate.Model = plan.Model
	}
	if candidate.Model.Path == "" {
		candidate.Model.Path = modelPath
	}
	if candidate.Runtime.Backend == "" {
		candidate.Runtime = plan.Runtime
	}
	if candidate.Adapter.Path == "" && plan.Adapter.Path != "" {
		candidate.Adapter = plan.Adapter
	}
	if candidate.Workload == "" {
		candidate.Workload = workload
	}
	score := result.Score
	if score.Workload == "" {
		score.Workload = workload
	}
	profileLabels := cliCloneStringLabels(labels)
	if profileLabels == nil {
		profileLabels = map[string]string{}
	}
	if profileLabels["source"] == "" {
		profileLabels["source"] = "lthn-mlx tune-run"
	}
	return inference.TuningProfile{
		Key: inference.TuningProfileKey{
			MachineHash: machineHash,
			Runtime:     candidate.Runtime,
			Model:       candidate.Model,
			Adapter:     candidate.Adapter,
			Workload:    workload,
		},
		Candidate:     candidate,
		Measurements:  result.Measurements,
		Score:         score,
		CreatedAtUnix: createdAt.Unix(),
		Labels:        profileLabels,
	}
}

func writeTuningProfile(path string, profile inference.TuningProfile) error {
	data := core.JSONMarshalIndent(profile, "", "  ")
	if !data.OK {
		return core.NewError("marshal tuning profile failed")
	}
	if result := core.MkdirAll(core.PathDir(path), 0o755); !result.OK {
		return core.Errorf("create profile directory: %v", result.Value)
	}
	if result := core.WriteFile(path, data.Value.([]byte), 0o600); !result.OK {
		return core.Errorf("write tuning profile: %v", result.Value)
	}
	return nil
}

func cliLimitTuningCandidates(candidates []inference.TuningCandidate, maxCandidates int) []inference.TuningCandidate {
	if maxCandidates > 0 && len(candidates) > maxCandidates {
		return append([]inference.TuningCandidate(nil), candidates[:maxCandidates]...)
	}
	return append([]inference.TuningCandidate(nil), candidates...)
}

func writeTuningEventJSONL(stdout io.Writer, event inference.TuningEvent) error {
	data := core.JSONMarshal(event)
	if !data.OK {
		return core.NewError("marshal tuning event failed")
	}
	core.WriteString(stdout, string(data.Value.([]byte)))
	core.WriteString(stdout, "\n")
	return nil
}

func printTuneRunSummary(stdout io.Writer, modelPath string, results []inference.TuningResult) {
	core.WriteString(stdout, core.Sprintf("tuning run: %s\n", modelPath))
	core.WriteString(stdout, core.Sprintf("  results: %d\n", len(results)))
	for _, result := range results {
		if result.Error != "" {
			core.WriteString(stdout, core.Sprintf("  candidate: %s error=%q\n", result.Candidate.ID, result.Error))
			continue
		}
		core.WriteString(stdout, core.Sprintf(
			"  candidate: %s score=%.2f decode=%.1f tok/s peak=%d MB\n",
			result.Candidate.ID,
			result.Score.Score,
			result.Measurements.DecodeTokensPerSec,
			result.Measurements.PeakMemoryBytes/1024/1024,
		))
	}
}

func cliTuningWorkloads(value string) ([]inference.TuningWorkload, error) {
	value = core.Trim(value)
	if value == "" {
		return nil, nil
	}
	workload := inference.TuningWorkload(value)
	if !cliValidTuningWorkload(workload) {
		return nil, core.Errorf("unsupported workload %q", value)
	}
	return []inference.TuningWorkload{workload}, nil
}

func cliValidTuningWorkload(workload inference.TuningWorkload) bool {
	switch workload {
	case inference.TuningWorkloadChat,
		inference.TuningWorkloadCoding,
		inference.TuningWorkloadLongContext,
		inference.TuningWorkloadAgentState,
		inference.TuningWorkloadThroughput,
		inference.TuningWorkloadLowLatency:
		return true
	default:
		return false
	}
}

func runSliceSmokeCommand(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	defaultBench := bench.DefaultConfig()
	fs := flag.NewFlagSet(cliCommandName("slice-smoke"), flag.ContinueOnError)
	fs.SetOutput(stderr)
	jsonOut := fs.Bool("json", false, "print JSON smoke report")
	preset := fs.String("preset", string(inference.ModelSlicePresetClient), "slice preset to materialise before reload")
	output := fs.String("output", "", "output directory for the materialised slice")
	prompt := fs.String("prompt", "Write one short sentence about local inference.", "tiny reload smoke prompt")
	maxTokens := fs.Int("max-tokens", 1, "generated tokens for the smoke pass")
	runs := fs.Int("runs", 1, "generation runs for the smoke pass")
	contextLen := fs.Int("context", 0, "override context length when loading the slice")
	device := fs.String("device", "", "execution device: gpu or cpu")
	split := fs.Bool("split", false, "run split executor for client slices instead of skipping reload")
	cpuFFNCache := fs.Int("cpu-ffn-cache", 0, "max CPU FFN layers to cache during split smoke; 0 caches all, negative disables cache")
	fs.Usage = func() {
		core.WriteString(stderr, core.Sprintf("Usage: %s slice-smoke [flags] <model-path>\n", cliName()))
		fs.VisitAll(func(f *flag.Flag) {
			if f.DefValue == "" {
				core.WriteString(stderr, core.Sprintf("  -%s\n\t%s\n", f.Name, f.Usage))
				return
			}
			core.WriteString(stderr, core.Sprintf("  -%s\n\t%s (default %q)\n", f.Name, f.Usage, f.DefValue))
		})
	}
	if err := fs.Parse(args); err != nil {
		if core.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	if fs.NArg() != 1 {
		core.WriteString(stderr, core.Sprintf("%s slice-smoke: expected exactly one model path\n", cliName()))
		fs.Usage()
		return 2
	}
	if core.Trim(*output) == "" {
		core.WriteString(stderr, core.Sprintf("%s slice-smoke: -output is required\n", cliName()))
		fs.Usage()
		return 2
	}

	source := fs.Arg(0)
	report := &sliceSmokeReport{
		Version:    1,
		SourcePath: source,
		OutputPath: *output,
		Preset:     inference.ModelSlicePreset(*preset),
	}
	sliceStart := time.Now()
	plan, err := mlx.SliceModel(ctx, inference.ModelSliceRequest{
		Preset:     inference.ModelSlicePreset(*preset),
		Model:      inference.ModelIdentity{Path: source},
		OutputPath: *output,
	})
	report.SliceDuration = time.Since(sliceStart)
	report.Slice = plan
	report.OutputWeightBytes = fileSize(core.PathJoin(*output, "model.safetensors"))
	if err != nil {
		report.Error = err.Error()
		return finishSliceSmokeReport(report, jsonOut, stdout, stderr)
	}
	placement, err := mlx.InspectModelSlice(*output)
	if err != nil {
		report.Error = err.Error()
		return finishSliceSmokeReport(report, jsonOut, stdout, stderr)
	}
	report.Placement = &placement
	if placement.RequiresSplitPlacement {
		estimate, estimateErr := runSliceSmokeEstimateCPUFFNMemory(ctx, source, *cpuFFNCache)
		report.CPUFFNMemoryEstimate = estimate
		if estimateErr != nil {
			report.CPUFFNMemoryEstimateError = estimateErr.Error()
		}
		if !*split {
			report.ReloadSkipped = true
			return finishSliceSmokeReport(report, jsonOut, stdout, stderr)
		}
		result, err := runSliceSmokeSplitGenerate(ctx, *output, *prompt, *maxTokens, *contextLen, *device, *cpuFFNCache)
		report.SplitDuration = result.Duration
		report.SplitOutput = result.Output
		report.CPUFFNMemory = result.CPUFFNMemory
		report.CPUFFNMemoryEstimate = result.CPUFFNMemoryEstimate
		if err != nil {
			report.Error = err.Error()
		}
		return finishSliceSmokeReport(report, jsonOut, stdout, stderr)
	}

	loadOptions := []mlx.LoadOption{}
	if *contextLen > 0 {
		loadOptions = append(loadOptions, mlx.WithContextLength(*contextLen))
	}
	if *device != "" {
		loadOptions = append(loadOptions, mlx.WithDevice(*device))
	}
	loadStart := time.Now()
	loaded, err := loadBenchModel(*output, loadOptions...)
	report.LoadDuration = time.Since(loadStart)
	if err != nil {
		report.Error = err.Error()
		return finishSliceSmokeReport(report, jsonOut, stdout, stderr)
	}
	if loaded != nil {
		defer loaded.Close()
	}

	cfg := defaultBench
	cfg.Model = core.PathBase(*output)
	cfg.ModelPath = *output
	cfg.Prompt = *prompt
	cfg.CachePrompt = ""
	cfg.MaxTokens = *maxTokens
	cfg.Runs = *runs
	cfg.IncludePromptCache = false
	cfg.IncludeKVRestore = false
	cfg.IncludeStateBundleRoundTrip = false
	cfg.IncludeProbeOverhead = false
	benchStart := time.Now()
	report.Bench, err = runBenchReport(ctx, loaded, cfg)
	report.BenchDuration = time.Since(benchStart)
	if err != nil {
		report.Error = err.Error()
		return finishSliceSmokeReport(report, jsonOut, stdout, stderr)
	}
	return finishSliceSmokeReport(report, jsonOut, stdout, stderr)
}

func finishSliceSmokeReport(report *sliceSmokeReport, jsonOut *bool, stdout, stderr io.Writer) int {
	if jsonOut != nil && *jsonOut {
		data := core.JSONMarshalIndent(report, "", "  ")
		if !data.OK {
			core.Print(stderr, "%s slice-smoke: marshal report failed", cliName())
			return 1
		}
		core.WriteString(stdout, string(data.Value.([]byte)))
		core.WriteString(stdout, "\n")
		if report.Error != "" {
			return 1
		}
		return 0
	}
	if report.Error != "" {
		core.Print(stderr, "%s slice-smoke: %s", cliName(), report.Error)
		return 1
	}
	printSliceSmokeSummary(stdout, report)
	return 0
}

func printSliceSmokeSummary(stdout io.Writer, report *sliceSmokeReport) {
	if report == nil {
		return
	}
	core.WriteString(stdout, core.Sprintf("slice smoke: %s\n", report.OutputPath))
	core.WriteString(stdout, core.Sprintf("  slice: %s, load: %s, bench: %s\n", report.SliceDuration, report.LoadDuration, report.BenchDuration))
	core.WriteString(stdout, core.Sprintf("  output weight bytes: %d\n", report.OutputWeightBytes))
	if report.Bench != nil {
		core.WriteString(stdout, core.Sprintf("  decode: %.1f tok/s, peak memory: %d MB\n", report.Bench.Generation.DecodeTokensPerSec, report.Bench.Generation.PeakMemoryBytes/1024/1024))
	}
	if report.SplitDuration > 0 {
		core.WriteString(stdout, core.Sprintf("  split: %s, output: %q\n", report.SplitDuration, report.SplitOutput))
	}
	if report.CPUFFNMemory != nil {
		mem := report.CPUFFNMemory
		core.WriteString(stdout, core.Sprintf("  cpu ffn: resident %d bytes, dense equivalent %d bytes, saved %d bytes\n", mem.ResidentBytes, mem.DenseEquivalentBytes, mem.SavedBytes))
	}
	if report.CPUFFNMemoryEstimate != nil {
		mem := report.CPUFFNMemoryEstimate
		core.WriteString(stdout, core.Sprintf("  cpu ffn estimate: peak %d bytes, resident %d bytes, loads %d, evictions %d\n", mem.PeakResidentBytes, mem.ResidentBytes, mem.LayerLoads, mem.EvictedLayers))
	}
}

var runCPUFFNMemoryEstimate = func(ctx context.Context, sourcePath string, cpuFFNCache int) (*mlx.CPUSplitFFNMemoryReport, error) {
	report, err := mlx.EstimateCPUSplitFFNMemory(ctx, sourcePath, mlx.WithCPUSplitFFNMaxCachedLayers(cpuFFNCache))
	if err != nil {
		return nil, err
	}
	return &report, nil
}

var runSliceSmokeEstimateCPUFFNMemory = runCPUFFNMemoryEstimate

var runDiscoverLocalRuntime = mlx.DiscoverLocalRuntime

var runPlanLocalTuning = mlx.PlanLocalTuning

var runLocalTuning = mlx.RunLocalTuning

var runGetDeviceInfo = mlx.GetDeviceInfo

var runSliceSmokeSplitGenerate = func(ctx context.Context, slicePath, prompt string, maxTokens, contextLen int, device string, cpuFFNCache int) (sliceSmokeSplitResult, error) {
	loadOptions := []mlx.LoadOption{}
	if contextLen > 0 {
		loadOptions = append(loadOptions, mlx.WithContextLength(contextLen))
	}
	if device != "" {
		loadOptions = append(loadOptions, mlx.WithDevice(device))
	}
	start := time.Now()
	executor, err := mlx.LoadSplitExecutor(
		ctx,
		slicePath,
		mlx.WithNativeSplitLocalRuntime(loadOptions...),
		mlx.WithCPUSplitFFNExecutor(mlx.WithCPUSplitFFNMaxCachedLayers(cpuFFNCache)),
	)
	if err != nil {
		return sliceSmokeSplitResult{Duration: time.Since(start)}, err
	}
	estimate, err := executor.CPUSplitFFNMemoryEstimate(ctx)
	if err != nil {
		return sliceSmokeSplitResult{Duration: time.Since(start)}, err
	}
	text, err := executor.Generate(ctx, prompt, mlx.GenerateConfig{MaxTokens: maxTokens, Temperature: 0})
	return sliceSmokeSplitResult{
		Output:               text,
		Duration:             time.Since(start),
		CPUFFNMemory:         executor.CPUSplitFFNMemoryReport(),
		CPUFFNMemoryEstimate: estimate,
	}, err
}

func fileSize(path string) int64 {
	stat := core.Stat(path)
	if !stat.OK {
		return 0
	}
	return stat.Value.(core.FsFileInfo).Size()
}

func runSliceCommand(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet(cliCommandName("slice"), flag.ContinueOnError)
	fs.SetOutput(stderr)
	jsonOut := fs.Bool("json", false, "print JSON slice plan")
	preset := fs.String("preset", string(inference.ModelSlicePresetClient), "slice preset: client, attention, embed, server, browse, router, expert_server, full")
	output := fs.String("output", "", "output directory for the materialised slice")
	fs.Usage = func() {
		core.WriteString(stderr, core.Sprintf("Usage: %s slice [flags] <model-path>\n", cliName()))
		fs.VisitAll(func(f *flag.Flag) {
			if f.DefValue == "" {
				core.WriteString(stderr, core.Sprintf("  -%s\n\t%s\n", f.Name, f.Usage))
				return
			}
			core.WriteString(stderr, core.Sprintf("  -%s\n\t%s (default %q)\n", f.Name, f.Usage, f.DefValue))
		})
	}
	if err := fs.Parse(args); err != nil {
		if core.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	if fs.NArg() != 1 {
		core.WriteString(stderr, core.Sprintf("%s slice: expected exactly one model path\n", cliName()))
		fs.Usage()
		return 2
	}
	if core.Trim(*output) == "" {
		core.WriteString(stderr, core.Sprintf("%s slice: -output is required\n", cliName()))
		fs.Usage()
		return 2
	}

	plan, err := mlx.SliceModel(ctx, inference.ModelSliceRequest{
		Preset:     inference.ModelSlicePreset(*preset),
		Model:      inference.ModelIdentity{Path: fs.Arg(0)},
		OutputPath: *output,
	})
	if err != nil {
		core.Print(stderr, "%s slice: %v", cliName(), err)
		return 1
	}
	if *jsonOut {
		data := core.JSONMarshalIndent(plan, "", "  ")
		if !data.OK {
			core.Print(stderr, "%s slice: marshal report failed", cliName())
			return 1
		}
		core.WriteString(stdout, string(data.Value.([]byte)))
		core.WriteString(stdout, "\n")
		return 0
	}
	printSliceSummary(stdout, plan)
	return 0
}

func printSliceSummary(stdout io.Writer, plan *inference.ModelSlicePlan) {
	if plan == nil {
		return
	}
	core.WriteString(stdout, core.Sprintf("model slice: %s\n", plan.OutputPath))
	core.WriteString(stdout, core.Sprintf("  preset: %s, components: %d\n", plan.Preset, len(plan.Components)))
	if plan.Labels != nil {
		core.WriteString(stdout, core.Sprintf("  tensors: %s, selected bytes: %s / %s\n", plan.Labels["tensor_count"], plan.Labels["selected_tensor_bytes"], plan.Labels["source_tensor_bytes"]))
		if plan.Labels["retained_tensor_ratio"] != "" {
			core.WriteString(stdout, core.Sprintf("  retained tensor ratio: %s\n", plan.Labels["retained_tensor_ratio"]))
		}
	}
}

var (
	loadBenchModel                    = mlx.LoadModel
	loadSpeculativePair               = mlx.LoadSpeculativePair
	runBenchReport                    = mlx.RunFastEvalBench
	runBenchReportWithDraft           = mlx.RunFastEvalBenchWithDraft
	runBenchReportWithSpeculativePair = mlx.RunFastEvalBenchWithSpeculativePair
)

func runBenchCommand(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	cfg := bench.DefaultConfig()
	fs := flag.NewFlagSet(cliCommandName("bench"), flag.ContinueOnError)
	fs.SetOutput(stderr)
	jsonOut := fs.Bool("json", false, "print JSON report")
	profilePath := fs.String("profile", "", "saved tuning profile to apply before loading the model")
	prompt := fs.String("prompt", cfg.Prompt, "baseline benchmark prompt")
	promptFile := fs.String("prompt-file", "", "read baseline benchmark prompt text from a file")
	promptRepeat := fs.Int("prompt-repeat", 1, "repeat the resolved benchmark prompt N times")
	promptSuffix := fs.String("prompt-suffix", "", "append extra text to the resolved benchmark prompt")
	promptSuffixFile := fs.String("prompt-suffix-file", "", "read prompt suffix text from a file")
	cachePrompt := fs.String("cache-prompt", "", "stable prompt used for prompt-cache and KV restore checks")
	maxTokens := fs.Int("max-tokens", cfg.MaxTokens, "generated tokens per pass")
	runs := fs.Int("runs", cfg.Runs, "baseline generation passes")
	contextLen := fs.Int("context", 0, "override context length")
	prefillChunkSize := fs.Int("prefill-chunk-size", 0, "override long-prompt prefill chunk size in tokens")
	cacheMode := fs.String("cache-mode", "", "override KV cache mode: fp16, q8, k-q8-v-q4, or paged")
	device := fs.String("device", "", "execution device: gpu or cpu")
	fastGemma4Lane := fs.Bool("fast-gemma4-lane", true, "enable the accepted Gemma 4 fast runtime gates by default; set false for baseline diagnostics")
	speculativeDraftModel := fs.String("speculative-draft-model", "", "assistant/draft model path for speculative decode metrics")
	speculativeDraftTokens := fs.Int("speculative-draft-tokens", 2, "draft tokens proposed per speculative decode pass")
	noCache := fs.Bool("no-cache", false, "skip prompt-cache warm/hit check")
	noRestore := fs.Bool("no-restore", false, "skip KV restore latency check")
	noBundle := fs.Bool("no-bundle", false, "skip state-bundle round trip check")
	noProbes := fs.Bool("no-probes", false, "skip probe overhead check")
	memvidKVWarm := fs.Bool("memvid-kv-warm", false, "include memvid KV block build, restore, and warmed generation check")
	memvidKVBlockSize := fs.Int("memvid-kv-block-size", 0, "memvid KV block size in tokens; 0 uses the runtime default")
	memvidKVPrefixTokens := fs.Int("memvid-kv-prefix-tokens", 0, "tokens to restore from memvid KV blocks; 0 restores the full captured prefix")
	memvidKVStore := fs.String("memvid-kv-store", "", "path for the memvid KV block store; empty uses a temporary file")
	fs.Usage = func() {
		core.WriteString(stderr, core.Sprintf("Usage: %s bench [flags] [model-path]\n", cliName()))
		fs.VisitAll(func(f *flag.Flag) {
			if f.DefValue == "" {
				core.WriteString(stderr, core.Sprintf("  -%s\n\t%s\n", f.Name, f.Usage))
				return
			}
			core.WriteString(stderr, core.Sprintf("  -%s\n\t%s (default %q)\n", f.Name, f.Usage, f.DefValue))
		})
	}
	if err := fs.Parse(args); err != nil {
		if core.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	visitedFlags := driverProfileVisitedFlags(fs)
	if driverProfileFastGemma4LaneEnabled(*fastGemma4Lane, visitedFlags, *profilePath) {
		for _, restore := range applyGemma4FastLaneDefaults(
			visitedFlags,
			contextLen,
			cacheMode,
			prefillChunkSize,
			nil,
			mlx.ProductionLaneContextLength,
		) {
			defer restore()
		}
	}
	if fs.NArg() > 1 || (fs.NArg() == 0 && core.Trim(*profilePath) == "") {
		core.WriteString(stderr, core.Sprintf("%s bench: expected one model path or -profile\n", cliName()))
		fs.Usage()
		return 2
	}
	if *promptRepeat < 1 {
		core.WriteString(stderr, core.Sprintf("%s bench: prompt repeat must be >= 1\n", cliName()))
		return 2
	}
	if *memvidKVBlockSize < 0 {
		core.WriteString(stderr, core.Sprintf("%s bench: memvid KV block size must be >= 0\n", cliName()))
		return 2
	}
	if *memvidKVPrefixTokens < 0 {
		core.WriteString(stderr, core.Sprintf("%s bench: memvid KV prefix tokens must be >= 0\n", cliName()))
		return 2
	}
	if *prefillChunkSize < 0 {
		core.WriteString(stderr, core.Sprintf("%s bench: prefill chunk size must be >= 0\n", cliName()))
		return 2
	}
	if core.Trim(*promptFile) != "" {
		read := core.ReadFile(*promptFile)
		if !read.OK {
			core.Print(stderr, "%s bench: prompt file: %v", cliName(), read.Value)
			return 1
		}
		*prompt = string(read.Value.([]byte))
	}
	if core.Trim(*promptSuffixFile) != "" {
		read := core.ReadFile(*promptSuffixFile)
		if !read.OK {
			core.Print(stderr, "%s bench: prompt suffix file: %v", cliName(), read.Value)
			return 1
		}
		*promptSuffix = string(read.Value.([]byte))
	}
	resolvedPrompt := appendDriverProfilePromptSuffix(repeatDriverProfilePrompt(*prompt, *promptRepeat), *promptSuffix)

	modelPath := ""
	loadOptions := []mlx.LoadOption{}
	if core.Trim(*profilePath) != "" {
		report, err := readTuneProfileReport(*profilePath)
		if err != nil {
			core.Print(stderr, "%s bench: profile: %v", cliName(), err)
			return 1
		}
		if report.Profile == nil {
			core.Print(stderr, "%s bench: profile payload missing", cliName())
			return 1
		}
		modelPath = report.ModelPath
		loadOptions = append(loadOptions, mlx.TuningCandidateLoadOptions(report.Profile.Candidate)...)
	}
	if fs.NArg() == 1 {
		modelPath = fs.Arg(0)
	}
	if core.Trim(modelPath) == "" {
		core.WriteString(stderr, core.Sprintf("%s bench: model path missing from profile\n", cliName()))
		fs.Usage()
		return 2
	}
	cfg.Model = core.PathBase(modelPath)
	cfg.ModelPath = modelPath
	cfg.Prompt = resolvedPrompt
	cfg.CachePrompt = *cachePrompt
	cfg.MaxTokens = *maxTokens
	cfg.Runs = *runs
	cfg.IncludePromptCache = !*noCache
	cfg.IncludeKVRestore = !*noRestore
	cfg.IncludeStateBundleRoundTrip = !*noBundle
	cfg.IncludeProbeOverhead = !*noProbes
	cfg.IncludeMemvidKVBlockWarm = *memvidKVWarm
	cfg.MemvidKVBlockSize = *memvidKVBlockSize
	cfg.MemvidKVPrefixTokens = *memvidKVPrefixTokens
	cfg.MemvidKVBlockStorePath = core.Trim(*memvidKVStore)
	if *speculativeDraftTokens < 0 {
		core.WriteString(stderr, core.Sprintf("%s bench: speculative draft tokens must be >= 0\n", cliName()))
		return 2
	}
	if core.Trim(*speculativeDraftModel) != "" {
		cfg.IncludeSpeculativeDecode = true
		cfg.SpeculativeDraftModelPath = core.Trim(*speculativeDraftModel)
		cfg.SpeculativeDraftTokens = *speculativeDraftTokens
	}

	if *contextLen > 0 {
		loadOptions = append(loadOptions, mlx.WithContextLength(*contextLen))
	}
	if *prefillChunkSize > 0 {
		loadOptions = append(loadOptions, mlx.WithPrefillChunkSize(*prefillChunkSize))
	}
	if core.Trim(*cacheMode) != "" {
		mode := memory.KVCacheMode(core.Trim(*cacheMode))
		switch mode {
		case memory.KVCacheModeFP16, memory.KVCacheModeQ8, memory.KVCacheModeKQ8VQ4, memory.KVCacheModePaged:
		default:
			core.WriteString(stderr, core.Sprintf("%s bench: unsupported cache mode %q\n", cliName(), string(mode)))
			return 2
		}
		loadOptions = append(loadOptions, mlx.WithKVCacheMode(mode))
	}
	if *device != "" {
		loadOptions = append(loadOptions, mlx.WithDevice(*device))
	}
	if cfg.IncludeSpeculativeDecode {
		pair, err := loadSpeculativePair(modelPath, cfg.SpeculativeDraftModelPath, mlx.SpeculativePairConfig{
			TargetOptions: loadOptions,
			DraftOptions:  loadOptions,
		})
		if err != nil {
			core.Print(stderr, "%s bench: load speculative pair: %v", cliName(), err)
			return 1
		}
		defer pair.Close()
		report, err := runBenchReportWithDraft(ctx, pair.Target, pair.Draft, cfg)
		if pair.Gemma4Assistant != nil {
			report, err = runBenchReportWithSpeculativePair(ctx, pair, cfg)
		}
		if err != nil {
			core.Print(stderr, "%s bench: %v", cliName(), err)
			return 1
		}
		if *jsonOut {
			data := core.JSONMarshalIndent(report, "", "  ")
			if !data.OK {
				core.Print(stderr, "%s bench: marshal report failed", cliName())
				return 1
			}
			core.WriteString(stdout, string(data.Value.([]byte)))
			core.WriteString(stdout, "\n")
			return 0
		}
		printBenchSummary(stdout, report)
		return 0
	}
	model, err := loadBenchModel(modelPath, loadOptions...)
	if err != nil {
		core.Print(stderr, "%s bench: load model: %v", cliName(), err)
		return 1
	}
	defer model.Close()

	report, err := runBenchReport(ctx, model, cfg)
	if err != nil {
		core.Print(stderr, "%s bench: %v", cliName(), err)
		return 1
	}
	if *jsonOut {
		data := core.JSONMarshalIndent(report, "", "  ")
		if !data.OK {
			core.Print(stderr, "%s bench: marshal report failed", cliName())
			return 1
		}
		core.WriteString(stdout, string(data.Value.([]byte)))
		core.WriteString(stdout, "\n")
		return 0
	}
	printBenchSummary(stdout, report)
	return 0
}

func printBenchSummary(stdout io.Writer, report *bench.Report) {
	if report == nil {
		return
	}
	core.WriteString(stdout, core.Sprintf("fast eval: %s\n", report.ModelPath))
	core.WriteString(stdout, core.Sprintf("  prefill: %.1f tok/s, decode: %.1f tok/s\n", report.Generation.PrefillTokensPerSec, report.Generation.DecodeTokensPerSec))
	core.WriteString(stdout, core.Sprintf("  peak memory: %d MB, active memory: %d MB\n", report.Generation.PeakMemoryBytes/1024/1024, report.Generation.ActiveMemoryBytes/1024/1024))
	if report.PromptCache.Attempted {
		core.WriteString(stdout, core.Sprintf("  prompt cache: %.0f%% hit rate (%d hit, %d miss)\n", report.PromptCache.HitRate*100, report.PromptCache.Hits, report.PromptCache.Misses))
	}
	if report.KVRestore.Attempted {
		core.WriteString(stdout, core.Sprintf("  KV restore: %s\n", report.KVRestore.Duration))
	}
	if report.StateBundle.Attempted {
		core.WriteString(stdout, core.Sprintf("  state bundle: %d bytes, %s round trip\n", report.StateBundle.Bytes, report.StateBundle.Duration))
	}
	if report.Probes.Attempted {
		core.WriteString(stdout, core.Sprintf("  probes: %d events, %.1f%% overhead\n", report.Probes.EventCount, report.Probes.OverheadRatio*100))
	}
	if report.SpeculativeDecode.Attempted {
		core.WriteString(stdout, core.Sprintf("  speculative: %.1f%% accepted (%d accepted, %d rejected), %.1f visible tok/s\n",
			report.SpeculativeDecode.Metrics.AcceptanceRate*100,
			report.SpeculativeDecode.Metrics.AcceptedTokens,
			report.SpeculativeDecode.Metrics.RejectedTokens,
			report.SpeculativeDecode.Metrics.VisibleTokensPerSec,
		))
	}
}

func runPackCommand(_ context.Context, args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet(cliCommandName("pack"), flag.ContinueOnError)
	fs.SetOutput(stderr)
	jsonOut := fs.Bool("json", false, "print JSON report")
	expectedQuant := fs.Int("quantization", 0, "required quantization bits")
	maxContext := fs.Int("max-context", 0, "maximum allowed context length")
	fs.Usage = func() {
		core.WriteString(stderr, core.Sprintf("Usage: %s pack [flags] <model-path>\n", cliName()))
		fs.VisitAll(func(f *flag.Flag) {
			if f.DefValue == "" {
				core.WriteString(stderr, core.Sprintf("  -%s\n\t%s\n", f.Name, f.Usage))
				return
			}
			core.WriteString(stderr, core.Sprintf("  -%s\n\t%s (default %q)\n", f.Name, f.Usage, f.DefValue))
		})
	}
	if err := fs.Parse(args); err != nil {
		if core.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	if fs.NArg() != 1 {
		core.WriteString(stderr, core.Sprintf("%s pack: expected exactly one model path\n", cliName()))
		fs.Usage()
		return 2
	}

	options := []pack.ModelPackOption{}
	if *expectedQuant > 0 {
		options = append(options, pack.WithPackQuantization(*expectedQuant))
	}
	if *maxContext > 0 {
		options = append(options, pack.WithPackMaxContextLength(*maxContext))
	}
	pack, err := model.Inspect(fs.Arg(0), options...)
	if err != nil {
		core.Print(stderr, "%s pack: %v", cliName(), err)
		return 1
	}
	if *jsonOut {
		data := core.JSONMarshal(pack)
		if !data.OK {
			core.Print(stderr, "%s pack: marshal report failed", cliName())
			return 1
		}
		core.WriteString(stdout, string(data.Value.([]byte)))
		core.WriteString(stdout, "\n")
		if !pack.Valid() {
			return 1
		}
		return 0
	}
	if !pack.Valid() {
		printPackIssues(stderr, pack)
		return 1
	}
	core.WriteString(stdout, core.Sprintf(
		"valid model pack: %s (%s, %s, quant=%d, context=%d)\n",
		pack.Root,
		pack.Architecture,
		pack.Format,
		pack.QuantBits,
		pack.ContextLength,
	))
	return 0
}

func printPackIssues(stderr io.Writer, p pack.ModelPack) {
	core.WriteString(stderr, core.Sprintf("%s pack: invalid model pack\n", cliName()))
	for _, issue := range p.Issues {
		if issue.Severity != pack.ModelPackIssueError {
			continue
		}
		core.WriteString(stderr, core.Sprintf("  %s: %s\n", issue.Code, issue.Message))
	}
}

func printUsage(w io.Writer) {
	core.WriteString(w, core.Sprintf("Usage: %s <command> [flags]\n", cliName()))
	core.WriteString(w, "\n")
	core.WriteString(w, "Commands:\n")
	core.WriteString(w, "  bench   run fast local eval/benchmark harness\n")
	core.WriteString(w, "  discover  report local MLX runtime and optional model candidates\n")
	core.WriteString(w, "  driver-profile  measure load, first-token, and decode timings for one question\n")
	core.WriteString(w, "  ffn-estimate  estimate split CPU FFN memory without loading the model\n")
	core.WriteString(w, "  pack    validate a local native model pack\n")
	core.WriteString(w, "  profile-list  list saved tuning profiles for a machine/model/workload\n")
	core.WriteString(w, "  profile-select  select the best saved tuning profile for a machine/model/workload\n")
	core.WriteString(w, "  replace-plan  plan state handling for a profile/model reload\n")
	core.WriteString(w, "  slice   materialise a local model slice for split/reload tests\n")
	core.WriteString(w, "  slice-smoke  materialise, reload, and benchmark a model slice\n")
	core.WriteString(w, "  state-ramp-profile  measure warm retained-state growth across append/generate turns\n")
	core.WriteString(w, "  tune-plan  plan local tuning candidates for a model\n")
	core.WriteString(w, "  tune-profile  read a saved tuning profile and print reusable load settings\n")
	core.WriteString(w, "  tune-run  run and stream local tuning candidate measurements\n")
}
