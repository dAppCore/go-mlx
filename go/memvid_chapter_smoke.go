// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"dappco.re/go/mlx/blockcache"
	"context"
	"time"

	core "dappco.re/go"
	memvid "dappco.re/go/inference/state"
	"dappco.re/go/mlx/kv"
	filestore "dappco.re/go/inference/state/filestore"
	memvidcli "dappco.re/go/mlx/pkg/memvid/cli"
)

const (
	DefaultMemvidKVChapterSmokeAnswerMaxTokens = 32

	MemvidKVChapterSmokeStoreFileLog = "file-log"
	MemvidKVChapterSmokeStoreCLI     = "cli"
)

// MemvidKVChapterRunner is the small driver surface the chapter-smoke
// orchestration needs. The callbacks deal with mlx-specific kv / memvid
// types that the driver-neutral bench package keeps opaque.
type MemvidKVChapterRunner struct {
	CaptureKVBlocksToMemvid  func(context.Context, string, memvid.Writer, kv.MemvidBlockOptions) (*kv.MemvidBlockBundle, error)
	GenerateWithMemvidPrefix func(context.Context, memvid.Store, *kv.MemvidBlockBundle, int, string, GenerateConfig) (ChapterGeneration, error)
}

// ChapterGeneration is one generation step's result inside the chapter-smoke flow.
type ChapterGeneration struct {
	Text    string  `json:"text,omitempty"`
	Tokens  []Token `json:"tokens,omitempty"`
	Metrics Metrics `json:"metrics"`
}

// NewModelMemvidKVChapterRunner builds the chapter-smoke runner from a loaded Model.
func NewModelMemvidKVChapterRunner(model *Model) MemvidKVChapterRunner {
	return MemvidKVChapterRunner{
		CaptureKVBlocksToMemvid: func(ctx context.Context, prompt string, store memvid.Writer, opts kv.MemvidBlockOptions) (*kv.MemvidBlockBundle, error) {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			session, err := model.NewSession()
			if err != nil {
				return nil, err
			}
			defer session.Close()
			if err := session.Prefill(prompt); err != nil {
				return nil, err
			}
			return session.SaveKVBlocksToMemvid(ctx, store, opts)
		},
		GenerateWithMemvidPrefix: func(ctx context.Context, store memvid.Store, bundle *kv.MemvidBlockBundle, prefixTokens int, suffix string, cfg GenerateConfig) (ChapterGeneration, error) {
			if err := ctx.Err(); err != nil {
				return ChapterGeneration{}, err
			}
			session, err := model.NewSession()
			if err != nil {
				return ChapterGeneration{}, err
			}
			defer session.Close()
			loadOpts := kv.LoadOptions{}
			if bundle != nil && bundle.KVEncoding == kv.EncodingNative {
				loadOpts.RawKVOnly = true
			}
			restoreStart := time.Now()
			snapshot, err := kv.LoadPrefixFromMemvidBlocksWithOptions(ctx, store, bundle, prefixTokens, loadOpts)
			if err != nil {
				return ChapterGeneration{}, err
			}
			if err := session.RestoreKV(snapshot); err != nil {
				return ChapterGeneration{}, err
			}
			restoreDuration := time.Since(restoreStart)
			if err := session.AppendPrompt(suffix); err != nil {
				return ChapterGeneration{}, err
			}
			text, err := session.Generate(memvidKVChapterGenerateOptions(cfg)...)
			metrics := model.Metrics()
			metrics.PromptCacheRestoreDuration = restoreDuration
			return ChapterGeneration{Text: text, Metrics: metrics}, err
		},
	}
}

func memvidKVChapterGenerateOptions(cfg GenerateConfig) []GenerateOption {
	out := []GenerateOption{
		WithMaxTokens(cfg.MaxTokens),
		WithTemperature(cfg.Temperature),
	}
	if cfg.TopK > 0 {
		out = append(out, WithTopK(cfg.TopK))
	}
	if cfg.TopP > 0 {
		out = append(out, WithTopP(cfg.TopP))
	}
	if cfg.MinP > 0 {
		out = append(out, WithMinP(cfg.MinP))
	}
	if len(cfg.StopTokens) > 0 {
		out = append(out, WithStopTokens(cfg.StopTokens...))
	}
	if cfg.RepeatPenalty > 0 {
		out = append(out, WithRepeatPenalty(cfg.RepeatPenalty))
	}
	if cfg.ProbeSink != nil {
		out = append(out, WithProbeSink(cfg.ProbeSink))
	}
	return out
}

type memvidChapterReadCountingStore struct {
	store  memvid.Store
	reads  int
	unique map[int]struct{}
}

func newMemvidChapterReadCountingStore(store memvid.Store) *memvidChapterReadCountingStore {
	return &memvidChapterReadCountingStore{store: store, unique: map[int]struct{}{}}
}

func (s *memvidChapterReadCountingStore) Get(ctx context.Context, chunkID int) (string, error) {
	s.record(chunkID)
	return s.store.Get(ctx, chunkID)
}

func (s *memvidChapterReadCountingStore) Resolve(ctx context.Context, chunkID int) (memvid.Chunk, error) {
	s.record(chunkID)
	return memvid.Resolve(ctx, s.store, chunkID)
}

func (s *memvidChapterReadCountingStore) ResolveBytes(ctx context.Context, chunkID int) (memvid.Chunk, error) {
	s.record(chunkID)
	return memvid.ResolveBytes(ctx, s.store, chunkID)
}

func (s *memvidChapterReadCountingStore) Reads() int {
	if s == nil {
		return 0
	}
	return s.reads
}

func (s *memvidChapterReadCountingStore) UniqueReads() int {
	if s == nil {
		return 0
	}
	return len(s.unique)
}

func (s *memvidChapterReadCountingStore) record(chunkID int) {
	s.reads++
	if s.unique == nil {
		s.unique = map[int]struct{}{}
	}
	s.unique[chunkID] = struct{}{}
}

func memvidChapterFileSize(path string) int64 {
	stat := core.Stat(path)
	if !stat.OK {
		return 0
	}
	return stat.Value.(core.FsFileInfo).Size()
}

// MemvidKVChapterSmokeConfig configures a small memvid-backed KV restore smoke
// over chapter-sized prompts.
type MemvidKVChapterSmokeConfig struct {
	StoreDir        string                      `json:"store_dir,omitempty"`
	StorePath       string                      `json:"store_path,omitempty"`
	StoreKind       string                      `json:"store_kind,omitempty"`
	MemvidBinary    string                      `json:"memvid_binary,omitempty"`
	BlockSize       int                         `json:"block_size,omitempty"`
	AnswerMaxTokens int                         `json:"answer_max_tokens,omitempty"`
	Temperature     float32                     `json:"temperature,omitempty"`
	Chapters        []MemvidKVChapterSmokeInput `json:"chapters,omitempty"`
	GenerateConfig  GenerateConfig              `json:"generate_config,omitempty"`
}

// MemvidKVChapterSmokeInput is one chapter-sized prefix and question.
type MemvidKVChapterSmokeInput struct {
	Name          string   `json:"name,omitempty"`
	Text          string   `json:"text"`
	Question      string   `json:"question"`
	ExpectedTerms []string `json:"expected_terms,omitempty"`
}

// MemvidKVChapterSmokeReport captures the full smoke result.
type MemvidKVChapterSmokeReport struct {
	StoreDir  string                        `json:"store_dir,omitempty"`
	StorePath string                        `json:"store_path,omitempty"`
	FileCount int                           `json:"file_count,omitempty"`
	BlockSize int                           `json:"block_size,omitempty"`
	Chapters  []MemvidKVChapterSmokeChapter `json:"chapters,omitempty"`
	Error     string                        `json:"error,omitempty"`
}

// MemvidKVChapterSmokeChapter reports one save, reopen, restore, and answer
// cycle from a memvid store.
type MemvidKVChapterSmokeChapter struct {
	Name                 string        `json:"name,omitempty"`
	Question             string        `json:"question,omitempty"`
	Source               string        `json:"source,omitempty"`
	StorePath            string        `json:"store_path,omitempty"`
	BundleURI            string        `json:"bundle_uri,omitempty"`
	StoreBytes           int64         `json:"store_bytes,omitempty"`
	BlockSize            int           `json:"block_size,omitempty"`
	TotalBlocks          int           `json:"total_blocks,omitempty"`
	BlocksRead           int           `json:"blocks_read,omitempty"`
	ChunksRead           int           `json:"chunks_read,omitempty"`
	PrefixTokensRestored int           `json:"prefix_tokens_restored,omitempty"`
	CaptureDuration      time.Duration `json:"capture_duration,omitempty"`
	SaveDuration         time.Duration `json:"save_duration,omitempty"`
	ReopenDuration       time.Duration `json:"reopen_duration,omitempty"`
	RestoreDuration      time.Duration `json:"restore_duration,omitempty"`
	AnswerDuration       time.Duration `json:"answer_duration,omitempty"`
	Answer               string        `json:"answer,omitempty"`
	Plausible            bool          `json:"plausible"`
	Error                string        `json:"error,omitempty"`
}

func RunModelMemvidKVChapterSmoke(ctx context.Context, model *Model, cfg MemvidKVChapterSmokeConfig) (*MemvidKVChapterSmokeReport, error) {
	if model == nil {
		return nil, core.NewError("mlx: model is nil")
	}
	return RunMemvidKVChapterSmoke(ctx, NewModelMemvidKVChapterRunner(model), cfg)
}

func RunMemvidKVChapterSmoke(ctx context.Context, runner MemvidKVChapterRunner, cfg MemvidKVChapterSmokeConfig) (*MemvidKVChapterSmokeReport, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	cfg = normalizeMemvidKVChapterSmokeConfig(cfg)
	if err := validateMemvidKVChapterSmokeStoreKind(cfg.StoreKind); err != nil {
		return nil, err
	}
	if runner.GenerateWithMemvidPrefix == nil {
		return nil, core.NewError("mlx: memvid chapter smoke requires GenerateWithMemvidPrefix")
	}
	if runner.CaptureKVBlocksToMemvid == nil {
		return nil, core.NewError("mlx: memvid chapter smoke requires CaptureKVBlocksToMemvid")
	}
	if len(cfg.Chapters) == 0 {
		return nil, core.NewError("mlx: memvid chapter smoke requires at least one chapter")
	}
	storeDir, storePath, err := memvidKVChapterSmokeStorePaths(cfg)
	if err != nil {
		return nil, err
	}
	report := &MemvidKVChapterSmokeReport{
		StoreDir:  storeDir,
		StorePath: storePath,
		BlockSize: cfg.BlockSize,
		Chapters:  make([]MemvidKVChapterSmokeChapter, 0, len(cfg.Chapters)),
	}
	defer func() {
		report.FileCount = memvidKVChapterSmokeFileCount(storeDir)
	}()
	for i, chapter := range cfg.Chapters {
		chapterReport, err := runMemvidKVChapterSmokeChapter(ctx, runner, cfg, storePath, i, chapter)
		report.Chapters = append(report.Chapters, chapterReport)
		if err != nil {
			report.Error = err.Error()
			return report, err
		}
	}
	return report, nil
}

func memvidKVChapterSmokeFileCount(dir string) int {
	count := 0
	for _, path := range core.PathGlob(core.PathJoin(dir, "*")) {
		stat := core.Stat(path)
		if !stat.OK {
			continue
		}
		info := stat.Value.(core.FsFileInfo)
		if !info.IsDir() {
			count++
		}
	}
	return count
}

func runMemvidKVChapterSmokeChapter(ctx context.Context, runner MemvidKVChapterRunner, cfg MemvidKVChapterSmokeConfig, storePath string, index int, chapter MemvidKVChapterSmokeInput) (MemvidKVChapterSmokeChapter, error) {
	report := MemvidKVChapterSmokeChapter{
		Name:      memvidKVChapterSmokeName(index, chapter.Name),
		Question:  chapter.Question,
		Source:    memvidKVChapterSmokeStoreSource(cfg),
		BlockSize: cfg.BlockSize,
		StorePath: storePath,
		BundleURI: memvidKVChapterSmokeBundleURI(index, chapter.Name),
	}
	if core.Trim(chapter.Text) == "" {
		return memvidKVChapterSmokeChapterError(report, "mlx: memvid chapter smoke chapter text is empty")
	}
	if core.Trim(chapter.Question) == "" {
		return memvidKVChapterSmokeChapterError(report, "mlx: memvid chapter smoke chapter question is empty")
	}

	store, err := memvidKVChapterSmokeOpenWriteStore(ctx, cfg, report.StorePath, index)
	if err != nil {
		return memvidKVChapterSmokeChapterError(report, err.Error())
	}
	captureStart := time.Now()
	bundle, err := runner.CaptureKVBlocksToMemvid(ctx, chapter.Text, store.Writer, kv.MemvidBlockOptions{
		BlockSize:  cfg.BlockSize,
		KVEncoding: kv.EncodingNative,
		URI:        "mlx://memvid-chapter-smoke/" + memvidKVChapterSmokeSlug(index, chapter.Name),
		Labels:     []string{"chapter-smoke", "memvid-kv"},
	})
	report.CaptureDuration = nonZeroDuration(time.Since(captureStart))
	if err == nil {
		_, err = kv.SaveMemvidBlockBundle(ctx, store.Writer, bundle, report.BundleURI)
	}
	closeErr := store.Close()
	report.SaveDuration = report.CaptureDuration
	if err != nil {
		return memvidKVChapterSmokeChapterError(report, err.Error())
	}
	if closeErr != nil {
		return memvidKVChapterSmokeChapterError(report, closeErr.Error())
	}
	report.TotalBlocks = len(bundle.Blocks)
	report.StoreBytes = memvidChapterFileSize(report.StorePath)
	report.PrefixTokensRestored = bundle.TokenCount
	if report.TotalBlocks == 0 {
		return memvidKVChapterSmokeChapterError(report, "mlx: memvid chapter smoke wrote no KV blocks")
	}
	if report.StoreBytes <= 0 {
		return memvidKVChapterSmokeChapterError(report, "mlx: memvid chapter smoke wrote empty file store")
	}

	reopenStart := time.Now()
	reader, err := memvidKVChapterSmokeOpenReadStore(ctx, cfg, report.StorePath)
	report.ReopenDuration = nonZeroDuration(time.Since(reopenStart))
	if err != nil {
		return memvidKVChapterSmokeChapterError(report, err.Error())
	}
	loadedBundle, err := kv.LoadMemvidBlockBundle(ctx, reader.Store, report.BundleURI)
	if err != nil {
		closeErr = reader.Close()
		if closeErr != nil {
			return memvidKVChapterSmokeChapterError(report, closeErr.Error())
		}
		return memvidKVChapterSmokeChapterError(report, err.Error())
	}
	countingStore := newMemvidChapterReadCountingStore(reader.Store)
	restoreStart := time.Now()
	generation, err := runner.GenerateWithMemvidPrefix(ctx, countingStore, loadedBundle, loadedBundle.TokenCount, memvidKVChapterSmokeQuestionPrompt(chapter), memvidKVChapterSmokeGenerateConfig(cfg))
	report.RestoreDuration = nonZeroDuration(time.Since(restoreStart))
	if generation.Metrics.PromptCacheRestoreDuration > 0 {
		report.RestoreDuration = generation.Metrics.PromptCacheRestoreDuration
	}
	report.BlocksRead = countingStore.UniqueReads()
	report.ChunksRead = countingStore.Reads()
	closeErr = reader.Close()
	if err != nil {
		return memvidKVChapterSmokeChapterError(report, err.Error())
	}
	if closeErr != nil {
		return memvidKVChapterSmokeChapterError(report, closeErr.Error())
	}

	report.AnswerDuration = generation.Metrics.DecodeDuration
	if report.AnswerDuration <= 0 {
		report.AnswerDuration = generation.Metrics.TotalDuration
	}
	report.AnswerDuration = nonZeroDuration(report.AnswerDuration)
	report.Answer = firstNonEmpty(generation.Text, renderTokensText(generation.Tokens))
	report.Plausible = memvidKVChapterSmokeAnswerPlausible(report.Answer, chapter.ExpectedTerms)
	return report, nil
}

func normalizeMemvidKVChapterSmokeConfig(cfg MemvidKVChapterSmokeConfig) MemvidKVChapterSmokeConfig {
	cfg.StoreKind = memvidKVChapterSmokeNormalizeStoreKind(cfg.StoreKind, cfg.StorePath)
	if cfg.BlockSize <= 0 {
		cfg.BlockSize = blockcache.DefaultBlockSize
	}
	if cfg.AnswerMaxTokens <= 0 && cfg.GenerateConfig.MaxTokens <= 0 {
		cfg.AnswerMaxTokens = DefaultMemvidKVChapterSmokeAnswerMaxTokens
	}
	cfg.Chapters = append([]MemvidKVChapterSmokeInput(nil), cfg.Chapters...)
	return cfg
}

func memvidKVChapterSmokeGenerateConfig(cfg MemvidKVChapterSmokeConfig) GenerateConfig {
	gen := cfg.GenerateConfig
	if gen.MaxTokens <= 0 {
		gen.MaxTokens = cfg.AnswerMaxTokens
	}
	if gen.Temperature == 0 {
		gen.Temperature = cfg.Temperature
	}
	return gen
}

func memvidKVChapterSmokeStorePaths(cfg MemvidKVChapterSmokeConfig) (string, string, error) {
	if core.Trim(cfg.StorePath) != "" {
		dir := core.PathDir(cfg.StorePath)
		if result := core.MkdirAll(dir, 0o755); !result.OK {
			return "", "", core.E("mlx.memvidKVChapterSmokeStoreDir", "create store path parent", memvidKVChapterSmokeResultError(result))
		}
		return dir, cfg.StorePath, nil
	}
	if core.Trim(cfg.StoreDir) != "" {
		if result := core.MkdirAll(cfg.StoreDir, 0o755); !result.OK {
			return "", "", core.E("mlx.memvidKVChapterSmokeStoreDir", "create store dir", memvidKVChapterSmokeResultError(result))
		}
		return cfg.StoreDir, core.PathJoin(cfg.StoreDir, memvidKVChapterSmokeStoreFileName(cfg.StoreKind)), nil
	}
	result := core.MkdirTemp("", "go-mlx-chapter-smoke-*")
	if !result.OK {
		return "", "", core.E("mlx.memvidKVChapterSmokeStoreDir", "create temp store dir", memvidKVChapterSmokeResultError(result))
	}
	dir := result.Value.(string)
	return dir, core.PathJoin(dir, memvidKVChapterSmokeStoreFileName(cfg.StoreKind)), nil
}

type memvidKVChapterSmokeStore struct {
	Store  memvid.Store
	Writer memvid.Writer
	close  func() error
}

func (s memvidKVChapterSmokeStore) Close() error {
	if s.close == nil {
		return nil
	}
	return s.close()
}

func memvidKVChapterSmokeOpenWriteStore(ctx context.Context, cfg MemvidKVChapterSmokeConfig, path string, index int) (memvidKVChapterSmokeStore, error) {
	switch cfg.StoreKind {
	case MemvidKVChapterSmokeStoreCLI:
		if index == 0 {
			store, err := memvidcli.Create(ctx, path, memvidKVChapterSmokeCLIOptions(cfg)...)
			return memvidKVChapterSmokeStore{Store: store, Writer: store}, err
		}
		store, err := memvidcli.Open(path, memvidKVChapterSmokeCLIOptions(cfg)...)
		return memvidKVChapterSmokeStore{Store: store, Writer: store}, err
	default:
		if index == 0 {
			store, err := filestore.Create(ctx, path)
			return memvidKVChapterSmokeStore{Store: store, Writer: store, close: store.Close}, err
		}
		store, err := filestore.Open(ctx, path)
		return memvidKVChapterSmokeStore{Store: store, Writer: store, close: store.Close}, err
	}
}

func memvidKVChapterSmokeOpenReadStore(ctx context.Context, cfg MemvidKVChapterSmokeConfig, path string) (memvidKVChapterSmokeStore, error) {
	switch cfg.StoreKind {
	case MemvidKVChapterSmokeStoreCLI:
		store, err := memvidcli.Open(path, memvidKVChapterSmokeCLIOptions(cfg)...)
		return memvidKVChapterSmokeStore{Store: store, Writer: store}, err
	default:
		store, err := filestore.Open(ctx, path)
		return memvidKVChapterSmokeStore{Store: store, Writer: store, close: store.Close}, err
	}
}

func memvidKVChapterSmokeCLIOptions(cfg MemvidKVChapterSmokeConfig) []memvidcli.Option {
	if core.Trim(cfg.MemvidBinary) == "" {
		return nil
	}
	return []memvidcli.Option{memvidcli.WithBinary(cfg.MemvidBinary)}
}

func memvidKVChapterSmokeNormalizeStoreKind(kind, path string) string {
	kind = core.Lower(core.Trim(kind))
	if kind != "" {
		switch kind {
		case "cli", "memvid", "mp4", "mv2":
			return MemvidKVChapterSmokeStoreCLI
		case "file", "file-log", "filestore", "mvlog":
			return MemvidKVChapterSmokeStoreFileLog
		default:
			return kind
		}
	}
	lowerPath := core.Lower(path)
	if core.HasSuffix(lowerPath, ".mp4") || core.HasSuffix(lowerPath, ".mv2") {
		return MemvidKVChapterSmokeStoreCLI
	}
	return MemvidKVChapterSmokeStoreFileLog
}

func validateMemvidKVChapterSmokeStoreKind(kind string) error {
	switch kind {
	case MemvidKVChapterSmokeStoreFileLog, MemvidKVChapterSmokeStoreCLI:
		return nil
	default:
		return core.NewError("mlx: unsupported memvid chapter smoke store kind")
	}
}

func memvidKVChapterSmokeStoreSource(cfg MemvidKVChapterSmokeConfig) string {
	if cfg.StoreKind == MemvidKVChapterSmokeStoreCLI {
		return memvid.CodecQRVideo
	}
	return filestore.CodecFile
}

func memvidKVChapterSmokeQuestionPrompt(chapter MemvidKVChapterSmokeInput) string {
	return "\n\nQuestion: " + chapter.Question + "\nAnswer:"
}

func memvidKVChapterSmokeAnswerPlausible(answer string, expected []string) bool {
	answer = core.Trim(answer)
	if answer == "" {
		return false
	}
	if len(expected) == 0 {
		return true
	}
	lower := core.Lower(answer)
	for _, term := range expected {
		if core.Trim(term) == "" {
			continue
		}
		if !core.Contains(lower, core.Lower(term)) {
			return false
		}
	}
	return true
}

func memvidKVChapterSmokeChapterError(report MemvidKVChapterSmokeChapter, message string) (MemvidKVChapterSmokeChapter, error) {
	report.Error = message
	return report, core.NewError(message)
}

func memvidKVChapterSmokeName(index int, name string) string {
	if core.Trim(name) != "" {
		return name
	}
	return core.Sprintf("chapter-%d", index+1)
}

func memvidKVChapterSmokeStoreFileName(kind string) string {
	if kind == MemvidKVChapterSmokeStoreCLI {
		return "memvid-kv-chapters.mp4"
	}
	return "memvid-kv-chapters.mvlog"
}

func memvidKVChapterSmokeBundleURI(index int, name string) string {
	return "mlx://memvid-chapter-smoke/" + memvidKVChapterSmokeSlug(index, name) + "/bundle"
}

func memvidKVChapterSmokeSlug(index int, name string) string {
	name = core.Lower(core.Trim(name))
	if name == "" {
		name = core.Sprintf("chapter-%d", index+1)
	}
	builder := core.NewBuilder()
	lastDash := false
	for _, r := range name {
		ok := (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9')
		if ok {
			builder.WriteRune(r)
			lastDash = false
			continue
		}
		if !lastDash {
			builder.WriteRune('-')
			lastDash = true
		}
	}
	slug := builder.String()
	for core.HasPrefix(slug, "-") {
		slug = core.TrimPrefix(slug, "-")
	}
	for core.HasSuffix(slug, "-") {
		slug = core.TrimSuffix(slug, "-")
	}
	if slug == "" {
		slug = core.Sprintf("chapter-%d", index+1)
	}
	return core.Sprintf("%02d-%s", index+1, slug)
}

func memvidKVChapterSmokeResultError(result core.Result) error {
	if result.OK {
		return nil
	}
	if err, ok := result.Value.(error); ok {
		return err
	}
	return core.NewError("core result failed")
}
