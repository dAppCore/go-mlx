// SPDX-Licence-Identifier: EUPL-1.2

// Package chaptersmoke runs chapter-sized State KV save/restore/generate
// smoke benchmarks. Driver-neutral — callers supply a Runner with the
// model-specific Capture/Generate callbacks.
//
//	runner := mlx.NewModelStateKVChapterRunner(model, baseGen)
//	report, err := chaptersmoke.Run(ctx, runner, chaptersmoke.Config{
//	    StoreDir: "/tmp/smoke",
//	    Chapters: []chaptersmoke.Input{{Text: chapter, Question: q}},
//	})
package chaptersmoke

import (
	"context"
	"strconv"
	"time"

	core "dappco.re/go"
	state "dappco.re/go/inference/state"
	filestore "dappco.re/go/inference/state/filestore"
	"dappco.re/go/mlx/blockcache"
	"dappco.re/go/mlx/kv"
	memvidcli "dappco.re/go/mlx/pkg/memvid/cli"
)

const (
	// DefaultAnswerMaxTokens caps the answer generation length when the
	// caller does not provide a higher MaxTokens setting.
	DefaultAnswerMaxTokens = 32

	// StoreFileLog selects the .mvlog filestore backend.
	StoreFileLog = "file-log"
	// StoreCLI selects the deprecated memvid CLI backend (.mp4 / .mv2 QR-video).
	StoreCLI = "cli"
)

// Runner is the small driver surface the chapter-smoke orchestration needs.
// Both callbacks close over caller-supplied model state — chaptersmoke does
// not import mlx and never sees its types directly.
type Runner struct {
	// Capture writes a chapter prompt's KV state into store as State blocks.
	Capture func(ctx context.Context, prompt string, store state.Writer, opts kv.StateBlockOptions) (*kv.StateBlockBundle, error)
	// Generate restores a State prefix, appends suffix, and decodes an answer.
	Generate func(ctx context.Context, store state.Store, bundle *kv.StateBlockBundle, prefixTokens int, suffix string) (Generation, error)
}

// Generation is one generation step's result inside the chapter-smoke flow.
type Generation struct {
	Text                       string        `json:"text,omitempty"`
	DecodeDuration             time.Duration `json:"decode_duration,omitempty"`
	TotalDuration              time.Duration `json:"total_duration,omitempty"`
	PromptCacheRestoreDuration time.Duration `json:"prompt_cache_restore_duration,omitempty"`
}

// Config configures a small State-backed KV restore smoke over
// chapter-sized prompts.
type Config struct {
	StoreDir        string  `json:"store_dir,omitempty"`
	StorePath       string  `json:"store_path,omitempty"`
	StoreKind       string  `json:"store_kind,omitempty"`
	StateBinary     string  `json:"state_binary,omitempty"`
	MemvidBinary    string  `json:"-"`
	BlockSize       int     `json:"block_size,omitempty"`
	AnswerMaxTokens int     `json:"answer_max_tokens,omitempty"`
	Temperature     float32 `json:"temperature,omitempty"`
	Chapters        []Input `json:"chapters,omitempty"`
}

// Input is one chapter-sized prefix and question.
type Input struct {
	Name          string   `json:"name,omitempty"`
	Text          string   `json:"text"`
	Question      string   `json:"question"`
	ExpectedTerms []string `json:"expected_terms,omitempty"`
}

// Report captures the full smoke result.
type Report struct {
	StoreDir  string          `json:"store_dir,omitempty"`
	StorePath string          `json:"store_path,omitempty"`
	FileCount int             `json:"file_count,omitempty"`
	BlockSize int             `json:"block_size,omitempty"`
	Chapters  []ChapterReport `json:"chapters,omitempty"`
	Error     string          `json:"error,omitempty"`
}

// ChapterReport reports one save, reopen, restore, and answer cycle from a
// State store.
type ChapterReport struct {
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

// Run executes the chapter-smoke harness. The runner's Capture and Generate
// callbacks supply all model-specific behaviour.
//
//	report, err := chaptersmoke.Run(ctx, runner, cfg)
func Run(ctx context.Context, runner Runner, cfg Config) (*Report, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	cfg = normalizeConfig(cfg)
	if err := validateStoreKind(cfg.StoreKind); err != nil {
		return nil, err
	}
	if runner.Generate == nil {
		return nil, core.NewError("chaptersmoke: runner requires Generate callback")
	}
	if runner.Capture == nil {
		return nil, core.NewError("chaptersmoke: runner requires Capture callback")
	}
	if len(cfg.Chapters) == 0 {
		return nil, core.NewError("chaptersmoke: requires at least one chapter")
	}
	storeDir, storePath, err := storePaths(cfg)
	if err != nil {
		return nil, err
	}
	report := &Report{
		StoreDir:  storeDir,
		StorePath: storePath,
		BlockSize: cfg.BlockSize,
		Chapters:  make([]ChapterReport, 0, len(cfg.Chapters)),
	}
	defer func() {
		report.FileCount = fileCount(storeDir)
	}()
	for i, chapter := range cfg.Chapters {
		chapterReport, err := runChapter(ctx, runner, cfg, storePath, i, chapter)
		report.Chapters = append(report.Chapters, chapterReport)
		if err != nil {
			report.Error = err.Error()
			return report, err
		}
	}
	return report, nil
}

func runChapter(ctx context.Context, runner Runner, cfg Config, storePath string, index int, chapter Input) (ChapterReport, error) {
	report := ChapterReport{
		Name:      chapterName(index, chapter.Name),
		Question:  chapter.Question,
		Source:    storeSource(cfg),
		BlockSize: cfg.BlockSize,
		StorePath: storePath,
		BundleURI: bundleURI(index, chapter.Name),
	}
	if core.Trim(chapter.Text) == "" {
		return chapterError(report, "chaptersmoke: chapter text is empty")
	}
	if core.Trim(chapter.Question) == "" {
		return chapterError(report, "chaptersmoke: chapter question is empty")
	}

	store, err := openWriteStore(ctx, cfg, report.StorePath, index)
	if err != nil {
		return chapterError(report, err.Error())
	}
	captureStart := time.Now()
	bundle, err := runner.Capture(ctx, chapter.Text, store.Writer, kv.StateBlockOptions{
		BlockSize:  cfg.BlockSize,
		KVEncoding: kv.EncodingNative,
		URI:        "mlx://state-chapter-smoke/" + slug(index, chapter.Name),
		Labels:     []string{"chapter-smoke", "state-kv"},
	})
	report.CaptureDuration = nonZeroDuration(time.Since(captureStart))
	if err == nil {
		_, err = kv.SaveStateBlockBundle(ctx, store.Writer, bundle, report.BundleURI)
	}
	closeErr := store.Close()
	report.SaveDuration = report.CaptureDuration
	if err != nil {
		return chapterError(report, err.Error())
	}
	if closeErr != nil {
		return chapterError(report, closeErr.Error())
	}
	report.TotalBlocks = len(bundle.Blocks)
	report.StoreBytes = fileSize(report.StorePath)
	report.PrefixTokensRestored = bundle.TokenCount
	if report.TotalBlocks == 0 {
		return chapterError(report, "chaptersmoke: wrote no KV blocks")
	}
	if report.StoreBytes <= 0 {
		return chapterError(report, "chaptersmoke: wrote empty file store")
	}

	reopenStart := time.Now()
	reader, err := openReadStore(ctx, cfg, report.StorePath)
	report.ReopenDuration = nonZeroDuration(time.Since(reopenStart))
	if err != nil {
		return chapterError(report, err.Error())
	}
	loadedBundle, err := kv.LoadStateBlockBundle(ctx, reader.Store, report.BundleURI)
	if err != nil {
		closeErr = reader.Close()
		if closeErr != nil {
			return chapterError(report, closeErr.Error())
		}
		return chapterError(report, err.Error())
	}
	counting := newCountingStore(reader.Store)
	restoreStart := time.Now()
	generation, err := runner.Generate(ctx, counting, loadedBundle, loadedBundle.TokenCount, questionPrompt(chapter))
	report.RestoreDuration = nonZeroDuration(time.Since(restoreStart))
	if generation.PromptCacheRestoreDuration > 0 {
		report.RestoreDuration = generation.PromptCacheRestoreDuration
	}
	report.BlocksRead = counting.UniqueReads()
	report.ChunksRead = counting.Reads()
	closeErr = reader.Close()
	if err != nil {
		return chapterError(report, err.Error())
	}
	if closeErr != nil {
		return chapterError(report, closeErr.Error())
	}

	report.AnswerDuration = generation.DecodeDuration
	if report.AnswerDuration <= 0 {
		report.AnswerDuration = generation.TotalDuration
	}
	report.AnswerDuration = nonZeroDuration(report.AnswerDuration)
	report.Answer = core.Trim(generation.Text)
	report.Plausible = answerPlausible(report.Answer, chapter.ExpectedTerms)
	return report, nil
}

func normalizeConfig(cfg Config) Config {
	cfg.StoreKind = normalizeStoreKind(cfg.StoreKind, cfg.StorePath)
	if cfg.BlockSize <= 0 {
		cfg.BlockSize = blockcache.DefaultBlockSize
	}
	if cfg.AnswerMaxTokens <= 0 {
		cfg.AnswerMaxTokens = DefaultAnswerMaxTokens
	}
	cfg.Chapters = core.SliceClone(cfg.Chapters)
	return cfg
}

func storePaths(cfg Config) (string, string, error) {
	if core.Trim(cfg.StorePath) != "" {
		dir := core.PathDir(cfg.StorePath)
		if result := core.MkdirAll(dir, 0o755); !result.OK {
			return "", "", core.E("chaptersmoke.storePaths", "create store path parent", resultError(result))
		}
		return dir, cfg.StorePath, nil
	}
	if core.Trim(cfg.StoreDir) != "" {
		if result := core.MkdirAll(cfg.StoreDir, 0o755); !result.OK {
			return "", "", core.E("chaptersmoke.storePaths", "create store dir", resultError(result))
		}
		return cfg.StoreDir, core.PathJoin(cfg.StoreDir, storeFileName(cfg.StoreKind)), nil
	}
	result := core.MkdirTemp("", "go-mlx-chapter-smoke-*")
	if !result.OK {
		return "", "", core.E("chaptersmoke.storePaths", "create temp store dir", resultError(result))
	}
	dir := result.Value.(string)
	return dir, core.PathJoin(dir, storeFileName(cfg.StoreKind)), nil
}

type storeHandle struct {
	Store  state.Store
	Writer state.Writer
	close  func() error
}

func (s storeHandle) Close() error {
	if s.close == nil {
		return nil
	}
	return s.close()
}

func openWriteStore(ctx context.Context, cfg Config, path string, index int) (storeHandle, error) {
	switch cfg.StoreKind {
	case StoreCLI:
		if index == 0 {
			store, err := memvidcli.Create(ctx, path, cliOptions(cfg)...)
			return storeHandle{Store: store, Writer: store}, err
		}
		store, err := memvidcli.Open(path, cliOptions(cfg)...)
		return storeHandle{Store: store, Writer: store}, err
	default:
		if index == 0 {
			store, err := filestore.Create(ctx, path)
			return storeHandle{Store: store, Writer: store, close: store.Close}, err
		}
		store, err := filestore.Open(ctx, path)
		return storeHandle{Store: store, Writer: store, close: store.Close}, err
	}
}

func openReadStore(ctx context.Context, cfg Config, path string) (storeHandle, error) {
	switch cfg.StoreKind {
	case StoreCLI:
		store, err := memvidcli.Open(path, cliOptions(cfg)...)
		return storeHandle{Store: store, Writer: store}, err
	default:
		store, err := filestore.Open(ctx, path)
		return storeHandle{Store: store, Writer: store, close: store.Close}, err
	}
}

func cliOptions(cfg Config) []memvidcli.Option {
	binary := core.Trim(cfg.StateBinary)
	if binary == "" {
		binary = core.Trim(cfg.MemvidBinary)
	}
	if binary == "" {
		return nil
	}
	return []memvidcli.Option{memvidcli.WithBinary(binary)}
}

func normalizeStoreKind(kind, path string) string {
	kind = core.Lower(core.Trim(kind))
	if kind != "" {
		switch kind {
		case "cli", "memvid", "mp4", "mv2":
			return StoreCLI
		case "file", "file-log", "filestore", "mvlog":
			return StoreFileLog
		default:
			return kind
		}
	}
	// Avoid lowering the entire path string just to check a 4-char
	// suffix — inspect the last 4 bytes directly and ASCII-lower them.
	if hasCaseInsensitiveSuffix(path, ".mp4") || hasCaseInsensitiveSuffix(path, ".mv2") {
		return StoreCLI
	}
	return StoreFileLog
}

// hasCaseInsensitiveSuffix reports whether path ends with suffix using
// ASCII-only case folding. Allocation-free.
func hasCaseInsensitiveSuffix(path, suffix string) bool {
	if len(path) < len(suffix) {
		return false
	}
	tail := path[len(path)-len(suffix):]
	for i := 0; i < len(suffix); i++ {
		c := tail[i]
		if c >= 'A' && c <= 'Z' {
			c += 'a' - 'A'
		}
		if c != suffix[i] {
			return false
		}
	}
	return true
}

func validateStoreKind(kind string) error {
	switch kind {
	case StoreFileLog, StoreCLI:
		return nil
	default:
		return core.NewError("chaptersmoke: unsupported store kind")
	}
}

func storeSource(cfg Config) string {
	if cfg.StoreKind == StoreCLI {
		return state.CodecQRVideo
	}
	return filestore.CodecFile
}

func questionPrompt(chapter Input) string {
	return "\n\nQuestion: " + chapter.Question + "\nAnswer:"
}

func answerPlausible(answer string, expected []string) bool {
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

func chapterError(report ChapterReport, message string) (ChapterReport, error) {
	report.Error = message
	return report, core.NewError(message)
}

func chapterName(index int, name string) string {
	if core.Trim(name) != "" {
		return name
	}
	// Hand-built "chapter-N" — avoids Sprintf("%d") interface boxing.
	buf := make([]byte, 0, 8+20)
	buf = append(buf, "chapter-"...)
	buf = strconv.AppendInt(buf, int64(index+1), 10)
	return core.AsString(buf)
}

func storeFileName(kind string) string {
	if kind == StoreCLI {
		return "state-kv-chapters.mp4"
	}
	return "state-kv-chapters.mvlog"
}

func bundleURI(index int, name string) string {
	return "mlx://state-chapter-smoke/" + slug(index, name) + "/bundle"
}

func slug(index int, name string) string {
	name = core.Lower(core.Trim(name))
	if name == "" {
		name = defaultChapterSlug(index)
	}
	builder := core.NewBuilder()
	// Pre-grow to the input rune count's upper bound (UTF-8 bytes) so
	// the builder skips its grow-and-copy ladder for typical chapter
	// names. Worst-case overestimate is fine — Builder.String() trims to
	// the actually-written length.
	builder.Grow(len(name))
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
	// Trim leading/trailing dashes in a single pass each — replaces two
	// HasPrefix/HasSuffix loops that each scanned the prefix on every
	// iteration. TrimLeft/TrimRight are single linear sweeps.
	out := core.TrimLeft(core.TrimRight(builder.String(), "-"), "-")
	if out == "" {
		out = defaultChapterSlug(index)
	}
	// Hand-built "%02d-out" — avoids Sprintf parsing + interface boxing.
	idx := index + 1
	buf := make([]byte, 0, 3+len(out))
	if idx < 10 {
		buf = append(buf, '0')
	}
	buf = strconv.AppendInt(buf, int64(idx), 10)
	buf = append(buf, '-')
	buf = append(buf, out...)
	return core.AsString(buf)
}

// defaultChapterSlug returns "chapter-N" without Sprintf boxing.
func defaultChapterSlug(index int) string {
	buf := make([]byte, 0, 8+20)
	buf = append(buf, "chapter-"...)
	buf = strconv.AppendInt(buf, int64(index+1), 10)
	return core.AsString(buf)
}

func fileCount(dir string) int {
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

func fileSize(path string) int64 {
	stat := core.Stat(path)
	if !stat.OK {
		return 0
	}
	return stat.Value.(core.FsFileInfo).Size()
}

func nonZeroDuration(d time.Duration) time.Duration {
	if d > 0 {
		return d
	}
	return 0
}

func resultError(result core.Result) error {
	if result.OK {
		return nil
	}
	if err, ok := result.Value.(error); ok {
		return err
	}
	return core.NewError("core result failed")
}

type countingStore struct {
	store  state.Store
	reads  int
	unique map[int]struct{}
}

func newCountingStore(store state.Store) *countingStore {
	return &countingStore{store: store, unique: map[int]struct{}{}}
}

func (s *countingStore) Get(ctx context.Context, chunkID int) (string, error) {
	s.record(chunkID)
	return s.store.Get(ctx, chunkID)
}

func (s *countingStore) Resolve(ctx context.Context, chunkID int) (state.Chunk, error) {
	s.record(chunkID)
	return state.Resolve(ctx, s.store, chunkID)
}

func (s *countingStore) ResolveBytes(ctx context.Context, chunkID int) (state.Chunk, error) {
	s.record(chunkID)
	return state.ResolveBytes(ctx, s.store, chunkID)
}

func (s *countingStore) Reads() int {
	if s == nil {
		return 0
	}
	return s.reads
}

func (s *countingStore) UniqueReads() int {
	if s == nil {
		return 0
	}
	return len(s.unique)
}

func (s *countingStore) record(chunkID int) {
	// newCountingStore is the only constructor and it initialises
	// s.unique, so the nil-guard is dead. Hot inner of every Get /
	// Resolve / ResolveBytes — strip the branch.
	s.reads++
	s.unique[chunkID] = struct{}{}
}
