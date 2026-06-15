// SPDX-Licence-Identifier: EUPL-1.2

package chaptersmoke

import (
	"context"
	"errors"
	"testing"
	"time"

	core "dappco.re/go"
	state "dappco.re/go/inference/state"
	filestore "dappco.re/go/inference/state/filestore"
	"dappco.re/go/mlx/blockcache"
	"dappco.re/go/mlx/kv"
)

func TestRun_Good_FileBackedChapterRestart(t *testing.T) {
	var capturedPrompts []string
	var streamedEncodings []kv.Encoding
	var restoredPaths []string
	var answeredSuffixes []string
	runner := Runner{
		Capture: func(ctx context.Context, prompt string, store state.Writer, opts kv.StateBlockOptions) (*kv.StateBlockBundle, error) {
			capturedPrompts = append(capturedPrompts, prompt)
			streamedEncodings = append(streamedEncodings, opts.KVEncoding)
			return testSnapshot().SaveStateBlocks(ctx, store, opts)
		},
		Generate: func(ctx context.Context, store state.Store, bundle *kv.StateBlockBundle, prefixTokens int, suffix string) (Generation, error) {
			if bundle.KVEncoding != kv.EncodingNative {
				return Generation{}, core.Errorf("bundle KVEncoding = %q, want native", bundle.KVEncoding)
			}
			if len(bundle.Blocks) == 0 || bundle.Blocks[0].State.Codec != filestore.CodecFile {
				return Generation{}, core.Errorf("bundle refs = %+v, want file-backed refs", bundle.Blocks)
			}
			if _, err := kv.LoadPrefixFromStateBlocksWithOptions(ctx, store, bundle, prefixTokens, kv.LoadOptions{RawKVOnly: true}); err != nil {
				return Generation{}, err
			}
			restoredPaths = append(restoredPaths, bundle.Blocks[0].State.Segment)
			answeredSuffixes = append(answeredSuffixes, suffix)
			answer := "Marcus identifies the chapter's pressure."
			if core.Contains(suffix, "Chapter 2") {
				answer = "Julia changes the plan in the second chapter."
			}
			return Generation{
				Text:                       answer,
				DecodeDuration:             time.Millisecond,
				PromptCacheRestoreDuration: time.Millisecond,
			}, nil
		},
	}

	report, err := Run(context.Background(), runner, Config{
		StoreDir:        t.TempDir(),
		BlockSize:       2,
		AnswerMaxTokens: 4,
		Chapters: []Input{
			{Name: "Chapter 1", Text: "Chapter 1. Marcus opens the sealed letter and names the risk.", Question: "Chapter 1: who opens the sealed letter?", ExpectedTerms: []string{"Marcus"}},
			{Name: "Chapter 2", Text: "Chapter 2. Julia changes the plan after the council leaves.", Question: "Chapter 2: who changes the plan?", ExpectedTerms: []string{"Julia"}},
		},
	})

	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if len(report.Chapters) != 2 {
		t.Fatalf("chapters = %d, want 2", len(report.Chapters))
	}
	if len(capturedPrompts) != 2 || capturedPrompts[0] == capturedPrompts[1] {
		t.Fatalf("captured prompts = %q, want chapter-specific prompts", capturedPrompts)
	}
	if len(streamedEncodings) != 2 || streamedEncodings[0] != kv.EncodingNative || streamedEncodings[1] != kv.EncodingNative {
		t.Fatalf("streamed encodings = %v, want native streaming for both chapters", streamedEncodings)
	}
	if len(restoredPaths) != 2 || restoredPaths[0] != restoredPaths[1] {
		t.Fatalf("restored paths = %q, want one reopened file store", restoredPaths)
	}
	if len(answeredSuffixes) != 2 || !core.Contains(answeredSuffixes[0], "Chapter 1") || !core.Contains(answeredSuffixes[1], "Chapter 2") {
		t.Fatalf("answered suffixes = %q, want chapter questions", answeredSuffixes)
	}
	for _, chapter := range report.Chapters {
		if chapter.Source != filestore.CodecFile {
			t.Fatalf("%s source = %q, want file-log", chapter.Name, chapter.Source)
		}
		if chapter.TotalBlocks == 0 || chapter.PrefixTokensRestored == 0 {
			t.Fatalf("%s blocks = total %d prefix %d, want restored prefix blocks", chapter.Name, chapter.TotalBlocks, chapter.PrefixTokensRestored)
		}
		if chapter.SaveDuration <= 0 || chapter.ReopenDuration <= 0 || chapter.RestoreDuration <= 0 || chapter.AnswerDuration <= 0 {
			t.Fatalf("%s timings = save %s reopen %s restore %s answer %s, want all measured", chapter.Name, chapter.SaveDuration, chapter.ReopenDuration, chapter.RestoreDuration, chapter.AnswerDuration)
		}
		if !chapter.Plausible || chapter.Answer == "" {
			t.Fatalf("%s answer = %q plausible=%v, want plausible answer", chapter.Name, chapter.Answer, chapter.Plausible)
		}
	}
}

func TestStoreKind_Good_SelectsCLIForStateFiles(t *testing.T) {
	cases := []struct {
		name string
		cfg  Config
		want string
		file string
	}{
		{name: "mp4 path", cfg: Config{StorePath: "/tmp/book.mp4"}, want: StoreCLI, file: "/tmp/book.mp4"},
		{name: "mv2 path", cfg: Config{StorePath: "/tmp/book.mv2"}, want: StoreCLI, file: "/tmp/book.mv2"},
		{name: "cli alias", cfg: Config{StoreDir: "/tmp/store", StoreKind: "mp4"}, want: StoreCLI, file: "/tmp/store/state-kv-chapters.mp4"},
		{name: "file log default", cfg: Config{StoreDir: "/tmp/store"}, want: StoreFileLog, file: "/tmp/store/state-kv-chapters.mvlog"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg := normalizeConfig(tc.cfg)
			if cfg.StoreKind != tc.want {
				t.Fatalf("StoreKind = %q, want %q", cfg.StoreKind, tc.want)
			}
			_, path, err := storePaths(cfg)
			if err != nil {
				t.Fatalf("storePaths() error = %v", err)
			}
			if path != tc.file {
				t.Fatalf("store path = %q, want %q", path, tc.file)
			}
		})
	}
}

func TestRun_Bad_ValidatesInputs(t *testing.T) {
	if _, err := Run(context.Background(), Runner{}, Config{Chapters: []Input{{Text: "x", Question: "q"}}}); err == nil {
		t.Fatal("Run(missing generator) error = nil")
	}
	if _, err := Run(context.Background(), Runner{
		Generate: func(context.Context, state.Store, *kv.StateBlockBundle, int, string) (Generation, error) {
			return Generation{}, nil
		},
	}, Config{Chapters: []Input{{Text: "x", Question: "q"}}}); err == nil {
		t.Fatal("Run(missing capture) error = nil")
	}
	if _, err := Run(context.Background(), Runner{
		Generate: func(context.Context, state.Store, *kv.StateBlockBundle, int, string) (Generation, error) {
			return Generation{}, nil
		},
		Capture: func(context.Context, string, state.Writer, kv.StateBlockOptions) (*kv.StateBlockBundle, error) {
			return nil, nil
		},
	}, Config{}); err == nil {
		t.Fatal("Run(no chapters) error = nil")
	}
}

func TestNormalizeConfig_Defaults(t *testing.T) {
	cfg := normalizeConfig(Config{
		StoreKind:       "filestore",
		AnswerMaxTokens: 0,
		Temperature:     0.25,
		Chapters:        []Input{{Text: "chapter", Question: "q"}},
	})
	if cfg.StoreKind != StoreFileLog {
		t.Fatalf("StoreKind = %q, want %q", cfg.StoreKind, StoreFileLog)
	}
	if cfg.BlockSize != blockcache.DefaultBlockSize {
		t.Fatalf("BlockSize = %d, want %d", cfg.BlockSize, blockcache.DefaultBlockSize)
	}
	if cfg.AnswerMaxTokens != DefaultAnswerMaxTokens {
		t.Fatalf("AnswerMaxTokens = %d, want %d", cfg.AnswerMaxTokens, DefaultAnswerMaxTokens)
	}
}

// referenceSlugBody reproduces the slug body shape as it was BEFORE the
// ASCII-inline lowering optimisation: lower the whole name unconditionally
// via core.Lower, then run the keep-walk with no inline fold (after a full
// lower there is no ASCII uppercase left). slug/bundleURI are gated on
// producing byte-identical output to this reference across ASCII-mixed,
// non-ASCII (whose lowercase is non-ASCII), and the U+212A KELVIN SIGN edge
// (Unicode lowercase IS ASCII 'k') inputs.
func referenceSlugBody(index int, name string) string {
	name = core.Lower(core.Trim(name))
	idx := index + 1
	var buf []byte
	if idx < 10 {
		buf = append(buf, '0')
	}
	buf = append(buf, []byte(core.Itoa(idx))...)
	buf = append(buf, '-')
	prefixEnd := len(buf)
	firstKept, lastKept := -1, -1
	lastDash := false
	for i := 0; i < len(name); i++ {
		c := name[i]
		if (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') {
			buf = append(buf, c)
			rel := len(buf) - 1 - prefixEnd
			if firstKept < 0 {
				firstKept = rel
			}
			lastKept = rel
			lastDash = false
			continue
		}
		if !lastDash {
			buf = append(buf, '-')
			lastDash = true
		}
	}
	if firstKept < 0 {
		buf = append(buf[:prefixEnd], "chapter-"...)
		return string(append(buf, []byte(core.Itoa(idx))...))
	}
	if firstKept != 0 || prefixEnd+lastKept+1 != len(buf) {
		copy(buf[prefixEnd:], buf[prefixEnd+firstKept:prefixEnd+lastKept+1])
		buf = buf[:prefixEnd+(lastKept+1-firstKept)]
	}
	return string(buf)
}

func TestSlug_ByteIdenticalAfterInlineLower(t *testing.T) {
	names := []string{
		"",
		"chapter-one",
		"Chapter 7: The Sealed Letter",
		"CAFÉ at the End",
		"Kelvin scale",            // U+212A KELVIN SIGN — Unicode-lowercase is ASCII 'k'
		"Æther & Ångström",        // non-ASCII whose lowercase stays non-ASCII
		"   Mixed 7 CASE name   ", // leading/trailing space + digits
		"!!!",                     // all-dropped → empty-name fallback
		"İstanbul",                // dotted capital I — Unicode-lower may produce ASCII 'i'
	}
	for _, idx := range []int{0, 5, 8, 9, 41} {
		for _, name := range names {
			want := referenceSlugBody(idx, name)
			if got := slug(idx, name); got != want {
				t.Fatalf("slug(%d, %q) = %q, want %q (byte-identicality broken)", idx, name, got, want)
			}
			wantURI := bundleURIPrefix + want + bundleURISuffix
			if got := bundleURI(idx, name); got != wantURI {
				t.Fatalf("bundleURI(%d, %q) = %q, want %q", idx, name, got, wantURI)
			}
		}
	}
}

// fullRunner builds a Runner whose Capture saves a real testSnapshot bundle to
// the supplied store and whose Generate returns a fixed, plausible answer. It
// is the hermetic stand-in for the model-backed mlx Runner: no model is ever
// loaded — Capture/Generate are caller-supplied callbacks (see Runner doc), so
// the whole orchestration is unit-coverable with fakes.
func fullRunner(answer string) Runner {
	return Runner{
		Capture: func(ctx context.Context, prompt string, store state.Writer, opts kv.StateBlockOptions) (*kv.StateBlockBundle, error) {
			return testSnapshot().SaveStateBlocks(ctx, store, opts)
		},
		Generate: func(ctx context.Context, store state.Store, bundle *kv.StateBlockBundle, prefixTokens int, suffix string) (Generation, error) {
			return Generation{Text: answer, DecodeDuration: time.Millisecond}, nil
		},
	}
}

// TestRun_Bad_SentinelErrors strengthens the validation-guard coverage by
// asserting the exact lifted sentinel each bad input returns — not merely that
// err != nil — so a future refactor that swaps the sentinel breaks the test.
func TestRun_Bad_SentinelErrors(t *testing.T) {
	gen := func(context.Context, state.Store, *kv.StateBlockBundle, int, string) (Generation, error) {
		return Generation{}, nil
	}
	cap := func(context.Context, string, state.Writer, kv.StateBlockOptions) (*kv.StateBlockBundle, error) {
		return nil, nil
	}
	cases := []struct {
		name   string
		runner Runner
		cfg    Config
		want   error
	}{
		{name: "missing generate", runner: Runner{}, cfg: Config{Chapters: []Input{{Text: "x", Question: "q"}}}, want: errGenerateRequired},
		{name: "missing capture", runner: Runner{Generate: gen}, cfg: Config{Chapters: []Input{{Text: "x", Question: "q"}}}, want: errCaptureRequired},
		{name: "no chapters", runner: Runner{Generate: gen, Capture: cap}, cfg: Config{}, want: errNoChapters},
		{name: "unsupported store kind", runner: Runner{Generate: gen, Capture: cap}, cfg: Config{StoreKind: "bogus", Chapters: []Input{{Text: "x", Question: "q"}}}, want: errUnsupportedStoreKind},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := Run(context.Background(), tc.runner, tc.cfg)
			if !errors.Is(err, tc.want) {
				t.Fatalf("Run() error = %v, want %v", err, tc.want)
			}
		})
	}
}

// TestRun_Ugly_PerChapterFaults drives the per-chapter validation faults
// (empty text, empty question). Both flow through chapterFault, so the lifted
// sentinel survives in the returned error AND lands in report.Error /
// ChapterReport.Error — assert all three.
func TestRun_Ugly_PerChapterFaults(t *testing.T) {
	runner := fullRunner("ignored")
	cases := []struct {
		name    string
		chapter Input
		want    error
	}{
		{name: "empty text", chapter: Input{Text: "   ", Question: "who?"}, want: errChapterTextEmpty},
		{name: "empty question", chapter: Input{Text: "Chapter 1. Marcus.", Question: "  "}, want: errChapterQuestionEmpty},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			report, err := Run(context.Background(), runner, Config{
				StoreDir: t.TempDir(),
				Chapters: []Input{tc.chapter},
			})
			if !errors.Is(err, tc.want) {
				t.Fatalf("Run() error = %v, want %v", err, tc.want)
			}
			if report == nil || report.Error != tc.want.Error() {
				t.Fatalf("report.Error = %v, want %q", report, tc.want.Error())
			}
			if len(report.Chapters) != 1 || report.Chapters[0].Error != tc.want.Error() {
				t.Fatalf("chapter report = %+v, want fault recorded", report.Chapters)
			}
		})
	}
}

// TestRun_Ugly_StorePathParentIsFile forces storePaths' MkdirAll to fail by
// pointing StorePath under a parent that is a regular file, exercising the
// storePaths error branch and resultError's error-extraction path.
func TestRun_Ugly_StorePathParentIsFile(t *testing.T) {
	dir := t.TempDir()
	parentAsFile := core.PathJoin(dir, "not-a-dir")
	if result := core.WriteFile(parentAsFile, []byte("x"), 0o644); !result.OK {
		t.Fatalf("seed regular file: %v", resultError(result))
	}
	// StorePath's parent (parentAsFile) is a file, so MkdirAll(parent) fails.
	_, err := Run(context.Background(), fullRunner("ignored"), Config{
		StorePath: core.PathJoin(parentAsFile, "sub", "state.mvlog"),
		Chapters:  []Input{{Text: "Chapter 1. Marcus.", Question: "who?"}},
	})
	if err == nil {
		t.Fatal("Run() error = nil, want store-path creation failure")
	}
	if !core.Contains(err.Error(), "chaptersmoke.storePaths") {
		t.Fatalf("Run() error = %v, want storePaths failure", err)
	}
}

// TestRun_Good_DefaultChapterName covers the unnamed-chapter path: an empty
// Name routes chapterName -> defaultChapterSlug, and the report carries the
// synthesised "chapter-1" name.
func TestRun_Good_DefaultChapterName(t *testing.T) {
	report, err := Run(context.Background(), fullRunner("Marcus opens the letter."), Config{
		StoreDir: t.TempDir(),
		Chapters: []Input{{Text: "Chapter 1. Marcus opens the letter.", Question: "who opens it?"}},
	})
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if len(report.Chapters) != 1 {
		t.Fatalf("chapters = %d, want 1", len(report.Chapters))
	}
	if report.Chapters[0].Name != "chapter-1" {
		t.Fatalf("chapter name = %q, want %q", report.Chapters[0].Name, "chapter-1")
	}
	if report.FileCount == 0 {
		t.Fatalf("FileCount = 0, want >0 after a successful write")
	}
}

// TestNormalizeStoreKind_Branches covers the alias map, the suffix-sniff path,
// the too-short-path early return in hasCaseInsensitiveSuffix, and the
// unknown-kind pass-through (which validateStoreKind later rejects).
func TestNormalizeStoreKind_Branches(t *testing.T) {
	cases := []struct {
		name, kind, path, want string
	}{
		{name: "mp4 suffix", path: "/tmp/book.MP4", want: StoreCLI},
		{name: "mv2 suffix", path: "/tmp/book.mv2", want: StoreCLI},
		{name: "mvlog suffix → file", path: "/tmp/book.mvlog", want: StoreFileLog},
		{name: "path shorter than suffix", path: "ab", want: StoreFileLog},
		{name: "empty path", path: "", want: StoreFileLog},
		{name: "memvid alias", kind: "memvid", want: StoreCLI},
		{name: "filestore alias", kind: "FileStore", want: StoreFileLog},
		{name: "unknown kind passthrough", kind: "bogus", want: "bogus"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := normalizeStoreKind(tc.kind, tc.path); got != tc.want {
				t.Fatalf("normalizeStoreKind(%q, %q) = %q, want %q", tc.kind, tc.path, got, tc.want)
			}
		})
	}
}

// TestStorePaths_Ugly_StoreDirParentIsFile forces the StoreDir-branch MkdirAll
// failure (sibling of the StorePath case in TestRun_Ugly_StorePathParentIsFile),
// exercising the second storePaths error return.
func TestStorePaths_Ugly_StoreDirParentIsFile(t *testing.T) {
	dir := t.TempDir()
	parentAsFile := core.PathJoin(dir, "blocker")
	if result := core.WriteFile(parentAsFile, []byte("x"), 0o644); !result.OK {
		t.Fatalf("seed regular file: %v", resultError(result))
	}
	cfg := normalizeConfig(Config{StoreDir: core.PathJoin(parentAsFile, "store")})
	_, _, err := storePaths(cfg)
	if err == nil {
		t.Fatal("storePaths() error = nil, want store dir creation failure")
	}
	if !core.Contains(err.Error(), "chaptersmoke.storePaths") {
		t.Fatalf("storePaths() error = %v, want storePaths failure", err)
	}
}

// TestRun_Ugly_GenerateFails exercises the in-chapter error path: Capture and
// the store reopen succeed (a real bundle is written + reloaded), then Generate
// returns an error, which runChapter wraps via chapterError into report.Error.
func TestRun_Ugly_GenerateFails(t *testing.T) {
	wantMsg := "decode exploded"
	runner := Runner{
		Capture: func(ctx context.Context, prompt string, store state.Writer, opts kv.StateBlockOptions) (*kv.StateBlockBundle, error) {
			return testSnapshot().SaveStateBlocks(ctx, store, opts)
		},
		Generate: func(context.Context, state.Store, *kv.StateBlockBundle, int, string) (Generation, error) {
			return Generation{}, core.NewError(wantMsg)
		},
	}
	report, err := Run(context.Background(), runner, Config{
		StoreDir: t.TempDir(),
		Chapters: []Input{{Text: "Chapter 1. Marcus opens the letter.", Question: "who opens it?"}},
	})
	if err == nil || err.Error() != wantMsg {
		t.Fatalf("Run() error = %v, want %q", err, wantMsg)
	}
	if report == nil || report.Error != wantMsg {
		t.Fatalf("report.Error = %v, want %q", report, wantMsg)
	}
	// Capture/save/reopen all ran before the fault, so the chapter records a
	// written bundle even though generation failed.
	if len(report.Chapters) != 1 || report.Chapters[0].TotalBlocks == 0 {
		t.Fatalf("chapter report = %+v, want a written bundle before the fault", report.Chapters)
	}
	if report.Chapters[0].Error != wantMsg {
		t.Fatalf("chapter Error = %q, want %q", report.Chapters[0].Error, wantMsg)
	}
}

// TestCountingStore_Good_PassThroughAndCounts asserts that the countingStore
// wrapper forwards Get/Resolve/ResolveBytes to the underlying store (returning
// its real values) while tallying total reads and unique chunk IDs.
func TestCountingStore_Good_PassThroughAndCounts(t *testing.T) {
	cs := newCountingStore(recordingStore{})
	ctx := context.Background()

	text, err := cs.Get(ctx, 7)
	if err != nil || text != "chunk-7" {
		t.Fatalf("Get(7) = %q, %v, want \"chunk-7\", nil", text, err)
	}
	chunk, err := cs.Resolve(ctx, 7) // repeat ID 7 — total reads up, unique flat
	if err != nil || chunk.Text != "chunk-7" {
		t.Fatalf("Resolve(7) = %+v, %v, want text \"chunk-7\"", chunk, err)
	}
	bchunk, err := cs.ResolveBytes(ctx, 9)
	if err != nil || string(bchunk.Data) != "chunk-9" {
		t.Fatalf("ResolveBytes(9) = %+v, %v, want data \"chunk-9\"", bchunk, err)
	}
	if cs.Reads() != 3 {
		t.Fatalf("Reads() = %d, want 3", cs.Reads())
	}
	if cs.UniqueReads() != 2 {
		t.Fatalf("UniqueReads() = %d, want 2 (IDs 7,9)", cs.UniqueReads())
	}
}

// TestCountingStore_Ugly_NilReceiverReads guards the nil-receiver accessor
// paths: Reads/UniqueReads must report 0 rather than panic when the store was
// never constructed (the orchestration leaves *countingStore nil on early
// chapter faults before Generate runs).
func TestCountingStore_Ugly_NilReceiverReads(t *testing.T) {
	var cs *countingStore
	if cs.Reads() != 0 || cs.UniqueReads() != 0 {
		t.Fatalf("nil countingStore reads = %d/%d, want 0/0", cs.Reads(), cs.UniqueReads())
	}
}

// TestCliOptions covers the StateBinary/MemvidBinary precedence and the
// no-binary nil-options path used when selecting the deprecated CLI store.
func TestCliOptions_GoodBadUgly(t *testing.T) {
	if opts := cliOptions(Config{}); opts != nil {
		t.Fatalf("cliOptions(empty) = %v, want nil", opts)
	}
	if opts := cliOptions(Config{StateBinary: "/bin/memvid"}); len(opts) != 1 {
		t.Fatalf("cliOptions(StateBinary) = %v, want 1 option", opts)
	}
	// MemvidBinary is the fallback when StateBinary is blank.
	if opts := cliOptions(Config{MemvidBinary: "/bin/memvid"}); len(opts) != 1 {
		t.Fatalf("cliOptions(MemvidBinary) = %v, want 1 option", opts)
	}
}

// TestStoreSource covers both store-source codec branches without touching the
// external memvid binary (storeSource is pure config logic).
func TestStoreSource_BothKinds(t *testing.T) {
	if got := storeSource(Config{StoreKind: StoreCLI}); got != state.CodecQRVideo {
		t.Fatalf("storeSource(cli) = %q, want %q", got, state.CodecQRVideo)
	}
	if got := storeSource(Config{StoreKind: StoreFileLog}); got != filestore.CodecFile {
		t.Fatalf("storeSource(file-log) = %q, want %q", got, filestore.CodecFile)
	}
}

// TestAnswerPlausible_GoodBadUgly covers the three answerPlausible verdicts:
// non-empty no-terms (true), empty answer (false), and a present term that is
// absent from the answer (false), plus the blank-term skip.
func TestAnswerPlausible_GoodBadUgly(t *testing.T) {
	if !answerPlausible("Marcus opens it.", nil) {
		t.Fatal("answerPlausible(non-empty, no terms) = false, want true")
	}
	if answerPlausible("   ", []string{"Marcus"}) {
		t.Fatal("answerPlausible(empty) = true, want false")
	}
	if answerPlausible("Julia changes the plan.", []string{"Marcus"}) {
		t.Fatal("answerPlausible(missing term) = true, want false")
	}
	// Blank terms are skipped; a hit on the real term still passes.
	if !answerPlausible("Marcus opens it.", []string{"  ", "marcus"}) {
		t.Fatal("answerPlausible(blank+hit term) = false, want true")
	}
}

// TestRun_Good_NilContext covers the ctx == nil guard at the top of Run:
// passing a nil context must not panic — Run substitutes context.Background()
// and completes the happy path. Also confirms the report carries the resolved
// store dir/path and a positive file count after the write.
func TestRun_Good_NilContext(t *testing.T) {
	//nolint:staticcheck // SA1012: passing nil is the branch under test.
	report, err := Run(nil, fullRunner("Marcus opens the letter."), Config{
		StoreDir: t.TempDir(),
		Chapters: []Input{{Name: "C1", Text: "Chapter 1. Marcus opens the letter.", Question: "who opens it?"}},
	})
	if err != nil {
		t.Fatalf("Run(nil ctx) error = %v", err)
	}
	if report.StoreDir == "" || report.StorePath == "" {
		t.Fatalf("report paths = dir %q path %q, want both populated", report.StoreDir, report.StorePath)
	}
	if report.FileCount == 0 {
		t.Fatalf("FileCount = 0, want >0 after a successful write")
	}
}

// TestRun_Ugly_CaptureFails drives the Capture-error branch of runChapter: the
// store opens, but the model's Capture callback returns an error before any
// bundle is saved. runChapter wraps the message via chapterError into both the
// returned error and report.Error / ChapterReport.Error, and the chapter shows
// no written blocks.
func TestRun_Ugly_CaptureFails(t *testing.T) {
	wantMsg := "capture exploded"
	runner := Runner{
		Capture: func(context.Context, string, state.Writer, kv.StateBlockOptions) (*kv.StateBlockBundle, error) {
			return nil, core.NewError(wantMsg)
		},
		Generate: func(context.Context, state.Store, *kv.StateBlockBundle, int, string) (Generation, error) {
			t.Fatal("Generate must not run after Capture fails")
			return Generation{}, nil
		},
	}
	report, err := Run(context.Background(), runner, Config{
		StoreDir: t.TempDir(),
		Chapters: []Input{{Text: "Chapter 1. Marcus.", Question: "who?"}},
	})
	if err == nil || err.Error() != wantMsg {
		t.Fatalf("Run() error = %v, want %q", err, wantMsg)
	}
	if report == nil || report.Error != wantMsg {
		t.Fatalf("report.Error = %v, want %q", report, wantMsg)
	}
	if len(report.Chapters) != 1 || report.Chapters[0].Error != wantMsg || report.Chapters[0].TotalBlocks != 0 {
		t.Fatalf("chapter report = %+v, want capture fault with no blocks", report.Chapters)
	}
}

// TestRun_Good_AnswerDurationFallsBackToTotal covers the AnswerDuration
// fallback: when Generate reports no DecodeDuration, runChapter records the
// generation's TotalDuration instead (both are non-zero-clamped). A genuine
// runner whose engine times only the whole step, not the decode phase.
func TestRun_Good_AnswerDurationFallsBackToTotal(t *testing.T) {
	const total = 5 * time.Millisecond
	runner := Runner{
		Capture: func(ctx context.Context, prompt string, store state.Writer, opts kv.StateBlockOptions) (*kv.StateBlockBundle, error) {
			return testSnapshot().SaveStateBlocks(ctx, store, opts)
		},
		Generate: func(context.Context, state.Store, *kv.StateBlockBundle, int, string) (Generation, error) {
			return Generation{Text: "Marcus opens it.", DecodeDuration: 0, TotalDuration: total}, nil
		},
	}
	report, err := Run(context.Background(), runner, Config{
		StoreDir:  t.TempDir(),
		BlockSize: 2,
		Chapters:  []Input{{Text: "Chapter 1. Marcus.", Question: "who?"}},
	})
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if got := report.Chapters[0].AnswerDuration; got != total {
		t.Fatalf("AnswerDuration = %s, want %s (TotalDuration fallback)", got, total)
	}
}

// TestRun_Ugly_StorePathIsDirectory forces openWriteStore's filestore.Create to
// fail: StorePath points at an existing directory, so the underlying file open
// returns "is a directory". storePaths succeeds (the parent already exists), so
// the fault surfaces inside runChapter's openWriteStore call rather than earlier.
func TestRun_Ugly_StorePathIsDirectory(t *testing.T) {
	dir := t.TempDir() // the directory we (mis)point StorePath at
	report, err := Run(context.Background(), fullRunner("ignored"), Config{
		StorePath: dir,
		Chapters:  []Input{{Text: "Chapter 1. Marcus.", Question: "who?"}},
	})
	if err == nil {
		t.Fatal("Run() error = nil, want filestore create failure")
	}
	if !core.Contains(err.Error(), "state.filestore.Create") {
		t.Fatalf("Run() error = %v, want filestore.Create failure", err)
	}
	if report == nil || len(report.Chapters) != 1 || report.Chapters[0].Error == "" {
		t.Fatalf("chapter report = %+v, want the open fault recorded", report)
	}
}

// TestStoreHandle_Close_NilCloseIsNoop covers the storeHandle.Close guard: the
// CLI-backed handle leaves close nil (memvid CLI stores carry no closer), so
// Close must be a no-op returning nil rather than dereferencing a nil func.
func TestStoreHandle_Close_NilCloseIsNoop(t *testing.T) {
	if err := (storeHandle{}).Close(); err != nil {
		t.Fatalf("storeHandle{}.Close() = %v, want nil (no-op)", err)
	}
}

// TestFileSize_GoodBadUgly covers fileSize's stat branches: a real file reports
// its byte length; a missing path reports 0 (stat fails, no panic).
func TestFileSize_GoodBadUgly(t *testing.T) {
	dir := t.TempDir()
	path := core.PathJoin(dir, "blob.bin")
	body := []byte("twelve bytes")
	if r := core.WriteFile(path, body, 0o644); !r.OK {
		t.Fatalf("seed file: %v", resultError(r))
	}
	if got := fileSize(path); got != int64(len(body)) {
		t.Fatalf("fileSize(real) = %d, want %d", got, len(body))
	}
	if got := fileSize(core.PathJoin(dir, "absent")); got != 0 {
		t.Fatalf("fileSize(absent) = %d, want 0", got)
	}
}

// TestNonZeroDuration_GoodAndZero covers both nonZeroDuration arms: a positive
// duration passes through unchanged; a zero or negative duration clamps to 0.
func TestNonZeroDuration_GoodAndZero(t *testing.T) {
	if got := nonZeroDuration(7 * time.Millisecond); got != 7*time.Millisecond {
		t.Fatalf("nonZeroDuration(7ms) = %s, want 7ms", got)
	}
	if got := nonZeroDuration(0); got != 0 {
		t.Fatalf("nonZeroDuration(0) = %s, want 0", got)
	}
	if got := nonZeroDuration(-3 * time.Millisecond); got != 0 {
		t.Fatalf("nonZeroDuration(-3ms) = %s, want 0", got)
	}
}

// TestResultError_GoodBadUgly covers all three resultError arms: an OK result
// yields nil; a failed result whose Value is an error returns that error
// verbatim; a failed result whose Value is not an error returns the lifted
// errCoreResultFailed sentinel.
func TestResultError_GoodBadUgly(t *testing.T) {
	if err := resultError(core.Result{OK: true}); err != nil {
		t.Fatalf("resultError(OK) = %v, want nil", err)
	}
	inner := core.NewError("disk on fire")
	if err := resultError(core.Result{OK: false, Value: inner}); err != inner {
		t.Fatalf("resultError(error-value) = %v, want the inner error", err)
	}
	if err := resultError(core.Result{OK: false, Value: "not-an-error"}); !errors.Is(err, errCoreResultFailed) {
		t.Fatalf("resultError(non-error value) = %v, want errCoreResultFailed", err)
	}
}

// TestFileCount_Ugly_DanglingSymlinkSkipped covers fileCount's stat-fail branch:
// a directory holding one real file plus one dangling symlink must count only
// the real file — the symlink's target is absent, so core.Stat fails and the
// entry is skipped rather than counted or panicked on.
func TestFileCount_Ugly_DanglingSymlinkSkipped(t *testing.T) {
	dir := t.TempDir()
	real := core.PathJoin(dir, "real.txt")
	if r := core.WriteFile(real, []byte("x"), 0o644); !r.OK {
		t.Fatalf("seed real file: %v", resultError(r))
	}
	link := core.PathJoin(dir, "dangling.lnk")
	if r := core.Symlink(core.PathJoin(dir, "absent-target"), link); !r.OK {
		t.Skipf("symlink unsupported on this platform: %v", resultError(r))
	}
	if got := fileCount(dir); got != 1 {
		t.Fatalf("fileCount(dir with 1 real + 1 dangling symlink) = %d, want 1", got)
	}
}

// recordingStore is a state.Store/Resolver/BinaryResolver fake returning
// deterministic per-ID values so countingStore pass-through can be asserted.
// Distinct from the bench file's noopStore (which returns empty values).
type recordingStore struct{}

func (recordingStore) Get(_ context.Context, id int) (string, error) {
	return "chunk-" + core.Itoa(id), nil
}

func (recordingStore) Resolve(_ context.Context, id int) (state.Chunk, error) {
	return state.Chunk{Ref: state.ChunkRef{ChunkID: id}, Text: "chunk-" + core.Itoa(id)}, nil
}

func (recordingStore) ResolveBytes(_ context.Context, id int) (state.Chunk, error) {
	return state.Chunk{Ref: state.ChunkRef{ChunkID: id}, Data: []byte("chunk-" + core.Itoa(id))}, nil
}

func testSnapshot() *kv.Snapshot {
	return &kv.Snapshot{
		Version:       kv.SnapshotVersion,
		Architecture:  "gemma4_text",
		Tokens:        []int32{1, 2, 3},
		TokenOffset:   3,
		NumLayers:     1,
		NumHeads:      1,
		SeqLen:        3,
		HeadDim:       2,
		NumQueryHeads: 1,
		Layers: []kv.LayerSnapshot{{
			Layer:      0,
			CacheIndex: 0,
			Heads: []kv.HeadSnapshot{{
				Key:   []float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6},
				Value: []float32{0.6, 0.5, 0.4, 0.3, 0.2, 0.1},
			}},
		}},
	}
}
