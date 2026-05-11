// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"testing"
	"time"

	core "dappco.re/go"
	filestore "dappco.re/go/inference/state/filestore"
	memvid "dappco.re/go/inference/state"
	"dappco.re/go/mlx/blockcache"
	"dappco.re/go/mlx/kv"
)

func TestRunMemvidKVChapterSmoke_Good_FileBackedChapterRestart(t *testing.T) {
	var capturedPrompts []string
	var streamedEncodings []kv.Encoding
	var restoredPaths []string
	var answeredSuffixes []string
	runner := MemvidKVChapterRunner{
		CaptureKVBlocksToMemvid: func(ctx context.Context, prompt string, store memvid.Writer, opts kv.MemvidBlockOptions) (*kv.MemvidBlockBundle, error) {
			capturedPrompts = append(capturedPrompts, prompt)
			streamedEncodings = append(streamedEncodings, opts.KVEncoding)
			return fastEvalTestSnapshot().SaveMemvidBlocks(ctx, store, opts)
		},
		GenerateWithMemvidPrefix: func(ctx context.Context, store memvid.Store, bundle *kv.MemvidBlockBundle, prefixTokens int, suffix string, _ GenerateConfig) (ChapterGeneration, error) {
			if bundle.KVEncoding != kv.EncodingNative {
				return ChapterGeneration{}, core.Errorf("bundle KVEncoding = %q, want native", bundle.KVEncoding)
			}
			if len(bundle.Blocks) == 0 || bundle.Blocks[0].Memvid.Codec != filestore.CodecFile {
				return ChapterGeneration{}, core.Errorf("bundle refs = %+v, want file-backed refs", bundle.Blocks)
			}
			if _, err := kv.LoadPrefixFromMemvidBlocksWithOptions(ctx, store, bundle, prefixTokens, kv.LoadOptions{RawKVOnly: true}); err != nil {
				return ChapterGeneration{}, err
			}
			restoredPaths = append(restoredPaths, bundle.Blocks[0].Memvid.Segment)
			answeredSuffixes = append(answeredSuffixes, suffix)
			answer := "Marcus identifies the chapter's pressure."
			if core.Contains(suffix, "Chapter 2") {
				answer = "Julia changes the plan in the second chapter."
			}
			return ChapterGeneration{
				Text: answer,
				Metrics: Metrics{
					GeneratedTokens:            4,
					DecodeDuration:             time.Millisecond,
					PromptCacheRestoreDuration: time.Millisecond,
				},
			}, nil
		},
	}

	report, err := RunMemvidKVChapterSmoke(context.Background(), runner, MemvidKVChapterSmokeConfig{
		StoreDir:        t.TempDir(),
		BlockSize:       2,
		AnswerMaxTokens: 4,
		Chapters: []MemvidKVChapterSmokeInput{
			{
				Name:          "Chapter 1",
				Text:          "Chapter 1. Marcus opens the sealed letter and names the risk.",
				Question:      "Chapter 1: who opens the sealed letter?",
				ExpectedTerms: []string{"Marcus"},
			},
			{
				Name:          "Chapter 2",
				Text:          "Chapter 2. Julia changes the plan after the council leaves.",
				Question:      "Chapter 2: who changes the plan?",
				ExpectedTerms: []string{"Julia"},
			},
		},
	})

	if err != nil {
		t.Fatalf("RunMemvidKVChapterSmoke() error = %v", err)
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
	for _, suffix := range answeredSuffixes {
		if core.Contains(suffix, "and names the risk") || core.Contains(suffix, "after the council leaves") {
			t.Fatalf("answered suffix %q contains chapter text, want question-only append", suffix)
		}
	}
	if report.StorePath == "" {
		t.Fatal("report StorePath is empty")
	}
	if report.FileCount != 1 {
		t.Fatalf("report FileCount = %d, want 1", report.FileCount)
	}
	if matches := core.PathGlob(core.PathJoin(report.StoreDir, "*")); len(matches) != 1 || matches[0] != report.StorePath {
		t.Fatalf("store files = %q, want only %q", matches, report.StorePath)
	}
	for _, chapter := range report.Chapters {
		if chapter.Source != filestore.CodecFile {
			t.Fatalf("%s source = %q, want file-log", chapter.Name, chapter.Source)
		}
		if chapter.StorePath != report.StorePath {
			t.Fatalf("%s StorePath = %q, want shared %q", chapter.Name, chapter.StorePath, report.StorePath)
		}
		if chapter.BundleURI == "" {
			t.Fatalf("%s BundleURI is empty, want restart manifest inside store", chapter.Name)
		}
		reopened, err := filestore.Open(context.Background(), chapter.StorePath)
		if err != nil {
			t.Fatalf("%s reopen file store from report: %v", chapter.Name, err)
		}
		bundle, err := kv.LoadMemvidBlockBundle(context.Background(), reopened, chapter.BundleURI)
		if err != nil {
			t.Fatalf("%s load bundle manifest from store URI: %v", chapter.Name, err)
		}
		if _, err := kv.LoadPrefixFromMemvidBlocksWithOptions(context.Background(), reopened, bundle, bundle.TokenCount, kv.LoadOptions{RawKVOnly: true}); err != nil {
			t.Fatalf("%s restore from durable manifest: %v", chapter.Name, err)
		}
		if err := reopened.Close(); err != nil {
			t.Fatalf("%s close reopened file store: %v", chapter.Name, err)
		}
		if chapter.StorePath == "" || chapter.StoreBytes <= 0 {
			t.Fatalf("%s store = path %q bytes %d, want real non-empty file", chapter.Name, chapter.StorePath, chapter.StoreBytes)
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
		if chapter.Error != "" {
			t.Fatalf("%s error = %q, want none", chapter.Name, chapter.Error)
		}
		if chapter.SaveDuration == time.Duration(0) {
			t.Fatalf("%s save duration was not normalised", chapter.Name)
		}
	}
}

func TestMemvidKVChapterSmokeStoreKind_Good_SelectsCLIForMemvidFiles(t *testing.T) {
	cases := []struct {
		name string
		cfg  MemvidKVChapterSmokeConfig
		want string
		file string
	}{
		{name: "mp4 path", cfg: MemvidKVChapterSmokeConfig{StorePath: "/tmp/book.mp4"}, want: MemvidKVChapterSmokeStoreCLI, file: "/tmp/book.mp4"},
		{name: "mv2 path", cfg: MemvidKVChapterSmokeConfig{StorePath: "/tmp/book.mv2"}, want: MemvidKVChapterSmokeStoreCLI, file: "/tmp/book.mv2"},
		{name: "cli alias", cfg: MemvidKVChapterSmokeConfig{StoreDir: "/tmp/store", StoreKind: "mp4"}, want: MemvidKVChapterSmokeStoreCLI, file: "/tmp/store/memvid-kv-chapters.mp4"},
		{name: "file log default", cfg: MemvidKVChapterSmokeConfig{StoreDir: "/tmp/store"}, want: MemvidKVChapterSmokeStoreFileLog, file: "/tmp/store/memvid-kv-chapters.mvlog"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg := normalizeMemvidKVChapterSmokeConfig(tc.cfg)
			if cfg.StoreKind != tc.want {
				t.Fatalf("StoreKind = %q, want %q", cfg.StoreKind, tc.want)
			}
			_, path, err := memvidKVChapterSmokeStorePaths(cfg)
			if err != nil {
				t.Fatalf("memvidKVChapterSmokeStorePaths() error = %v", err)
			}
			if path != tc.file {
				t.Fatalf("store path = %q, want %q", path, tc.file)
			}
		})
	}
}

func TestMemvidKVChapterSmokeStoreKind_Bad_RejectsUnknown(t *testing.T) {
	cfg := normalizeMemvidKVChapterSmokeConfig(MemvidKVChapterSmokeConfig{StoreKind: "sqlite"})

	err := validateMemvidKVChapterSmokeStoreKind(cfg.StoreKind)

	if err == nil {
		t.Fatal("expected unsupported store kind error")
	}
}

func TestRunMemvidKVChapterSmoke_Bad_ValidatesInputs(t *testing.T) {
	if _, err := RunModelMemvidKVChapterSmoke(context.Background(), nil, MemvidKVChapterSmokeConfig{}); err == nil {
		t.Fatal("RunModelMemvidKVChapterSmoke(nil model) error = nil")
	}
	if _, err := RunMemvidKVChapterSmoke(context.Background(), MemvidKVChapterRunner{}, MemvidKVChapterSmokeConfig{Chapters: []MemvidKVChapterSmokeInput{{Text: "x", Question: "q"}}}); err == nil {
		t.Fatal("RunMemvidKVChapterSmoke(missing generator) error = nil")
	}
	if _, err := RunMemvidKVChapterSmoke(context.Background(), MemvidKVChapterRunner{
		GenerateWithMemvidPrefix: func(context.Context, memvid.Store, *kv.MemvidBlockBundle, int, string, GenerateConfig) (ChapterGeneration, error) {
			return ChapterGeneration{}, nil
		},
	}, MemvidKVChapterSmokeConfig{Chapters: []MemvidKVChapterSmokeInput{{Text: "x", Question: "q"}}}); err == nil {
		t.Fatal("RunMemvidKVChapterSmoke(missing capture) error = nil")
	}
	if _, err := RunMemvidKVChapterSmoke(context.Background(), MemvidKVChapterRunner{
		GenerateWithMemvidPrefix: func(context.Context, memvid.Store, *kv.MemvidBlockBundle, int, string, GenerateConfig) (ChapterGeneration, error) {
			return ChapterGeneration{}, nil
		},
		CaptureKVBlocksToMemvid: func(context.Context, string, memvid.Writer, kv.MemvidBlockOptions) (*kv.MemvidBlockBundle, error) {
			return nil, nil
		},
	}, MemvidKVChapterSmokeConfig{}); err == nil {
		t.Fatal("RunMemvidKVChapterSmoke(no chapters) error = nil")
	}
}

func TestRunMemvidKVChapterSmoke_Bad_ChapterValidation(t *testing.T) {
	runner := MemvidKVChapterRunner{
		GenerateWithMemvidPrefix: func(context.Context, memvid.Store, *kv.MemvidBlockBundle, int, string, GenerateConfig) (ChapterGeneration, error) {
			return ChapterGeneration{}, nil
		},
		CaptureKVBlocksToMemvid: func(context.Context, string, memvid.Writer, kv.MemvidBlockOptions) (*kv.MemvidBlockBundle, error) {
			return fastEvalTestSnapshot().SaveMemvidBlocks(context.Background(), memvid.NewInMemoryStore(nil), kv.MemvidBlockOptions{BlockSize: 2})
		},
	}
	for _, chapter := range []MemvidKVChapterSmokeInput{
		{Question: "who?"},
		{Text: "text"},
	} {
		report, err := RunMemvidKVChapterSmoke(context.Background(), runner, MemvidKVChapterSmokeConfig{
			StoreDir: t.TempDir(),
			Chapters: []MemvidKVChapterSmokeInput{
				chapter,
			},
		})
		if err == nil {
			t.Fatalf("RunMemvidKVChapterSmoke(%+v) error = nil", chapter)
		}
		if report == nil || len(report.Chapters) != 1 || report.Chapters[0].Error == "" {
			t.Fatalf("report = %+v, want chapter-level error", report)
		}
	}
}

func TestMemvidKVChapterSmokeHelpers_Good(t *testing.T) {
	cfg := normalizeMemvidKVChapterSmokeConfig(MemvidKVChapterSmokeConfig{
		StoreKind:       "filestore",
		AnswerMaxTokens: 0,
		Temperature:     0.25,
		Chapters:        []MemvidKVChapterSmokeInput{{Text: "chapter", Question: "q"}},
	})
	cfg.Chapters[0].Text = "mutated"
	if cfg.StoreKind != MemvidKVChapterSmokeStoreFileLog || cfg.BlockSize != blockcache.DefaultBlockSize || cfg.AnswerMaxTokens != DefaultMemvidKVChapterSmokeAnswerMaxTokens {
		t.Fatalf("normalised config = %+v", cfg)
	}
	if gen := memvidKVChapterSmokeGenerateConfig(cfg); gen.MaxTokens != DefaultMemvidKVChapterSmokeAnswerMaxTokens || gen.Temperature != 0.25 {
		t.Fatalf("generate config = %+v", gen)
	}
	if got := memvidKVChapterSmokeStoreSource(MemvidKVChapterSmokeConfig{StoreKind: MemvidKVChapterSmokeStoreCLI}); got != memvid.CodecQRVideo {
		t.Fatalf("CLI source = %q", got)
	}
	if got := memvidKVChapterSmokeStoreFileName(MemvidKVChapterSmokeStoreCLI); got != "memvid-kv-chapters.mp4" {
		t.Fatalf("CLI store file name = %q", got)
	}
	if got := memvidKVChapterSmokeName(0, " Named "); got != " Named " {
		t.Fatalf("chapter name = %q", got)
	}
	if got := memvidKVChapterSmokeSlug(0, " *** "); got != "01-chapter-1" {
		t.Fatalf("empty slug = %q", got)
	}
	if got := memvidKVChapterSmokeBundleURI(1, "My Chapter!"); got != "mlx://memvid-chapter-smoke/02-my-chapter/bundle" {
		t.Fatalf("bundle URI = %q", got)
	}
	if got := memvidKVChapterSmokeQuestionPrompt(MemvidKVChapterSmokeInput{Question: "who?"}); got != "\n\nQuestion: who?\nAnswer:" {
		t.Fatalf("question prompt = %q", got)
	}
	if !memvidKVChapterSmokeAnswerPlausible("Marcus Verus", []string{"marcus", "verus"}) {
		t.Fatal("expected answer with both terms to be plausible")
	}
	if memvidKVChapterSmokeAnswerPlausible("Marcus", []string{"marcus", "verus"}) {
		t.Fatal("expected missing term to be implausible")
	}
	if memvidKVChapterSmokeAnswerPlausible("   ", nil) {
		t.Fatal("expected blank answer to be implausible")
	}
	report, err := memvidKVChapterSmokeChapterError(MemvidKVChapterSmokeChapter{Name: "chapter"}, "boom")
	if err == nil || report.Error != "boom" {
		t.Fatalf("chapter error report = %+v err=%v", report, err)
	}
	if err := (memvidKVChapterSmokeStore{}).Close(); err != nil {
		t.Fatalf("empty store Close() = %v", err)
	}
	if opts := memvidKVChapterSmokeCLIOptions(MemvidKVChapterSmokeConfig{}); opts != nil {
		t.Fatalf("empty CLI options = %+v, want nil", opts)
	}
	if opts := memvidKVChapterSmokeCLIOptions(MemvidKVChapterSmokeConfig{MemvidBinary: "/bin/memvid"}); len(opts) != 1 {
		t.Fatalf("CLI options = %d, want binary option", len(opts))
	}
}

func TestMemvidKVChapterSmokeOpenStore_Good_FileLogAppendAndRead(t *testing.T) {
	ctx := context.Background()
	path := core.PathJoin(t.TempDir(), "chapters.mvlog")
	cfg := normalizeMemvidKVChapterSmokeConfig(MemvidKVChapterSmokeConfig{StorePath: path})
	first, err := memvidKVChapterSmokeOpenWriteStore(ctx, cfg, path, 0)
	if err != nil {
		t.Fatalf("open first write store: %v", err)
	}
	if _, err := first.Writer.Put(ctx, "first", memvid.PutOptions{URI: "mlx://first"}); err != nil {
		t.Fatalf("write first: %v", err)
	}
	if err := first.Close(); err != nil {
		t.Fatalf("close first: %v", err)
	}
	second, err := memvidKVChapterSmokeOpenWriteStore(ctx, cfg, path, 1)
	if err != nil {
		t.Fatalf("open append write store: %v", err)
	}
	if _, err := second.Writer.Put(ctx, "second", memvid.PutOptions{URI: "mlx://second"}); err != nil {
		t.Fatalf("write second: %v", err)
	}
	if err := second.Close(); err != nil {
		t.Fatalf("close second: %v", err)
	}
	reader, err := memvidKVChapterSmokeOpenReadStore(ctx, cfg, path)
	if err != nil {
		t.Fatalf("open read store: %v", err)
	}
	defer reader.Close()
	chunk, err := memvid.ResolveURI(ctx, reader.Store, "mlx://second")
	if err != nil {
		t.Fatalf("resolve appended chunk: %v", err)
	}
	if chunk.Text != "second" {
		t.Fatalf("resolved appended chunk = %q, want second", chunk.Text)
	}
}

func TestMemvidKVChapterSmokeResultError_Good(t *testing.T) {
	if err := memvidKVChapterSmokeResultError(core.Result{OK: true}); err != nil {
		t.Fatalf("resultError(OK) = %v", err)
	}
	if err := memvidKVChapterSmokeResultError(core.Result{Value: core.NewError("explicit")}); err == nil || err.Error() != "explicit" {
		t.Fatalf("resultError(error) = %v", err)
	}
	if err := memvidKVChapterSmokeResultError(core.Result{}); err == nil {
		t.Fatal("resultError(empty) = nil")
	}
}

func fastEvalTestSnapshot() *kv.Snapshot {
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
