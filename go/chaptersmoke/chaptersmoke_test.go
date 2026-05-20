// SPDX-Licence-Identifier: EUPL-1.2

package chaptersmoke

import (
	"context"
	"testing"
	"time"

	core "dappco.re/go"
	memvid "dappco.re/go/inference/state"
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
		Capture: func(ctx context.Context, prompt string, store memvid.Writer, opts kv.MemvidBlockOptions) (*kv.MemvidBlockBundle, error) {
			capturedPrompts = append(capturedPrompts, prompt)
			streamedEncodings = append(streamedEncodings, opts.KVEncoding)
			return testSnapshot().SaveMemvidBlocks(ctx, store, opts)
		},
		Generate: func(ctx context.Context, store memvid.Store, bundle *kv.MemvidBlockBundle, prefixTokens int, suffix string) (Generation, error) {
			if bundle.KVEncoding != kv.EncodingNative {
				return Generation{}, core.Errorf("bundle KVEncoding = %q, want native", bundle.KVEncoding)
			}
			if len(bundle.Blocks) == 0 || bundle.Blocks[0].Memvid.Codec != filestore.CodecFile {
				return Generation{}, core.Errorf("bundle refs = %+v, want file-backed refs", bundle.Blocks)
			}
			if _, err := kv.LoadPrefixFromMemvidBlocksWithOptions(ctx, store, bundle, prefixTokens, kv.LoadOptions{RawKVOnly: true}); err != nil {
				return Generation{}, err
			}
			restoredPaths = append(restoredPaths, bundle.Blocks[0].Memvid.Segment)
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

func TestStoreKind_Good_SelectsCLIForMemvidFiles(t *testing.T) {
	cases := []struct {
		name string
		cfg  Config
		want string
		file string
	}{
		{name: "mp4 path", cfg: Config{StorePath: "/tmp/book.mp4"}, want: StoreCLI, file: "/tmp/book.mp4"},
		{name: "mv2 path", cfg: Config{StorePath: "/tmp/book.mv2"}, want: StoreCLI, file: "/tmp/book.mv2"},
		{name: "cli alias", cfg: Config{StoreDir: "/tmp/store", StoreKind: "mp4"}, want: StoreCLI, file: "/tmp/store/memvid-kv-chapters.mp4"},
		{name: "file log default", cfg: Config{StoreDir: "/tmp/store"}, want: StoreFileLog, file: "/tmp/store/memvid-kv-chapters.mvlog"},
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
		Generate: func(context.Context, memvid.Store, *kv.MemvidBlockBundle, int, string) (Generation, error) {
			return Generation{}, nil
		},
	}, Config{Chapters: []Input{{Text: "x", Question: "q"}}}); err == nil {
		t.Fatal("Run(missing capture) error = nil")
	}
	if _, err := Run(context.Background(), Runner{
		Generate: func(context.Context, memvid.Store, *kv.MemvidBlockBundle, int, string) (Generation, error) {
			return Generation{}, nil
		},
		Capture: func(context.Context, string, memvid.Writer, kv.MemvidBlockOptions) (*kv.MemvidBlockBundle, error) {
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
