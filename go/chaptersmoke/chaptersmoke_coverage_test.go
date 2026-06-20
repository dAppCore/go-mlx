// SPDX-Licence-Identifier: EUPL-1.2

package chaptersmoke

import (
	"context"
	"errors"
	"os"
	"testing"
	"time"

	core "dappco.re/go"
	state "dappco.re/go/inference/state"
	"dappco.re/go/mlx/kv"
)

// TestRun_Ugly_EmptyFileStoreAfterUnlink drives the StoreBytes<=0 fault
// (errChapterEmptyFileStore). The Capture callback writes a real bundle to the
// open store, then unlinks the store path while the file descriptor is still
// open: SaveStateBlockBundle + Close (which operate on the live fd) both
// succeed, but runChapter's subsequent fileSize() stats the now-absent path and
// reports 0 bytes — tripping the empty-store guard before any reopen.
func TestRun_Ugly_EmptyFileStoreAfterUnlink(t *testing.T) {
	dir := t.TempDir()
	storePath := core.PathJoin(dir, "state.mvlog")
	runner := Runner{
		Capture: func(ctx context.Context, prompt string, store state.Writer, opts kv.StateBlockOptions) (*kv.StateBlockBundle, error) {
			bundle, err := testSnapshot().SaveStateBlocks(ctx, store, opts)
			if err != nil {
				return nil, err
			}
			// Unlink the backing file while the store fd stays open. The
			// pending SaveStateBlockBundle write + Close still hit the open
			// inode, so they succeed; only the later path-based fileSize sees
			// the file as gone.
			if r := core.Remove(storePath); !r.OK {
				return nil, core.NewError("unlink store path: " + resultError(r).Error())
			}
			return bundle, nil
		},
		Generate: func(context.Context, state.Store, *kv.StateBlockBundle, int, string) (Generation, error) {
			t.Fatal("Generate must not run after the empty-store fault")
			return Generation{}, nil
		},
	}

	report, err := Run(context.Background(), runner, Config{
		StorePath: storePath,
		BlockSize: 2,
		Chapters:  []Input{{Text: "Chapter 1. Marcus opens the letter.", Question: "who opens it?"}},
	})
	if !errors.Is(err, errChapterEmptyFileStore) {
		t.Fatalf("Run() error = %v, want errChapterEmptyFileStore", err)
	}
	if report == nil || report.Error != errChapterEmptyFileStore.Error() {
		t.Fatalf("report.Error = %v, want %q", report, errChapterEmptyFileStore.Error())
	}
	if len(report.Chapters) != 1 || report.Chapters[0].Error != errChapterEmptyFileStore.Error() {
		t.Fatalf("chapter report = %+v, want empty-store fault recorded", report.Chapters)
	}
	// The bundle was captured + saved before the size check, so the chapter
	// still records the written blocks even though the store size read 0.
	if report.Chapters[0].TotalBlocks == 0 {
		t.Fatalf("chapter TotalBlocks = 0, want the written bundle before the size fault")
	}
	if report.Chapters[0].StoreBytes != 0 {
		t.Fatalf("chapter StoreBytes = %d, want 0 after the unlink", report.Chapters[0].StoreBytes)
	}
}

// TestRun_Ugly_ReopenFails drives the openReadStore error branch (235-237). The
// Capture callback writes a real bundle (so the size guard at 228 passes), then
// chmods the store file to 0000. runChapter closes the write store, and the
// follow-up filestore.Open(O_RDONLY) is denied — surfacing the reopen fault.
// Skipped when the process can read mode-0000 files (e.g. running as root),
// matching the platform-skip discipline of the dangling-symlink test.
func TestRun_Ugly_ReopenFails(t *testing.T) {
	if os.Geteuid() == 0 {
		t.Skip("running as root bypasses 0000 permission; reopen-denied branch unreachable")
	}
	dir := t.TempDir()
	storePath := core.PathJoin(dir, "state.mvlog")
	runner := Runner{
		Capture: func(ctx context.Context, prompt string, store state.Writer, opts kv.StateBlockOptions) (*kv.StateBlockBundle, error) {
			bundle, err := testSnapshot().SaveStateBlocks(ctx, store, opts)
			if err != nil {
				return nil, err
			}
			// Strip all permissions. The open write fd is unaffected (Save +
			// Close still succeed), but the subsequent path-based reopen for
			// reading is denied.
			if r := core.Chmod(storePath, 0o000); !r.OK {
				return nil, core.NewError("chmod store path: " + resultError(r).Error())
			}
			return bundle, nil
		},
		Generate: func(context.Context, state.Store, *kv.StateBlockBundle, int, string) (Generation, error) {
			t.Fatal("Generate must not run after the reopen fault")
			return Generation{}, nil
		},
	}

	report, err := Run(context.Background(), runner, Config{
		StorePath: storePath,
		BlockSize: 2,
		Chapters:  []Input{{Text: "Chapter 1. Marcus opens the letter.", Question: "who opens it?"}},
	})
	// Restore permissions so t.TempDir cleanup can remove the file.
	_ = core.Chmod(storePath, 0o644)

	if err == nil {
		t.Fatal("Run() error = nil, want reopen-denied failure")
	}
	if !core.Contains(err.Error(), "state.filestore.Open") {
		t.Fatalf("Run() error = %v, want filestore.Open failure", err)
	}
	if report == nil || len(report.Chapters) != 1 || report.Chapters[0].Error == "" {
		t.Fatalf("chapter report = %+v, want the reopen fault recorded", report)
	}
	// Capture + save succeeded before the reopen, so the chapter records a
	// written bundle and a positive store size; only the reopen failed.
	if report.Chapters[0].TotalBlocks == 0 || report.Chapters[0].StoreBytes <= 0 {
		t.Fatalf("chapter = %+v, want a written bundle before the reopen fault", report.Chapters[0])
	}
	if report.Chapters[0].ReopenDuration <= 0 {
		t.Fatalf("ReopenDuration = %s, want measured before the fault", report.Chapters[0].ReopenDuration)
	}
}

// TestRun_Ugly_TempDirCreationFails drives the storePaths temp-dir branch
// failure (306-308): with neither StorePath nor StoreDir set, storePaths falls
// through to core.MkdirTemp(""), which resolves the temp root from $TMPDIR.
// Pointing TMPDIR at a regular file makes the temp-dir create fail (ENOTDIR),
// exercising the otherwise-unhit MkdirTemp error return.
func TestRun_Ugly_TempDirCreationFails(t *testing.T) {
	// Capture a usable temp dir BEFORE repointing TMPDIR — t.TempDir itself
	// reads $TMPDIR, so it must run while the env var is still valid.
	base := t.TempDir()
	blocker := core.PathJoin(base, "not-a-dir")
	if r := core.WriteFile(blocker, []byte("x"), 0o644); !r.OK {
		t.Fatalf("seed regular file: %v", resultError(r))
	}
	// MkdirTemp("") -> os.TempDir() -> $TMPDIR. A file there yields ENOTDIR.
	t.Setenv("TMPDIR", blocker)

	_, err := Run(context.Background(), fullRunner("ignored"), Config{
		// No StorePath, no StoreDir -> the temp-dir branch of storePaths.
		Chapters: []Input{{Text: "Chapter 1. Marcus.", Question: "who?"}},
	})
	if err == nil {
		t.Fatal("Run() error = nil, want temp store dir creation failure")
	}
	if !core.Contains(err.Error(), "chaptersmoke.storePaths") {
		t.Fatalf("Run() error = %v, want storePaths temp-dir failure", err)
	}
	if !core.Contains(err.Error(), "create temp store dir") {
		t.Fatalf("Run() error = %v, want the temp-dir create message", err)
	}
}

// loadFailingStore wraps a real read store so chunk Get / Resolve still work,
// but ResolveURI — the call kv.LoadStateBlockBundle uses — returns a forced
// error. Lets the close-then-load arms of runChapter be driven without
// corrupting the on-disk bundle.
type loadFailingStore struct {
	inner state.Store
	err   error
}

func (s loadFailingStore) Get(ctx context.Context, chunkID int) (string, error) {
	return s.inner.Get(ctx, chunkID)
}

func (s loadFailingStore) ResolveURI(context.Context, string) (state.Chunk, error) {
	return state.Chunk{}, s.err
}

// fullSeamRunner is the happy-path runner used by the store-seam fault tests:
// Capture writes a real bundle (so save + size guards pass) and Generate
// returns a plausible answer, isolating the injected store fault as the only
// failure on the path.
func fullSeamRunner() Runner {
	return Runner{
		Capture: func(ctx context.Context, _ string, store state.Writer, opts kv.StateBlockOptions) (*kv.StateBlockBundle, error) {
			return testSnapshot().SaveStateBlocks(ctx, store, opts)
		},
		Generate: func(context.Context, state.Store, *kv.StateBlockBundle, int, string) (Generation, error) {
			return Generation{Text: "Marcus opens the letter.", DecodeDuration: time.Millisecond}, nil
		},
	}
}

// TestRun_Ugly_WriteStoreCloseFails drives the post-capture close-error arm
// (lines 219-221): Capture + SaveStateBlockBundle succeed against the real
// write store, but the store handle's Close reports an error. runChapter must
// surface that close error via chapterError after the save, before any reopen.
func TestRun_Ugly_WriteStoreCloseFails(t *testing.T) {
	origWrite := openWriteStore
	t.Cleanup(func() { openWriteStore = origWrite })
	wantMsg := "write store close exploded"
	openWriteStore = func(ctx context.Context, cfg Config, path string, index int) (storeHandle, error) {
		h, err := origWrite(ctx, cfg, path, index)
		if err != nil {
			return h, err
		}
		realClose := h.close
		h.close = func() error {
			_ = realClose() // close the real fd so the temp file is released
			return core.NewError(wantMsg)
		}
		return h, nil
	}

	report, err := Run(context.Background(), fullSeamRunner(), Config{
		StoreDir: t.TempDir(),
		Chapters: []Input{{Text: "Chapter 1. Marcus opens the letter.", Question: "who opens it?"}},
	})
	if err == nil || err.Error() != wantMsg {
		t.Fatalf("Run() error = %v, want %q", err, wantMsg)
	}
	if report == nil || len(report.Chapters) != 1 || report.Chapters[0].Error != wantMsg {
		t.Fatalf("chapter report = %+v, want the write-close fault recorded", report)
	}
	// The close check fires before report.TotalBlocks / StoreBytes are
	// populated, so the chapter early-returns with those still zero and the
	// reopen never started.
	if report.Chapters[0].TotalBlocks != 0 {
		t.Fatalf("chapter TotalBlocks = %d, want 0 — close fault returns before TotalBlocks is set", report.Chapters[0].TotalBlocks)
	}
	if report.Chapters[0].ReopenDuration != 0 {
		t.Fatalf("ReopenDuration = %s, want 0 — reopen must not run after write-close fault", report.Chapters[0].ReopenDuration)
	}
	// CaptureDuration was measured before the fault, so the early-return report
	// still records that the capture + save ran.
	if report.Chapters[0].CaptureDuration <= 0 {
		t.Fatalf("CaptureDuration = %s, want measured before the close fault", report.Chapters[0].CaptureDuration)
	}
}

// TestRun_Ugly_LoadBundleFailsCloseOK drives the load-error arm where the
// follow-up reader.Close() succeeds (lines 239-241 entry + 244): the read
// store opens, but ResolveURI is forced to fail so LoadStateBlockBundle
// returns an error; the clean Close means runChapter surfaces the load error
// itself.
func TestRun_Ugly_LoadBundleFailsCloseOK(t *testing.T) {
	origRead := openReadStore
	t.Cleanup(func() { openReadStore = origRead })
	wantMsg := "resolve bundle uri exploded"
	openReadStore = func(ctx context.Context, cfg Config, path string) (storeHandle, error) {
		h, err := origRead(ctx, cfg, path)
		if err != nil {
			return h, err
		}
		h.Store = loadFailingStore{inner: h.Store, err: core.NewError(wantMsg)}
		return h, nil
	}

	report, err := Run(context.Background(), fullSeamRunner(), Config{
		StoreDir: t.TempDir(),
		Chapters: []Input{{Text: "Chapter 1. Marcus opens the letter.", Question: "who opens it?"}},
	})
	if err == nil || !core.Contains(err.Error(), wantMsg) {
		t.Fatalf("Run() error = %v, want load failure containing %q", err, wantMsg)
	}
	if report == nil || len(report.Chapters) != 1 || !core.Contains(report.Chapters[0].Error, wantMsg) {
		t.Fatalf("chapter report = %+v, want the load fault recorded", report)
	}
	// The reopen succeeded and was measured before the load failed.
	if report.Chapters[0].ReopenDuration <= 0 {
		t.Fatalf("ReopenDuration = %s, want measured before the load fault", report.Chapters[0].ReopenDuration)
	}
}

// TestRun_Ugly_LoadBundleFailsCloseAlsoFails drives the nested close-error arm
// (lines 241-243): both LoadStateBlockBundle and the follow-up reader.Close()
// fail. runChapter must surface the close error in that arm, taking priority
// over the load error.
func TestRun_Ugly_LoadBundleFailsCloseAlsoFails(t *testing.T) {
	origRead := openReadStore
	t.Cleanup(func() { openReadStore = origRead })
	loadMsg := "resolve bundle uri exploded"
	closeMsg := "read store close exploded"
	openReadStore = func(ctx context.Context, cfg Config, path string) (storeHandle, error) {
		h, err := origRead(ctx, cfg, path)
		if err != nil {
			return h, err
		}
		h.Store = loadFailingStore{inner: h.Store, err: core.NewError(loadMsg)}
		realClose := h.close
		h.close = func() error {
			_ = realClose()
			return core.NewError(closeMsg)
		}
		return h, nil
	}

	report, err := Run(context.Background(), fullSeamRunner(), Config{
		StoreDir: t.TempDir(),
		Chapters: []Input{{Text: "Chapter 1. Marcus opens the letter.", Question: "who opens it?"}},
	})
	// The close error wins over the load error in this arm.
	if err == nil || err.Error() != closeMsg {
		t.Fatalf("Run() error = %v, want the read-close error %q (priority over load error)", err, closeMsg)
	}
	if report == nil || len(report.Chapters) != 1 || report.Chapters[0].Error != closeMsg {
		t.Fatalf("chapter report = %+v, want the read-close fault recorded", report)
	}
}

// TestRun_Ugly_ReadStoreCloseFailsAfterGenerate drives the final close-error
// arm (lines 263-265): load + Generate both succeed, then the read store's
// Close reports an error. runChapter must surface that close error even though
// generation produced a valid answer.
func TestRun_Ugly_ReadStoreCloseFailsAfterGenerate(t *testing.T) {
	origRead := openReadStore
	t.Cleanup(func() { openReadStore = origRead })
	wantMsg := "read store close after generate exploded"
	openReadStore = func(ctx context.Context, cfg Config, path string) (storeHandle, error) {
		h, err := origRead(ctx, cfg, path)
		if err != nil {
			return h, err
		}
		realClose := h.close
		h.close = func() error {
			_ = realClose()
			return core.NewError(wantMsg)
		}
		return h, nil
	}

	report, err := Run(context.Background(), fullSeamRunner(), Config{
		StoreDir: t.TempDir(),
		Chapters: []Input{{Text: "Chapter 1. Marcus opens the letter.", Question: "who opens it?"}},
	})
	if err == nil || err.Error() != wantMsg {
		t.Fatalf("Run() error = %v, want the post-generate read-close error %q", err, wantMsg)
	}
	if report == nil || len(report.Chapters) != 1 || report.Chapters[0].Error != wantMsg {
		t.Fatalf("chapter report = %+v, want the post-generate read-close fault recorded", report)
	}
	// The close check (lines 259-265) returns before report.Answer is set
	// (line 272), so the early-return report carries the fault but no answer.
	if report.Chapters[0].Answer != "" {
		t.Fatalf("chapter Answer = %q, want empty — close fault returns before Answer is set", report.Chapters[0].Answer)
	}
	// The restore phase ran (Generate succeeded) before the close fault, so its
	// duration was measured into the report.
	if report.Chapters[0].RestoreDuration <= 0 {
		t.Fatalf("RestoreDuration = %s, want measured before the close fault", report.Chapters[0].RestoreDuration)
	}
}
