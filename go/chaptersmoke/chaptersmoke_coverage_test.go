// SPDX-Licence-Identifier: EUPL-1.2

package chaptersmoke

import (
	"context"
	"errors"
	"os"
	"testing"

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
