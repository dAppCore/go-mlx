// SPDX-Licence-Identifier: EUPL-1.2

package metal_test

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	mlx "dappco.re/go/mlx"
)

func TestMlx_GC_Good(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("GC panicked: %v", r)
		}
	}()

	mlx.GC()
}

func TestMlx_GC_Bad(t *testing.T) {
	got := goFilesContaining(t, "run"+"time.GC(")
	want := []string{"internal/metal/gc.go"}
	if strings.Join(got, "\n") != strings.Join(want, "\n") {
		t.Fatalf("direct GC callsites = %v, want %v", got, want)
	}
}

func TestMlx_GC_Ugly(t *testing.T) {
	source := readSourceFile(t, filepath.Join(repoRoot(), filepath.FromSlash("internal/metal/gc.go")))

	wantComment := "AX-6-exception: " + "run" + "time import scoped here so consumers can call mlx.GC() instead of " + "run" + "time.GC() directly."
	if !strings.Contains(source, wantComment) {
		t.Fatalf("missing AX-6 confinement comment in internal/metal/gc.go")
	}

	wantWrapper := "func RuntimeGC() { " + "run" + "time.GC() }"
	if !strings.Contains(source, wantWrapper) {
		t.Fatalf("missing RuntimeGC wrapper in internal/metal/gc.go")
	}
}

func goFilesContaining(t *testing.T, needle string) []string {
	t.Helper()

	root := repoRoot()
	var matches []string
	err := filepath.WalkDir(root, func(path string, entry os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if entry.IsDir() {
			switch entry.Name() {
			case ".git", "build", "dist":
				return filepath.SkipDir
			default:
				return nil
			}
		}
		if filepath.Ext(path) != ".go" {
			return nil
		}
		if strings.Contains(readSourceFile(t, path), needle) {
			rel, err := filepath.Rel(root, path)
			if err != nil {
				return err
			}
			matches = append(matches, filepath.ToSlash(rel))
		}
		return nil
	})
	if err != nil {
		t.Fatalf("walk source files: %v", err)
	}
	return matches
}

func readSourceFile(t *testing.T, path string) string {
	t.Helper()

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	return string(data)
}

func repoRoot() string {
	return filepath.Clean(filepath.Join("..", ".."))
}
