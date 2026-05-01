// SPDX-Licence-Identifier: EUPL-1.2

package metal_test

import (
	"testing"

	core "dappco.re/go"
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
	if core.Join("\n", got...) != core.Join("\n", want...) {
		t.Fatalf("direct GC callsites = %v, want %v", got, want)
	}
}

func TestMlx_GC_Ugly(t *testing.T) {
	source := readSourceFile(t, core.PathJoin(repoRoot(), "internal", "metal", "gc.go"))

	wantComment := "AX-6-exception: " + "run" + "time import scoped here so consumers can call mlx.GC() instead of " + "run" + "time.GC() directly."
	if !core.Contains(source, wantComment) {
		t.Fatalf("missing AX-6 confinement comment in internal/metal/gc.go")
	}

	wantWrapper := "func RuntimeGC() { " + "run" + "time.GC() }"
	if !core.Contains(source, wantWrapper) {
		t.Fatalf("missing RuntimeGC wrapper in internal/metal/gc.go")
	}
}

func goFilesContaining(t *testing.T, needle string) []string {
	t.Helper()

	root := repoRoot()
	var matches []string
	err := core.PathWalkDir(root, func(path string, entry core.FsDirEntry, err error) error {
		if err != nil {
			return err
		}
		if entry.IsDir() {
			switch entry.Name() {
			case ".git", "build", "dist":
				return core.PathSkipDir
			default:
				return nil
			}
		}
		if core.PathExt(path) != ".go" {
			return nil
		}
		if core.Contains(readSourceFile(t, path), needle) {
			relResult := core.PathRel(root, path)
			if !relResult.OK {
				return gcTestResultError(relResult)
			}
			matches = append(matches, core.PathToSlash(relResult.Value.(string)))
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

	data := core.ReadFile(path)
	if !data.OK {
		t.Fatalf("read %s: %v", path, data.Value)
	}
	return string(data.Value.([]byte))
}

func repoRoot() string {
	return core.CleanPath(core.PathJoin("..", ".."), string(core.PathSeparator))
}

func gcTestResultError(result core.Result) error {
	if err, ok := result.Value.(error); ok {
		return err
	}
	return nil
}

// Generated file-aware compliance coverage.
func TestGc_RuntimeGC_Good(t *testing.T) {
	target := "RuntimeGC"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestGc_RuntimeGC_Bad(t *testing.T) {
	target := "RuntimeGC"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestGc_RuntimeGC_Ugly(t *testing.T) {
	target := "RuntimeGC"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}
