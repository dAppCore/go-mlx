// SPDX-Licence-Identifier: EUPL-1.2

package main

import "testing"

func TestClassifyMetallibSource_Good(t *testing.T) {
	const tmp = "/tmp"
	cases := []struct {
		name    string
		path    string
		fromEnv bool
		want    string
	}{
		{"embed extract", "/tmp/lthn-mlx/abc12345/mlx.metallib", true, "embedded"},
		{"app bundle", "/Applications/lthn-mlx.app/Contents/Resources/mlx.metallib", false, "bundle"},
		{"helper inside host app", "/Applications/LEM Runtime.app/Contents/Resources/mlx.metallib", false, "bundle"},
		{"dev tree walk", "/Users/x/Code/core/go-mlx/dist/lib/mlx.metallib", false, "dev-tree"},
		{"operator env", "/opt/custom/mlx.metallib", true, "env"},
		{"bare fallback", "mlx.metallib", false, "external"},
		{"unresolved", "", false, "unresolved"},
	}
	for _, tc := range cases {
		if got := classifyMetallibSource(tc.path, tc.fromEnv, tmp); got != tc.want {
			t.Fatalf("%s: classifyMetallibSource(%q, %t) = %q, want %q", tc.name, tc.path, tc.fromEnv, got, tc.want)
		}
	}
}

// A user env var pointing INTO a dev tree or bundle classifies by the path
// shape, not the env origin — the label answers "where is the metallib",
// with fromEnv only breaking the tie for unrecognised locations.
func TestClassifyMetallibSource_EnvPointingAtKnownShapes_Ugly(t *testing.T) {
	if got := classifyMetallibSource("/Users/x/go-mlx/dist/lib/mlx.metallib", true, "/tmp"); got != "dev-tree" {
		t.Fatalf("env→dist/lib = %q, want dev-tree (path shape wins)", got)
	}
	if got := classifyMetallibSource("/Apps/X.app/Contents/Resources/mlx.metallib", true, "/tmp"); got != "bundle" {
		t.Fatalf("env→bundle = %q, want bundle (path shape wins)", got)
	}
}
