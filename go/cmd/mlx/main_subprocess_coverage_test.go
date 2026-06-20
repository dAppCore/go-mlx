// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"os"
	"os/exec"
	"strings"
	"testing"
)

// main() reads os.Args + the signal context and dispatches via runCommand, then
// calls core.Exit — none of which is stubbable from outside the core package.
// It's covered by re-executing the test binary with GO_MLX_RUN_MAIN=1 set: the
// guarded branch below calls main() directly in the child process with a benign
// "help" arg (prints usage, exits 0). The parent asserts the child's exit code
// and output. os/exec is fine in tests — the banned-imports rule is
// production-only.
func TestMain_HelpViaSubprocess_Good(t *testing.T) {
	if os.Getenv("GO_MLX_RUN_MAIN") == "1" {
		// Child process: drive the real main() with a help arg. os.Args[0] is
		// the test binary path; arg "help" routes runCommand → printUsage → exit 0.
		os.Args = []string{os.Args[0], "help"}
		main()
		return // unreached — main() calls core.Exit
	}

	cmd := exec.Command(os.Args[0], "-test.run=TestMain_HelpViaSubprocess_Good")
	cmd.Env = append(os.Environ(), "GO_MLX_RUN_MAIN=1")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("subprocess main() exited non-zero: %v\noutput:\n%s", err, out)
	}
	if !strings.Contains(string(out), "Usage:") {
		t.Fatalf("subprocess main() output = %q, want the usage block from printUsage", string(out))
	}
}
