// SPDX-Licence-Identifier: EUPL-1.2

// Differential + boundary tests for buildSFTExample. The function builds the
// inputs/targets/mask triple from a tokenised sample; the prompt/response
// path stitches promptIDs ++ responseIDs ++ EOS into a virtual sequence and
// emits inputs = V[0..n), targets = V[1..n+1), mask = 0 over the prompt
// region and 1 over the response region. buildSFTExampleRef is a literal,
// allocation-naive transcription of that contract (build the explicit []int32
// sequence, then walk it) — the production function is held byte-identical to
// it across the {EOS,NoEOS} × P × R × {truncation} matrix. The interesting
// seams are targets[promptLen-1] (crosses the prompt/response boundary) and
// targets[n-1] (EOS vs last response token), which a "tests still pass" check
// would silently miss.
//
// Run:    go test -run='TestBuildSFTExample' ./go

package train

import (
	"testing"

	"dappco.re/go/mlx/dataset"
	"dappco.re/go/mlx/spine"
)

// buildSFTExampleRef is the reference (pre-optimisation) algorithm: it
// materialises the full virtual sequence as a fresh []int32, then derives
// inputs/targets/mask from it exactly as the original code did. Kept here as
// the differential oracle for the seq-free production path.
func buildSFTExampleRef(tok *spine.Tokenizer, sample dataset.Sample, cfg SFTConfig) (sftExample, bool, error) {
	var seq []int32
	var promptLen int
	trainWholeText := sample.Text != ""
	if trainWholeText {
		ids, err := tok.Encode(sample.Text)
		if err != nil {
			return sftExample{}, false, err
		}
		seq = ids
	} else {
		promptIDs, err := tok.Encode(sample.Prompt)
		if err != nil {
			return sftExample{}, false, err
		}
		responseIDs, err := tok.Encode(sample.Response)
		if err != nil {
			return sftExample{}, false, err
		}
		promptLen = len(promptIDs)
		extra := 0
		if !cfg.NoEOS {
			extra = 1
		}
		seq = make([]int32, 0, len(promptIDs)+len(responseIDs)+extra)
		seq = append(seq, promptIDs...)
		seq = append(seq, responseIDs...)
	}
	if !cfg.NoEOS {
		seq = append(seq, tok.EOS())
	}
	if len(seq) < 2 {
		return sftExample{}, false, nil
	}
	n := len(seq) - 1
	inputs := make([]int, n)
	targets := make([]int, n)
	for i := range n {
		inputs[i] = int(seq[i])
		targets[i] = int(seq[i+1])
	}
	mask := make([]float32, n)
	if trainWholeText {
		for i := range mask {
			mask[i] = 1
		}
	} else {
		start := max(promptLen-1, 0)
		if start < len(mask) {
			for i := start; i < len(mask); i++ {
				mask[i] = 1
			}
		}
	}
	if cfg.MaxSeqLen > 0 && len(inputs) > cfg.MaxSeqLen {
		start := len(inputs) - cfg.MaxSeqLen
		inputs = append([]int(nil), inputs[start:]...)
		targets = append([]int(nil), targets[start:]...)
		mask = append([]float32(nil), mask[start:]...)
	}
	if !hasTrainingTargetRef(mask) {
		return sftExample{}, false, nil
	}
	return sftExample{inputs: inputs, targets: targets, mask: mask}, true, nil
}

func hasTrainingTargetRef(mask []float32) bool {
	for _, m := range mask {
		if m != 0 {
			return true
		}
	}
	return false
}

// buildExampleTestTokenizer maps prompt/response/text strings to caller-chosen
// token IDs so the matrix can drive P and R lengths precisely (incl. zero).
type buildExampleTestTokenizer struct {
	m   map[string][]int32
	eos int32
}

func (t buildExampleTestTokenizer) Encode(text string) []int32 {
	return append([]int32(nil), t.m[text]...)
}
func (t buildExampleTestTokenizer) Decode([]int32) string      { return "" }
func (t buildExampleTestTokenizer) DecodeOne(int32) string     { return "" }
func (buildExampleTestTokenizer) TokenID(string) (int32, bool) { return 0, false }
func (buildExampleTestTokenizer) IDToken(int32) string         { return "" }
func (buildExampleTestTokenizer) BOS() int32                   { return 0 }
func (t buildExampleTestTokenizer) EOS() int32                 { return t.eos }
func (buildExampleTestTokenizer) HasBOSToken() bool            { return false }

func makeIDs(start, n int) []int32 {
	out := make([]int32, n)
	for i := range out {
		out[i] = int32(start + i)
	}
	return out
}

func equalFloat32Slices(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// TestBuildSFTExample_DifferentialMatrix holds the production path
// byte-identical to the reference across the seam-sensitive matrix.
func TestBuildSFTExample_DifferentialMatrix(t *testing.T) {
	lengths := []int{0, 1, 2, 5, 16}
	maxSeqLens := []int{0, 1, 3, 8} // 0 = no truncation
	for _, noEOS := range []bool{false, true} {
		for _, p := range lengths {
			for _, r := range lengths {
				for _, maxSeq := range maxSeqLens {
					tok := spine.NewTokenizer(buildExampleTestTokenizer{
						m: map[string][]int32{
							"P": makeIDs(100, p),
							"R": makeIDs(500, r),
						},
						eos: 9,
					})
					sample := dataset.Sample{Prompt: "P", Response: "R"}
					cfg := SFTConfig{NoEOS: noEOS, MaxSeqLen: maxSeq}

					gotEx, gotOK, gotErr := buildSFTExample(tok, sample, cfg)
					wantEx, wantOK, wantErr := buildSFTExampleRef(tok, sample, cfg)

					ctx := func() string {
						return "noEOS=" + boolStr(noEOS) + " P=" + intStr(p) + " R=" + intStr(r) + " maxSeq=" + intStr(maxSeq)
					}
					if (gotErr == nil) != (wantErr == nil) {
						t.Fatalf("%s: err mismatch got=%v want=%v", ctx(), gotErr, wantErr)
					}
					if gotOK != wantOK {
						t.Fatalf("%s: usable got=%v want=%v", ctx(), gotOK, wantOK)
					}
					if !gotOK {
						continue
					}
					if !equalIntSlices(gotEx.inputs, wantEx.inputs) {
						t.Fatalf("%s: inputs got=%v want=%v", ctx(), gotEx.inputs, wantEx.inputs)
					}
					if !equalIntSlices(gotEx.targets, wantEx.targets) {
						t.Fatalf("%s: targets got=%v want=%v", ctx(), gotEx.targets, wantEx.targets)
					}
					if !equalFloat32Slices(gotEx.mask, wantEx.mask) {
						t.Fatalf("%s: mask got=%v want=%v", ctx(), gotEx.mask, wantEx.mask)
					}
				}
			}
		}
	}
}

// TestBuildSFTExample_TextBranchUnchanged guards the trainWholeText path,
// which the optimisation leaves untouched.
func TestBuildSFTExample_TextBranchUnchanged(t *testing.T) {
	for _, noEOS := range []bool{false, true} {
		for _, n := range []int{0, 1, 2, 7} {
			for _, maxSeq := range []int{0, 1, 4} {
				tok := spine.NewTokenizer(buildExampleTestTokenizer{
					m:   map[string][]int32{"T": makeIDs(900, n)},
					eos: 9,
				})
				sample := dataset.Sample{Text: "T"}
				cfg := SFTConfig{NoEOS: noEOS, MaxSeqLen: maxSeq}

				gotEx, gotOK, _ := buildSFTExample(tok, sample, cfg)
				wantEx, wantOK, _ := buildSFTExampleRef(tok, sample, cfg)
				if gotOK != wantOK {
					t.Fatalf("text n=%d noEOS=%v maxSeq=%d: usable got=%v want=%v", n, noEOS, maxSeq, gotOK, wantOK)
				}
				if !gotOK {
					continue
				}
				if !equalIntSlices(gotEx.inputs, wantEx.inputs) ||
					!equalIntSlices(gotEx.targets, wantEx.targets) ||
					!equalFloat32Slices(gotEx.mask, wantEx.mask) {
					t.Fatalf("text n=%d noEOS=%v maxSeq=%d: triple mismatch got=%+v want=%+v", n, noEOS, maxSeq, gotEx, wantEx)
				}
			}
		}
	}
}

func boolStr(b bool) string {
	if b {
		return "true"
	}
	return "false"
}

func intStr(i int) string {
	if i == 0 {
		return "0"
	}
	neg := i < 0
	if neg {
		i = -i
	}
	var buf [20]byte
	pos := len(buf)
	for i > 0 {
		pos--
		buf[pos] = byte('0' + i%10)
		i /= 10
	}
	if neg {
		pos--
		buf[pos] = '-'
	}
	return string(buf[pos:])
}
