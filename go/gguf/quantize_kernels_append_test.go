// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import (
	"bytes"
	"testing"
)

// TestAppendQuantize_ByteIdentical proves the append-into kernel forms
// (used by the streaming writer with a reused output buffer) produce
// bytes identical to the make-a-fresh-buffer quantizeQX wrappers, for
// every supported format. Two properties are checked:
//
//	appendQuantizeQX(nil, v)      == quantizeQX(v)
//	appendQuantizeQX(prefix, v)   == prefix ++ quantizeQX(v)
//
// The second is the property the writer relies on: appending into a
// non-empty (reused) buffer must not perturb the quantised payload.
func TestAppendQuantize_ByteIdentical(t *testing.T) {
	cases := []struct {
		name      string
		blockSize int
		append    func(out []byte, values []float32) []byte
		whole     func(values []float32) []byte
	}{
		{"Q8_0", 32, appendQuantizeQ8_0, quantizeQ8_0},
		{"Q4_0", 32, appendQuantizeQ4_0, quantizeQ4_0},
		{"Q5_0", 32, appendQuantizeQ5_0, quantizeQ5_0},
		{"Q4_K", 256, appendQuantizeQ4_K, quantizeQ4_K},
		{"Q5_K", 256, appendQuantizeQ5_K, quantizeQ5_K},
		{"Q6_K", 256, appendQuantizeQ6_K, quantizeQ6_K},
		{"Q8_K", 256, appendQuantizeQ8_K, quantizeQ8_K},
		{"Q3_K", 256, appendQuantizeQ3_K, quantizeQ3_K},
		{"Q2_K", 256, appendQuantizeQ2_K, quantizeQ2_K},
	}
	prefix := []byte("\x00\x01\x02\x03prefix-bytes-that-must-survive")
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			// Two blocks of varied data so the per-block scale path runs.
			values := benchQuantizeValues(tc.blockSize * 2)
			want := tc.whole(values)

			// Property 1: append into an empty slice == whole.
			gotNil := tc.append(nil, values)
			if !bytes.Equal(gotNil, want) {
				t.Fatalf("%s: appendQuantize(nil) != quantize (len %d vs %d)", tc.name, len(gotNil), len(want))
			}

			// Property 2: append into a reused/non-empty buffer leaves the
			// prefix intact and re-emits the identical payload.
			buf := append([]byte(nil), prefix...)
			buf = tc.append(buf, values)
			if !bytes.Equal(buf[:len(prefix)], prefix) {
				t.Fatalf("%s: append clobbered prefix bytes", tc.name)
			}
			if !bytes.Equal(buf[len(prefix):], want) {
				t.Fatalf("%s: append-into-prefix payload != quantize", tc.name)
			}

			// Property 3: reusing the same buffer via [:0] across two
			// rounds (the writer's exact pattern) yields the same payload
			// both times.
			reuse := tc.append(nil, values)
			reuse = tc.append(reuse[:0], values)
			if !bytes.Equal(reuse, want) {
				t.Fatalf("%s: reused-buffer (out[:0]) payload != quantize", tc.name)
			}
		})
	}
}
