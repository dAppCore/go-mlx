// SPDX-Licence-Identifier: EUPL-1.2

package probe

import (
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
)

func lpClock(step time.Duration) func() time.Time {
	t := time.Unix(1000, 0)
	return func() time.Time {
		t = t.Add(step)
		return t
	}
}

// The v0 schema, verbatim: these golden strings are the contract with the
// existing LEM dashboards (lthn/LEM pkg/lem ingest) — a drift here is a
// broken instrument, not a style choice.
func TestLineProtocolSink_TrainLine_V0Schema_Good(t *testing.T) {
	var posts []string
	s := NewLineProtocolSink(LineProtocolConfig{
		Model: "LEM-gemma3-1b", RunID: "gold-1",
		Post:       func(body string) error { posts = append(posts, body); return nil },
		BatchLines: 1,
		now:        lpClock(2 * time.Second),
	})

	// First train step: no interval yet — rates read 0.
	s.EmitProbe(Event{Kind: KindTraining, Training: &Training{
		Step: 1, Loss: 2.5, LearningRate: 0.0001, Tokens: 512, LossType: LossTypeTrain,
	}})
	want := "training_loss,model=LEM-gemma3-1b,run_id=gold-1,loss_type=train loss=2.500000,learning_rate=0.000100,iterations_per_sec=0.000000,tokens_per_sec=0.000000,iteration=1i"
	if len(posts) != 1 || posts[0] != want {
		t.Fatalf("first train line:\n got %q\nwant %q", posts, want)
	}

	// Second step 2s later: 0.5 it/s, 256 tok/s.
	s.EmitProbe(Event{Kind: KindTraining, Training: &Training{
		Step: 2, Loss: 2.25, LearningRate: 0.0001, Tokens: 512, LossType: LossTypeTrain,
	}})
	want = "training_loss,model=LEM-gemma3-1b,run_id=gold-1,loss_type=train loss=2.250000,learning_rate=0.000100,iterations_per_sec=0.500000,tokens_per_sec=256.000000,iteration=2i"
	if posts[1] != want {
		t.Fatalf("second train line:\n got %q\nwant %q", posts[1], want)
	}
}

func TestLineProtocolSink_ValLine_V0Schema_Good(t *testing.T) {
	var posts []string
	s := NewLineProtocolSink(LineProtocolConfig{
		Model: "LEM-gemma3-1b", RunID: "gold-1",
		Post:       func(body string) error { posts = append(posts, body); return nil },
		BatchLines: 1,
	})
	s.EmitProbe(Event{Kind: KindTraining, Training: &Training{Step: 25, Loss: 1.75, LossType: LossTypeVal}})
	want := "training_loss,model=LEM-gemma3-1b,run_id=gold-1,loss_type=val loss=1.750000,iteration=25i"
	if len(posts) != 1 || posts[0] != want {
		t.Fatalf("val line:\n got %q\nwant %q", posts, want)
	}
}

// Score events become content_score lines — one per dimension, sorted for
// determinism, has_kernel derived from the label exactly as v0 does.
func TestLineProtocolSink_ScoreLines_V0Schema_Good(t *testing.T) {
	var posts []string
	s := NewLineProtocolSink(LineProtocolConfig{
		Model: "m", RunID: "r",
		Post:       func(body string) error { posts = append(posts, body); return nil },
		BatchLines: 2,
	})
	s.EmitProbe(Event{Kind: KindScore, Step: 50, Score: &Score{
		Label:  "sft-eval-kernel",
		Values: map[string]float64{"lek": 61.5, "composite": 58.25},
	}})
	if len(posts) != 1 {
		t.Fatalf("posts = %d, want 1 (two lines, batch of 2)", len(posts))
	}
	want := "content_score,model=m,run_id=r,label=sft-eval-kernel,dimension=composite,has_kernel=true score=58.250000,iteration=50i\n" +
		"content_score,model=m,run_id=r,label=sft-eval-kernel,dimension=lek,has_kernel=true score=61.500000,iteration=50i"
	if posts[0] != want {
		t.Fatalf("score lines:\n got %q\nwant %q", posts[0], want)
	}
}

// Tag escaping mirrors v0's escapeLp: comma, equals, space.
func TestLineProtocolSink_TagEscaping_Good(t *testing.T) {
	var posts []string
	s := NewLineProtocolSink(LineProtocolConfig{
		Model: "LEM gemma3,1b=x", RunID: "r",
		Post:       func(body string) error { posts = append(posts, body); return nil },
		BatchLines: 1,
	})
	s.EmitProbe(Event{Kind: KindTraining, Training: &Training{Step: 1, Loss: 1, LossType: LossTypeVal}})
	want := `training_loss,model=LEM\ gemma3\,1b\=x,run_id=r,loss_type=val loss=1.000000,iteration=1i`
	if posts[0] != want {
		t.Fatalf("escaped line:\n got %q\nwant %q", posts[0], want)
	}
}

// The file copy is appended per line as it happens — durable and
// tail-able mid-run; the poster batches; Close flushes the remainder; a
// failing poster counts drops but never errors outward.
func TestLineProtocolSink_FileBatchingAndDrops_Good(t *testing.T) {
	path := t.TempDir() + "/metrics.lp"
	posted := 0
	s := NewLineProtocolSink(LineProtocolConfig{
		Model: "m", RunID: "r", FilePath: path,
		Post:       func(string) error { posted++; return nil },
		BatchLines: 3,
	})
	for step := 1; step <= 4; step++ {
		s.EmitProbe(Event{Kind: KindTraining, Training: &Training{Step: step, Loss: 1, LossType: LossTypeVal}})
	}
	if posted != 1 {
		t.Fatalf("posts before close = %d, want 1 (batch of 3)", posted)
	}
	s.Close()
	if posted != 2 {
		t.Fatalf("posts after close = %d, want 2 (flush remainder)", posted)
	}
	read, err := coreio.Local.Read(path)
	if err != nil {
		t.Fatalf("file copy: %v", err)
	}
	lines := 0
	for _, b := range []byte(read) {
		if b == '\n' {
			lines++
		}
	}
	if lines != 4 || s.Lines() != 4 {
		t.Fatalf("file lines = %d, sink lines = %d, want 4/4", lines, s.Lines())
	}

	failing := NewLineProtocolSink(LineProtocolConfig{
		Model: "m", RunID: "r",
		Post:       func(string) error { return core.NewError("dashboard down") },
		BatchLines: 1,
	})
	failing.EmitProbe(Event{Kind: KindTraining, Training: &Training{Step: 1, Loss: 1, LossType: LossTypeVal}})
	if failing.Dropped() != 1 {
		t.Fatalf("dropped = %d, want 1 — post failures count, never propagate", failing.Dropped())
	}
}

// Non-training, non-score events and nil shapes are ignored — this sink
// is the training instrument, not a firehose.
func TestLineProtocolSink_IgnoresOtherKinds_Ugly(t *testing.T) {
	s := NewLineProtocolSink(LineProtocolConfig{Model: "m", RunID: "r"})
	s.EmitProbe(Event{Kind: KindToken, Token: &Token{ID: 1}})
	s.EmitProbe(Event{Kind: KindTraining})        // nil payload
	s.EmitProbe(Event{Kind: KindScore})           // nil payload
	s.EmitProbe(Event{Kind: KindScore, Score: &Score{}}) // empty values
	if s.Lines() != 0 {
		t.Fatalf("lines = %d, want 0", s.Lines())
	}
	var nilSink *LineProtocolSink
	nilSink.EmitProbe(Event{Kind: KindTraining, Training: &Training{}})
	nilSink.Flush()
	if nilSink.Lines() != 0 || nilSink.Dropped() != 0 {
		t.Fatal("nil sink must no-op")
	}
}

// A score with no label defaults the label tag to "unknown" — the
// instrument never emits an empty label tag.
func TestLineProtocolSink_ScoreEmptyLabel_Ugly(t *testing.T) {
	var posts []string
	s := NewLineProtocolSink(LineProtocolConfig{
		Model: "m", RunID: "r",
		Post:       func(body string) error { posts = append(posts, body); return nil },
		BatchLines: 1,
	})
	s.EmitProbe(Event{Kind: KindScore, Step: 7, Score: &Score{Values: map[string]float64{"lek": 1.0}}})
	want := "content_score,model=m,run_id=r,label=unknown,dimension=lek,has_kernel=false score=1.000000,iteration=7i"
	if len(posts) != 1 || posts[0] != want {
		t.Fatalf("empty-label score line:\n got %q\nwant %q", posts, want)
	}
}

// A file-only sink (FilePath set, Post nil) writes the durable copy and
// no-ops the poster path. Flush on a Post-less sink is a clean no-op.
func TestLineProtocolSink_FileOnlyNoPoster_Good(t *testing.T) {
	path := t.TempDir() + "/metrics.lp"
	s := NewLineProtocolSink(LineProtocolConfig{Model: "m", RunID: "r", FilePath: path})
	s.EmitProbe(Event{Kind: KindTraining, Training: &Training{Step: 1, Loss: 1, LossType: LossTypeVal}})
	s.Flush() // Post is nil — flushLocked must early-return without panic
	read, err := coreio.Local.Read(path)
	if err != nil {
		t.Fatalf("file copy: %v", err)
	}
	if read == "" || s.Lines() != 1 || s.Dropped() != 0 {
		t.Fatalf("file-only sink: read=%q lines=%d dropped=%d", read, s.Lines(), s.Dropped())
	}
}

// A file destination that can't be written (the path is a directory) makes
// the append fail — the file side of the honesty counter increments
// Dropped without ever erroring outward.
func TestLineProtocolSink_FileAppendFailureCountsDrop_Ugly(t *testing.T) {
	dir := t.TempDir() // a directory is not an appendable file → EISDIR
	s := NewLineProtocolSink(LineProtocolConfig{Model: "m", RunID: "r", FilePath: dir})
	s.EmitProbe(Event{Kind: KindTraining, Training: &Training{Step: 1, Loss: 1, LossType: LossTypeVal}})
	if s.Dropped() != 1 {
		t.Fatalf("dropped = %d, want 1 — a failed file append counts a drop", s.Dropped())
	}
}

// NewInfluxPoster returns the HTTP write closure. The Good path POSTs the
// body, sets the text/plain content type, and carries the token as a
// "Token <tok>" Authorization header — the InfluxDB v2 write contract.
func TestNewInfluxPoster_PostsBodyWithTokenHeader_Good(t *testing.T) {
	var gotBody, gotAuth, gotContentType, gotMethod string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotMethod = r.Method
		gotAuth = r.Header.Get("Authorization")
		gotContentType = r.Header.Get("Content-Type")
		body, _ := io.ReadAll(r.Body)
		gotBody = string(body)
		w.WriteHeader(http.StatusNoContent)
	}))
	defer srv.Close()

	post := NewInfluxPoster(srv.URL, "secret-token")
	if err := post("training_loss,model=m loss=1.0 1i"); err != nil {
		t.Fatalf("post returned error: %v", err)
	}
	if gotMethod != http.MethodPost {
		t.Fatalf("method = %q, want POST", gotMethod)
	}
	if gotBody != "training_loss,model=m loss=1.0 1i" {
		t.Fatalf("body = %q", gotBody)
	}
	if gotAuth != "Token secret-token" {
		t.Fatalf("auth header = %q, want %q", gotAuth, "Token secret-token")
	}
	if gotContentType != "text/plain; charset=utf-8" {
		t.Fatalf("content-type = %q", gotContentType)
	}
}

// An empty token sends no Authorization header — the unauthenticated
// write branch.
func TestNewInfluxPoster_EmptyTokenSendsNoAuthHeader_Good(t *testing.T) {
	authPresent := true
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, authPresent = r.Header["Authorization"]
		w.WriteHeader(http.StatusNoContent)
	}))
	defer srv.Close()

	post := NewInfluxPoster(srv.URL, "")
	if err := post("line"); err != nil {
		t.Fatalf("post returned error: %v", err)
	}
	if authPresent {
		t.Fatal("Authorization header sent for empty token, want none")
	}
}

// A non-2xx response (>=300) is reported as an error carrying the status
// code — the dashboard rejected the write.
func TestNewInfluxPoster_HTTPErrorStatus_Bad(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer srv.Close()

	post := NewInfluxPoster(srv.URL, "tok")
	err := post("line")
	if err == nil {
		t.Fatal("post to 500 endpoint returned nil, want error")
	}
	if !core.Contains(err.Error(), "500") {
		t.Fatalf("error = %q, want it to mention status 500", err.Error())
	}
}

// A dead endpoint (server closed before the call) fails the transport —
// client.Do returns an error and the poster surfaces it.
func TestNewInfluxPoster_UnreachableEndpoint_Ugly(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	url := srv.URL
	srv.Close() // close before posting → connection refused

	post := NewInfluxPoster(url, "tok")
	if err := post("line"); err == nil {
		t.Fatal("post to closed server returned nil, want transport error")
	}
}

// A malformed target URL fails request construction (core.NewHTTPRequest's
// !OK branch) before any network call.
func TestNewInfluxPoster_MalformedURL_Ugly(t *testing.T) {
	post := NewInfluxPoster("://no-scheme\x7f", "tok")
	if err := post("line"); err == nil {
		t.Fatal("post with malformed URL returned nil, want request-build error")
	}
}
