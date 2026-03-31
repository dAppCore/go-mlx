// SPDX-Licence-Identifier: EUPL-1.2

//go:build !nomlxlm

// Package mlxlm provides a subprocess-based inference backend using Python's mlx-lm.
//
// It implements the [inference.Backend] interface by spawning a Python process
// that communicates over JSON Lines (one JSON object per line on stdin/stdout).
// This allows using mlx-lm models without CGO or native Metal bindings.
//
// The backend auto-registers as "mlx_lm" via init(). Consumers can opt out
// with the build tag "nomlxlm".
//
// # Usage
//
//	import _ "forge.lthn.ai/core/go-mlx/mlxlm"
//
//	m, err := inference.LoadModel("/path/to/model", inference.WithBackend("mlx_lm"))
//	defer m.Close()
//
//	for tok := range m.Generate(ctx, "Hello", inference.WithMaxTokens(64)) {
//	    fmt.Print(tok.Text)
//	}
package mlxlm

import (
	"bufio"
	"context"
	"embed"
	"encoding/binary"
	"io"
	"iter"
	"math"
	"os"
	"os/exec"
	"sync"
	"time"

	"dappco.re/go/core"

	"forge.lthn.ai/core/go-inference"
	coreio "forge.lthn.ai/core/go-io"
	coreerr "forge.lthn.ai/core/go-log"
)

//go:embed bridge.py
var bridgeFS embed.FS

var (
	scriptOnce sync.Once
	scriptPath string // extracted bridge.py temp path (created once per process)
	scriptErr  error
)

// extractScript writes the embedded bridge.py to a temp file and returns its path.
//
//	path, err := extractScript() // called automatically by LoadModel
func extractScript() (string, error) {
	scriptOnce.Do(func() {
		data, err := bridgeFS.ReadFile("bridge.py")
		if err != nil {
			scriptErr = coreerr.E("mlxlm.extractScript", "read embedded bridge.py", err)
			return
		}
		dir, err := os.MkdirTemp("", "mlxlm-*")
		if err != nil {
			scriptErr = coreerr.E("mlxlm.extractScript", "create temp dir", err)
			return
		}
		p := core.JoinPath(dir, "bridge.py")
		if err := coreio.Local.Write(p, string(data)); err != nil {
			scriptErr = coreerr.E("mlxlm.extractScript", "write bridge.py", err)
			return
		}
		scriptPath = p
	})
	return scriptPath, scriptErr
}

func init() {
	inference.Register(&mlxlmBackend{})
}

type mlxlmBackend struct{}

func (b *mlxlmBackend) Name() string { return "mlx_lm" }

// Available reports whether python3 is on PATH.
func (b *mlxlmBackend) Available() bool {
	_, err := exec.LookPath("python3")
	return err == nil
}

// LoadModel spawns bridge.py as a subprocess and returns a TextModel backed by it.
//
//	m, err := inference.LoadModel("/path/to/model", inference.WithBackend("mlx_lm"))
func (b *mlxlmBackend) LoadModel(path string, opts ...inference.LoadOption) (inference.TextModel, error) {
	return loadModel(context.Background(), path, "", opts...)
}

// loadModel is the internal implementation, accepting an optional scriptOverride
// for testing (uses the mock bridge script instead of the embedded one).
func loadModel(ctx context.Context, modelPath, scriptOverride string, opts ...inference.LoadOption) (inference.TextModel, error) {
	cfg := inference.ApplyLoadOpts(opts)
	_ = cfg // reserved for future use (context length, etc.)

	var pyScript string
	if scriptOverride != "" {
		pyScript = scriptOverride
	} else {
		var err error
		pyScript, err = extractScript()
		if err != nil {
			return nil, err
		}
	}

	cmd := exec.CommandContext(ctx, "python3", "-u", pyScript)
	cmd.Stderr = nil // let stderr go to parent for debugging

	stdinPipe, err := cmd.StdinPipe()
	if err != nil {
		return nil, coreerr.E("mlxlm.loadModel", "stdin pipe", err)
	}
	stdoutPipe, err := cmd.StdoutPipe()
	if err != nil {
		return nil, coreerr.E("mlxlm.loadModel", "stdout pipe", err)
	}

	if err := cmd.Start(); err != nil {
		return nil, coreerr.E("mlxlm.loadModel", "start python3", err)
	}

	scanner := bufio.NewScanner(stdoutPipe)
	scanner.Buffer(make([]byte, 0, 1024*1024), 1024*1024) // 1MB line buffer

	m := &mlxlmModel{
		cmd:    cmd,
		stdin:  stdinPipe,
		stdout: scanner,
		raw:    stdoutPipe,
	}

	// Send load command.
	loadReq := map[string]any{
		"cmd":  "load",
		"path": modelPath,
	}
	if err := m.send(loadReq); err != nil {
		m.kill()
		return nil, coreerr.E("mlxlm.loadModel", "send load", err)
	}

	resp, err := m.recv()
	if err != nil {
		m.kill()
		return nil, coreerr.E("mlxlm.loadModel", "recv load response", err)
	}

	if errMsg, ok := resp["error"].(string); ok {
		m.kill()
		return nil, coreerr.E("mlxlm.loadModel", errMsg, nil)
	}

	if modelType, ok := resp["model_type"].(string); ok {
		m.modelType = modelType
	}
	if vocabSize, ok := resp["vocab_size"].(float64); ok {
		m.vocabSize = int(vocabSize)
	}

	return m, nil
}

// mlxlmModel is a subprocess-backed TextModel.
type mlxlmModel struct {
	cmd    *exec.Cmd
	stdin  io.WriteCloser
	stdout *bufio.Scanner
	raw    io.ReadCloser

	modelType string
	vocabSize int

	lastErr error
	mu      sync.Mutex // serialise Generate/Chat calls
}

// send writes a JSON object as a single line to the subprocess stdin.
func (m *mlxlmModel) send(obj map[string]any) error {
	r := core.JSONMarshal(obj)
	if !r.OK {
		return coreerr.E("mlxlm.send", "marshal", nil)
	}
	data := append(r.Value.([]byte), '\n')
	_, err := m.stdin.Write(data)
	return err
}

// recv reads and parses a single JSON line from the subprocess stdout.
func (m *mlxlmModel) recv() (map[string]any, error) {
	if !m.stdout.Scan() {
		if err := m.stdout.Err(); err != nil {
			return nil, coreerr.E("mlxlm.recv", "scanner", err)
		}
		return nil, coreerr.E("mlxlm.recv", "subprocess closed stdout", nil)
	}
	var obj map[string]any
	if r := core.JSONUnmarshal(m.stdout.Bytes(), &obj); !r.OK {
		return nil, coreerr.E("mlxlm.recv", "parse response", nil)
	}
	return obj, nil
}

// Generate streams tokens from the subprocess for the given prompt.
// Only one Generate or Chat call runs at a time per model (mu lock).
func (m *mlxlmModel) Generate(ctx context.Context, prompt string, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	cfg := inference.ApplyGenerateOpts(opts)

	return func(yield func(inference.Token) bool) {
		m.mu.Lock()
		defer m.mu.Unlock()
		m.lastErr = nil

		req := map[string]any{
			"cmd":        "generate",
			"prompt":     prompt,
			"max_tokens": cfg.MaxTokens,
		}
		if cfg.Temperature > 0 {
			req["temperature"] = cfg.Temperature
		}
		if cfg.TopK > 0 {
			req["top_k"] = cfg.TopK
		}
		if cfg.TopP > 0 {
			req["top_p"] = cfg.TopP
		}
		if cfg.RepeatPenalty > 1.0 {
			req["repeat_penalty"] = cfg.RepeatPenalty
		}

		if err := m.send(req); err != nil {
			m.lastErr = coreerr.E("mlxlm.Generate", "send generate", err)
			return
		}

		for {
			select {
			case <-ctx.Done():
				m.lastErr = ctx.Err()
				// Tell subprocess to stop.
				_ = m.send(map[string]any{"cmd": "cancel"})
				// Drain until done or error.
				m.drain()
				return
			default:
			}

			resp, err := m.recv()
			if err != nil {
				m.lastErr = err
				return
			}

			if errMsg, ok := resp["error"].(string); ok {
				m.lastErr = coreerr.E("mlxlm.Generate", errMsg, nil)
				return
			}

			if _, ok := resp["done"]; ok {
				return
			}

			text, _ := resp["token"].(string)
			var id int32
			if fid, ok := resp["token_id"].(float64); ok {
				id = int32(fid)
			}

			if !yield(inference.Token{ID: id, Text: text}) {
				// Consumer stopped early — send cancel and drain.
				_ = m.send(map[string]any{"cmd": "cancel"})
				m.drain()
				return
			}
		}
	}
}

// Chat streams tokens from a multi-turn conversation.
func (m *mlxlmModel) Chat(ctx context.Context, messages []inference.Message, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	cfg := inference.ApplyGenerateOpts(opts)

	return func(yield func(inference.Token) bool) {
		m.mu.Lock()
		defer m.mu.Unlock()
		m.lastErr = nil

		// Convert messages to JSON-safe format.
		msgs := make([]map[string]string, len(messages))
		for i, msg := range messages {
			msgs[i] = map[string]string{
				"role":    msg.Role,
				"content": msg.Content,
			}
		}

		req := map[string]any{
			"cmd":        "chat",
			"messages":   msgs,
			"max_tokens": cfg.MaxTokens,
		}
		if cfg.Temperature > 0 {
			req["temperature"] = cfg.Temperature
		}
		if cfg.TopK > 0 {
			req["top_k"] = cfg.TopK
		}
		if cfg.TopP > 0 {
			req["top_p"] = cfg.TopP
		}
		if cfg.RepeatPenalty > 1.0 {
			req["repeat_penalty"] = cfg.RepeatPenalty
		}

		if err := m.send(req); err != nil {
			m.lastErr = coreerr.E("mlxlm.Chat", "send chat", err)
			return
		}

		for {
			select {
			case <-ctx.Done():
				m.lastErr = ctx.Err()
				_ = m.send(map[string]any{"cmd": "cancel"})
				m.drain()
				return
			default:
			}

			resp, err := m.recv()
			if err != nil {
				m.lastErr = err
				return
			}

			if errMsg, ok := resp["error"].(string); ok {
				m.lastErr = coreerr.E("mlxlm.Chat", errMsg, nil)
				return
			}

			if _, ok := resp["done"]; ok {
				return
			}

			text, _ := resp["token"].(string)
			var id int32
			if fid, ok := resp["token_id"].(float64); ok {
				id = int32(fid)
			}

			if !yield(inference.Token{ID: id, Text: text}) {
				_ = m.send(map[string]any{"cmd": "cancel"})
				m.drain()
				return
			}
		}
	}
}

// Classify is not supported by the subprocess backend.
// Returns an error indicating that classification requires the native Metal backend.
func (m *mlxlmModel) Classify(_ context.Context, _ []string, _ ...inference.GenerateOption) ([]inference.ClassifyResult, error) {
	return nil, coreerr.E("mlxlm.Classify", "not supported (use native Metal backend)", nil)
}

// BatchGenerate is not supported by the subprocess backend.
// Returns an error indicating that batch generation requires the native Metal backend.
func (m *mlxlmModel) BatchGenerate(_ context.Context, _ []string, _ ...inference.GenerateOption) ([]inference.BatchResult, error) {
	return nil, coreerr.E("mlxlm.BatchGenerate", "not supported (use native Metal backend)", nil)
}

// ModelType returns the architecture identifier reported by the subprocess.
func (m *mlxlmModel) ModelType() string { return m.modelType }

// Info returns metadata about the loaded model.
func (m *mlxlmModel) Info() inference.ModelInfo {
	m.mu.Lock()
	defer m.mu.Unlock()

	if err := m.send(map[string]any{"cmd": "info"}); err != nil {
		return inference.ModelInfo{}
	}
	resp, err := m.recv()
	if err != nil {
		return inference.ModelInfo{}
	}
	if _, ok := resp["error"]; ok {
		return inference.ModelInfo{}
	}

	info := inference.ModelInfo{
		Architecture: m.modelType,
		VocabSize:    m.vocabSize,
	}
	if layers, ok := resp["layers"].(float64); ok {
		info.NumLayers = int(layers)
	}
	if hidden, ok := resp["hidden_size"].(float64); ok {
		info.HiddenSize = int(hidden)
	}
	return info
}

// Metrics returns empty metrics — the subprocess backend does not track timing.
func (m *mlxlmModel) Metrics() inference.GenerateMetrics {
	return inference.GenerateMetrics{}
}

// Err returns the error from the last Generate or Chat call.
func (m *mlxlmModel) Err() error { return m.lastErr }

// Close sends a quit command and waits for the subprocess to exit.
// If the subprocess does not exit within 2 seconds, it is killed.
func (m *mlxlmModel) Close() error {
	// Send quit — ignore errors (subprocess may already be dead).
	_ = m.send(map[string]any{"cmd": "quit"})
	_ = m.stdin.Close()

	// Wait with timeout.
	done := make(chan error, 1)
	go func() { done <- m.cmd.Wait() }()

	select {
	case err := <-done:
		return err
	case <-time.After(2 * time.Second):
		_ = m.cmd.Process.Kill()
		return <-done
	}
}

// drain reads and discards subprocess output until a "done" or "error" line,
// or the scanner stops. Used after cancellation to keep the protocol in sync.
func (m *mlxlmModel) drain() {
	for {
		resp, err := m.recv()
		if err != nil {
			return
		}
		if _, ok := resp["done"]; ok {
			return
		}
		if _, ok := resp["error"]; ok {
			return
		}
	}
}

// InspectAttention implements inference.AttentionInspector.
func (m *mlxlmModel) InspectAttention(ctx context.Context, prompt string, opts ...inference.GenerateOption) (*inference.AttentionSnapshot, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	req := map[string]any{
		"cmd":    "inspect",
		"prompt": prompt,
	}
	if err := m.send(req); err != nil {
		return nil, coreerr.E("mlxlm.InspectAttention", "send inspect", err)
	}

	resp, err := m.recv()
	if err != nil {
		return nil, coreerr.E("mlxlm.InspectAttention", "recv inspect", err)
	}
	if errMsg, ok := resp["error"].(string); ok {
		return nil, coreerr.E("mlxlm.InspectAttention", errMsg, nil)
	}

	// Parse metadata.
	dir, _ := resp["dir"].(string)
	numLayers := int(resp["num_layers"].(float64))
	numKVHeads := int(resp["num_kv_heads"].(float64))
	numQHeads := int(resp["num_q_heads"].(float64))
	seqLen := int(resp["seq_len"].(float64))
	headDim := int(resp["head_dim"].(float64))
	arch, _ := resp["architecture"].(string)

	// Read binary files from temp dir.
	keys := make([][][]float32, numLayers)
	queries := make([][][]float32, numLayers)

	for layer := range numLayers {
		kPath := core.JoinPath(dir, core.Sprintf("keys_%02d.bin", layer))
		kStr, err := coreio.Local.Read(kPath)
		if err != nil {
			continue
		}
		keys[layer] = reshapeFloat32([]byte(kStr), numKVHeads, seqLen*headDim)

		qPath := core.JoinPath(dir, core.Sprintf("queries_%02d.bin", layer))
		qStr, err := coreio.Local.Read(qPath)
		if err != nil {
			continue
		}
		queries[layer] = reshapeFloat32([]byte(qStr), numQHeads, seqLen*headDim)
	}

	// Clean up temp dir.
	coreio.Local.DeleteAll(dir)

	return &inference.AttentionSnapshot{
		NumLayers:     numLayers,
		NumHeads:      numKVHeads,
		NumQueryHeads: numQHeads,
		SeqLen:        seqLen,
		HeadDim:       headDim,
		Keys:          keys,
		Queries:       queries,
		Architecture:  arch,
	}, nil
}

// reshapeFloat32 reads raw little-endian float32 bytes and reshapes into
// [numHeads][stride] slices.
func reshapeFloat32(data []byte, numHeads, stride int) [][]float32 {
	total := len(data) / 4
	flat := make([]float32, total)
	for i := range flat {
		bits := binary.LittleEndian.Uint32(data[i*4 : i*4+4])
		flat[i] = math.Float32frombits(bits)
	}

	heads := make([][]float32, numHeads)
	for h := range numHeads {
		start := h * stride
		end := start + stride
		if end > len(flat) {
			break
		}
		head := make([]float32, stride)
		copy(head, flat[start:end])
		heads[h] = head
	}
	return heads
}

// kill terminates the subprocess immediately. Used during load failures.
func (m *mlxlmModel) kill() {
	_ = m.stdin.Close()
	if m.cmd.Process != nil {
		_ = m.cmd.Process.Kill()
	}
	_ = m.cmd.Wait()
}
