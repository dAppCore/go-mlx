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
//	import _ "dappco.re/go/mlx/mlxlm"
//
//	ctx := context.Background()
//	model, err := inference.LoadModel("/path/to/model", inference.WithBackend("mlx_lm"))
//	defer model.Close()
//
//	for token := range model.Generate(ctx, "Hello", inference.WithMaxTokens(64)) {
//	    fmt.Print(token.Text)
//	}
package mlxlm

import (
	"bytes"
	"context"
	"embed"
	"encoding/binary"
	"io"
	"iter"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"time"

	"dappco.re/go/core"

	"dappco.re/go/core/inference"
	coreio "dappco.re/go/core/io"
)

//go:embed bridge.py
var bridgeFS embed.FS

var (
	mlxlmCore         = newMLXLMCore()
	bridgeScriptLock  = mlxlmCore.Lock("mlxlm.bridgeScript").Mutex
	bridgeScriptReady bool
	bridgeScriptPath  string // extracted bridge.py temp path (created once per process)
	bridgeScriptError error
)

// extractScript writes the embedded bridge.py to a temp file and returns its path.
//
//	bridgePath, err := extractScript() // called automatically by LoadModel
func extractScript() (string, error) {
	bridgeScriptLock.Lock()
	defer bridgeScriptLock.Unlock()

	if bridgeScriptReady {
		return bridgeScriptPath, bridgeScriptError
	}
	bridgeScriptReady = true

	data, err := bridgeFS.ReadFile("bridge.py")
	if err != nil {
		bridgeScriptError = core.E("mlxlm.extractScript", "read embedded bridge.py", err)
		return bridgeScriptPath, bridgeScriptError
	}
	dir := (&core.Fs{}).New("/").TempDir("mlxlm-")
	if dir == "" {
		bridgeScriptError = core.E("mlxlm.extractScript", "create temp dir", nil)
		return bridgeScriptPath, bridgeScriptError
	}
	p := core.JoinPath(dir, "bridge.py")
	if err := coreio.Local.Write(p, string(data)); err != nil {
		bridgeScriptError = core.E("mlxlm.extractScript", "write bridge.py", err)
		return bridgeScriptPath, bridgeScriptError
	}
	bridgeScriptPath = p
	return bridgeScriptPath, bridgeScriptError
}

func init() {
	inference.Register(&mlxlmBackend{})
}

type mlxlmBackend struct{}

func (backend *mlxlmBackend) Name() string { return "mlx_lm" }

// Available reports whether python3 is on PATH.
func (backend *mlxlmBackend) Available() bool {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	return mlxlmCore.Process().Run(ctx, "python3", "--version").OK
}

// LoadModel spawns bridge.py as a subprocess and returns a TextModel backed by it.
//
//	model, err := inference.LoadModel("/path/to/model", inference.WithBackend("mlx_lm"))
func (backend *mlxlmBackend) LoadModel(modelPath string, opts ...inference.LoadOption) (inference.TextModel, error) {
	return loadModel(context.Background(), modelPath, "", opts...)
}

// loadModel is the internal implementation. scriptPathOverride substitutes the embedded
// bridge.py for testing.
func loadModel(ctx context.Context, modelPath, scriptPathOverride string, opts ...inference.LoadOption) (inference.TextModel, error) {
	loadOptions := inference.ApplyLoadOpts(opts)
	_ = loadOptions // reserved for future use (context length, etc.)

	var bridgePath string
	if scriptPathOverride != "" {
		bridgePath = scriptPathOverride
	} else {
		var err error
		bridgePath, err = extractScript()
		if err != nil {
			return nil, err
		}
	}

	result := mlxlmCore.Process().Start(ctx, core.NewOptions(
		core.Option{Key: "command", Value: "python3"},
		core.Option{Key: "args", Value: []string{"-u", bridgePath}},
	))
	if !result.OK {
		return nil, core.E("mlxlm.loadModel", "start python3", resultError(result))
	}
	proc, ok := result.Value.(*mlxlmProcess)
	if !ok {
		return nil, core.E("mlxlm.loadModel", "process.start returned unexpected handle", nil)
	}

	model := &mlxlmModel{
		process: proc,
		stdin:   proc.stdin,
		stdout:  newJSONLineReader(proc.stdout),
		mu:      mlxlmCore.Lock("mlxlm.model." + core.ID()).Mutex,
	}

	loadRequest := map[string]any{
		"cmd":  "load",
		"path": modelPath,
	}
	if err := model.send(loadRequest); err != nil {
		model.kill()
		return nil, core.E("mlxlm.loadModel", "send load", err)
	}

	response, err := model.recv()
	if err != nil {
		model.kill()
		return nil, core.E("mlxlm.loadModel", "recv load response", err)
	}

	if errMsg, ok := response["error"].(string); ok {
		model.kill()
		return nil, core.E("mlxlm.loadModel", errMsg, nil)
	}

	if modelType, ok := response["model_type"].(string); ok {
		model.modelType = modelType
	}
	if vocabSize, ok := response["vocab_size"].(float64); ok {
		model.vocabSize = int(vocabSize)
	}

	return model, nil
}

type mlxlmModel struct {
	process *mlxlmProcess
	stdin   io.WriteCloser
	stdout  *jsonLineReader

	modelType string
	vocabSize int

	lastErr error
	mu      mutex // serialise Generate/Chat calls
}

type mutex interface {
	Lock()
	Unlock()
}

func optionalFloat32Field(v any, fieldName string) (float32, bool) {
	field := reflect.ValueOf(v).FieldByName(fieldName)
	if !field.IsValid() {
		return 0, false
	}
	switch field.Kind() {
	case reflect.Float32, reflect.Float64:
		return float32(field.Float()), true
	default:
		return 0, false
	}
}

// send writes a JSON object as a newline-terminated line to subprocess stdin.
func (model *mlxlmModel) send(obj map[string]any) error {
	encoded := core.JSONMarshal(obj)
	if !encoded.OK {
		return core.E("mlxlm.send", "marshal", nil)
	}
	data := append(encoded.Value.([]byte), '\n')
	_, err := model.stdin.Write(data)
	return err
}

// recv reads and parses one JSON line from subprocess stdout.
func (model *mlxlmModel) recv() (map[string]any, error) {
	line, err := model.stdout.ReadLine()
	if err != nil {
		if err == io.EOF {
			return nil, core.E("mlxlm.recv", "subprocess closed stdout", nil)
		}
		return nil, core.E("mlxlm.recv", "read subprocess stdout", err)
	}
	var obj map[string]any
	if r := core.JSONUnmarshal(line, &obj); !r.OK {
		return nil, core.E("mlxlm.recv", "parse response", nil)
	}
	return obj, nil
}

// Generate streams tokens from the subprocess for the given prompt.
// Calls are serialised per model (mu lock).
//
//	for token := range model.Generate(ctx, "Hello", inference.WithMaxTokens(64)) {
//	    fmt.Print(token.Text)
//	}
func (model *mlxlmModel) Generate(ctx context.Context, prompt string, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	generateOptions := inference.ApplyGenerateOpts(opts)

	return func(yield func(inference.Token) bool) {
		model.mu.Lock()
		defer model.mu.Unlock()
		model.lastErr = nil

		request := map[string]any{
			"cmd":        "generate",
			"prompt":     prompt,
			"max_tokens": generateOptions.MaxTokens,
		}
		if generateOptions.Temperature > 0 {
			request["temperature"] = generateOptions.Temperature
		}
		if generateOptions.TopK > 0 {
			request["top_k"] = generateOptions.TopK
		}
		if generateOptions.TopP > 0 {
			request["top_p"] = generateOptions.TopP
		}
		if minP, ok := optionalFloat32Field(generateOptions, "MinP"); ok && minP > 0 {
			request["min_p"] = minP
		}
		if generateOptions.RepeatPenalty > 1.0 {
			request["repeat_penalty"] = generateOptions.RepeatPenalty
		}

		if err := model.send(request); err != nil {
			model.lastErr = core.E("mlxlm.Generate", "send generate", err)
			return
		}

		for {
			select {
			case <-ctx.Done():
				model.lastErr = ctx.Err()
				_ = model.send(map[string]any{"cmd": "cancel"})
				model.drain()
				return
			default:
			}

			response, err := model.recv()
			if err != nil {
				model.lastErr = err
				return
			}

			if errMsg, ok := response["error"].(string); ok {
				model.lastErr = core.E("mlxlm.Generate", errMsg, nil)
				return
			}

			if _, ok := response["done"]; ok {
				return
			}

			text, _ := response["token"].(string)
			var id int32
			if fid, ok := response["token_id"].(float64); ok {
				id = int32(fid)
			}

			if !yield(inference.Token{ID: id, Text: text}) {
				_ = model.send(map[string]any{"cmd": "cancel"})
				model.drain()
				return
			}
		}
	}
}

// Chat streams tokens from a multi-turn conversation via the subprocess.
//
//	for token := range model.Chat(ctx, []inference.Message{{Role: "user", Content: "Hello"}}, opts...) {
//	    fmt.Print(token.Text)
//	}
func (model *mlxlmModel) Chat(ctx context.Context, messages []inference.Message, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	generateOptions := inference.ApplyGenerateOpts(opts)

	return func(yield func(inference.Token) bool) {
		model.mu.Lock()
		defer model.mu.Unlock()
		model.lastErr = nil

		messagePayloads := make([]map[string]string, len(messages))
		for i, msg := range messages {
			messagePayloads[i] = map[string]string{
				"role":    msg.Role,
				"content": msg.Content,
			}
		}

		request := map[string]any{
			"cmd":        "chat",
			"messages":   messagePayloads,
			"max_tokens": generateOptions.MaxTokens,
		}
		if generateOptions.Temperature > 0 {
			request["temperature"] = generateOptions.Temperature
		}
		if generateOptions.TopK > 0 {
			request["top_k"] = generateOptions.TopK
		}
		if generateOptions.TopP > 0 {
			request["top_p"] = generateOptions.TopP
		}
		if minP, ok := optionalFloat32Field(generateOptions, "MinP"); ok && minP > 0 {
			request["min_p"] = minP
		}
		if generateOptions.RepeatPenalty > 1.0 {
			request["repeat_penalty"] = generateOptions.RepeatPenalty
		}

		if err := model.send(request); err != nil {
			model.lastErr = core.E("mlxlm.Chat", "send chat", err)
			return
		}

		for {
			select {
			case <-ctx.Done():
				model.lastErr = ctx.Err()
				_ = model.send(map[string]any{"cmd": "cancel"})
				model.drain()
				return
			default:
			}

			response, err := model.recv()
			if err != nil {
				model.lastErr = err
				return
			}

			if errMsg, ok := response["error"].(string); ok {
				model.lastErr = core.E("mlxlm.Chat", errMsg, nil)
				return
			}

			if _, ok := response["done"]; ok {
				return
			}

			text, _ := response["token"].(string)
			var id int32
			if fid, ok := response["token_id"].(float64); ok {
				id = int32(fid)
			}

			if !yield(inference.Token{ID: id, Text: text}) {
				_ = model.send(map[string]any{"cmd": "cancel"})
				model.drain()
				return
			}
		}
	}
}

// Classify is not supported by the subprocess backend.
// Use the native Metal backend for classification.
func (model *mlxlmModel) Classify(_ context.Context, _ []string, _ ...inference.GenerateOption) ([]inference.ClassifyResult, error) {
	return nil, core.E("mlxlm.Classify", "not supported (use native Metal backend)", nil)
}

// BatchGenerate is not supported by the subprocess backend.
// Use the native Metal backend for batch generation.
func (model *mlxlmModel) BatchGenerate(_ context.Context, _ []string, _ ...inference.GenerateOption) ([]inference.BatchResult, error) {
	return nil, core.E("mlxlm.BatchGenerate", "not supported (use native Metal backend)", nil)
}

// ModelType returns the architecture identifier reported by the subprocess.
func (model *mlxlmModel) ModelType() string { return model.modelType }

func (model *mlxlmModel) Info() inference.ModelInfo {
	model.mu.Lock()
	defer model.mu.Unlock()

	if err := model.send(map[string]any{"cmd": "info"}); err != nil {
		return inference.ModelInfo{}
	}
	response, err := model.recv()
	if err != nil {
		return inference.ModelInfo{}
	}
	if _, ok := response["error"]; ok {
		return inference.ModelInfo{}
	}

	info := inference.ModelInfo{
		Architecture: model.modelType,
		VocabSize:    model.vocabSize,
	}
	if layers, ok := response["layers"].(float64); ok {
		info.NumLayers = int(layers)
	}
	if hidden, ok := response["hidden_size"].(float64); ok {
		info.HiddenSize = int(hidden)
	}
	return info
}

// Metrics returns empty metrics; the subprocess backend does not track timing.
func (model *mlxlmModel) Metrics() inference.GenerateMetrics {
	return inference.GenerateMetrics{}
}

// Err returns the error from the last Generate or Chat call.
func (model *mlxlmModel) Err() error { return model.lastErr }

// Close sends quit and waits up to 2 seconds for the subprocess to exit, then kills it.
func (model *mlxlmModel) Close() error {
	_ = model.send(map[string]any{"cmd": "quit"}) // ignore errors — subprocess may be dead
	_ = model.stdin.Close()
	done := make(chan error, 1)
	go func() { done <- model.process.Wait() }()

	select {
	case err := <-done:
		return err
	case <-time.After(2 * time.Second):
		_ = model.process.Kill()
		return <-done
	}
}

// drain discards subprocess output until "done" or "error", keeping the protocol in sync.
func (model *mlxlmModel) drain() {
	for {
		response, err := model.recv()
		if err != nil {
			return
		}
		if _, ok := response["done"]; ok {
			return
		}
		if _, ok := response["error"]; ok {
			return
		}
	}
}

// InspectAttention implements inference.AttentionInspector.
func (model *mlxlmModel) InspectAttention(ctx context.Context, prompt string, opts ...inference.GenerateOption) (*inference.AttentionSnapshot, error) {
	model.mu.Lock()
	defer model.mu.Unlock()

	request := map[string]any{
		"cmd":    "inspect",
		"prompt": prompt,
	}
	if err := model.send(request); err != nil {
		return nil, core.E("mlxlm.InspectAttention", "send inspect", err)
	}

	response, err := model.recv()
	if err != nil {
		return nil, core.E("mlxlm.InspectAttention", "recv inspect", err)
	}
	if errMsg, ok := response["error"].(string); ok {
		return nil, core.E("mlxlm.InspectAttention", errMsg, nil)
	}

	snapshotDir, _ := response["dir"].(string)
	numLayers := int(response["num_layers"].(float64))
	numKeyValueHeads := int(response["num_kv_heads"].(float64))
	numQueryHeads := int(response["num_q_heads"].(float64))
	seqLen := int(response["seq_len"].(float64))
	headDim := int(response["head_dim"].(float64))
	architecture, _ := response["architecture"].(string)

	keys := make([][][]float32, numLayers)
	queries := make([][][]float32, numLayers)

	for layerIndex := range numLayers {
		keyPath := core.JoinPath(snapshotDir, core.Sprintf("keys_%02d.bin", layerIndex))
		keyData, err := coreio.Local.Read(keyPath)
		if err != nil {
			continue
		}
		keys[layerIndex] = reshapeFloat32([]byte(keyData), numKeyValueHeads, seqLen*headDim)

		queryPath := core.JoinPath(snapshotDir, core.Sprintf("queries_%02d.bin", layerIndex))
		queryData, err := coreio.Local.Read(queryPath)
		if err != nil {
			continue
		}
		queries[layerIndex] = reshapeFloat32([]byte(queryData), numQueryHeads, seqLen*headDim)
	}

	coreio.Local.DeleteAll(snapshotDir)

	return &inference.AttentionSnapshot{
		NumLayers:     numLayers,
		NumHeads:      numKeyValueHeads,
		NumQueryHeads: numQueryHeads,
		SeqLen:        seqLen,
		HeadDim:       headDim,
		Keys:          keys,
		Queries:       queries,
		Architecture:  architecture,
	}, nil
}

// reshapeFloat32 reads raw little-endian float32 bytes and reshapes them into
// [numHeads][stride] slices, one slice per attention head.
//
//	// 8 heads, seqLen=5, headDim=64 → stride=320 floats per head
//	heads := reshapeFloat32(rawBytes, 8, 5*64)
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

// kill terminates the subprocess immediately (used during load failures).
func (model *mlxlmModel) kill() {
	_ = model.stdin.Close()
	_ = model.process.Kill()
	_ = model.process.Wait()
}

const maxJSONLineBytes = 1024 * 1024

type jsonLineReader struct {
	reader  io.Reader
	pending []byte
	scratch []byte
}

func newJSONLineReader(reader io.Reader) *jsonLineReader {
	return &jsonLineReader{
		reader:  reader,
		pending: make([]byte, 0, 32*1024),
		scratch: make([]byte, 32*1024),
	}
}

func (reader *jsonLineReader) ReadLine() ([]byte, error) {
	for {
		if index := bytes.IndexByte(reader.pending, '\n'); index >= 0 {
			line := make([]byte, index)
			copy(line, reader.pending[:index])
			if len(line) > 0 && line[len(line)-1] == '\r' {
				line = line[:len(line)-1]
			}
			reader.pending = reader.pending[index+1:]
			return line, nil
		}

		if len(reader.pending) >= maxJSONLineBytes {
			return nil, core.E("mlxlm.recv", "JSONL line exceeds 1 MiB", nil)
		}

		chunk := reader.scratch
		if remaining := maxJSONLineBytes - len(reader.pending); remaining < len(chunk) {
			chunk = chunk[:remaining]
		}
		n, err := reader.reader.Read(chunk)
		if n > 0 {
			reader.pending = append(reader.pending, chunk[:n]...)
			continue
		}
		if err != nil {
			if err == io.EOF && len(reader.pending) > 0 {
				line := make([]byte, len(reader.pending))
				copy(line, reader.pending)
				reader.pending = reader.pending[:0]
				return line, nil
			}
			return nil, err
		}
	}
}

type mlxlmProcess struct {
	process *os.Process
	stdin   io.WriteCloser
	stdout  io.ReadCloser
	done    chan struct{}
	state   *os.ProcessState
	err     error
}

func newMLXLMCore() *core.Core {
	c := core.New()
	c.Action("process.run", mlxlmProcessRun)
	c.Action("process.start", mlxlmProcessStart)
	return c
}

func mlxlmProcessRun(ctx context.Context, opts core.Options) core.Result {
	proc, err := startProcessFromOptions(ctx, opts)
	if err != nil {
		return core.Result{Value: err, OK: false}
	}
	_ = proc.stdin.Close()

	drained := make(chan struct{})
	go func() {
		_, _ = io.Copy(io.Discard, proc.stdout)
		close(drained)
	}()

	err = proc.Wait()
	<-drained
	if err != nil {
		return core.Result{Value: err, OK: false}
	}
	return core.Result{Value: "", OK: true}
}

func mlxlmProcessStart(ctx context.Context, opts core.Options) core.Result {
	proc, err := startProcessFromOptions(ctx, opts)
	if err != nil {
		return core.Result{Value: err, OK: false}
	}
	return core.Result{Value: proc, OK: true}
}

func startProcessFromOptions(ctx context.Context, opts core.Options) (*mlxlmProcess, error) {
	command := opts.String("command")
	args, err := stringSliceOption(opts, "args")
	if err != nil {
		return nil, err
	}
	return startMLXLMProcess(ctx, command, args...)
}

func stringSliceOption(opts core.Options, key string) ([]string, error) {
	result := opts.Get(key)
	if !result.OK {
		return nil, nil
	}
	args, ok := result.Value.([]string)
	if !ok {
		return nil, core.E("mlxlm.process", key+" must be []string", nil)
	}
	return append([]string(nil), args...), nil
}

func startMLXLMProcess(ctx context.Context, command string, args ...string) (*mlxlmProcess, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if command == "" {
		return nil, core.E("mlxlm.process", "command is required", nil)
	}

	path, err := lookPath(command)
	if err != nil {
		return nil, err
	}

	stdinRead, stdinWrite, err := os.Pipe()
	if err != nil {
		return nil, core.E("mlxlm.process", "stdin pipe", err)
	}
	stdoutRead, stdoutWrite, err := os.Pipe()
	if err != nil {
		_ = stdinRead.Close()
		_ = stdinWrite.Close()
		return nil, core.E("mlxlm.process", "stdout pipe", err)
	}
	stderr, err := os.OpenFile(os.DevNull, os.O_WRONLY, 0)
	if err != nil {
		_ = stdinRead.Close()
		_ = stdinWrite.Close()
		_ = stdoutRead.Close()
		_ = stdoutWrite.Close()
		return nil, core.E("mlxlm.process", "stderr pipe", err)
	}

	files := []*os.File{stdinRead, stdoutWrite, stderr}
	process, err := os.StartProcess(path, append([]string{command}, args...), &os.ProcAttr{
		Env:   os.Environ(),
		Files: files,
	})
	_ = stdinRead.Close()
	_ = stdoutWrite.Close()
	_ = stderr.Close()
	if err != nil {
		_ = stdinWrite.Close()
		_ = stdoutRead.Close()
		return nil, core.E("mlxlm.process", "start "+command, err)
	}

	proc := &mlxlmProcess{
		process: process,
		stdin:   stdinWrite,
		stdout:  stdoutRead,
		done:    make(chan struct{}),
	}
	go proc.wait()
	go proc.killOnContextDone(ctx)
	return proc, nil
}

func (proc *mlxlmProcess) wait() {
	proc.state, proc.err = proc.process.Wait()
	close(proc.done)
}

func (proc *mlxlmProcess) killOnContextDone(ctx context.Context) {
	select {
	case <-ctx.Done():
		_ = proc.Kill()
	case <-proc.done:
	}
}

func (proc *mlxlmProcess) Wait() error {
	<-proc.done
	if proc.err != nil {
		return proc.err
	}
	if proc.state != nil && !proc.state.Success() {
		return core.E("mlxlm.process.Wait", proc.state.String(), nil)
	}
	return nil
}

func (proc *mlxlmProcess) Kill() error {
	if proc == nil || proc.process == nil {
		return nil
	}
	return proc.process.Kill()
}

func lookPath(command string) (string, error) {
	if dir, _ := filepath.Split(command); dir != "" {
		if executable(command) {
			return command, nil
		}
		return "", core.E("mlxlm.process", "executable not found: "+command, nil)
	}

	for _, dir := range filepath.SplitList(os.Getenv("PATH")) {
		if dir == "" {
			dir = "."
		}
		path := filepath.Join(dir, command)
		if executable(path) {
			return path, nil
		}
	}
	return "", core.E("mlxlm.process", "executable not found: "+command, nil)
}

func executable(path string) bool {
	info, err := os.Stat(path)
	return err == nil && !info.IsDir() && info.Mode()&0111 != 0
}

func resultError(result core.Result) error {
	if err, ok := result.Value.(error); ok {
		return err
	}
	return nil
}
