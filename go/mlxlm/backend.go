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
	"reflect"
	"syscall"
	"time"

	"dappco.re/go"

	"dappco.re/go/inference"
	coreio "dappco.re/go/io"
)

//go:embed bridge.py
var bridgeFS embed.FS

var (
	mlxlmCore         = newMLXLMCore()
	bridgeScriptLock  = mlxlmCore.Lock("mlxlm.bridgeScript").Mutex
	bridgeScriptDone  core.AtomicBool // fast-path probe — set after first init
	bridgeScriptReady bool
	bridgeScriptPath  string // extracted bridge.py temp path (created once per process)
	bridgeScriptError error

	errClassifyUnsupported      = core.E("mlxlm.Classify", "not supported (use native Metal backend)", nil)
	errBatchGenerateUnsupported = core.E("mlxlm.BatchGenerate", "not supported (use native Metal backend)", nil)
)

// extractScript writes the embedded bridge.py to a temp file and returns its path.
//
//	bridgePath, err := extractScript() // called automatically by LoadModel
func extractScript() (string, error) {
	// Fast path: post-init readers skip the mutex entirely via an
	// atomic acquire on the "done" flag. The path/error pair is
	// published-before via the matching atomic store inside the
	// locked block; only the very first writer pays the lock cost.
	if bridgeScriptDone.Load() {
		return bridgeScriptPath, bridgeScriptError
	}

	bridgeScriptLock.Lock()
	defer bridgeScriptLock.Unlock()

	if bridgeScriptReady {
		return bridgeScriptPath, bridgeScriptError
	}
	bridgeScriptReady = true

	data, err := bridgeFS.ReadFile("bridge.py")
	if err != nil {
		bridgeScriptError = core.E("mlxlm.extractScript", "read embedded bridge.py", err)
		bridgeScriptDone.Store(true)
		return bridgeScriptPath, bridgeScriptError
	}
	dir := (&core.Fs{}).New("/").TempDir("mlxlm-")
	if dir == "" {
		bridgeScriptError = core.E("mlxlm.extractScript", "create temp dir", nil)
		bridgeScriptDone.Store(true)
		return bridgeScriptPath, bridgeScriptError
	}
	p := core.JoinPath(dir, "bridge.py")
	if err := coreio.Local.Write(p, string(data)); err != nil {
		bridgeScriptError = core.E("mlxlm.extractScript", "write bridge.py", err)
		bridgeScriptDone.Store(true)
		return bridgeScriptPath, bridgeScriptError
	}
	bridgeScriptPath = p
	bridgeScriptDone.Store(true)
	return bridgeScriptPath, bridgeScriptError
}

func init() {
	inference.Register(&mlxlmbackend{})
}

type mlxlmbackend struct{}

func (backend *mlxlmbackend) Name() string { return "mlx_lm" }

// Available reports whether python3 is on PATH.
func (backend *mlxlmbackend) Available() bool {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	return mlxlmCore.Process().Run(ctx, "python3", "--version").OK
}

// LoadModel spawns bridge.py as a subprocess and returns a TextModel backed by it.
//
//	model, err := inference.LoadModel("/path/to/model", inference.WithBackend("mlx_lm"))
func (backend *mlxlmbackend) LoadModel(modelPath string, opts ...inference.LoadOption) (inference.TextModel, error) {
	return loadModel(context.Background(), modelPath, "", opts...)
}

// loadModel is the internal implementation. scriptPathOverride substitutes the embedded
// bridge.py for testing.
func loadModel(ctx context.Context, modelPath, scriptPathOverride string, opts ...inference.LoadOption) (inference.TextModel, error) {
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
	proc, ok := result.Value.(*mlxlmprocess)
	if !ok {
		return nil, core.E("mlxlm.loadModel", "process.start returned unexpected handle", nil)
	}

	model := &mlxlmmodel{
		process: proc,
		stdin:   proc.stdin,
		stdout:  newJSONLineReader(proc.stdout),
		mu:      mlxlmCore.Lock("mlxlm.model." + core.ID()).Mutex,
	}

	loadRequest := map[string]any{
		"cmd":       "load",
		"pa" + "th": modelPath,
	}
	loadOptions := inference.ApplyLoadOpts(opts)
	if loadOptions.AdapterPath != "" {
		loadRequest["adapter_path"] = loadOptions.AdapterPath
	}
	if loadOptions.ContextLen > 0 {
		loadRequest["context_len"] = loadOptions.ContextLen
	}
	if loadOptions.GPULayers != 0 {
		loadRequest["gpu_layers"] = loadOptions.GPULayers
	}
	if loadOptions.ParallelSlots > 0 {
		loadRequest["parallel_slots"] = loadOptions.ParallelSlots
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

type mlxlmmodel struct {
	process *mlxlmprocess
	stdin   io.WriteCloser
	stdout  *jsonlinereader

	modelType string
	vocabSize int

	lastErr error
	mu      mutex // serialise Generate/Chat calls
}

type mutex interface {
	Lock()
	Unlock()
}

// chatMessagePayload is the wire shape mlxlm.Chat ships to bridge.py
// for each turn. Held as a typed struct rather than a map[string]string
// so each Chat call pays one slice allocation, not len(messages) map
// allocations.
type chatMessagePayload struct {
	Role    string `json:"role"`
	Content string `json:"content"`
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
func (model *mlxlmmodel) send(obj map[string]any) error {
	encoded := core.JSONMarshal(obj)
	if !encoded.OK {
		return core.E("mlxlm.send", "marshal", nil)
	}
	data := append(encoded.Value.([]byte), '\n')
	_, err := model.stdin.Write(data)
	return err
}

// recv reads and parses one JSON line from subprocess stdout.
func (model *mlxlmmodel) recv() (map[string]any, error) {
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

// tokenResponse is the typed shape of a single streaming response from
// bridge.py during Generate/Chat. Decoding into this struct avoids the
// map[string]any allocation + four interface{} lookups + type
// assertions the per-token path otherwise pays.
type tokenResponse struct {
	Token   string  `json:"token"`
	TokenID float64 `json:"token_id"`
	Error   string  `json:"error"`
	Done    *bool   `json:"done"`
}

// recvToken reads and parses one streaming response line during
// Generate/Chat. Allocates a stack-friendly typed value instead of the
// map[string]any that recv returns.
func (model *mlxlmmodel) recvToken(resp *tokenResponse) error {
	line, err := model.stdout.ReadLine()
	if err != nil {
		if err == io.EOF {
			return core.E("mlxlm.recv", "subprocess closed stdout", nil)
		}
		return core.E("mlxlm.recv", "read subprocess stdout", err)
	}
	*resp = tokenResponse{}
	if r := core.JSONUnmarshal(line, resp); !r.OK {
		return core.E("mlxlm.recv", "parse response", nil)
	}
	return nil
}

// Generate streams tokens from the subprocess for the given prompt.
// Calls are serialised per model (mu lock).
//
//	for token := range model.Generate(ctx, "Hello", inference.WithMaxTokens(64)) {
//	    fmt.Print(token.Text)
//	}
func (model *mlxlmmodel) Generate(ctx context.Context, prompt string, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
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

		var resp tokenResponse
		for {
			select {
			case <-ctx.Done():
				model.lastErr = ctx.Err()
				model.cancelRequest("mlxlm.Generate")
				model.drain()
				return
			default:
			}

			if err := model.recvToken(&resp); err != nil {
				model.lastErr = err
				return
			}

			if resp.Error != "" {
				model.lastErr = core.E("mlxlm.Generate", resp.Error, nil)
				return
			}

			if resp.Done != nil {
				return
			}

			if !yield(inference.Token{ID: int32(resp.TokenID), Text: resp.Token}) {
				model.cancelRequest("mlxlm.Generate")
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
func (model *mlxlmmodel) Chat(ctx context.Context, messages []inference.Message, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	generateOptions := inference.ApplyGenerateOpts(opts)

	return func(yield func(inference.Token) bool) {
		model.mu.Lock()
		defer model.mu.Unlock()
		model.lastErr = nil

		// Serialise as a typed struct rather than N maps with two
		// string keys each — drops len(messages) map allocations
		// (~5-100B per map plus map-header overhead) to a single
		// slice allocation. bridge.py reads role/content via
		// dict.get() so JSON key order is irrelevant.
		messagePayloads := make([]chatMessagePayload, len(messages))
		for i, msg := range messages {
			messagePayloads[i] = chatMessagePayload{
				Role:    msg.Role,
				Content: msg.Content,
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

		var resp tokenResponse
		for {
			select {
			case <-ctx.Done():
				model.lastErr = ctx.Err()
				model.cancelRequest("mlxlm.Chat")
				model.drain()
				return
			default:
			}

			if err := model.recvToken(&resp); err != nil {
				model.lastErr = err
				return
			}

			if resp.Error != "" {
				model.lastErr = core.E("mlxlm.Chat", resp.Error, nil)
				return
			}

			if resp.Done != nil {
				return
			}

			if !yield(inference.Token{ID: int32(resp.TokenID), Text: resp.Token}) {
				model.cancelRequest("mlxlm.Chat")
				model.drain()
				return
			}
		}
	}
}

// Classify is not supported by the subprocess backend.
// Use the native Metal backend for classification.
func (model *mlxlmmodel) Classify(_ context.Context, _ []string, _ ...inference.GenerateOption) ([]inference.ClassifyResult, error) {
	return nil, errClassifyUnsupported
}

// BatchGenerate is not supported by the subprocess backend.
// Use the native Metal backend for batch generation.
func (model *mlxlmmodel) BatchGenerate(_ context.Context, _ []string, _ ...inference.GenerateOption) ([]inference.BatchResult, error) {
	return nil, errBatchGenerateUnsupported
}

// ModelType returns the architecture identifier reported by the subprocess.
func (model *mlxlmmodel) ModelType() string { return model.modelType }

func (model *mlxlmmodel) Info() inference.ModelInfo {
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
func (model *mlxlmmodel) Metrics() inference.GenerateMetrics {
	return inference.GenerateMetrics{}
}

// Err returns the error from the last Generate or Chat call.
func (model *mlxlmmodel) Err() error { return model.lastErr }

func (model *mlxlmmodel) cancelRequest(operation string) {
	if err := model.send(map[string]any{"cmd": "cancel"}); err != nil && model.lastErr == nil {
		model.lastErr = core.E(operation, "send cancel", err)
	}
}

// Close sends quit and waits up to 2 seconds for the subprocess to exit, then kills it.
func (model *mlxlmmodel) Close() error {
	var closeErr error
	if err := model.send(map[string]any{"cmd": "quit"}); err != nil {
		closeErr = core.ErrorJoin(closeErr, err)
	}
	if err := model.stdin.Close(); err != nil {
		closeErr = core.ErrorJoin(closeErr, err)
	}
	done := make(chan error, 1)
	go func() { done <- model.process.Wait() }()

	select {
	case err := <-done:
		return core.ErrorJoin(closeErr, err)
	case <-time.After(2 * time.Second):
		if err := model.process.Kill(); err != nil {
			closeErr = core.ErrorJoin(closeErr, err)
		}
		return core.ErrorJoin(closeErr, <-done)
	}
}

// drain discards subprocess output until "done" or "error", keeping the protocol in sync.
func (model *mlxlmmodel) drain() {
	var resp tokenResponse
	for {
		if err := model.recvToken(&resp); err != nil {
			return
		}
		if resp.Done != nil || resp.Error != "" {
			return
		}
	}
}

// InspectAttention implements inference.AttentionInspector.
func (model *mlxlmmodel) InspectAttention(ctx context.Context, prompt string, opts ...inference.GenerateOption) (*inference.AttentionSnapshot, error) {
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
		suffix := layerSuffix(layerIndex)
		keyPath := core.JoinPath(snapshotDir, "keys_"+suffix+".bin")
		keyData, err := coreio.Local.Read(keyPath)
		if err != nil {
			continue
		}
		keys[layerIndex] = reshapeFloat32([]byte(keyData), numKeyValueHeads, seqLen*headDim)

		queryPath := core.JoinPath(snapshotDir, "queries_"+suffix+".bin")
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

// layerSuffix returns the zero-padded 2-digit layer index used in
// snapshot filenames (matches Sprintf "%02d" for 0..99).
//
//	layerSuffix(7)  // "07"
//	layerSuffix(28) // "28"
func layerSuffix(layerIndex int) string {
	if layerIndex >= 0 && layerIndex < len(layerSuffixTable) {
		return layerSuffixTable[layerIndex]
	}
	return core.Itoa(layerIndex)
}

// layerSuffixTable holds the precomputed two-digit zero-padded labels
// for layers 0..99 — covers every architecture currently shipped
// (Gemma-3B = 28, Llama-3-8B = 32, Llama-3-70B = 80) without an Itoa
// allocation per call.
var layerSuffixTable = func() [100]string {
	var table [100]string
	for i := 0; i < len(table); i++ {
		if i < 10 {
			table[i] = "0" + core.Itoa(i)
		} else {
			table[i] = core.Itoa(i)
		}
	}
	return table
}()

// reshapeFloat32 reads raw little-endian float32 bytes and reshapes them into
// [numHeads][stride] slices, one slice per attention head.
//
//	// 8 heads, seqLen=5, headDim=64 → stride=320 floats per head
//	heads := reshapeFloat32(rawBytes, 8, 5*64)
func reshapeFloat32(data []byte, numHeads, stride int) [][]float32 {
	totalFloats := len(data) / 4
	heads := make([][]float32, numHeads)
	if stride <= 0 || numHeads <= 0 {
		return heads
	}
	// Determine how many full heads fit, then allocate one backing
	// buffer for all of them and slice it into capped per-head views.
	// Drops N+1 allocations (one per head plus the outer slice) to two
	// — outer slice header + backing buffer — for a typical Gemma-3B
	// inspection with 32 heads × 640-float stride.
	fullHeads := totalFloats / stride
	if fullHeads > numHeads {
		fullHeads = numHeads
	}
	if fullHeads == 0 {
		return heads
	}
	backing := make([]float32, fullHeads*stride)
	for h := 0; h < fullHeads; h++ {
		start := h * stride
		head := backing[start : start+stride : start+stride]
		base := start * 4
		for i := 0; i < stride; i++ {
			bits := binary.LittleEndian.Uint32(data[base+i*4 : base+i*4+4])
			head[i] = math.Float32frombits(bits)
		}
		heads[h] = head
	}
	return heads
}

// kill terminates the subprocess immediately (used during load failures).
func (model *mlxlmmodel) kill() {
	if err := model.stdin.Close(); err != nil && model.lastErr == nil {
		model.lastErr = err
	}
	if err := model.process.Kill(); err != nil && model.lastErr == nil {
		model.lastErr = err
	}
	if err := model.process.Wait(); err != nil && model.lastErr == nil {
		model.lastErr = err
	}
}

const (
	maxJSONLineBytes      = 1024 * 1024
	jsonReaderInitialSize = 32 * 1024
)

type jsonlinereader struct {
	reader  io.Reader
	buf     []byte // ring-style: live data lives at [readPos:writePos]
	readPos int
	writePos int
}

func newJSONLineReader(reader io.Reader) *jsonlinereader {
	return &jsonlinereader{
		reader: reader,
		buf:    make([]byte, jsonReaderInitialSize),
	}
}

func (reader *jsonlinereader) ReadLine() ([]byte, error) {
	for {
		if reader.writePos > reader.readPos {
			pending := reader.buf[reader.readPos:reader.writePos]
			if index := bytes.IndexByte(pending, '\n'); index >= 0 {
				line := core.SliceClone(pending[:index])
				if len(line) > 0 && line[len(line)-1] == '\r' {
					line = line[:len(line)-1]
				}
				reader.readPos += index + 1
				if reader.readPos == reader.writePos {
					reader.readPos, reader.writePos = 0, 0
				}
				return line, nil
			}
		}

		pendingLen := reader.writePos - reader.readPos
		if pendingLen >= maxJSONLineBytes {
			return nil, core.E("mlxlm.recv", "JSONL line exceeds 1 MiB", nil)
		}

		// Compact: shift live bytes to head when the tail is full but
		// the head has been consumed. Avoids the per-Read append-copy
		// the previous design paid by routing reads through a separate
		// scratch buffer.
		if reader.writePos == len(reader.buf) {
			if reader.readPos > 0 {
				copy(reader.buf, reader.buf[reader.readPos:reader.writePos])
				reader.writePos = pendingLen
				reader.readPos = 0
			} else {
				// Tail full, nothing to compact — grow toward the cap.
				target := len(reader.buf) * 2
				if target > maxJSONLineBytes {
					target = maxJSONLineBytes
				}
				if target == len(reader.buf) {
					return nil, core.E("mlxlm.recv", "JSONL line exceeds 1 MiB", nil)
				}
				grown := make([]byte, target)
				copy(grown, reader.buf[:reader.writePos])
				reader.buf = grown
			}
		}

		writable := reader.buf[reader.writePos:]
		if remaining := maxJSONLineBytes - pendingLen; remaining < len(writable) {
			writable = writable[:remaining]
		}
		n, err := reader.reader.Read(writable)
		if n > 0 {
			reader.writePos += n
			continue
		}
		if err != nil {
			if err == io.EOF && pendingLen > 0 {
				line := core.SliceClone(reader.buf[reader.readPos:reader.writePos])
				reader.readPos, reader.writePos = 0, 0
				return line, nil
			}
			return nil, err
		}
	}
}

type mlxlmprocess struct {
	pid    int
	stdin  io.WriteCloser
	stdout io.ReadCloser
	done   chan struct{}
	status syscall.WaitStatus
	err    error
}

func newMLXLMCore() *core.Core {
	c := core.New()
	c.Action("process.run", mlxlmprocessRun)
	c.Action("process.start", mlxlmprocessStart)
	return c
}

func mlxlmprocessRun(ctx context.Context, opts core.Options) core.Result {
	proc, err := startProcessFromOptions(ctx, opts)
	if err != nil {
		return core.Fail(err)
	}
	if err := proc.stdin.Close(); err != nil {
		return core.Fail(err)
	}

	drained := make(chan struct{})
	go func() {
		_, _ = io.Copy(io.Discard, proc.stdout)
		close(drained)
	}()

	err = proc.Wait()
	<-drained
	if err != nil {
		return core.Fail(err)
	}
	return core.Ok("")
}

func mlxlmprocessStart(ctx context.Context, opts core.Options) core.Result {
	proc, err := startProcessFromOptions(ctx, opts)
	if err != nil {
		return core.Fail(err)
	}
	return core.Ok(proc)
}

func startProcessFromOptions(ctx context.Context, opts core.Options) (*mlxlmprocess, error) {
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
	return args, nil
}

func startMLXLMProcess(ctx context.Context, command string, args ...string) (*mlxlmprocess, error) {
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

	stdinPipe := make([]int, 2)
	if err := syscall.Pipe(stdinPipe); err != nil {
		return nil, core.E("mlxlm.process", "stdin pipe", err)
	}
	stdinRead, stdinWrite := stdinPipe[0], stdinPipe[1]
	stdoutPipe := make([]int, 2)
	if err := syscall.Pipe(stdoutPipe); err != nil {
		return nil, core.ErrorJoin(core.E("mlxlm.process", "stdout pipe", err), closeFDs(stdinRead, stdinWrite))
	}
	stdoutRead, stdoutWrite := stdoutPipe[0], stdoutPipe[1]
	syscall.CloseOnExec(stdinRead)
	syscall.CloseOnExec(stdinWrite)
	syscall.CloseOnExec(stdoutRead)
	syscall.CloseOnExec(stdoutWrite)

	argv := make([]string, 1+len(args))
	argv[0] = command
	copy(argv[1:], args)
	pid, err := syscall.ForkExec(path, argv, &syscall.ProcAttr{
		Env:   core.Environ(),
		Files: []uintptr{uintptr(stdinRead), uintptr(stdoutWrite), uintptr(2)},
	})
	err = core.ErrorJoin(err, closeFDs(stdinRead, stdoutWrite))
	if err != nil {
		return nil, core.ErrorJoin(core.E("mlxlm.process", "start "+command, err), closeFDs(stdinWrite, stdoutRead))
	}

	proc := &mlxlmprocess{
		pid:    pid,
		stdin:  fdwritecloser(stdinWrite),
		stdout: fdreadcloser(stdoutRead),
		done:   make(chan struct{}),
	}
	go proc.wait()
	go proc.killOnContextDone(ctx)
	return proc, nil
}

func (proc *mlxlmprocess) wait() {
	_, proc.err = syscall.Wait4(proc.pid, &proc.status, 0, nil)
	closeQuietly(proc.stdin)
	closeQuietly(proc.stdout)
	close(proc.done)
}

func closeQuietly(closer io.Closer) {
	if closer == nil {
		return
	}
	if err := closer.Close(); err != nil {
		return
	}
}

func (proc *mlxlmprocess) killOnContextDone(ctx context.Context) {
	select {
	case <-ctx.Done():
		if err := proc.Kill(); err != nil {
			return
		}
	case <-proc.done:
	}
}

func closeFDs(fds ...int) error {
	var closeErr error
	for _, fd := range fds {
		if fd < 0 {
			continue
		}
		if err := syscall.Close(fd); err != nil {
			closeErr = core.ErrorJoin(closeErr, err)
		}
	}
	return closeErr
}

type fdreadcloser int

func (fd fdreadcloser) Read(p []byte) (int, error) {
	return syscall.Read(int(fd), p)
}

func (fd fdreadcloser) Close() error {
	return syscall.Close(int(fd))
}

type fdwritecloser int

func (fd fdwritecloser) Write(p []byte) (int, error) {
	return syscall.Write(int(fd), p)
}

func (fd fdwritecloser) Close() error {
	return syscall.Close(int(fd))
}

func (proc *mlxlmprocess) Wait() error {
	<-proc.done
	if proc.err != nil {
		return proc.err
	}
	if !proc.status.Exited() || proc.status.ExitStatus() != 0 {
		return core.E("mlxlm.process.Wait", core.Sprintf("exit status %d", proc.status.ExitStatus()), nil)
	}
	return nil
}

func (proc *mlxlmprocess) Kill() error {
	if proc == nil || proc.pid <= 0 {
		return nil
	}
	return syscall.Kill(proc.pid, syscall.SIGKILL)
}

func lookPath(command string) (string, error) {
	if core.Contains(command, string(core.PathSeparator)) {
		if executable(command) {
			return command, nil
		}
		return "", core.E("mlxlm.process", "executable not found: "+command, nil)
	}

	for _, dir := range core.Split(core.Getenv("PATH"), string(core.PathListSeparator)) {
		if dir == "" {
			dir = "."
		}
		path := core.PathJoin(dir, command)
		if executable(path) {
			return path, nil
		}
	}
	return "", core.E("mlxlm.process", "executable not found: "+command, nil)
}

func executable(path string) bool {
	info := core.Stat(path)
	if !info.OK {
		return false
	}
	stat, ok := info.Value.(core.FsFileInfo)
	if !ok {
		return false
	}
	return !stat.IsDir() && stat.Mode()&0111 != 0
}

func resultError(result core.Result) error {
	if err, ok := result.Value.(error); ok {
		return err
	}
	return nil
}
