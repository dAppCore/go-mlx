// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"strconv"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

// RemoteSplitFFNConfig configures an HTTP-backed FFN placement for split
// inference. The endpoint URL receives JSON RemoteSplitFFNRequest payloads and
// returns RemoteSplitFFNResponse payloads.
type RemoteSplitFFNConfig struct {
	Endpoint inference.SplitEndpoint `json:"endpoint,omitempty"`
	URL      string                  `json:"url,omitempty"`
	Headers  map[string]string       `json:"headers,omitempty"`
	Client   *core.HTTPClient        `json:"-"`
}

// RemoteSplitFFNRequest is the stable wire shape sent to a remote FFN
// placement.
type RemoteSplitFFNRequest struct {
	EndpointID string            `json:"endpoint_id,omitempty"`
	Layer      int               `json:"layer"`
	Hidden     []float32         `json:"hidden,omitempty"`
	Labels     map[string]string `json:"labels,omitempty"`
}

// RemoteSplitFFNResponse is the stable wire shape returned by a remote FFN
// placement.
type RemoteSplitFFNResponse struct {
	Hidden []float32 `json:"hidden,omitempty"`
	Error  string    `json:"error,omitempty"`
}

// RemoteSplitFFNExecutor calls a remote HTTP endpoint for omitted FFN layers.
type RemoteSplitFFNExecutor struct {
	endpoint inference.SplitEndpoint
	url      string
	headers  map[string]string
	client   *core.HTTPClient
}

// Sentinel errors for the remote FFN executor hot paths. Built once at
// package init instead of per-call so the steady-state ForwardFFN cost
// excludes the core.NewError allocation triplet (errors.New + struct +
// interface header) for each guard the call cannot avoid checking.
var (
	errRemoteSplitFFNExecutorNil   = core.NewError("mlx: remote split FFN executor is nil")
	errRemoteSplitFFNBodyShape     = core.NewError("mlx: remote split FFN response body shape is invalid")
	errRemoteSplitFFNEmptyHidden   = core.NewError("mlx: remote split FFN endpoint returned empty hidden state")
)

// NewRemoteSplitFFNExecutor creates a network-backed SplitFFNExecutor.
func NewRemoteSplitFFNExecutor(cfg RemoteSplitFFNConfig) (*RemoteSplitFFNExecutor, error) {
	url := core.Trim(firstNonEmpty(cfg.URL, cfg.Endpoint.URL))
	if url == "" {
		return nil, core.NewError("mlx: remote split FFN endpoint URL is required")
	}
	if cfg.Endpoint.Role != "" && cfg.Endpoint.Role != inference.SplitEndpointRoleFFN {
		return nil, core.NewError("mlx: remote split FFN endpoint role must be ffn")
	}
	client := cfg.Client
	if client == nil {
		client = &core.HTTPClient{}
	}
	return &RemoteSplitFFNExecutor{
		endpoint: cfg.Endpoint,
		url:      url,
		headers:  cloneStringMap(cfg.Headers),
		client:   client,
	}, nil
}

// ForwardFFN sends one FFN layer request to the configured remote endpoint.
func (executor *RemoteSplitFFNExecutor) ForwardFFN(ctx context.Context, req SplitFFNRequest) (SplitFFNResult, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return SplitFFNResult{}, err
	}
	if executor == nil {
		return SplitFFNResult{}, errRemoteSplitFFNExecutorNil
	}
	// NewRemoteSplitFFNExecutor already trims + validates the URL and
	// stores the trimmed form on the receiver. Re-running core.Trim on
	// every ForwardFFN call walked the URL string each invocation for
	// a guarantee the constructor had already proven; drop the loop.
	payload := RemoteSplitFFNRequest{
		EndpointID: executor.endpoint.ID,
		Layer:      req.Layer,
		Hidden:     cloneSplitHidden(req.Hidden),
		Labels:     cloneStringMap(executor.endpoint.Labels),
	}
	encoded := core.JSONMarshal(payload)
	if !encoded.OK {
		return SplitFFNResult{}, core.E("RemoteSplitFFNExecutor.ForwardFFN", "marshal request", modelSliceResultError(encoded))
	}
	// core.NewBufferReader → bytes.Reader directly over the JSON bytes
	// avoids the []byte → string copy the prior core.NewReader path forced.
	// JSONMarshal already owns a fresh []byte, so handing it straight to
	// the request body costs one fewer allocation per ForwardFFN call.
	httpReqResult := core.NewHTTPRequestContext(ctx, "POST", executor.url, core.NewBufferReader(encoded.Value.([]byte)))
	if !httpReqResult.OK {
		return SplitFFNResult{}, core.E("RemoteSplitFFNExecutor.ForwardFFN", "build request", modelSliceResultError(httpReqResult))
	}
	httpReq := httpReqResult.Value.(*core.Request)
	httpReq.Header.Set("Accept", "application/json")
	httpReq.Header.Set("Content-Type", "application/json")
	for key, value := range executor.headers {
		httpReq.Header.Set(key, value)
	}
	resp, err := executor.client.Do(httpReq)
	if err != nil {
		return SplitFFNResult{}, core.E("RemoteSplitFFNExecutor.ForwardFFN", "post request", err)
	}
	defer resp.Body.Close()
	read := core.ReadAll(resp.Body)
	if !read.OK {
		return SplitFFNResult{}, core.E("RemoteSplitFFNExecutor.ForwardFFN", "read response", modelSliceResultError(read))
	}
	body, ok := read.Value.(string)
	if !ok {
		return SplitFFNResult{}, errRemoteSplitFFNBodyShape
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		// core.Sprintf("%d: %s", ...) routed through fmt's reflection-driven
		// formatter — strconv.Itoa is direct ascii conversion with zero
		// reflection; core.Concat fuses the parts without a fmt.State.
		return SplitFFNResult{}, core.NewError(core.Concat("mlx: remote split FFN endpoint returned ", strconv.Itoa(resp.StatusCode), ": ", core.Trim(body)))
	}
	var remote RemoteSplitFFNResponse
	// core.ReadAll handed us a string built from a fresh []byte buffer the
	// HTTP transport owns alone; core.AsBytes returns the same backing
	// array without copying. JSONUnmarshal does not retain references past
	// the call (it consumes tokens into target fields), so the read-only
	// alias is safe here. Saves one alloc the size of the response body
	// on every successful ForwardFFN call.
	if result := core.JSONUnmarshal(core.AsBytes(body), &remote); !result.OK {
		return SplitFFNResult{}, core.E("RemoteSplitFFNExecutor.ForwardFFN", "parse response", modelSliceResultError(result))
	}
	if remote.Error != "" {
		return SplitFFNResult{}, core.NewError("mlx: remote split FFN endpoint error: " + remote.Error)
	}
	if len(remote.Hidden) == 0 {
		return SplitFFNResult{}, errRemoteSplitFFNEmptyHidden
	}
	return SplitFFNResult{Hidden: cloneSplitHidden(remote.Hidden)}, nil
}
