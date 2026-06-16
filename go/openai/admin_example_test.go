// SPDX-Licence-Identifier: EUPL-1.2

package openai_test

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"

	"dappco.re/go/inference"
	openaicompat "dappco.re/go/inference/openai"
	"dappco.re/go/mlx/openai"
)

// ExampleHealth shows the small health payload the local compatibility mux
// serves. Hosts return it from an AdminConfig.Health callback; the handler
// fills "ok"/"go-mlx"/time defaults for any field left blank.
func ExampleHealth() {
	h := openai.Health{Status: "ok", Runtime: "go-mlx", Models: []string{"qwen3"}}
	fmt.Println(h.Status)
	fmt.Println(h.Runtime)
	fmt.Println(h.Models[0])
	// Output:
	// ok
	// go-mlx
	// qwen3
}

// ExampleActionResponse shows the envelope a runtime wake/sleep callback
// produces — the action name and its status.
func ExampleActionResponse() {
	resp := openai.ActionResponse{Action: "wake", Status: "ok"}
	fmt.Println(resp.Action, resp.Status)
	// Output: wake ok
}

// ExampleAdminConfig wires a host-owned health callback onto the compatibility
// mux and drives the health route, printing the status the handler returns.
func ExampleAdminConfig() {
	resolver := openaicompat.NewStaticResolver(map[string]inference.TextModel{"qwen": exampleModel{}})
	handler := openai.NewMuxWithAdmin(resolver, openai.AdminConfig{
		Health: func(context.Context) (openai.Health, error) {
			return openai.Health{Status: "ok", Models: []string{"qwen3"}}, nil
		},
	})

	req := httptest.NewRequest(http.MethodGet, openai.DefaultHealthPath, nil)
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)

	fmt.Println(rec.Code)
	fmt.Println(strings.Contains(rec.Body.String(), `"status":"ok"`))
	// Output:
	// 200
	// true
}
