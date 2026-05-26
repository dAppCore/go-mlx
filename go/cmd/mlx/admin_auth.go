// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"crypto/rand"
	"crypto/subtle"
	"encoding/base64"
	"io"
	"net/http"

	core "dappco.re/go"
)

// adminTokenPrefix marks the token as a lthn-mlx admin secret so
// future secret-scanners (gitleaks, trufflehog) recognise leaked
// tokens in repos. Matches the gh_pat_/sk-/ghp_ convention.
const adminTokenPrefix = "lthn-mlx_"

// standardAdminTokenPath returns ~/Lethean/data/admin.token — the
// canonical location for the Bearer auth secret. Mode 0600 enforced
// on write so other local users can't read it.
func standardAdminTokenPath() string {
	return core.PathJoin(core.Env("HOME"), "Lethean", "data", "admin.token")
}

// generateAdminToken returns a fresh opaque 256-bit token, base64url-
// encoded, with the lthn-mlx_ prefix. 256 bits of entropy is
// unbreakable in practice.
//
//	tok, err := generateAdminToken()
//	// → "lthn-mlx_K7gH..." (52 chars total)
func generateAdminToken() (string, error) {
	var raw [32]byte
	if _, err := rand.Read(raw[:]); err != nil {
		return "", core.E("admin.generateToken", "rand", err)
	}
	return adminTokenPrefix + base64.RawURLEncoding.EncodeToString(raw[:]), nil
}

// loadAdminToken reads the existing token at path. Returns ("",false,nil)
// for any read failure including file-not-found — the caller treats that
// as "no token yet, generate one" rather than fatal.
func loadAdminToken(path string) (token string, exists bool, err error) {
	res := core.ReadFile(path)
	if !res.OK {
		return "", false, nil
	}
	data, ok := res.Value.([]byte)
	if !ok {
		return "", false, nil
	}
	tok := core.Trim(string(data))
	if tok == "" {
		return "", false, nil
	}
	return tok, true, nil
}

// writeAdminToken writes the token to path with 0o600 perms. Parent
// dir is created if missing. Per Cerberus §5.1 this is the fail-
// closed checkpoint — caller MUST abort serve startup if write fails
// (better to refuse to boot than to bind a listener with an unprotected
// admin surface).
func writeAdminToken(path, token string) error {
	if dir := core.PathDir(path); dir != "" {
		if r := core.MkdirAll(dir, 0o755); !r.OK {
			return core.E("admin.writeToken", "mkdir parent", r.Value.(error))
		}
	}
	if r := core.WriteFile(path, []byte(token+"\n"), 0o600); !r.OK {
		return core.E("admin.writeToken", "write", r.Value.(error))
	}
	return nil
}

// ensureAdminToken loads the existing token or generates + writes a
// fresh one. Returns the token + whether it was freshly generated
// (so serve can print a one-line notice the first time).
func ensureAdminToken(path string) (token string, generated bool, err error) {
	existing, exists, err := loadAdminToken(path)
	if err != nil {
		return "", false, err
	}
	if exists {
		return existing, false, nil
	}
	tok, err := generateAdminToken()
	if err != nil {
		return "", false, err
	}
	if err := writeAdminToken(path, tok); err != nil {
		return "", false, err
	}
	return tok, true, nil
}

// requireBearerOnAdmin wraps next with Bearer-token auth on any path
// starting with /v1/admin/. Other paths (/v1/chat/completions, etc.)
// pass through unauthenticated — the localhost / tunnel-trust model
// still applies to inference, only admin verbs need explicit auth.
//
// Uses crypto/subtle constant-time compare to defeat timing side
// channels. Every 401 audit-emits to stderr so brute-force attempts
// against the token are visible in operator logs.
func requireBearerOnAdmin(next http.Handler, token string, stderr io.Writer) http.Handler {
	expected := []byte("Bearer " + token)
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !core.HasPrefix(r.URL.Path, "/v1/admin/") {
			next.ServeHTTP(w, r)
			return
		}
		got := []byte(r.Header.Get("Authorization"))
		if len(got) != len(expected) || subtle.ConstantTimeCompare(got, expected) != 1 {
			core.Print(stderr, "%s admin: auth deny path=%s remote=%s",
				cliName(), r.URL.Path, r.RemoteAddr)
			w.Header().Set("www-authenticate", `Bearer realm="lthn-mlx-admin"`)
			http.Error(w, "admin endpoint requires Authorization: Bearer <token>", http.StatusUnauthorized)
			return
		}
		next.ServeHTTP(w, r)
	})
}
