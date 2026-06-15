// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	core "dappco.re/go"
	mp "dappco.re/go/mlx/pack"
)

// inspectModelPackChatTemplate resolves the pack's chat template, preferring a
// tokenizer_config.json chat_template, then a chat_template.jinja sidecar, then
// the architecture profile's native template — recording the source and raising
// a warning/error when none is found (gated on cfg.RequireChatTemplate).
func inspectModelPackChatTemplate(pack *mp.ModelPack, root string, cfg mp.ModelPackConfig, dir *modelPackDirIndex) {
	if dir.has("tokenizer_config.json") {
		tokenizerConfigPath := core.PathJoin(root, "tokenizer_config.json")
		if template, ok, err := readTokenizerChatTemplate(tokenizerConfigPath); ok {
			pack.TokenizerConfigPath = tokenizerConfigPath
			pack.ChatTemplate = template
			pack.ChatTemplateSource = mp.ModelPackChatTemplateFile
			pack.HasChatTemplate = true
			return
		} else if err != nil {
			pack.AddIssue(mp.ModelPackIssueWarning, mp.ModelPackIssueMissingChatTemplate, err.Error(), tokenizerConfigPath)
		}
	}

	if dir.has("chat_template.jinja") {
		jinjaPath := core.PathJoin(root, "chat_template.jinja")
		if template, ok, err := readJinjaChatTemplate(jinjaPath); ok {
			pack.TokenizerConfigPath = jinjaPath
			pack.ChatTemplate = template
			pack.ChatTemplateSource = mp.ModelPackChatTemplateJinja
			pack.HasChatTemplate = true
			return
		} else if err != nil {
			pack.AddIssue(mp.ModelPackIssueWarning, mp.ModelPackIssueMissingChatTemplate, err.Error(), jinjaPath)
		}
	}

	// inspectModelPackArchitecture has already resolved
	// pack.ArchitectureProfile when the architecture is known; consult
	// it directly so we don't re-enter profile.LookupArchitectureProfile
	// once for the native template and again for the requires-template
	// predicate.
	archProfile := pack.ArchitectureProfile
	if archProfile != nil && archProfile.ChatTemplate != "" {
		pack.ChatTemplate = archProfile.ChatTemplate
		pack.ChatTemplateSource = mp.ModelPackChatTemplateNative
		pack.HasChatTemplate = true
		return
	}
	requiresTemplate := true
	if archProfile != nil {
		requiresTemplate = archProfile.RequiresChatTemplate
	}
	if !requiresTemplate {
		return
	}
	if cfg.RequireChatTemplate {
		pack.AddIssue(mp.ModelPackIssueError, mp.ModelPackIssueMissingChatTemplate, "no tokenizer_config.json chat_template or native chat template is available", root)
	}
}

func readTokenizerChatTemplate(path string) (string, bool, error) {
	read := core.ReadFile(path)
	if !read.OK {
		if core.IsNotExist(read.Value.(error)) {
			return "", false, nil
		}
		return "", false, read.Value.(error)
	}
	// chat_template is usually a single Jinja string but can also be a
	// list of {name, template} dicts. Defer the decode via RawMessage
	// so we don't pay the any-decoding cost — the common path is a
	// single string which only needs a string-unmarshal afterwards.
	var config struct {
		ChatTemplate core.RawMessage `json:"chat_template"`
	}
	if result := core.JSONUnmarshal(read.Value.([]byte), &config); !result.OK {
		return "", false, result.Value.(error)
	}
	raw := config.ChatTemplate
	if len(raw) == 0 || core.AsString(raw) == "null" {
		return "", false, nil
	}
	switch raw[0] {
	case '"':
		var template string
		if result := core.JSONUnmarshal(raw, &template); !result.OK {
			return "", false, result.Value.(error)
		}
		template = core.Trim(template)
		return template, template != "", nil
	case '[':
		// Non-empty arrays start with '[' followed by something other
		// than ']'. The whitespace shapes JSON allows are space/tab/
		// newline/carriage-return per RFC 8259.
		for i := 1; i < len(raw); i++ {
			c := raw[i]
			if c == ' ' || c == '\t' || c == '\n' || c == '\r' {
				continue
			}
			if c == ']' {
				return "", false, nil
			}
			return "named_chat_templates", true, nil
		}
	}
	return "", false, nil
}

func readJinjaChatTemplate(path string) (string, bool, error) {
	read := core.ReadFile(path)
	if !read.OK {
		if core.IsNotExist(read.Value.(error)) {
			return "", false, nil
		}
		return "", false, read.Value.(error)
	}
	template := core.Trim(core.AsString(read.Value.([]byte)))
	return template, template != "", nil
}
