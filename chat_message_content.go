package mistralai

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
)

// Message is the public name for a chat message.
type Message = ChatMessage

// TextContent is a typed text content value. It encodes as a JSON string.
type TextContent string

// ChatMessageContentPart is one multimodal content item for Chat Completions.
type ChatMessageContentPart struct {
	Type        string `json:"type"`
	Text        string `json:"text,omitempty"`
	FileID      string `json:"file_id,omitempty"`
	DocumentURL string `json:"document_url,omitempty"`
	ImageURL    string `json:"image_url,omitempty"`
}

func (m *ChatMessage) UnmarshalJSON(data []byte) error {
	type messageWire struct {
		Role       string          `json:"role"`
		Content    json.RawMessage `json:"content"`
		ToolCalls  []ToolCall      `json:"tool_calls,omitempty"`
		ToolCallID string          `json:"tool_call_id,omitempty"`
		Name       string          `json:"name,omitempty"`
	}
	var wire messageWire
	if err := json.Unmarshal(data, &wire); err != nil {
		return err
	}
	var content any
	switch raw := bytes.TrimSpace(wire.Content); {
	case len(raw) == 0 || bytes.Equal(raw, []byte("null")):
		content = nil
	case raw[0] == '"':
		var text string
		if err := json.Unmarshal(raw, &text); err != nil {
			return err
		}
		content = TextContent(text)
	case raw[0] == '[':
		var parts []ChatMessageContentPart
		if err := json.Unmarshal(raw, &parts); err != nil {
			return err
		}
		content = parts
	default:
		return fmt.Errorf("mistral: message content must be null, text, or content parts")
	}
	*m = ChatMessage{Role: wire.Role, Content: content, ToolCalls: wire.ToolCalls, ToolCallID: wire.ToolCallID, Name: wire.Name}
	return nil
}

func (m ChatMessage) validate() error {
	if strings.TrimSpace(m.Role) == "" {
		return fmt.Errorf("role is required")
	}
	switch content := m.Content.(type) {
	case string:
		if content == "" && (m.Role != RoleAssistant || len(m.ToolCalls) == 0) {
			return fmt.Errorf("text content is empty")
		}
	case TextContent:
		if content == "" && (m.Role != RoleAssistant || len(m.ToolCalls) == 0) {
			return fmt.Errorf("text content is empty")
		}
	case []ChatMessageContentPart:
		if len(content) == 0 {
			return fmt.Errorf("content parts are empty")
		}
		for index, part := range content {
			if err := part.validate(); err != nil {
				return fmt.Errorf("content part %d: %w", index, err)
			}
		}
	case nil:
		if m.Role != RoleAssistant || len(m.ToolCalls) == 0 {
			return fmt.Errorf("null content requires assistant tool calls")
		}
	default:
		return fmt.Errorf("unsupported content type %T", content)
	}
	return nil
}

func (p ChatMessageContentPart) validate() error {
	switch p.Type {
	case ContentPartText:
		if p.Text == "" || p.FileID != "" || p.DocumentURL != "" || p.ImageURL != "" {
			return fmt.Errorf("invalid text part")
		}
	case ContentPartFile:
		if p.FileID == "" || p.Text != "" || p.DocumentURL != "" || p.ImageURL != "" {
			return fmt.Errorf("invalid file part")
		}
	case ContentPartDocumentURL:
		if p.DocumentURL == "" || p.Text != "" || p.FileID != "" || p.ImageURL != "" {
			return fmt.Errorf("invalid document URL part")
		}
	case ContentPartImageURL:
		if p.ImageURL == "" || p.Text != "" || p.FileID != "" || p.DocumentURL != "" {
			return fmt.Errorf("invalid image URL part")
		}
	default:
		return fmt.Errorf("unsupported content part type %q", p.Type)
	}
	return nil
}
