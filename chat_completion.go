package mistralai

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
)

// Chat message role values for ChatMessage.Role.
const (
	RoleSystem    = "system"
	RoleUser      = "user"
	RoleAssistant = "assistant"
)

// Content part type values for ChatMessageContentPart.Type.
const (
	ContentPartText        = "text"
	ContentPartFile        = "file"
	ContentPartImageURL    = "image_url"
	ContentPartDocumentURL = "document_url"
)

// ChatMessage is one message in a chat completion request or response.
type ChatMessage struct {
	Role       string     `json:"role"`
	Content    any        `json:"content"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
	Name       string     `json:"name,omitempty"`
}

// TextMessage builds a plain-text ChatMessage for the given role
// (RoleSystem, RoleUser, or RoleAssistant).
func TextMessage(role, text string) ChatMessage {
	return ChatMessage{Role: role, Content: text}
}

// MultipartMessage builds a ChatMessage whose content is a list of multimodal
// parts (see TextPart, FilePart, ImageURLPart, DocumentURLPart).
func MultipartMessage(role string, parts ...ChatMessageContentPart) ChatMessage {
	return ChatMessage{Role: role, Content: parts}
}

// TextPart is a text content part.
func TextPart(text string) ChatMessageContentPart {
	return ChatMessageContentPart{Type: ContentPartText, Text: text}
}

// FilePart references a previously uploaded file by its API file id
// (see Client.UploadFile).
func FilePart(fileID string) ChatMessageContentPart {
	return ChatMessageContentPart{Type: ContentPartFile, FileID: fileID}
}

// ImageURLPart references an image by URL.
func ImageURLPart(url string) ChatMessageContentPart {
	return ChatMessageContentPart{Type: ContentPartImageURL, ImageURL: url}
}

// DocumentURLPart references a document by URL.
func DocumentURLPart(url string) ChatMessageContentPart {
	return ChatMessageContentPart{Type: ContentPartDocumentURL, DocumentURL: url}
}

// ChatCompletionRequest is the body for POST /v1/chat/completions.
type ChatCompletionRequest struct {
	Model             string          `json:"model"`
	Messages          []ChatMessage   `json:"messages"`
	Temperature       *float64        `json:"temperature,omitempty"`
	MaxTokens         int             `json:"max_tokens,omitempty"`
	TopP              *float64        `json:"top_p,omitempty"`
	ResponseFormat    *ResponseFormat `json:"response_format,omitempty"`
	Tools             []Tool          `json:"tools,omitempty"`
	ToolChoice        any             `json:"tool_choice,omitempty"`
	ParallelToolCalls *bool           `json:"parallel_tool_calls,omitempty"`
}

// ChatCompletionResponseChoice is one completion choice.
type ChatCompletionResponseChoice struct {
	Index        int         `json:"index"`
	Message      ChatMessage `json:"message"`
	FinishReason string      `json:"finish_reason"`
}

// ToolMessage builds a tool result message for a prior tool_call.
func ToolMessage(toolCallID, name, content string) ChatMessage {
	return ChatMessage{
		Role:       RoleTool,
		ToolCallID: toolCallID,
		Name:       name,
		Content:    content,
	}
}

// AssistantToolCallsMessage builds an assistant message that requests tool calls.
func AssistantToolCallsMessage(calls []ToolCall) ChatMessage {
	return ChatMessage{
		Role:      RoleAssistant,
		Content:   nil,
		ToolCalls: calls,
	}
}

// HasToolCalls reports whether the choice message contains tool calls.
func (c ChatCompletionResponseChoice) HasToolCalls() bool {
	return len(c.Message.ToolCalls) > 0
}

// FirstChoice returns the first completion choice or an error if missing.
func (r ChatCompletionResponse) FirstChoice() (ChatCompletionResponseChoice, error) {
	if len(r.Choices) == 0 {
		return ChatCompletionResponseChoice{}, fmt.Errorf("mistral: chat response has no choices")
	}
	return r.Choices[0], nil
}

// UsageInfo reports token usage from a chat completion.
type UsageInfo struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// ChatCompletionResponse is the successful chat completions API payload.
type ChatCompletionResponse struct {
	ID      string                         `json:"id"`
	Model   string                         `json:"model"`
	Choices []ChatCompletionResponseChoice `json:"choices"`
	Object  string                         `json:"object"`
	Usage   UsageInfo                      `json:"usage"`
}

// AllChoicesContent concatenates assistant content from every choice.
func (r ChatCompletionResponse) AllChoicesContent() string {
	var b strings.Builder
	for _, c := range r.Choices {
		if c.Message.Content == nil {
			continue
		}
		if s, ok := c.Message.Content.(string); ok {
			b.WriteString(s)
			continue
		}
		encoded, err := json.Marshal(c.Message.Content)
		if err != nil {
			continue
		}
		b.Write(encoded)
	}
	return b.String()
}

// FirstChoiceContent returns trimmed content from the first choice, or an error if missing/empty.
func (r ChatCompletionResponse) FirstChoiceContent() (string, error) {
	choice, err := r.FirstChoice()
	if err != nil {
		return "", err
	}
	if choice.HasToolCalls() {
		return "", fmt.Errorf("mistral: choice has tool_calls, no text content yet")
	}
	raw := choice.Message.Content
	content, ok := raw.(string)
	if !ok {
		if raw == nil {
			return "", fmt.Errorf("mistral: chat response has empty content")
		}
		encoded, err := json.Marshal(raw)
		if err != nil {
			return "", fmt.Errorf("mistral: chat response has unsupported content type")
		}
		content = string(encoded)
	}
	content = strings.TrimSpace(content)
	if content == "" {
		return "", fmt.Errorf("mistral: chat response has empty content")
	}
	return content, nil
}

func (c *client) ChatCompletion(ctx context.Context, req ChatCompletionRequest) (ChatCompletionResponse, error) {
	if strings.TrimSpace(req.Model) == "" {
		return ChatCompletionResponse{}, fmt.Errorf("mistral: model is required")
	}
	if len(req.Messages) == 0 {
		return ChatCompletionResponse{}, fmt.Errorf("mistral: messages are required")
	}

	var resp ChatCompletionResponse
	if err := c.postJSON(ctx, "/v1/chat/completions", req, &resp); err != nil {
		return ChatCompletionResponse{}, fmt.Errorf("mistral: chat completion: %w", err)
	}
	return resp, nil
}
