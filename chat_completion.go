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

// PromptMode values for ChatCompletionRequest.PromptMode.
const (
	// PromptModeReasoning enables the reasoning system prompt on reasoning models.
	// Deprecated by the API in favor of ReasoningEffort.
	PromptModeReasoning = "reasoning"
)

// ReasoningEffort values for ChatCompletionRequest.ReasoningEffort.
const (
	ReasoningEffortHigh = "high"
	ReasoningEffortNone = "none"
)

// PredictionTypeContent is the only prediction type the API accepts.
const PredictionTypeContent = "content"

// Prediction supplies expected completion content so the API can skip
// generating tokens that match it, reducing latency for edit-style tasks
// (see PredictionContent).
type Prediction struct {
	Type    string `json:"type"`
	Content string `json:"content"`
}

// PredictionContent builds a content prediction for ChatCompletionRequest.Prediction.
func PredictionContent(content string) *Prediction {
	return &Prediction{Type: PredictionTypeContent, Content: content}
}

// ChatCompletionRequest is the body for POST /v1/chat/completions.
type ChatCompletionRequest struct {
	Model       string        `json:"model"`
	Messages    []ChatMessage `json:"messages"`
	Temperature *float64      `json:"temperature,omitempty"`
	MaxTokens   int           `json:"max_tokens,omitempty"`
	TopP        *float64      `json:"top_p,omitempty"`
	// Stream is reserved for SSE streaming, which this client does not support
	// yet; ChatCompletion returns an error when it is set.
	Stream bool `json:"stream,omitempty"`
	// Stop lists tokens that end generation when detected.
	Stop []string `json:"stop,omitempty"`
	// RandomSeed makes sampling deterministic across calls when set.
	RandomSeed *int `json:"random_seed,omitempty"`
	// Metadata is arbitrary metadata stored with the request.
	Metadata       map[string]any  `json:"metadata,omitempty"`
	ResponseFormat *ResponseFormat `json:"response_format,omitempty"`
	Tools          []Tool          `json:"tools,omitempty"`
	ToolChoice     any             `json:"tool_choice,omitempty"`
	// PresencePenalty (-2..2) penalizes words that already appeared at all;
	// pointer so an explicit 0 differs from unset.
	PresencePenalty *float64 `json:"presence_penalty,omitempty"`
	// FrequencyPenalty (-2..2) penalizes words by how often they already appeared.
	FrequencyPenalty *float64 `json:"frequency_penalty,omitempty"`
	// N is the number of completions to return; input tokens are billed once.
	N *int `json:"n,omitempty"`
	// Prediction supplies expected output content to reduce latency (see PredictionContent).
	Prediction        *Prediction `json:"prediction,omitempty"`
	ParallelToolCalls *bool       `json:"parallel_tool_calls,omitempty"`
	// PromptMode toggles the reasoning system prompt (PromptModeReasoning).
	PromptMode string `json:"prompt_mode,omitempty"`
	// ReasoningEffort controls reasoning traces on reasoning models
	// (ReasoningEffortHigh or ReasoningEffortNone).
	ReasoningEffort string `json:"reasoning_effort,omitempty"`
	// PromptCacheKey enables prompt caching: requests sharing the same key and
	// prompt prefix reuse computed tokens at reduced input cost.
	PromptCacheKey string `json:"prompt_cache_key,omitempty"`
	// SafePrompt injects a safety prompt before the conversation.
	SafePrompt bool `json:"safe_prompt,omitempty"`
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

// ChatCompletion runs POST /v1/chat/completions with full message control.
func (c *Client) ChatCompletion(ctx context.Context, req ChatCompletionRequest) (ChatCompletionResponse, error) {
	if strings.TrimSpace(req.Model) == "" {
		return ChatCompletionResponse{}, fmt.Errorf("%w: model is required", ErrInvalidRequest)
	}
	if len(req.Messages) == 0 {
		return ChatCompletionResponse{}, fmt.Errorf("%w: messages are required", ErrInvalidRequest)
	}
	if req.Stream {
		return ChatCompletionResponse{}, fmt.Errorf("%w: streaming is not supported yet; leave Stream unset", ErrInvalidRequest)
	}

	var resp ChatCompletionResponse
	if err := c.postJSON(ctx, "/v1/chat/completions", req, &resp); err != nil {
		return ChatCompletionResponse{}, fmt.Errorf("mistral: chat completion: %w", err)
	}
	return resp, nil
}
