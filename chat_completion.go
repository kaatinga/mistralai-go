package mistralai

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
)

// ChatMessage is one message in a chat completion request or response.
type ChatMessage struct {
	Role    string `json:"role"`
	Content any    `json:"content"`
}

// ChatCompletionRequest is the body for POST /v1/chat/completions.
type ChatCompletionRequest struct {
	Model          string          `json:"model"`
	Messages       []ChatMessage   `json:"messages"`
	Temperature    *float64        `json:"temperature,omitempty"`
	MaxTokens      int             `json:"max_tokens,omitempty"`
	TopP           *float64        `json:"top_p,omitempty"`
	Stream         bool            `json:"stream,omitempty"`
	ResponseFormat *ResponseFormat `json:"response_format,omitempty"`
}

// ChatCompletionResponseChoice is one completion choice.
type ChatCompletionResponseChoice struct {
	Index   int         `json:"index"`
	Message ChatMessage `json:"message"`
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
	if len(r.Choices) == 0 {
		return "", fmt.Errorf("mistral: chat response has no choices")
	}
	raw := r.Choices[0].Message.Content
	content, ok := raw.(string)
	if !ok {
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

	body := req
	body.Stream = false

	var resp ChatCompletionResponse
	if err := c.postJSON(ctx, "/v1/chat/completions", body, &resp); err != nil {
		return ChatCompletionResponse{}, fmt.Errorf("mistral: chat completion: %w", err)
	}
	return resp, nil
}
