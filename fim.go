package mistralai

import (
	"context"
	"fmt"
	"strings"
)

// FIMCompletionRequest is the request body for /v1/fim/completions.
type FIMCompletionRequest struct {
	Model       string   `json:"model"`
	Prompt      string   `json:"prompt"`
	Suffix      string   `json:"suffix,omitempty"`
	Temperature *float64 `json:"temperature,omitempty"`
	TopP        *float64 `json:"top_p,omitempty"`
	MaxTokens   int      `json:"max_tokens,omitempty"`
	Stop        []string `json:"stop,omitempty"`
	RandomSeed  *int     `json:"random_seed,omitempty"`
	MinTokens   int      `json:"min_tokens,omitempty"`
}

// FIMChoice is a buffered FIM completion choice.
type FIMChoice struct {
	Index        int    `json:"index"`
	Text         string `json:"text"`
	FinishReason string `json:"finish_reason"`
}

// FIMCompletionResponse is the buffered FIM response.
type FIMCompletionResponse struct {
	ID      string      `json:"id"`
	Model   string      `json:"model"`
	Choices []FIMChoice `json:"choices"`
	Usage   UsageInfo   `json:"usage"`
}

func (r FIMCompletionRequest) validate() error {
	if strings.TrimSpace(r.Model) == "" {
		return fmt.Errorf("%w: model is required", ErrInvalidRequest)
	}
	if strings.TrimSpace(r.Prompt) == "" {
		return fmt.Errorf("%w: prompt is required", ErrInvalidRequest)
	}
	return nil
}

// FIMCompletion runs a buffered fill-in-the-middle completion.
func (c *Client) FIMCompletion(ctx context.Context, req FIMCompletionRequest) (FIMCompletionResponse, error) {
	if err := req.validate(); err != nil {
		return FIMCompletionResponse{}, err
	}
	var response FIMCompletionResponse
	if err := c.postJSON(ctx, pathFIMCompletions, req, &response); err != nil {
		return FIMCompletionResponse{}, fmt.Errorf("mistral: FIM: %w", err)
	}
	return response, nil
}

// FIMCompletionStream starts a streaming fill-in-the-middle completion.
func (c *Client) FIMCompletionStream(ctx context.Context, req FIMCompletionRequest) (*ChatCompletionStream, error) {
	if err := req.validate(); err != nil {
		return nil, err
	}
	payload := struct {
		FIMCompletionRequest
		Stream bool `json:"stream"`
	}{FIMCompletionRequest: req, Stream: true}
	return c.newJSONStream(ctx, pathFIMCompletions, payload)
}
