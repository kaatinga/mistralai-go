package mistralai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
)

const (
	ChatModelMistralSmallLatest  = "mistral-small-latest"
	ChatModelMistralMediumLatest = "mistral-medium-latest"
	ChatModelPixtral12BLatest    = "pixtral-12b-latest"
	ChatModelPixtralLargeLatest  = "pixtral-large-latest"

	// DefaultChatModel is used when ChatRequest.Model is empty.
	DefaultChatModel   = ChatModelMistralSmallLatest
	markdownSystemHint = "Format your entire reply as Markdown."
	jsonSystemHint     = "Reply with a single valid JSON value only. Do not wrap it in markdown fences or add commentary."
)

// OutputFormat selects how the model should structure its reply.
type OutputFormat string

const (
	OutputText     OutputFormat = "text"
	OutputMarkdown OutputFormat = "markdown"
	OutputJSON     OutputFormat = "json"
)

// ChatRequest is a chat completion with arbitrary user text.
type ChatRequest struct {
	// Input is the user message (required).
	Input string
	// System is an optional system instruction prepended before Input.
	System string
	// Model overrides DefaultChatModel when non-empty.
	Model string
	// Format requests text, markdown, or json output. Defaults to OutputText.
	Format OutputFormat
	// ResponseFormat optionally refines JSON output (e.g. json_schema). When Format is
	// OutputJSON and this is nil, json_object mode is used.
	ResponseFormat *ResponseFormat
}

// ChatResponse is the assistant message from a chat completion.
type ChatResponse struct {
	Content string
	Format  OutputFormat
	Model   string
}

// JSON unmarshals Content into dest when Format is OutputJSON.
func (r ChatResponse) JSON(dest any) error {
	if r.Format != OutputJSON {
		return fmt.Errorf("mistral: response format is %q, not json", r.Format)
	}
	if err := json.Unmarshal([]byte(r.Content), dest); err != nil {
		return fmt.Errorf("mistral: parse json content: %w", err)
	}
	return nil
}

func (c *client) Chat(ctx context.Context, req ChatRequest) (ChatResponse, error) {
	if err := req.validate(); err != nil {
		return ChatResponse{}, err
	}
	result, err := c.processChat(ctx, req)
	if err != nil {
		return ChatResponse{}, err
	}
	return *result, nil
}

func (r ChatRequest) validate() error {
	if strings.TrimSpace(r.Input) == "" {
		return errors.New("mistral: input is required")
	}
	switch r.Format {
	case "", OutputText, OutputMarkdown, OutputJSON:
		return nil
	default:
		return fmt.Errorf("mistral: unsupported output format %q", r.Format)
	}
}

func (c *client) processChat(ctx context.Context, req ChatRequest) (*ChatResponse, error) {
	format := req.Format
	if format == "" {
		format = OutputText
	}

	model := req.Model
	if model == "" {
		model = DefaultChatModel
	}

	system, responseFormat, err := chatPromptConfig(req, format)
	if err != nil {
		return nil, err
	}

	messages := make([]ChatMessage, 0, 2)
	if system != "" {
		messages = append(messages, TextMessage(RoleSystem, system))
	}
	messages = append(messages, TextMessage(RoleUser, req.Input))

	resp, err := c.ChatCompletion(ctx, ChatCompletionRequest{
		Model:          model,
		Messages:       messages,
		ResponseFormat: responseFormat,
	})
	if err != nil {
		return nil, err
	}

	content, err := resp.FirstChoiceContent()
	if err != nil {
		return nil, err
	}

	outModel := model
	if resp.Model != "" {
		outModel = resp.Model
	}

	return &ChatResponse{
		Content: content,
		Format:  format,
		Model:   outModel,
	}, nil
}

func chatPromptConfig(req ChatRequest, format OutputFormat) (system string, responseFormat *ResponseFormat, err error) {
	system = strings.TrimSpace(req.System)

	switch format {
	case OutputText:
		return system, nil, nil
	case OutputMarkdown:
		system = joinSystem(system, markdownSystemHint)
		return system, nil, nil
	case OutputJSON:
		system = joinSystem(system, jsonSystemHint)
		if req.ResponseFormat != nil {
			switch req.ResponseFormat.Type {
			case ResponseFormatJSONSchema, ResponseFormatJSONObject:
				return system, req.ResponseFormat, nil
			case ResponseFormatText, "":
				return "", nil, errors.New("mistral: ResponseFormat type must be json_object or json_schema for json output")
			default:
				return "", nil, fmt.Errorf("mistral: unsupported ResponseFormat type %q for json output", req.ResponseFormat.Type)
			}
		}
		return system, &ResponseFormat{Type: ResponseFormatJSONObject}, nil
	default:
		return "", nil, fmt.Errorf("mistral: unsupported output format %q", format)
	}
}

func joinSystem(parts ...string) string {
	var b strings.Builder
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		if b.Len() > 0 {
			b.WriteString("\n\n")
		}
		b.WriteString(p)
	}
	return b.String()
}
