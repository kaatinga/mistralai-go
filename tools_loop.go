package mistralai

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
)

// ToolHandler executes one tool call and returns the result content string
// (typically JSON) sent back to the model in a tool message.
type ToolHandler func(ctx context.Context, call ToolCall) (content string, err error)

// ToolLoopOptions controls ChatCompletionWithToolsOptions.
type ToolLoopOptions struct {
	// MaxRounds is how many rounds of tool calls may be executed. The loop
	// sends at most MaxRounds+1 completions: the initial one plus a follow-up
	// after each executed round, and the final follow-up is always inspected
	// for an answer before the budget is reported as exceeded.
	MaxRounds                int
	ForceToolChoiceEachRound bool
}

// ChatCompletionWithToolsOptions runs a validated tool loop. A named tool
// choice is applied only to the first request unless explicitly repeated.
func ChatCompletionWithToolsOptions(
	ctx context.Context,
	c ChatCompleter,
	req ChatCompletionRequest,
	handler ToolHandler,
	options ToolLoopOptions,
) (ChatCompletionResponse, error) {
	if handler == nil {
		return ChatCompletionResponse{}, fmt.Errorf("%w: tool handler is required", ErrInvalidRequest)
	}
	if options.MaxRounds <= 0 {
		return ChatCompletionResponse{}, fmt.Errorf("%w: maxRounds must be positive", ErrInvalidRequest)
	}
	if len(req.Tools) == 0 {
		return ChatCompletionResponse{}, fmt.Errorf("%w: at least one tool is required", ErrInvalidRequest)
	}
	known := make(map[string]struct{}, len(req.Tools))
	for _, tool := range req.Tools {
		if strings.TrimSpace(tool.Function.Name) == "" {
			return ChatCompletionResponse{}, fmt.Errorf("%w: tool name is required", ErrInvalidRequest)
		}
		known[tool.Function.Name] = struct{}{}
	}
	req.Messages = append([]ChatMessage(nil), req.Messages...)
	initialChoice := req.ToolChoice
	resp, err := c.ChatCompletion(ctx, req)
	if err != nil {
		return ChatCompletionResponse{}, err
	}
	for round := 0; ; round++ {
		choice, err := resp.FirstChoice()
		if err != nil {
			return ChatCompletionResponse{}, err
		}
		if !choice.HasToolCalls() {
			return resp, nil
		}
		if round == options.MaxRounds {
			return ChatCompletionResponse{}, fmt.Errorf("mistral: exceeded max tool rounds (%d)", options.MaxRounds)
		}
		seen := make(map[string]struct{}, len(choice.Message.ToolCalls))
		for _, call := range choice.Message.ToolCalls {
			if call.ID == "" {
				return ChatCompletionResponse{}, fmt.Errorf("%w: tool call id is required", ErrInvalidRequest)
			}
			if _, duplicate := seen[call.ID]; duplicate {
				return ChatCompletionResponse{}, fmt.Errorf("%w: duplicate tool call id %q", ErrInvalidRequest, call.ID)
			}
			seen[call.ID] = struct{}{}
			if _, ok := known[call.Function.Name]; !ok {
				return ChatCompletionResponse{}, fmt.Errorf("%w: unknown tool %q", ErrInvalidRequest, call.Function.Name)
			}
			if !json.Valid([]byte(call.Function.Arguments)) {
				return ChatCompletionResponse{}, fmt.Errorf("%w: malformed arguments for tool %q", ErrInvalidRequest, call.Function.Name)
			}
		}
		req.Messages = append(req.Messages, choice.Message)
		for _, call := range choice.Message.ToolCalls {
			result, err := handler(ctx, call)
			if err != nil {
				return ChatCompletionResponse{}, fmt.Errorf("mistral: tool %q: %w", call.Function.Name, err)
			}
			if strings.TrimSpace(result) == "" {
				return ChatCompletionResponse{}, fmt.Errorf("%w: empty result for tool %q", ErrInvalidRequest, call.Function.Name)
			}
			req.Messages = append(req.Messages, ToolMessage(call.ID, call.Function.Name, result))
		}
		if !options.ForceToolChoiceEachRound && round == 0 && !initialChoice.IsZero() {
			req.ToolChoice = ToolChoiceMode(ToolChoiceAuto)
		}
		resp, err = c.ChatCompletion(ctx, req)
		if err != nil {
			return ChatCompletionResponse{}, err
		}
	}
}

// ChatCompletionWithTools runs chat completion in a tool-calling loop until the
// model returns a final text response or maxRounds is exceeded.
//
// Each round where the model returns tool_calls: the assistant message and one
// tool message per call are appended to req.Messages, then ChatCompletion runs
// again with the same tools configuration. Parallel tool calls in one assistant
// turn are all executed before the next completion.
//
// maxRounds bounds how many rounds of tool calls are executed, so at most
// maxRounds+1 completions are sent; the answer returned after the last executed
// round is never discarded. A maxRounds of 3 is usually sufficient for chained
// tool use.
//
// Do not set ResponseFormat on req when using tools; structured output and tool
// calling should not be combined in the same request.
func ChatCompletionWithTools(
	ctx context.Context,
	c ChatCompleter,
	req ChatCompletionRequest,
	handler ToolHandler,
	maxRounds int,
) (ChatCompletionResponse, error) {
	return ChatCompletionWithToolsOptions(ctx, c, req, handler, ToolLoopOptions{MaxRounds: maxRounds})
}
