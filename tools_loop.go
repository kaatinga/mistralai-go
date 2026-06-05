package mistralai

import (
	"context"
	"fmt"
)

// ToolHandler executes one tool call and returns the result content string
// (typically JSON) sent back to the model in a tool message.
type ToolHandler func(ctx context.Context, call ToolCall) (content string, err error)

// ChatCompletionWithTools runs chat completion in a tool-calling loop until the
// model returns a final text response or maxRounds is exceeded.
//
// Each round where the model returns tool_calls: the assistant message and one
// tool message per call are appended to req.Messages, then ChatCompletion runs
// again with the same tools configuration. Parallel tool calls in one assistant
// turn are all executed before the next completion.
//
// Do not set ResponseFormat on req when using tools; structured output and tool
// calling should not be combined in the same request. A maxRounds of 3 is
// usually sufficient for chained tool use.
func ChatCompletionWithTools(
	ctx context.Context,
	c Client,
	req ChatCompletionRequest,
	handler ToolHandler,
	maxRounds int,
) (ChatCompletionResponse, error) {
	if handler == nil {
		return ChatCompletionResponse{}, fmt.Errorf("mistral: tool handler is required")
	}
	if maxRounds <= 0 {
		return ChatCompletionResponse{}, fmt.Errorf("mistral: maxRounds must be positive")
	}

	req.Messages = append([]ChatMessage(nil), req.Messages...)

	resp, err := c.ChatCompletion(ctx, req)
	if err != nil {
		return ChatCompletionResponse{}, err
	}

	for range maxRounds {
		choice, err := resp.FirstChoice()
		if err != nil {
			return ChatCompletionResponse{}, err
		}
		if !choice.HasToolCalls() {
			return resp, nil
		}

		req.Messages = append(req.Messages, choice.Message)
		for _, call := range choice.Message.ToolCalls {
			result, err := handler(ctx, call)
			if err != nil {
				return ChatCompletionResponse{}, fmt.Errorf("mistral: tool %q: %w", call.Function.Name, err)
			}
			req.Messages = append(req.Messages, ToolMessage(call.ID, call.Function.Name, result))
		}

		resp, err = c.ChatCompletion(ctx, req)
		if err != nil {
			return ChatCompletionResponse{}, err
		}
	}

	choice, err := resp.FirstChoice()
	if err != nil {
		return ChatCompletionResponse{}, err
	}
	if choice.HasToolCalls() {
		return ChatCompletionResponse{}, fmt.Errorf("mistral: exceeded max tool rounds (%d)", maxRounds)
	}
	return resp, nil
}
