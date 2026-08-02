package mistralai

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
)

func TestChatCompletionWithTools_singleCall(t *testing.T) {
	var calls int
	var mu sync.Mutex
	var lastBody ChatCompletionRequest

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		calls++
		n := calls
		mu.Unlock()

		if err := json.NewDecoder(r.Body).Decode(&lastBody); err != nil {
			t.Fatal(err)
		}

		var resp ChatCompletionResponse
		switch n {
		case 1:
			resp = ChatCompletionResponse{
				Choices: []ChatCompletionResponseChoice{{
					FinishReason: FinishReasonToolCalls,
					Message: AssistantToolCallsMessage([]ToolCall{{
						ID:   "call_1",
						Type: ToolTypeFunction,
						Function: FunctionCall{
							Name:      "count_apartments",
							Arguments: `{"building_id":12}`,
						},
					}}),
				}},
			}
		case 2:
			if len(lastBody.Messages) < 3 {
				t.Fatalf("messages = %+v", lastBody.Messages)
			}
			toolMsg := lastBody.Messages[len(lastBody.Messages)-1]
			if toolMsg.Role != RoleTool || toolMsg.ToolCallID != "call_1" {
				t.Fatalf("tool message = %+v", toolMsg)
			}
			resp = ChatCompletionResponse{
				Choices: []ChatCompletionResponseChoice{{
					FinishReason: FinishReasonStop,
					Message:      TextMessage(RoleAssistant, "There are 8 apartments."),
				}},
			}
		default:
			t.Fatalf("unexpected call %d", n)
		}
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer srv.Close()

	cl, err := NewClient("test-key", WithBaseURL(srv.URL))
	if err != nil {
		t.Fatal(err)
	}

	var handlerCalls []ToolCall
	resp, err := ChatCompletionWithTools(context.Background(), cl, ChatCompletionRequest{
		Model: DefaultChatModel,
		Messages: []ChatMessage{
			TextMessage(RoleUser, "How many apartments?"),
		},
		Tools: []Tool{
			FunctionTool("count_apartments", "Count apartments", map[string]any{"type": "object"}),
		},
		ToolChoice: ToolChoiceMode(ToolChoiceAuto),
	}, func(_ context.Context, call ToolCall) (string, error) {
		handlerCalls = append(handlerCalls, call)
		return `{"count":8}`, nil
	}, 3)
	if err != nil {
		t.Fatal(err)
	}
	if calls != 2 {
		t.Fatalf("http calls = %d, want 2", calls)
	}
	if len(handlerCalls) != 1 || handlerCalls[0].Function.Name != "count_apartments" {
		t.Fatalf("handler calls = %+v", handlerCalls)
	}
	got, err := resp.FirstText()
	if err != nil || got != "There are 8 apartments." {
		t.Fatalf("answer = %q err = %v", got, err)
	}
}

func TestChatCompletionWithTools_parallelCalls(t *testing.T) {
	var calls int
	var mu sync.Mutex

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		calls++
		n := calls
		mu.Unlock()

		var resp ChatCompletionResponse
		if n == 1 {
			resp = ChatCompletionResponse{
				Choices: []ChatCompletionResponseChoice{{
					FinishReason: FinishReasonToolCalls,
					Message: AssistantToolCallsMessage([]ToolCall{
						{
							ID:   "call_a",
							Type: ToolTypeFunction,
							Function: FunctionCall{
								Name:      "count_apartments",
								Arguments: `{}`,
							},
						},
						{
							ID:   "call_b",
							Type: ToolTypeFunction,
							Function: FunctionCall{
								Name:      "count_thermostats",
								Arguments: `{}`,
							},
						},
					}),
				}},
			}
		} else {
			resp = ChatCompletionResponse{
				Choices: []ChatCompletionResponseChoice{{
					FinishReason: FinishReasonStop,
					Message:      TextMessage(RoleAssistant, "done"),
				}},
			}
		}
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer srv.Close()

	cl, err := NewClient("test-key", WithBaseURL(srv.URL))
	if err != nil {
		t.Fatal(err)
	}

	var handled []string
	_, err = ChatCompletionWithTools(context.Background(), cl, ChatCompletionRequest{
		Model:    DefaultChatModel,
		Messages: []ChatMessage{TextMessage(RoleUser, "counts?")},
		Tools: []Tool{
			FunctionTool("count_apartments", "", map[string]any{"type": "object"}),
			FunctionTool("count_thermostats", "", map[string]any{"type": "object"}),
		},
	}, func(_ context.Context, call ToolCall) (string, error) {
		handled = append(handled, call.Function.Name)
		return "{}", nil
	}, 3)
	if err != nil {
		t.Fatal(err)
	}
	if len(handled) != 2 {
		t.Fatalf("handled = %v", handled)
	}
	if calls != 2 {
		t.Fatalf("http calls = %d", calls)
	}
}

func TestChatCompletionWithTools_forcesNamedChoiceOnce(t *testing.T) {
	var calls int
	var mu sync.Mutex
	var toolChoices []ToolChoice

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body ChatCompletionRequest
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatal(err)
		}
		mu.Lock()
		calls++
		n := calls
		toolChoices = append(toolChoices, body.ToolChoice)
		mu.Unlock()

		var resp ChatCompletionResponse
		if n == 1 {
			resp = ChatCompletionResponse{
				Choices: []ChatCompletionResponseChoice{{
					FinishReason: FinishReasonToolCalls,
					Message: AssistantToolCallsMessage([]ToolCall{{
						ID:       "call_1",
						Type:     ToolTypeFunction,
						Function: FunctionCall{Name: "fn", Arguments: `{}`},
					}}),
				}},
			}
		} else {
			resp = ChatCompletionResponse{
				Choices: []ChatCompletionResponseChoice{{
					FinishReason: FinishReasonStop,
					Message:      TextMessage(RoleAssistant, "done"),
				}},
			}
		}
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer srv.Close()

	cl, err := NewClient("test-key", WithBaseURL(srv.URL))
	if err != nil {
		t.Fatal(err)
	}

	_, err = ChatCompletionWithTools(context.Background(), cl, ChatCompletionRequest{
		Model:      DefaultChatModel,
		Messages:   []ChatMessage{TextMessage(RoleUser, "go")},
		Tools:      []Tool{FunctionTool("fn", "", nil)},
		ToolChoice: ToolChoiceFunction("fn"),
	}, func(context.Context, ToolCall) (string, error) {
		return "{}", nil
	}, 3)
	if err != nil {
		t.Fatal(err)
	}
	if len(toolChoices) != 2 {
		t.Fatalf("calls = %d", len(toolChoices))
	}
	if toolChoices[0] != ToolChoiceFunction("fn") {
		t.Fatalf("first tool_choice = %#v", toolChoices[0])
	}
	if toolChoices[1] != ToolChoiceMode(ToolChoiceAuto) {
		t.Fatalf("second tool_choice = %#v", toolChoices[1])
	}
}

func TestChatCompletionWithTools_doesNotMutateCallerMessages(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(ChatCompletionResponse{
			Choices: []ChatCompletionResponseChoice{{
				FinishReason: FinishReasonToolCalls,
				Message: AssistantToolCallsMessage([]ToolCall{{
					ID:       "call_1",
					Type:     ToolTypeFunction,
					Function: FunctionCall{Name: "fn", Arguments: `{}`},
				}}),
			}},
		})
	}))
	defer srv.Close()

	cl, err := NewClient("test-key", WithBaseURL(srv.URL))
	if err != nil {
		t.Fatal(err)
	}

	// Give the slice spare capacity so a naive append would clobber it.
	msgs := make([]ChatMessage, 1, 8)
	msgs[0] = TextMessage(RoleUser, "go")

	_, _ = ChatCompletionWithTools(context.Background(), cl, ChatCompletionRequest{
		Model:    DefaultChatModel,
		Messages: msgs,
		Tools:    []Tool{FunctionTool("fn", "", nil)},
	}, func(context.Context, ToolCall) (string, error) {
		return "{}", nil
	}, 1)

	if len(msgs) != 1 || msgs[0].Role != RoleUser {
		t.Fatalf("caller messages were mutated: %+v", msgs)
	}
}

func TestChatCompletionWithTools_maxRoundsExceeded(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(ChatCompletionResponse{
			Choices: []ChatCompletionResponseChoice{{
				FinishReason: FinishReasonToolCalls,
				Message: AssistantToolCallsMessage([]ToolCall{{
					ID:   "call_1",
					Type: ToolTypeFunction,
					Function: FunctionCall{
						Name:      "fn",
						Arguments: `{}`,
					},
				}}),
			}},
		})
	}))
	defer srv.Close()

	cl, err := NewClient("test-key", WithBaseURL(srv.URL))
	if err != nil {
		t.Fatal(err)
	}

	_, err = ChatCompletionWithTools(context.Background(), cl, ChatCompletionRequest{
		Model:    DefaultChatModel,
		Messages: []ChatMessage{TextMessage(RoleUser, "go")},
		Tools:    []Tool{FunctionTool("fn", "", map[string]any{"type": "object"})},
	}, func(_ context.Context, _ ToolCall) (string, error) {
		return "{}", nil
	}, 1)
	if err == nil || !strings.Contains(err.Error(), "exceeded max tool rounds") {
		t.Fatalf("err = %v", err)
	}
}

func TestChatCompletionWithTools_validation(t *testing.T) {
	cl, err := NewClient("k", WithBaseURL("http://127.0.0.1:1"))
	if err != nil {
		t.Fatal(err)
	}

	req := ChatCompletionRequest{
		Model:    DefaultChatModel,
		Messages: []ChatMessage{TextMessage(RoleUser, "hi")},
	}
	if _, err := ChatCompletionWithTools(context.Background(), cl, req, nil, 3); err == nil {
		t.Fatal("expected handler error")
	}
	if _, err := ChatCompletionWithTools(context.Background(), cl, req, func(context.Context, ToolCall) (string, error) {
		return "", nil
	}, 0); err == nil {
		t.Fatal("expected maxRounds error")
	}
}

// The response produced by the last permitted round must be returned, not
// discarded: with maxRounds=N the loop may send N+1 completions and the final
// one carries the answer.
func TestChatCompletionWithTools_answerOnLastRound(t *testing.T) {
	const maxRounds = 2

	var mu sync.Mutex
	var calls int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		calls++
		n := calls
		mu.Unlock()

		var resp ChatCompletionResponse
		if n <= maxRounds {
			resp = ChatCompletionResponse{
				Choices: []ChatCompletionResponseChoice{{
					FinishReason: FinishReasonToolCalls,
					Message: AssistantToolCallsMessage([]ToolCall{{
						ID:       fmt.Sprintf("call_%d", n),
						Type:     ToolTypeFunction,
						Function: FunctionCall{Name: "fn", Arguments: `{}`},
					}}),
				}},
			}
		} else {
			resp = ChatCompletionResponse{
				Choices: []ChatCompletionResponseChoice{{
					FinishReason: FinishReasonStop,
					Message:      TextMessage(RoleAssistant, "done"),
				}},
			}
		}
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer srv.Close()

	cl, err := NewClient("test-key", WithBaseURL(srv.URL))
	if err != nil {
		t.Fatal(err)
	}

	resp, err := ChatCompletionWithTools(context.Background(), cl, ChatCompletionRequest{
		Model:    DefaultChatModel,
		Messages: []ChatMessage{TextMessage(RoleUser, "go")},
		Tools:    []Tool{FunctionTool("fn", "", nil)},
	}, func(context.Context, ToolCall) (string, error) {
		return `{"ok":true}`, nil
	}, maxRounds)
	if err != nil {
		t.Fatalf("err = %v", err)
	}
	if got, err := resp.FirstText(); err != nil || got != "done" {
		t.Fatalf("answer = %q err = %v", got, err)
	}
	if calls != maxRounds+1 {
		t.Fatalf("http calls = %d, want %d", calls, maxRounds+1)
	}
}
