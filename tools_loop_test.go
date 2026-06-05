package mistralai

import (
	"context"
	"encoding/json"
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
	defer cl.Close()

	var handlerCalls []ToolCall
	resp, err := ChatCompletionWithTools(context.Background(), cl, ChatCompletionRequest{
		Model: DefaultChatModel,
		Messages: []ChatMessage{
			TextMessage(RoleUser, "How many apartments?"),
		},
		Tools: []Tool{
			FunctionTool("count_apartments", "Count apartments", map[string]any{"type": "object"}),
		},
		ToolChoice: ToolChoiceAuto,
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
	got, err := resp.FirstChoiceContent()
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
	defer cl.Close()

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

func TestChatCompletionWithTools_preservesToolChoice(t *testing.T) {
	var calls int
	var mu sync.Mutex
	var toolChoices []any

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
	defer cl.Close()

	_, err = ChatCompletionWithTools(context.Background(), cl, ChatCompletionRequest{
		Model:      DefaultChatModel,
		Messages:   []ChatMessage{TextMessage(RoleUser, "go")},
		Tools:      []Tool{FunctionTool("fn", "", nil)},
		ToolChoice: ToolChoiceNamed("fn"),
	}, func(context.Context, ToolCall) (string, error) {
		return "{}", nil
	}, 3)
	if err != nil {
		t.Fatal(err)
	}
	if len(toolChoices) != 2 {
		t.Fatalf("calls = %d", len(toolChoices))
	}
	// req is sent verbatim every round: the named (object) choice must persist,
	// not be silently rewritten by the loop.
	for i, tc := range toolChoices {
		if _, ok := tc.(map[string]any); !ok {
			t.Fatalf("tool_choice on call %d should be the named (object) choice, got %#v", i+1, tc)
		}
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
	defer cl.Close()

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
	defer cl.Close()

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
	defer cl.Close()

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
