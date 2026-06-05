package mistralai

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestToolJSONRoundTrip(t *testing.T) {
	tool := FunctionTool("count_apartments", "Count apartments", map[string]any{
		"type": "object",
		"properties": map[string]any{
			"building_id": map[string]any{"type": "integer"},
		},
	})
	data, err := json.Marshal(tool)
	if err != nil {
		t.Fatal(err)
	}
	var decoded Tool
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatal(err)
	}
	if decoded.Type != ToolTypeFunction || decoded.Function.Name != "count_apartments" {
		t.Fatalf("tool = %+v", decoded)
	}

	call := ToolCall{
		ID:   "call_abc",
		Type: ToolTypeFunction,
		Function: FunctionCall{
			Name:      "count_apartments",
			Arguments: `{"building_id":12}`,
		},
	}
	data, err = json.Marshal(call)
	if err != nil {
		t.Fatal(err)
	}
	var decodedCall ToolCall
	if err := json.Unmarshal(data, &decodedCall); err != nil {
		t.Fatal(err)
	}
	if decodedCall.Function.Arguments != `{"building_id":12}` {
		t.Fatalf("call = %+v", decodedCall)
	}

	msg := ToolMessage("call_abc", "count_apartments", `{"count":8}`)
	data, err = json.Marshal(msg)
	if err != nil {
		t.Fatal(err)
	}
	var decodedMsg ChatMessage
	if err := json.Unmarshal(data, &decodedMsg); err != nil {
		t.Fatal(err)
	}
	if decodedMsg.Role != RoleTool || decodedMsg.ToolCallID != "call_abc" || decodedMsg.Name != "count_apartments" {
		t.Fatalf("msg = %+v", decodedMsg)
	}

	named := ToolChoiceNamed("count_apartments")
	data, err = json.Marshal(named)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(data), `"name":"count_apartments"`) {
		t.Fatalf("named tool choice = %s", data)
	}
}

func TestHasToolCallsAndFirstChoice(t *testing.T) {
	withTools := ChatCompletionResponseChoice{
		FinishReason: FinishReasonToolCalls,
		Message: AssistantToolCallsMessage([]ToolCall{{
			ID:   "c1",
			Type: ToolTypeFunction,
			Function: FunctionCall{
				Name:      "fn",
				Arguments: `{}`,
			},
		}}),
	}
	if !withTools.HasToolCalls() {
		t.Fatal("expected tool calls")
	}

	text := ChatCompletionResponseChoice{
		FinishReason: FinishReasonStop,
		Message:      TextMessage(RoleAssistant, "hello"),
	}
	if text.HasToolCalls() {
		t.Fatal("expected no tool calls")
	}

	resp := ChatCompletionResponse{
		Choices: []ChatCompletionResponseChoice{withTools},
	}
	choice, err := resp.FirstChoice()
	if err != nil {
		t.Fatal(err)
	}
	if !choice.HasToolCalls() {
		t.Fatal("expected tool calls from FirstChoice")
	}

	if _, err := resp.FirstChoiceContent(); err == nil || !strings.Contains(err.Error(), "tool_calls") {
		t.Fatalf("FirstChoiceContent err = %v", err)
	}

	textResp := ChatCompletionResponse{
		Choices: []ChatCompletionResponseChoice{text},
	}
	got, err := textResp.FirstChoiceContent()
	if err != nil || got != "hello" {
		t.Fatalf("content = %q err = %v", got, err)
	}

	empty := ChatCompletionResponse{}
	if _, err := empty.FirstChoice(); err == nil {
		t.Fatal("expected error for empty choices")
	}
}

func TestChatCompletionRequest_toolsFields(t *testing.T) {
	parallel := false
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" || r.Method != http.MethodPost {
			http.NotFound(w, r)
			return
		}
		var body ChatCompletionRequest
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatal(err)
		}
		if len(body.Tools) != 1 || body.Tools[0].Function.Name != "get_weather" {
			t.Fatalf("tools = %+v", body.Tools)
		}
		if body.ToolChoice != ToolChoiceAuto {
			t.Fatalf("tool_choice = %v", body.ToolChoice)
		}
		if body.ParallelToolCalls == nil || *body.ParallelToolCalls {
			t.Fatalf("parallel_tool_calls = %v", body.ParallelToolCalls)
		}
		_ = json.NewEncoder(w).Encode(ChatCompletionResponse{
			Choices: []ChatCompletionResponseChoice{{
				FinishReason: FinishReasonStop,
				Message:      TextMessage(RoleAssistant, "ok"),
			}},
		})
	}))
	defer srv.Close()

	cl, err := NewClient("test-key", WithBaseURL(srv.URL))
	if err != nil {
		t.Fatal(err)
	}
	defer cl.Close()

	_, err = cl.ChatCompletion(context.Background(), ChatCompletionRequest{
		Model: DefaultChatModel,
		Messages: []ChatMessage{
			TextMessage(RoleUser, "weather?"),
		},
		Tools: []Tool{
			FunctionTool("get_weather", "Get weather", map[string]any{
				"type":       "object",
				"properties": map[string]any{},
			}),
		},
		ToolChoice:        ToolChoiceAuto,
		ParallelToolCalls: &parallel,
	})
	if err != nil {
		t.Fatal(err)
	}
}
