package mistralai

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"
)

func TestChatMessageContentRoundTrip(t *testing.T) {
	fixtures := []string{
		`{"role":"user","content":"hello"}`,
		`{"role":"user","content":[{"type":"text","text":"look"},{"type":"image_url","image_url":"https://example.test/image.png"}]}`,
		`{"role":"assistant","content":null,"tool_calls":[{"id":"call-1","type":"function","function":{"name":"lookup","arguments":"{}"}}]}`,
	}
	for _, fixture := range fixtures {
		var message ChatMessage
		if err := json.Unmarshal([]byte(fixture), &message); err != nil {
			t.Fatalf("unmarshal %s: %v", fixture, err)
		}
		encoded, err := json.Marshal(message)
		if err != nil {
			t.Fatal(err)
		}
		var got, want any
		_ = json.Unmarshal(encoded, &got)
		_ = json.Unmarshal([]byte(fixture), &want)
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("round trip:\n got %s\nwant %s", encoded, fixture)
		}
	}
}

func TestChatCompletionRejectsInvalidContentUnion(t *testing.T) {
	client, err := NewClient("key", WithBaseURL("http://127.0.0.1:1"))
	if err != nil {
		t.Fatal(err)
	}
	_, err = client.ChatCompletion(context.Background(), ChatCompletionRequest{
		Model: "model", Messages: []ChatMessage{{Role: RoleUser, Content: 42}},
	})
	if !errors.Is(err, ErrInvalidRequest) {
		t.Fatalf("err = %v", err)
	}
}

func TestFirstTextMultipart(t *testing.T) {
	response := ChatCompletionResponse{Choices: []ChatCompletionResponseChoice{{
		Message: MultipartMessage(RoleAssistant, TextPart("hello "), ImageURLPart("https://example.test/image"), TextPart("world")),
	}}}
	text, err := response.FirstText()
	if err != nil || text != "hello world" {
		t.Fatalf("text = %q err = %v", text, err)
	}
}

func TestAssistantToolCallsAllowEmptyText(t *testing.T) {
	message := AssistantToolCallsMessage([]ToolCall{{
		ID: "call-1", Type: ToolTypeFunction, Function: FunctionCall{Name: "ping", Arguments: `{}`},
	}})
	message.Content = TextContent("")
	request := ChatCompletionRequest{Model: "model", Messages: []ChatMessage{
		TextMessage(RoleUser, "ping"), message,
	}}
	if err := request.validate(); err != nil {
		t.Fatalf("validate: %v", err)
	}
}

func TestChatCompletion_requestFields(t *testing.T) {
	var sent map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &sent)
		_ = json.NewEncoder(w).Encode(ChatCompletionResponse{
			Choices: []ChatCompletionResponseChoice{{Message: TextMessage(RoleAssistant, "ok")}},
		})
	}))
	defer srv.Close()

	cl, err := NewClient("k", WithBaseURL(srv.URL))
	if err != nil {
		t.Fatal(err)
	}

	_, err = cl.ChatCompletion(context.Background(), ChatCompletionRequest{
		Model:            "mistral-small-latest",
		Messages:         []ChatMessage{TextMessage(RoleUser, "hi")},
		Stop:             []string{"END"},
		RandomSeed:       new(42),
		Metadata:         map[string]any{"trace": "t1"},
		PresencePenalty:  new(0.5),
		FrequencyPenalty: new(-0.5),
		N:                new(2),
		Prediction:       PredictionContent("expected output"),
		PromptMode:       PromptModeReasoning,
		ReasoningEffort:  ReasoningEffortNone,
		PromptCacheKey:   "cache-1",
		SafePrompt:       true,
	})
	if err != nil {
		t.Fatal(err)
	}

	want := map[string]any{
		"stop":              []any{"END"},
		"random_seed":       float64(42),
		"metadata":          map[string]any{"trace": "t1"},
		"presence_penalty":  0.5,
		"frequency_penalty": -0.5,
		"n":                 float64(2),
		"prediction":        map[string]any{"type": "content", "content": "expected output"},
		"prompt_mode":       "reasoning",
		"reasoning_effort":  "none",
		"prompt_cache_key":  "cache-1",
		"safe_prompt":       true,
	}
	for key, w := range want {
		got, ok := sent[key]
		if !ok {
			t.Errorf("request body missing %q", key)
			continue
		}
		gj, _ := json.Marshal(got)
		wj, _ := json.Marshal(w)
		if string(gj) != string(wj) {
			t.Errorf("%s = %s want %s", key, gj, wj)
		}
	}
	if _, ok := sent["stream"]; ok {
		t.Error("stream must be omitted when unset")
	}
}

func TestChatCompletion_zeroValueFieldsOmitted(t *testing.T) {
	var sent map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &sent)
		_ = json.NewEncoder(w).Encode(ChatCompletionResponse{
			Choices: []ChatCompletionResponseChoice{{Message: TextMessage(RoleAssistant, "ok")}},
		})
	}))
	defer srv.Close()

	cl, err := NewClient("k", WithBaseURL(srv.URL))
	if err != nil {
		t.Fatal(err)
	}

	_, err = cl.ChatCompletion(context.Background(), ChatCompletionRequest{
		Model:    "mistral-small-latest",
		Messages: []ChatMessage{TextMessage(RoleUser, "hi")},
	})
	if err != nil {
		t.Fatal(err)
	}

	for _, key := range []string{
		"stream", "stop", "random_seed", "metadata", "presence_penalty",
		"frequency_penalty", "n", "prediction", "prompt_mode",
		"reasoning_effort", "prompt_cache_key", "safe_prompt",
	} {
		if _, ok := sent[key]; ok {
			t.Errorf("unset field %q must be omitted, got %v", key, sent[key])
		}
	}
}
