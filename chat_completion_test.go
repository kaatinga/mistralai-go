package mistralai

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

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

func TestChatCompletion_streamRejected(t *testing.T) {
	cl, err := NewClient("k", WithBaseURL("http://127.0.0.1:1"))
	if err != nil {
		t.Fatal(err)
	}

	_, err = cl.ChatCompletion(context.Background(), ChatCompletionRequest{
		Model:    "mistral-small-latest",
		Messages: []ChatMessage{TextMessage(RoleUser, "hi")},
		Stream:   true,
	})
	if err == nil || !strings.Contains(err.Error(), "streaming is not supported") {
		t.Fatalf("err = %v", err)
	}
}
