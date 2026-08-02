package mistralai

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
	"time"
)

func TestModerate(t *testing.T) {
	var request ModerationRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != BatchEndpointModerations || r.Method != http.MethodPost {
			t.Fatalf("request = %s %s", r.Method, r.URL.Path)
		}
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Fatal(err)
		}
		_ = json.NewEncoder(w).Encode(ModerationResponse{
			ID: "mod-1", Model: request.Model,
			Results: []ModerationResult{{
				Categories:     map[string]bool{"future_category": true},
				CategoryScores: map[string]float64{"future_category": 0.75},
			}},
		})
	}))
	defer srv.Close()
	client, err := NewClient("key", WithBaseURL(srv.URL))
	if err != nil {
		t.Fatal(err)
	}
	response, err := client.Moderate(context.Background(), ModerationRequest{
		Model: DefaultModerationModel, Input: []string{"one", "two"}, Metadata: map[string]any{"trace": "test"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(request.Input, []string{"one", "two"}) || request.Metadata["trace"] != "test" {
		t.Fatalf("request = %+v", request)
	}
	if !response.Results[0].Categories["future_category"] {
		t.Fatalf("response = %+v", response)
	}
}

func TestClassify(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != BatchEndpointClassifications || r.Method != http.MethodPost {
			t.Fatalf("request = %s %s", r.Method, r.URL.Path)
		}
		_ = json.NewEncoder(w).Encode(ClassificationResponse{
			ID: "cls-1", Model: "classifier",
			Results: []map[string]ClassificationTargetResult{{
				"topic": {Scores: map[string]float64{"invoice": 0.9}},
			}},
		})
	}))
	defer srv.Close()
	client, err := NewClient("key", WithBaseURL(srv.URL))
	if err != nil {
		t.Fatal(err)
	}
	response, err := client.Classify(context.Background(), ClassificationRequest{Model: "classifier", Input: []string{"invoice"}})
	if err != nil {
		t.Fatal(err)
	}
	if response.Results[0]["topic"].Scores["invoice"] != 0.9 {
		t.Fatalf("response = %+v", response)
	}
}

func TestClassifierValidationAndNoRetry(t *testing.T) {
	client, err := NewClient("key", WithBaseURL("http://127.0.0.1:1"))
	if err != nil {
		t.Fatal(err)
	}
	for _, req := range []ModerationRequest{
		{Input: []string{"text"}},
		{Model: DefaultModerationModel},
		{Model: DefaultModerationModel, Input: []string{" "}},
	} {
		if _, err := client.Moderate(context.Background(), req); !errors.Is(err, ErrInvalidRequest) {
			t.Fatalf("err = %v", err)
		}
	}

	var attempts int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		attempts++
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer srv.Close()
	client, err = NewClient("key", WithBaseURL(srv.URL), WithRetryPolicy(RetryPolicy{
		MaxAttempts: 4, InitialDelay: time.Millisecond, MaxDelay: time.Millisecond,
	}))
	if err != nil {
		t.Fatal(err)
	}
	_, err = client.Moderate(context.Background(), ModerationRequest{Model: DefaultModerationModel, Input: []string{"text"}})
	if err == nil || attempts != 1 {
		t.Fatalf("err = %v attempts = %d", err, attempts)
	}
}

func TestClassifierBatchEntries(t *testing.T) {
	var output strings.Builder
	err := EncodeBatchEntries(&output, []BatchEntry{
		ModerationEntry("m", ModerationRequest{Model: DefaultModerationModel, Input: []string{"text"}}),
		ClassificationEntry("c", ClassificationRequest{Model: "classifier", Input: []string{"text"}}),
	})
	if err != nil {
		t.Fatal(err)
	}
	if got := strings.Count(output.String(), "\n"); got != 2 {
		t.Fatalf("lines = %d", got)
	}
}
