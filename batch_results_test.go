package mistralai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestParseBatchResults_chat(t *testing.T) {
	jsonl := strings.Join([]string{
		`{"id":"r0","custom_id":"0","response":{"status_code":200,"body":{"id":"cmpl-0","model":"mistral-small","choices":[{"index":0,"message":{"role":"assistant","content":"Hello there"}}]}},"error":null}`,
		``, // blank line should be skipped
		`{"id":"r1","custom_id":"1","response":{"status_code":429,"body":null},"error":{"message":"rate limited"}}`,
	}, "\n")

	results, err := ParseBatchResults[ChatCompletionResponse]([]byte(jsonl))
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 2 {
		t.Fatalf("got %d results, want 2", len(results))
	}

	// order preserved
	if results[0].CustomID != "0" || results[1].CustomID != "1" {
		t.Errorf("custom ids = %q, %q", results[0].CustomID, results[1].CustomID)
	}

	// success line: body decoded
	if results[0].StatusCode != 200 {
		t.Errorf("status = %d", results[0].StatusCode)
	}
	content, err := results[0].Body.FirstChoiceContent()
	if err != nil || content != "Hello there" {
		t.Errorf("content = %q err = %v", content, err)
	}
	if results[0].Error != nil {
		t.Errorf("unexpected error on success line: %v", results[0].Error)
	}

	// failure line: zero body, error + status populated
	if results[1].StatusCode != 429 {
		t.Errorf("status = %d", results[1].StatusCode)
	}
	if len(results[1].Body.Choices) != 0 {
		t.Errorf("expected zero body on failure, got %+v", results[1].Body)
	}
	if results[1].Error == nil {
		t.Error("expected error on failure line")
	}
}

func TestParseBatchResults_ocrAndLargeLine(t *testing.T) {
	bigMarkdown := strings.Repeat("A", 100*1024) // 100 KiB, exceeds default scanner line cap
	body := OCRResponse{Model: "mistral-ocr", Pages: []OCRPage{{Index: 0, Markdown: bigMarkdown}}}
	bodyJSON, _ := json.Marshal(body)
	line := fmt.Sprintf(`{"id":"r0","custom_id":"0","response":{"status_code":200,"body":%s},"error":null}`, bodyJSON)

	results, err := ParseBatchResults[OCRResponse]([]byte(line))
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 1 || len(results[0].Body.Pages) != 1 {
		t.Fatalf("results = %+v", results)
	}
	if results[0].Body.Pages[0].Markdown != bigMarkdown {
		t.Error("large markdown body not parsed intact")
	}
}

func TestResultsByCustomID(t *testing.T) {
	results := []BatchResult[ChatCompletionResponse]{
		{CustomID: "a", StatusCode: 200},
		{CustomID: "b", StatusCode: 500},
	}
	m := ResultsByCustomID(results)
	if len(m) != 2 || m["a"].StatusCode != 200 || m["b"].StatusCode != 500 {
		t.Errorf("map = %+v", m)
	}
}

func TestDownloadFile(t *testing.T) {
	const wantFileID = "file-out"
	const raw = "{\"custom_id\":\"0\"}\n{\"custom_id\":\"1\"}\n"

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/files/"+wantFileID+"/content" || r.Method != http.MethodGet {
			http.NotFound(w, r)
			return
		}
		if r.Header.Get("Authorization") != "Bearer test-key" {
			t.Errorf("authorization %q", r.Header.Get("Authorization"))
		}
		_, _ = w.Write([]byte(raw))
	}))
	defer srv.Close()

	cl, err := NewClient("test-key", WithBaseURL(srv.URL))
	if err != nil {
		t.Fatal(err)
	}

	body, err := cl.DownloadFile(context.Background(), wantFileID)
	if err != nil {
		t.Fatal(err)
	}
	if string(body) != raw {
		t.Errorf("body = %q want %q", body, raw)
	}

	if _, err := cl.DownloadFile(context.Background(), " "); err == nil {
		t.Error("expected validation error for empty file id")
	}
}

func TestDownloadFile_apiError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
		_, _ = w.Write([]byte(`{"message":"file not found","type":"not_found"}`))
	}))
	defer srv.Close()

	cl, err := NewClient("test-key", WithBaseURL(srv.URL))
	if err != nil {
		t.Fatal(err)
	}

	_, err = cl.DownloadFile(context.Background(), "missing")
	var apiErr *APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("expected *APIError, got %v", err)
	}
	if apiErr.StatusCode != http.StatusNotFound {
		t.Errorf("status = %d", apiErr.StatusCode)
	}
}
