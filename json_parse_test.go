package mistralai

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestParseJSON(t *testing.T) {
	type payload struct {
		Name string `json:"name"`
	}
	got, err := ParseJSON[payload](`{"name":"x"}`)
	if err != nil {
		t.Fatal(err)
	}
	if got.Name != "x" {
		t.Fatalf("got %q", got.Name)
	}
	_, err = ParseJSON[payload]("")
	if err == nil {
		t.Fatal("expected error for empty json")
	}
}

func TestDocumentAnnotationInto(t *testing.T) {
	raw := `{"value":42}`
	resp := OCRResponse{DocumentAnnotation: &raw}
	type out struct {
		Value int `json:"value"`
	}
	got, err := DocumentAnnotationInto[out](resp)
	if err != nil {
		t.Fatal(err)
	}
	if got.Value != 42 {
		t.Fatalf("got %d", got.Value)
	}
	empty := OCRResponse{}
	if _, err = DocumentAnnotationInto[out](empty); err == nil {
		t.Fatal("expected error")
	}
}

func TestChatStructured(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(ChatCompletionResponse{
			Choices: []ChatCompletionResponseChoice{{
				Message: ChatMessage{Role: "assistant", Content: `{"count":7}`},
			}},
		})
	}))
	defer srv.Close()

	cl, err := NewClient("key", WithBaseURL(srv.URL))
	if err != nil {
		t.Fatal(err)
	}

	type result struct {
		Count int `json:"count"`
	}
	got, resp, err := ChatStructured[result](context.Background(), cl, ChatCompletionRequest{
		Model:    "mistral-small-latest",
		Messages: []ChatMessage{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if got.Count != 7 {
		t.Fatalf("got count=%d", got.Count)
	}
	if len(resp.Choices) != 1 {
		t.Fatalf("choices=%d", len(resp.Choices))
	}
}

func TestOCRStructured(t *testing.T) {
	const annotation = `{"ok":true}`
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodPost && r.URL.Path == "/v1/files":
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"id":"file-1"}`))
		case r.Method == http.MethodPost && r.URL.Path == "/v1/ocr":
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(OCRResponse{
				DocumentAnnotation: new(annotation),
				Pages:              []OCRPage{{Index: 0, Markdown: "hi"}},
			})
		case r.Method == http.MethodDelete && r.URL.Path == "/v1/files/file-1":
			w.WriteHeader(http.StatusNoContent)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()

	cl, err := NewClient("key", WithBaseURL(srv.URL))
	if err != nil {
		t.Fatal(err)
	}

	type extracted struct {
		OK bool `json:"ok"`
	}
	got, resp, err := OCRStructured[extracted](context.Background(), cl, OCRRequest{
		Source: LocalFile{Name: "doc.pdf", Reader: strings.NewReader("%PDF")},
	})
	if err != nil {
		t.Fatal(err)
	}
	if !got.OK || len(resp.Pages) != 1 {
		t.Fatalf("got %+v pages=%d", got, len(resp.Pages))
	}
}
