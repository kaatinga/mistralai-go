package mistralai

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestOCR_uploadAndOCR(t *testing.T) {
	const wantFileID = "497f6eca-6276-4993-bfeb-53cbbbba6f09"
	var uploadedBody []byte

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/files":
			if r.Method != http.MethodPost {
				t.Errorf("files: method %s", r.Method)
			}
			if !strings.HasPrefix(r.Header.Get("Content-Type"), "multipart/form-data") {
				t.Errorf("files: content-type %q", r.Header.Get("Content-Type"))
			}
			if err := r.ParseMultipartForm(1 << 20); err != nil {
				t.Fatal(err)
			}
			if r.FormValue("purpose") != filePurposeOCR {
				t.Errorf("purpose = %q", r.FormValue("purpose"))
			}
			f, hdr, err := r.FormFile("file")
			if err != nil {
				t.Fatal(err)
			}
			uploadedBody, _ = io.ReadAll(f)
			f.Close()
			if hdr.Filename != "doc.pdf" {
				t.Errorf("filename = %q", hdr.Filename)
			}
			_ = json.NewEncoder(w).Encode(uploadFileResponse{
				ID:       wantFileID,
				Object:   "file",
				Filename: "doc.pdf",
				Purpose:  filePurposeOCR,
			})
		case "/v1/ocr":
			if r.Method != http.MethodPost {
				t.Errorf("ocr: method %s", r.Method)
			}
			var body ocrRequestBody
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
				t.Fatal(err)
			}
			if body.Model != DefaultOCRModel {
				t.Errorf("model = %q", body.Model)
			}
			if body.Document.Type != documentTypeFile || body.Document.FileID != wantFileID {
				t.Errorf("document = %+v", body.Document)
			}
			_ = json.NewEncoder(w).Encode(OCRResponse{
				Model:     "mistral-ocr-2503-completion",
				Pages:     []OCRPage{{Index: 0, Markdown: "# Title"}},
				UsageInfo: OCRUsageInfo{PagesProcessed: 1},
			})
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()

	cl, err := NewClient("test-key", WithBaseURL(srv.URL))
	if err != nil {
		t.Fatal(err)
	}
	defer cl.Close()

	ctx := context.Background()
	result, err := cl.OCR(ctx, OCRRequest{
		Filename:    "doc.pdf",
		Content:     bytes.NewReader([]byte("%PDF-1.4 test")),
		ContentType: "application/pdf",
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Pages) != 1 || result.Pages[0].Markdown != "# Title" {
		t.Fatalf("response: %+v", result)
	}
	if string(uploadedBody) != "%PDF-1.4 test" {
		t.Fatalf("uploaded %q", uploadedBody)
	}
}

func TestOCR_apiError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/files":
			_ = json.NewEncoder(w).Encode(uploadFileResponse{ID: "id", Object: "file"})
		case "/v1/ocr":
			w.WriteHeader(http.StatusBadRequest)
			_ = json.NewEncoder(w).Encode(apiErrorResponse{
				Object:  "error",
				Message: "invalid document",
			})
		}
	}))
	defer srv.Close()

	cl, err := NewClient("k", WithBaseURL(srv.URL))
	if err != nil {
		t.Fatal(err)
	}
	defer cl.Close()

	_, err = cl.OCR(context.Background(), OCRRequest{
		Filename: "x.pdf",
		Content:  bytes.NewReader([]byte("x")),
	})
	if err == nil || !strings.Contains(err.Error(), "invalid document") {
		t.Fatalf("err = %v", err)
	}
}

func TestNewClient_requiresAPIKey(t *testing.T) {
	if _, err := NewClient(""); err == nil {
		t.Fatal("expected error")
	}
	if _, err := NewClient("   "); err == nil {
		t.Fatal("expected error for whitespace key")
	}
}

func TestChat_textMarkdownJSON(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" || r.Method != http.MethodPost {
			http.NotFound(w, r)
			return
		}
		var body chatCompletionRequest
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatal(err)
		}
		if body.Model != DefaultChatModel {
			t.Errorf("model = %q", body.Model)
		}
		if len(body.Messages) < 1 || body.Messages[len(body.Messages)-1].Role != "user" {
			t.Fatalf("messages = %+v", body.Messages)
		}

		var content string
		switch body.Messages[len(body.Messages)-1].Content {
		case "plain":
			if body.ResponseFormat != nil {
				t.Fatalf("text: response_format = %+v", body.ResponseFormat)
			}
			content = "hello"
		case "md":
			if body.ResponseFormat != nil {
				t.Fatalf("markdown: response_format = %+v", body.ResponseFormat)
			}
			if !strings.Contains(body.Messages[0].Content, "Markdown") {
				t.Errorf("system = %q", body.Messages[0].Content)
			}
			content = "# Title"
		case "json":
			if body.ResponseFormat == nil || body.ResponseFormat.Type != "json_object" {
				t.Fatalf("json: response_format = %+v", body.ResponseFormat)
			}
			content = `{"ok":true}`
		default:
			t.Fatalf("unexpected input %q", body.Messages[len(body.Messages)-1].Content)
		}

		_ = json.NewEncoder(w).Encode(chatCompletionResponse{
			Model: "mistral-small-latest",
			Choices: []struct {
				Message struct {
					Content string `json:"content"`
				} `json:"message"`
			}{{Message: struct {
				Content string `json:"content"`
			}{Content: content}}},
		})
	}))
	defer srv.Close()

	cl, err := NewClient("test-key", WithBaseURL(srv.URL))
	if err != nil {
		t.Fatal(err)
	}
	defer cl.Close()

	ctx := context.Background()

	text, err := cl.Chat(ctx, ChatRequest{Input: "plain"})
	if err != nil || text.Content != "hello" || text.Format != OutputText {
		t.Fatalf("text: %+v err=%v", text, err)
	}

	md, err := cl.Chat(ctx, ChatRequest{Input: "md", Format: OutputMarkdown})
	if err != nil || md.Content != "# Title" || md.Format != OutputMarkdown {
		t.Fatalf("markdown: %+v err=%v", md, err)
	}

	js, err := cl.Chat(ctx, ChatRequest{Input: "json", Format: OutputJSON})
	if err != nil || js.Format != OutputJSON {
		t.Fatalf("json: %+v err=%v", js, err)
	}
	var parsed map[string]bool
	if err = js.JSON(&parsed); err != nil || !parsed["ok"] {
		t.Fatalf("json parse: %v parsed=%v", err, parsed)
	}
}

func TestChat_requiresInput(t *testing.T) {
	cl, err := NewClient("k", WithBaseURL("http://127.0.0.1:1"))
	if err != nil {
		t.Fatal(err)
	}
	defer cl.Close()

	_, err = cl.Chat(context.Background(), ChatRequest{})
	if err == nil || !strings.Contains(err.Error(), "input is required") {
		t.Fatalf("err = %v", err)
	}
}

func TestClose_idempotent(t *testing.T) {
	cl, err := NewClient("k", WithBaseURL("http://127.0.0.1:1"))
	if err != nil {
		t.Fatal(err)
	}
	if err := cl.Close(); err != nil {
		t.Fatalf("first close: %v", err)
	}
	if err := cl.Close(); err != nil {
		t.Fatalf("second close: %v", err)
	}
}
