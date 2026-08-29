package mistralai

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestOCR_uploadAndOCR(t *testing.T) {
	const wantFileID = "497f6eca-6276-4993-bfeb-53cbbbba6f09"
	var uploadedBody []byte
	var deletedFileID string

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.URL.Path == "/v1/files/"+wantFileID && r.Method == http.MethodDelete:
			if r.Header.Get("Authorization") != "Bearer test-key" {
				t.Errorf("delete: authorization %q", r.Header.Get("Authorization"))
			}
			deletedFileID = wantFileID
			w.WriteHeader(http.StatusOK)
		case r.URL.Path == "/v1/files":
			if r.Method != http.MethodPost {
				t.Errorf("files: method %s", r.Method)
			}
			if !strings.HasPrefix(r.Header.Get("Content-Type"), "multipart/form-data") {
				t.Errorf("files: content-type %q", r.Header.Get("Content-Type"))
			}
			if err := r.ParseMultipartForm(1 << 20); err != nil {
				t.Fatal(err)
			}
			if r.FormValue("purpose") != FilePurposeOCR {
				t.Errorf("purpose = %q", r.FormValue("purpose"))
			}
			f, hdr, err := r.FormFile("file")
			if err != nil {
				t.Fatal(err)
			}
			uploadedBody, _ = io.ReadAll(f)
			_ = f.Close()
			if hdr.Filename != "doc.pdf" {
				t.Errorf("filename = %q", hdr.Filename)
			}
			_ = json.NewEncoder(w).Encode(uploadFileResponse{
				ID:       wantFileID,
				Object:   "file",
				Filename: "doc.pdf",
				Purpose:  FilePurposeOCR,
			})
		case r.URL.Path == "/v1/ocr":
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

	ctx := context.Background()
	result, err := cl.OCR(ctx, OCRRequest{
		Source: LocalFile{Name: "doc.pdf", Reader: bytes.NewReader([]byte("%PDF-1.4 test")), ContentType: "application/pdf"},
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
	if deletedFileID != wantFileID {
		t.Fatalf("delete file_id = %q want %q", deletedFileID, wantFileID)
	}
}

func TestDeleteFile(t *testing.T) {
	const wantFileID = "497f6eca-6276-4993-bfeb-53cbbbba6f09"

	t.Run("success", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path != "/v1/files/"+wantFileID || r.Method != http.MethodDelete {
				http.NotFound(w, r)
				return
			}
			if r.Header.Get("Authorization") != "Bearer test-key" {
				t.Errorf("authorization %q", r.Header.Get("Authorization"))
			}
			w.WriteHeader(http.StatusOK)
		}))
		defer srv.Close()

		cl, err := NewClient("test-key", WithBaseURL(srv.URL))
		if err != nil {
			t.Fatal(err)
		}

		if err := cl.DeleteFile(context.Background(), wantFileID); err != nil {
			t.Fatal(err)
		}
	})

	t.Run("api_error", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path != "/v1/files/"+wantFileID {
				http.NotFound(w, r)
				return
			}
			w.WriteHeader(http.StatusNotFound)
			_ = json.NewEncoder(w).Encode(apiErrorResponse{
				Object:  "error",
				Message: json.RawMessage(`"file not found"`),
				Type:    "invalid_request_error",
			})
		}))
		defer srv.Close()

		cl, err := NewClient("k", WithBaseURL(srv.URL))
		if err != nil {
			t.Fatal(err)
		}

		err = cl.DeleteFile(context.Background(), wantFileID)
		if err == nil {
			t.Fatal("expected error")
		}
		var apiErr *APIError
		if !errors.As(err, &apiErr) || apiErr.StatusCode != http.StatusNotFound {
			t.Fatalf("err = %v", err)
		}
	})

	t.Run("empty_id", func(t *testing.T) {
		cl, err := NewClient("k", WithBaseURL("http://127.0.0.1:1"))
		if err != nil {
			t.Fatal(err)
		}

		if err := cl.DeleteFile(context.Background(), ""); err == nil || !strings.Contains(err.Error(), "file id is required") {
			t.Fatalf("err = %v", err)
		}
	})
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
				Message: json.RawMessage(`"invalid document"`),
				Type:    "invalid_request_error",
				Code:    "1100",
			})
		}
	}))
	defer srv.Close()

	cl, err := NewClient("k", WithBaseURL(srv.URL))
	if err != nil {
		t.Fatal(err)
	}

	_, err = cl.OCR(context.Background(), OCRRequest{
		Source: LocalFile{Name: "x.pdf", Reader: bytes.NewReader([]byte("x"))},
	})
	if err == nil || !strings.Contains(err.Error(), "invalid document") {
		t.Fatalf("err = %v", err)
	}

	var apiErr *APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("want *APIError, got %T: %v", err, err)
	}
	if apiErr.StatusCode != http.StatusBadRequest {
		t.Errorf("status = %d", apiErr.StatusCode)
	}
	if apiErr.Message != "invalid document" {
		t.Errorf("message = %q", apiErr.Message)
	}
	if apiErr.Type != "invalid_request_error" {
		t.Errorf("type = %q", apiErr.Type)
	}
	if apiErr.Code != "1100" {
		t.Errorf("code = %q", apiErr.Code)
	}
	if apiErr.Retryable() {
		t.Error("400 should not be retryable")
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

func TestChatCompletion_multiMessageAndTemperature(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" || r.Method != http.MethodPost {
			http.NotFound(w, r)
			return
		}
		var body ChatCompletionRequest
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatal(err)
		}
		if body.Model != "mistral-large-latest" {
			t.Errorf("model = %q", body.Model)
		}
		if body.Temperature == nil || *body.Temperature != 0.7 {
			t.Errorf("temperature = %v", body.Temperature)
		}
		if len(body.Messages) != 2 {
			t.Fatalf("messages = %+v", body.Messages)
		}
		_ = json.NewEncoder(w).Encode(ChatCompletionResponse{
			Model: body.Model,
			Choices: []ChatCompletionResponseChoice{{
				Message: ChatMessage{Role: "assistant", Content: "done"},
			}},
		})
	}))
	defer srv.Close()

	cl, err := NewClient("test-key", WithBaseURL(srv.URL))
	if err != nil {
		t.Fatal(err)
	}

	resp, err := cl.ChatCompletion(context.Background(), ChatCompletionRequest{
		Model: "mistral-large-latest",
		Messages: []ChatMessage{
			{Role: "system", Content: "sys"},
			{Role: "user", Content: "hi"},
		},
		Temperature: new(0.7),
	})
	if err != nil {
		t.Fatal(err)
	}
	if got, err := resp.FirstText(); err != nil || got != "done" {
		t.Fatalf("content = %q", got)
	}
}

func TestListFiles(t *testing.T) {
	t.Run("default", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path != "/v1/files" || r.Method != http.MethodGet {
				http.NotFound(w, r)
				return
			}
			if r.URL.RawQuery != "" {
				t.Errorf("query = %q", r.URL.RawQuery)
			}
			_ = json.NewEncoder(w).Encode(FileList{
				Object: "list",
				Data: []File{{
					ID:         "497f6eca-6276-4993-bfeb-53cbbbba6f09",
					Object:     "file",
					Filename:   "doc.pdf",
					Purpose:    FilePurposeOCR,
					SampleType: "ocr",
					Source:     "upload",
				}},
				Total: new(1),
			})
		}))
		defer srv.Close()

		cl, err := NewClient("test-key", WithBaseURL(srv.URL))
		if err != nil {
			t.Fatal(err)
		}

		list, err := cl.ListFiles(context.Background(), ListFilesRequest{})
		if err != nil {
			t.Fatal(err)
		}
		if list.Object != "list" || len(list.Data) != 1 || list.Data[0].ID != "497f6eca-6276-4993-bfeb-53cbbbba6f09" {
			t.Fatalf("list = %+v", list)
		}
		if list.Total == nil || *list.Total != 1 {
			t.Fatalf("total = %v", list.Total)
		}
	})

	t.Run("filters", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path != "/v1/files" || r.Method != http.MethodGet {
				http.NotFound(w, r)
				return
			}
			q := r.URL.Query()
			if q.Get("page") != "1" || q.Get("page_size") != "50" {
				t.Errorf("pagination: %v", q)
			}
			if q.Get("include_total") != "false" {
				t.Errorf("include_total = %q", q.Get("include_total"))
			}
			if q.Get("purpose") != FilePurposeOCR {
				t.Errorf("purpose = %q", q.Get("purpose"))
			}
			if q.Get("search") != "invoice" {
				t.Errorf("search = %q", q.Get("search"))
			}
			if got := q["sample_type"]; len(got) != 2 || got[0] != "instruct" || got[1] != "batch_result" {
				t.Errorf("sample_type = %v", got)
			}
			if got := q["source"]; len(got) != 1 || got[0] != "upload" {
				t.Errorf("source = %v", got)
			}
			if got := q["mimetypes"]; len(got) != 1 || got[0] != "application/pdf" {
				t.Errorf("mimetypes = %v", got)
			}
			_ = json.NewEncoder(w).Encode(FileList{Object: "list", Data: nil})
		}))
		defer srv.Close()

		cl, err := NewClient("k", WithBaseURL(srv.URL))
		if err != nil {
			t.Fatal(err)
		}

		_, err = cl.ListFiles(context.Background(), ListFilesRequest{
			Page:         new(1),
			PageSize:     new(50),
			IncludeTotal: new(false),
			Purpose:      FilePurposeOCR,
			Search:       "invoice",
			SampleType:   []string{"instruct", "batch_result"},
			Source:       []string{"upload"},
			Mimetypes:    []string{"application/pdf"},
		})
		if err != nil {
			t.Fatal(err)
		}
	})
}

func TestListModels(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/models" || r.Method != http.MethodGet {
			http.NotFound(w, r)
			return
		}
		_ = json.NewEncoder(w).Encode(ModelList{
			Object: "list",
			Data: []ModelCard{
				{ID: "mistral-small-latest", Object: "model"},
				{ID: "mistral-embed", Object: "model"},
			},
		})
	}))
	defer srv.Close()

	cl, err := NewClient("test-key", WithBaseURL(srv.URL))
	if err != nil {
		t.Fatal(err)
	}

	list, err := cl.ListModels(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(list.Data) != 2 || list.Data[0].ID != "mistral-small-latest" {
		t.Fatalf("list = %+v", list)
	}
}

func TestDoJSON_retriesOn429(t *testing.T) {
	var attempts int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/models" {
			http.NotFound(w, r)
			return
		}
		attempts++
		if attempts < 2 {
			w.WriteHeader(http.StatusTooManyRequests)
			_, _ = w.Write([]byte(`{"message":"rate limit"}`))
			return
		}
		_ = json.NewEncoder(w).Encode(ModelList{Object: "list", Data: []ModelCard{{ID: "ok"}}})
	}))
	defer srv.Close()

	cl, err := NewClient("k", WithBaseURL(srv.URL), WithMaxRetries(3))
	if err != nil {
		t.Fatal(err)
	}

	list, err := cl.ListModels(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if attempts != 2 {
		t.Fatalf("attempts = %d want 2", attempts)
	}
	if len(list.Data) != 1 || list.Data[0].ID != "ok" {
		t.Fatalf("list = %+v", list)
	}
}

// uploadFileResponse is the subset of the upload response that test servers
// echo back. The client decodes the full File model; this exists only so tests
// can write a realistic partial payload.
type uploadFileResponse struct {
	ID       string `json:"id"`
	Object   string `json:"object"`
	Bytes    int64  `json:"bytes"`
	Filename string `json:"filename"`
	Purpose  string `json:"purpose"`
}
