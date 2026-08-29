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

func TestUploadFileMultipartFilename(t *testing.T) {
	const filename = `résumé "Q1";.pdf`
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := r.ParseMultipartForm(1 << 20); err != nil {
			t.Fatal(err)
		}
		file, header, err := r.FormFile("file")
		if err != nil {
			t.Fatal(err)
		}
		defer func() { _ = file.Close() }()
		if header.Filename != filename {
			t.Fatalf("filename = %q", header.Filename)
		}
		_ = json.NewEncoder(w).Encode(File{ID: "file-1", Filename: filename})
	}))
	defer srv.Close()
	client, err := NewClient("key", WithBaseURL(srv.URL))
	if err != nil {
		t.Fatal(err)
	}
	file, err := client.UploadFile(context.Background(), UploadFileRequest{
		Filename: filename, Content: bytes.NewReader([]byte("content")), ContentType: "application/pdf",
	})
	if err != nil {
		t.Fatal(err)
	}
	if file.ID != "file-1" {
		t.Fatalf("file = %+v", file)
	}
}

func TestUploadFileReaderError(t *testing.T) {
	sentinel := errors.New("source read failed")
	httpClient := &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
		_, err := io.Copy(io.Discard, request.Body)
		return nil, err
	})}
	client, err := NewClient("key", WithHTTPClient(httpClient))
	if err != nil {
		t.Fatal(err)
	}
	_, err = client.UploadFile(context.Background(), UploadFileRequest{
		Filename: "broken.bin", Content: io.MultiReader(strings.NewReader("prefix"), errorReader{sentinel}),
	})
	if !errors.Is(err, sentinel) {
		t.Fatalf("err = %v", err)
	}
}

func TestOCRReturnsResponseWithCleanupError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodPost && r.URL.Path == "/v1/files":
			_ = json.NewEncoder(w).Encode(File{ID: "temporary-file"})
		case r.Method == http.MethodPost && r.URL.Path == "/v1/ocr":
			_ = json.NewEncoder(w).Encode(OCRResponse{Model: DefaultOCRModel, Pages: []OCRPage{{Index: 0}}})
		case r.Method == http.MethodDelete && r.URL.Path == "/v1/files/temporary-file":
			w.WriteHeader(http.StatusInternalServerError)
			_, _ = w.Write([]byte(`{"message":"cleanup failed"}`))
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	client, err := NewClient("key", WithBaseURL(srv.URL), WithRetryPolicy(RetryPolicy{MaxAttempts: 1}))
	if err != nil {
		t.Fatal(err)
	}
	response, err := client.OCR(context.Background(), OCRRequest{
		Source: LocalFile{Name: "tiny.png", ContentType: "image/png", Reader: bytes.NewReader([]byte("png"))},
	})
	var cleanupError *OCRCleanupError
	if !errors.As(err, &cleanupError) || cleanupError.FileID != "temporary-file" {
		t.Fatalf("err = %v", err)
	}
	if len(response.Pages) != 1 {
		t.Fatalf("response = %+v", response)
	}
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(request *http.Request) (*http.Response, error) { return f(request) }
