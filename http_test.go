package mistralai

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestUploadFile_retriesOn429(t *testing.T) {
	var attempts int
	var lastBody []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/files" {
			http.NotFound(w, r)
			return
		}
		attempts++
		body, _ := io.ReadAll(r.Body)
		if attempts < 2 {
			w.WriteHeader(http.StatusTooManyRequests)
			_, _ = w.Write([]byte(`{"message":"rate limit"}`))
			return
		}
		lastBody = body
		_ = json.NewEncoder(w).Encode(uploadFileResponse{ID: "file-1", Object: "file"})
	}))
	defer srv.Close()

	cl, err := NewClient("k", WithBaseURL(srv.URL), WithMaxRetries(3))
	if err != nil {
		t.Fatal(err)
	}

	id, err := cl.UploadFile(context.Background(), UploadFileRequest{
		Filename: "a.pdf",
		Content:  bytes.NewReader([]byte("%PDF-1.4 retry")),
	})
	if err != nil {
		t.Fatal(err)
	}
	if id != "file-1" {
		t.Fatalf("id = %q", id)
	}
	if attempts != 2 {
		t.Fatalf("attempts = %d want 2", attempts)
	}
	if !bytes.Contains(lastBody, []byte("%PDF-1.4 retry")) {
		t.Fatalf("retried multipart body missing file content: %q", lastBody)
	}
}

func TestWithMaxRetries_countsRetriesNotAttempts(t *testing.T) {
	run := func(t *testing.T, retries, wantAttempts int) {
		t.Helper()
		var attempts int
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			attempts++
			w.WriteHeader(http.StatusServiceUnavailable)
		}))
		defer srv.Close()

		cl, err := NewClient("k", WithBaseURL(srv.URL), WithMaxRetries(retries))
		if err != nil {
			t.Fatal(err)
		}

		if _, err := cl.ListModels(context.Background()); err == nil {
			t.Fatal("expected error")
		}
		if attempts != wantAttempts {
			t.Fatalf("attempts = %d want %d", attempts, wantAttempts)
		}
	}

	t.Run("zero_disables_retries", func(t *testing.T) { run(t, 0, 1) })
	t.Run("one_retry_two_attempts", func(t *testing.T) { run(t, 1, 2) })
	t.Run("negative_treated_as_zero", func(t *testing.T) { run(t, -3, 1) })
}

func TestDoRetry_acceptsNon200Success(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNoContent)
	}))
	defer srv.Close()

	cl, err := NewClient("k", WithBaseURL(srv.URL))
	if err != nil {
		t.Fatal(err)
	}

	if err := cl.DeleteFile(context.Background(), "file-1"); err != nil {
		t.Fatalf("DeleteFile on 204: %v", err)
	}
}

func TestDoRetry_setsUserAgent(t *testing.T) {
	var gotUA string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotUA = r.Header.Get("User-Agent")
		_ = json.NewEncoder(w).Encode(ModelList{Object: "list"})
	}))
	defer srv.Close()

	cl, err := NewClient("k", WithBaseURL(srv.URL))
	if err != nil {
		t.Fatal(err)
	}

	if _, err := cl.ListModels(context.Background()); err != nil {
		t.Fatal(err)
	}
	if gotUA != userAgent {
		t.Fatalf("User-Agent = %q want %q", gotUA, userAgent)
	}
}

func TestRetryAfter(t *testing.T) {
	h := http.Header{}
	if _, ok := retryAfter(h); ok {
		t.Fatal("empty header should not parse")
	}

	h.Set("Retry-After", "7")
	if d, ok := retryAfter(h); !ok || d != 7*time.Second {
		t.Fatalf("seconds form: d=%v ok=%v", d, ok)
	}

	h.Set("Retry-After", time.Now().Add(30*time.Second).UTC().Format(http.TimeFormat))
	if d, ok := retryAfter(h); !ok || d <= 0 || d > 31*time.Second {
		t.Fatalf("date form: d=%v ok=%v", d, ok)
	}

	h.Set("Retry-After", "garbage")
	if _, ok := retryAfter(h); ok {
		t.Fatal("garbage should not parse")
	}
}

func TestJittered_bounds(t *testing.T) {
	d := time.Second
	for range 100 {
		j := jittered(d)
		if j < d || j > d+d/4 {
			t.Fatalf("jittered(%v) = %v out of [d, 1.25d]", d, j)
		}
	}
}

func TestAPIError_validationDetail(t *testing.T) {
	t.Run("top_level_detail", func(t *testing.T) {
		body := []byte(`{"detail":[{"loc":["body","model"],"msg":"field required","type":"missing"}]}`)
		e := apiError(http.StatusUnprocessableEntity, body)
		if e.Message != "body.model: field required" {
			t.Fatalf("Message = %q", e.Message)
		}
	})

	t.Run("detail_nested_in_message", func(t *testing.T) {
		body := []byte(`{"object":"error","message":{"detail":[{"loc":["body","messages"],"msg":"value error"}]},"type":"invalid_request_error"}`)
		e := apiError(http.StatusUnprocessableEntity, body)
		if e.Message != "body.messages: value error" {
			t.Fatalf("Message = %q", e.Message)
		}
	})

	t.Run("plain_message_still_works", func(t *testing.T) {
		body := []byte(`{"object":"error","message":"bad request","type":"invalid_request_error"}`)
		e := apiError(http.StatusBadRequest, body)
		if e.Message != "bad request" {
			t.Fatalf("Message = %q", e.Message)
		}
	})
}
