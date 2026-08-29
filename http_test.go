package mistralai

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestUploadFile_doesNotRetryOn429(t *testing.T) {
	var attempts int
	var lastBody []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/files" {
			http.NotFound(w, r)
			return
		}
		attempts++
		body, _ := io.ReadAll(r.Body)
		lastBody = body
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"message":"rate limit"}`))
	}))
	defer srv.Close()

	cl, err := NewClient("k", WithBaseURL(srv.URL), WithMaxRetries(3))
	if err != nil {
		t.Fatal(err)
	}

	_, err = cl.UploadFile(context.Background(), UploadFileRequest{
		Filename: "a.pdf",
		Content:  bytes.NewReader([]byte("%PDF-1.4 retry")),
	})
	if err == nil {
		t.Fatal("expected rate-limit error")
	}
	if attempts != 1 {
		t.Fatalf("attempts = %d want 1", attempts)
	}
	if !bytes.Contains(lastBody, []byte("%PDF-1.4 retry")) {
		t.Fatalf("multipart body missing file content: %q", lastBody)
	}
}

func TestUploadFile_rejectsOversizedErrorBodyAtErrorLimit(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		w.WriteHeader(http.StatusBadRequest)
		_, _ = io.WriteString(w, strings.Repeat("x", maxErrorBody+1))
	}))
	defer srv.Close()

	cl, err := NewClient("k", WithBaseURL(srv.URL))
	if err != nil {
		t.Fatal(err)
	}
	_, err = cl.UploadFile(context.Background(), UploadFileRequest{
		Filename: "a.pdf",
		Content:  strings.NewReader("small"),
	})
	if !errors.Is(err, ErrResponseTooLarge) {
		t.Fatalf("error = %v", err)
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
	if !strings.HasPrefix(gotUA, "mistralai-go/") {
		t.Fatalf("User-Agent = %q", gotUA)
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

func TestEndpointURL_preservesPrefixEscapingAndQuery(t *testing.T) {
	cl, err := NewClient("k", WithBaseURL("https://example.test/gateway/v1/?tenant=acme"))
	if err != nil {
		t.Fatal(err)
	}
	got := cl.endpointURL("/files/"+url.PathEscape("a/b"), url.Values{"q": {"x y"}})
	if !strings.Contains(got, "/gateway/v1/files/a%2Fb") ||
		!strings.Contains(got, "tenant=acme") ||
		!strings.Contains(got, "q=x+y") {
		t.Fatalf("endpoint URL = %q", got)
	}
}

func TestDoJSON_rejectsOversizedSuccess(t *testing.T) {
	var mu sync.Mutex
	var requests int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		requests++
		mu.Unlock()
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, `{"data":"`+strings.Repeat("x", maxJSONResponse)+`"}`)
	}))
	defer srv.Close()
	cl, err := NewClient("k", WithBaseURL(srv.URL))
	if err != nil {
		t.Fatal(err)
	}
	_, err = cl.ListModels(context.Background())
	if !errors.Is(err, ErrResponseTooLarge) {
		t.Fatalf("error = %v", err)
	}
	// An oversized body fails the same way every time, so a retryable GET must
	// still send exactly one request rather than re-downloading it per attempt.
	mu.Lock()
	defer mu.Unlock()
	if requests != 1 {
		t.Fatalf("requests = %d, want 1", requests)
	}
}

func TestDoStream_returnsOwnedBody(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, "stream")
	}))
	defer srv.Close()
	cl, err := NewClient("k", WithBaseURL(srv.URL))
	if err != nil {
		t.Fatal(err)
	}
	resp, err := cl.doStream(context.Background(), http.MethodGet, "/stream", nil, true)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = resp.Body.Close() }()
	body, err := io.ReadAll(resp.Body)
	if err != nil || string(body) != "stream" {
		t.Fatalf("body=%q err=%v", body, err)
	}
}

func TestRetryBackoff_respectsCancellation(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer srv.Close()
	cl, err := NewClient("k",
		WithBaseURL(srv.URL),
		WithRetryPolicy(RetryPolicy{MaxAttempts: 3, InitialDelay: time.Hour, MaxDelay: time.Hour}),
	)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go func() {
		time.Sleep(10 * time.Millisecond)
		cancel()
	}()
	if _, err := cl.ListModels(ctx); !errors.Is(err, context.Canceled) {
		t.Fatalf("error = %v", err)
	}
}

// url.PathEscape leaves "." alone and endpointURL cleans the joined path, so a
// "." or ".." identifier would address a different resource. Every path
// parameter must reject it instead of quietly issuing the wrong request.
func TestPathID_rejectsPathSegments(t *testing.T) {
	var requests int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests++
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, `{}`)
	}))
	defer srv.Close()
	cl, err := NewClient("k", WithBaseURL(srv.URL))
	if err != nil {
		t.Fatal(err)
	}

	ctx := context.Background()
	calls := map[string]func(string) error{
		"GetFile":          func(id string) error { _, err := cl.GetFile(ctx, id); return err },
		"DeleteFile":       func(id string) error { return cl.DeleteFile(ctx, id) },
		"DownloadFile":     func(id string) error { _, err := cl.DownloadFile(ctx, id); return err },
		"GetFileSignedURL": func(id string) error { _, err := cl.GetFileSignedURL(ctx, id, nil); return err },
		"GetModel":         func(id string) error { _, err := cl.GetModel(ctx, id); return err },
		"GetBatchJob":      func(id string) error { _, err := cl.GetBatchJob(ctx, id); return err },
		"CancelBatchJob":   func(id string) error { _, err := cl.CancelBatchJob(ctx, id); return err },
	}
	for name, call := range calls {
		t.Run(name, func(t *testing.T) {
			for _, id := range []string{"", "   ", ".", "..", " .. "} {
				if err := call(id); !errors.Is(err, ErrInvalidRequest) {
					t.Errorf("id %q: err = %v", id, err)
				}
			}
		})
	}
	if requests != 0 {
		t.Fatalf("requests = %d, want 0 — nothing should reach the server", requests)
	}
}
