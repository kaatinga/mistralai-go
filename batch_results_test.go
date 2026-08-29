package mistralai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
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

	var results []BatchResult[ChatCompletionResponse]
	err := DecodeBatchResults[ChatCompletionResponse](strings.NewReader(jsonl), func(result BatchResult[ChatCompletionResponse]) error {
		results = append(results, result)
		return nil
	})
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
	content, err := results[0].Body.FirstText()
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

	var results []BatchResult[OCRResponse]
	err := DecodeBatchResults[OCRResponse](strings.NewReader(line), func(result BatchResult[OCRResponse]) error {
		results = append(results, result)
		return nil
	})
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
	defer func() { _ = body.Close() }()
	data, err := io.ReadAll(body)
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != raw {
		t.Errorf("body = %q want %q", data, raw)
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

func TestDecodeBatchResults_largeLineAndReaderError(t *testing.T) {
	large := strings.Repeat("x", 17*1024*1024)
	line := fmt.Sprintf(`{"id":"r0","custom_id":"large","response":{"status_code":200,"body":{"id":"%s"}}}`, large)
	var got []BatchResult[struct {
		ID string `json:"id"`
	}]
	if err := DecodeBatchResults(strings.NewReader(line+"\n"), func(result BatchResult[struct {
		ID string `json:"id"`
	}]) error {
		got = append(got, result)
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got[0].CustomID != "large" || len(got[0].Body.ID) != len(large) {
		t.Fatalf("decoded result = %+v", got)
	}

	wantErr := errors.New("reader failed")
	err := DecodeBatchResults[ChatCompletionResponse](errorReader{err: wantErr}, func(BatchResult[ChatCompletionResponse]) error {
		return nil
	})
	if !errors.Is(err, wantErr) {
		t.Fatalf("error = %v, want %v", err, wantErr)
	}
}

type errorReader struct {
	err error
}

func (r errorReader) Read([]byte) (int, error) { return 0, r.err }

// A single record must not be read into memory without a bound: an output file
// whose line never terminates has to fail fast, not consume the whole reader.
func TestDecodeBatchResults_recordSizeLimit(t *testing.T) {
	line := `{"custom_id":"a","response":{"status_code":200,"body":{"x":1}}}` + "\n"

	if err := DecodeBatchResultsWithOptions(
		strings.NewReader(line),
		BatchResultsOptions{MaxRecordBytes: 16},
		func(BatchResult[map[string]int]) error { return nil },
	); err == nil || !strings.Contains(err.Error(), "line 1 exceeds 16 bytes") {
		t.Fatalf("err = %v", err)
	}

	// The line number in the message points at the offending record.
	doc := line + strings.Repeat("y", 4096) + "\n"
	err := DecodeBatchResultsWithOptions(
		strings.NewReader(doc),
		BatchResultsOptions{MaxRecordBytes: 1024},
		func(BatchResult[map[string]int]) error { return nil },
	)
	if err == nil || !strings.Contains(err.Error(), "line 2 exceeds 1024 bytes") {
		t.Fatalf("err = %v", err)
	}

	// Positive overrides remain available for unusually large records.
	var seen int
	if err := DecodeBatchResultsWithOptions(
		strings.NewReader(line),
		BatchResultsOptions{MaxRecordBytes: int64(len(line))},
		func(BatchResult[map[string]int]) error { seen++; return nil },
	); err != nil || seen != 1 {
		t.Fatalf("overridden decode: seen=%d err=%v", seen, err)
	}

	// Negative bounds are invalid rather than an escape hatch from the package's
	// bounded-read contract.
	err = DecodeBatchResultsWithOptions(
		strings.NewReader(line),
		BatchResultsOptions{MaxRecordBytes: -1},
		func(BatchResult[map[string]int]) error { return nil },
	)
	if !errors.Is(err, ErrInvalidRequest) {
		t.Fatalf("negative limit err = %v", err)
	}
}
