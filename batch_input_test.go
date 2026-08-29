package mistralai

import (
	"bufio"
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

func TestBuildBatchInputJSONL(t *testing.T) {
	entries := []BatchEntry{
		ChatCompletionEntry("0", ChatCompletionRequest{
			Model:    ChatModelMistralSmallLatest,
			Messages: []ChatMessage{TextMessage(RoleUser, "Hello")},
		}),
		Entry("1", map[string]any{"input": "raw body"}),
	}

	var output bytes.Buffer
	err := EncodeBatchEntries(&output, entries)
	if err != nil {
		t.Fatal(err)
	}
	data := output.Bytes()

	var lines []batchResultLineStub
	sc := bufio.NewScanner(bytes.NewReader(data))
	for sc.Scan() {
		line := bytes.TrimSpace(sc.Bytes())
		if len(line) == 0 {
			continue
		}
		var l batchResultLineStub
		if err := json.Unmarshal(line, &l); err != nil {
			t.Fatalf("line not valid json: %v (%s)", err, line)
		}
		lines = append(lines, l)
	}
	if len(lines) != 2 {
		t.Fatalf("got %d lines, want 2", len(lines))
	}
	if lines[0].CustomID != "0" || lines[1].CustomID != "1" {
		t.Errorf("custom ids = %q, %q", lines[0].CustomID, lines[1].CustomID)
	}

	// First entry body round-trips into ChatCompletionRequest.
	var req ChatCompletionRequest
	if err := json.Unmarshal(lines[0].Body, &req); err != nil {
		t.Fatal(err)
	}
	if req.Model != ChatModelMistralSmallLatest || len(req.Messages) != 1 {
		t.Errorf("body = %+v", req)
	}
}

func TestBuildBatchInputJSONL_toolsRoundTrip(t *testing.T) {
	entries := []BatchEntry{
		ChatCompletionEntry("tools", ChatCompletionRequest{
			Model: ChatModelMistralSmallLatest,
			Messages: []ChatMessage{
				TextMessage(RoleUser, "How many?"),
			},
			Tools: []Tool{
				FunctionTool("count_apartments", "Count apartments", map[string]any{
					"type": "object",
					"properties": map[string]any{
						"building_id": map[string]any{"type": "integer"},
					},
				}),
			},
			ToolChoice: ToolChoiceMode(ToolChoiceAuto),
		}),
	}

	var output bytes.Buffer
	err := EncodeBatchEntries(&output, entries)
	if err != nil {
		t.Fatal(err)
	}
	data := output.Bytes()

	var lines []batchResultLineStub
	sc := bufio.NewScanner(bytes.NewReader(data))
	for sc.Scan() {
		var l batchResultLineStub
		if err := json.Unmarshal(sc.Bytes(), &l); err != nil {
			t.Fatal(err)
		}
		lines = append(lines, l)
	}
	if len(lines) != 1 {
		t.Fatalf("got %d lines", len(lines))
	}

	var req ChatCompletionRequest
	if err := json.Unmarshal(lines[0].Body, &req); err != nil {
		t.Fatal(err)
	}
	if len(req.Tools) != 1 || req.Tools[0].Function.Name != "count_apartments" {
		t.Fatalf("tools = %+v", req.Tools)
	}
	if req.ToolChoice != ToolChoiceMode(ToolChoiceAuto) {
		t.Fatalf("tool_choice = %v", req.ToolChoice)
	}
}

// batchResultLineStub mirrors the {custom_id, body} input line for assertions.
type batchResultLineStub struct {
	CustomID string          `json:"custom_id"`
	Body     json.RawMessage `json:"body"`
}

func TestBuildBatchInputJSONL_errors(t *testing.T) {
	cases := map[string][]BatchEntry{
		"empty":           {},
		"blank custom id": {{CustomID: " ", Body: 1}},
		"nil body":        {{CustomID: "0", Body: nil}},
		"duplicate id":    {{CustomID: "0", Body: 1}, {CustomID: "0", Body: 2}},
	}
	for name, entries := range cases {
		t.Run(name, func(t *testing.T) {
			if err := EncodeBatchEntries(io.Discard, entries); err == nil {
				t.Fatal("expected error")
			}
		})
	}
}

func TestOCREntry(t *testing.T) {
	entry, err := OCREntry("0", OCRRequest{
		Source:        UploadedFile{FileID: "file-doc"},
		Pages:         []int{0, 1},
		ExtractHeader: new(true),
	})
	if err != nil {
		t.Fatal(err)
	}
	body, ok := entry.Body.(ocrRequestBody)
	if !ok {
		t.Fatalf("body type = %T", entry.Body)
	}
	if body.Model != DefaultOCRModel {
		t.Errorf("model = %q", body.Model)
	}
	if body.Document.Type != documentTypeFile || body.Document.FileID != "file-doc" {
		t.Errorf("document = %+v", body.Document)
	}
	if len(body.Pages) != 2 || body.ExtractH == nil || !*body.ExtractH {
		t.Errorf("options not applied: %+v", body)
	}
}

// A rejected OCR entry must say why at construction, not surface later as a
// generic "body is required" from the JSONL encoder.
func TestOCREntry_rejectsInvalidRequests(t *testing.T) {
	cases := map[string]OCRRequest{
		"local file":      {Source: LocalFile{Name: "a.pdf", Reader: strings.NewReader("x")}},
		"nil source":      {},
		"empty file id":   {Source: UploadedFile{}},
		"prompt w/o fmt":  {Source: UploadedFile{FileID: "f"}, DocumentAnnotationPrompt: "extract"},
		"negative limit":  {Source: UploadedFile{FileID: "f"}, ImageLimit: new(-1)},
		"unknown sources": {Source: DocumentURL{}},
	}
	for name, req := range cases {
		t.Run(name, func(t *testing.T) {
			entry, err := OCREntry("0", req)
			if !errors.Is(err, ErrInvalidRequest) {
				t.Fatalf("err = %v", err)
			}
			if entry.Body != nil || entry.CustomID != "" {
				t.Fatalf("entry = %+v, want zero value", entry)
			}
		})
	}

	if _, err := OCREntry("0", OCRRequest{Source: LocalFile{Name: "a.pdf", Reader: strings.NewReader("x")}}); !strings.Contains(err.Error(), "LocalFile") {
		t.Fatalf("LocalFile rejection should name the source: %v", err)
	}
}

func TestUploadBatchInput(t *testing.T) {
	const wantFileID = "file-batch-1"
	var uploaded []byte
	var gotPurpose string

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/files" || r.Method != http.MethodPost {
			http.NotFound(w, r)
			return
		}
		if err := r.ParseMultipartForm(1 << 20); err != nil {
			t.Fatal(err)
		}
		gotPurpose = r.FormValue("purpose")
		f, hdr, err := r.FormFile("file")
		if err != nil {
			t.Fatal(err)
		}
		uploaded, _ = io.ReadAll(f)
		_ = f.Close()
		if hdr.Filename != "batch.jsonl" {
			t.Errorf("filename = %q", hdr.Filename)
		}
		_ = json.NewEncoder(w).Encode(uploadFileResponse{ID: wantFileID, Object: "file", Purpose: FilePurposeBatch})
	}))
	defer srv.Close()

	cl, err := NewClient("test-key", WithBaseURL(srv.URL))
	if err != nil {
		t.Fatal(err)
	}

	entries := []BatchEntry{
		ChatCompletionEntry("0", ChatCompletionRequest{
			Model:    ChatModelMistralSmallLatest,
			Messages: []ChatMessage{TextMessage(RoleUser, "Hi")},
		}),
	}
	fileID, err := cl.UploadBatchInput(context.Background(), "batch.jsonl", entries)
	if err != nil {
		t.Fatal(err)
	}
	if fileID != wantFileID {
		t.Errorf("file id = %q", fileID)
	}
	if gotPurpose != FilePurposeBatch {
		t.Errorf("purpose = %q", gotPurpose)
	}

	var want bytes.Buffer
	_ = EncodeBatchEntries(&want, entries)
	if !bytes.Equal(uploaded, want.Bytes()) {
		t.Errorf("uploaded bytes mismatch:\n got %q\nwant %q", uploaded, want.Bytes())
	}

	if _, err := cl.UploadBatchInput(context.Background(), " ", entries); err == nil {
		t.Error("expected error for blank filename")
	}
	if _, err := cl.UploadBatchInput(context.Background(), "batch.jsonl", nil); err == nil {
		t.Error("expected error for empty entries")
	}
}

// ensure JSONL has one object per line (no embedded newlines within an object).
func TestBuildBatchInputJSONL_oneObjectPerLine(t *testing.T) {
	var output bytes.Buffer
	err := EncodeBatchEntries(&output, []BatchEntry{
		Entry("0", map[string]any{"a": 1}),
		Entry("1", map[string]any{"b": 2}),
	})
	if err != nil {
		t.Fatal(err)
	}
	if got := strings.Count(output.String(), "\n"); got != 2 {
		t.Errorf("newline count = %d, want 2", got)
	}
}
