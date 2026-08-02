package mistralai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"strings"
)

// BatchEntry is one line of a batch input JSONL file: a unique custom id and the
// request body for the job's endpoint. Build entries with ChatCompletionEntry,
// OCREntry, or the generic Entry, then pass them to UploadBatchInput.
type BatchEntry struct {
	// CustomID uniquely identifies the entry within the file and is echoed back
	// on the matching result line (see DecodeBatchResults).
	CustomID string
	// Body is the endpoint request payload (e.g. ChatCompletionRequest).
	Body any
}

type batchInputLine struct {
	CustomID string `json:"custom_id"`
	Body     any    `json:"body"`
}

// EncodeBatchEntries writes batch JSONL incrementally. It retains only the set
// of custom IDs for validation and never materializes the complete file.
func EncodeBatchEntries(w io.Writer, entries []BatchEntry) error {
	if w == nil {
		return fmt.Errorf("%w: output writer is required", ErrInvalidRequest)
	}
	if err := validateBatchEntries(entries); err != nil {
		return err
	}
	enc := json.NewEncoder(w)
	for _, e := range entries {
		if err := enc.Encode(batchInputLine(e)); err != nil {
			return fmt.Errorf("mistral: encode batch entry %q: %w", e.CustomID, err)
		}
	}
	return nil
}

func validateBatchEntries(entries []BatchEntry) error {
	if len(entries) == 0 {
		return fmt.Errorf("%w: at least one batch entry is required", ErrInvalidRequest)
	}
	seen := make(map[string]struct{}, len(entries))
	for i, e := range entries {
		if strings.TrimSpace(e.CustomID) == "" {
			return fmt.Errorf("%w: batch entry %d: custom_id is required", ErrInvalidRequest, i)
		}
		if _, dup := seen[e.CustomID]; dup {
			return fmt.Errorf("%w: duplicate custom_id %q", ErrInvalidRequest, e.CustomID)
		}
		seen[e.CustomID] = struct{}{}
		if e.Body == nil {
			return fmt.Errorf("%w: batch entry %q: body is required", ErrInvalidRequest, e.CustomID)
		}
	}
	return nil
}

type batchJSONLReader struct {
	entries []BatchEntry
	index   int
	buffer  bytes.Buffer
}

func newBatchJSONLReader(entries []BatchEntry) (*batchJSONLReader, error) {
	if err := validateBatchEntries(entries); err != nil {
		return nil, err
	}
	return &batchJSONLReader{entries: entries}, nil
}

func (r *batchJSONLReader) Read(p []byte) (int, error) {
	for r.buffer.Len() == 0 && r.index < len(r.entries) {
		if err := json.NewEncoder(&r.buffer).Encode(batchInputLine(r.entries[r.index])); err != nil {
			return 0, fmt.Errorf("mistral: encode batch entry %q: %w", r.entries[r.index].CustomID, err)
		}
		r.index++
	}
	if r.buffer.Len() == 0 {
		return 0, io.EOF
	}
	return r.buffer.Read(p)
}

// Entry builds a batch entry from an arbitrary request body. Use it for
// endpoints this package does not type yet; the body is JSON-encoded as-is.
func Entry(customID string, body any) BatchEntry {
	return BatchEntry{CustomID: customID, Body: body}
}

// ChatCompletionEntry builds a batch entry for the /v1/chat/completions endpoint
// from a typed ChatCompletionRequest.
func ChatCompletionEntry(customID string, req ChatCompletionRequest) BatchEntry {
	return BatchEntry{CustomID: customID, Body: req}
}

// EmbeddingEntry builds a batch entry for the /v1/embeddings endpoint from a
// typed EmbeddingRequest.
func EmbeddingEntry(customID string, req EmbeddingRequest) BatchEntry {
	return BatchEntry{CustomID: customID, Body: req}
}

// ModerationEntry builds a /v1/moderations batch entry.
func ModerationEntry(customID string, req ModerationRequest) BatchEntry {
	return BatchEntry{CustomID: customID, Body: req}
}

// ClassificationEntry builds a /v1/classifications batch entry.
func ClassificationEntry(customID string, req ClassificationRequest) BatchEntry {
	return BatchEntry{CustomID: customID, Body: req}
}

// OCREntry builds a batch entry using the same typed source and options as
// synchronous OCR, and validates the request the same way. LocalFile is
// rejected because a batch entry cannot own an upload lifecycle: upload the
// file first and reference it with UploadedFile.
func OCREntry(customID string, req OCRRequest) (BatchEntry, error) {
	if _, ok := req.Source.(LocalFile); ok {
		return BatchEntry{}, fmt.Errorf("%w: LocalFile cannot be a batch OCR source; upload it first and use UploadedFile", ErrInvalidRequest)
	}
	if err := req.validate(); err != nil {
		return BatchEntry{}, err
	}
	return BatchEntry{CustomID: customID, Body: ocrBody(req, req.document(""))}, nil
}

// UploadBatchInput builds a JSONL input file from entries and uploads it with
// purpose "batch", returning the file id for CreateBatchJobRequest.InputFiles.
func (c *Client) UploadBatchInput(ctx context.Context, filename string, entries []BatchEntry) (string, error) {
	if strings.TrimSpace(filename) == "" {
		return "", fmt.Errorf("%w: filename is required", ErrInvalidRequest)
	}
	reader, err := newBatchJSONLReader(entries)
	if err != nil {
		return "", err
	}
	file, err := c.UploadFile(ctx, UploadFileRequest{
		Filename:    filename,
		Content:     reader,
		ContentType: "application/jsonl",
		Purpose:     FilePurposeBatch,
	})
	if err != nil {
		return "", err
	}
	return file.ID, nil
}
