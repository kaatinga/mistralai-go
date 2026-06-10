package mistralai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"strings"
)

// BatchEntry is one line of a batch input JSONL file: a unique custom id and the
// request body for the job's endpoint. Build entries with ChatCompletionEntry,
// OCREntry, or the generic Entry, then pass them to UploadBatchInput.
type BatchEntry struct {
	// CustomID uniquely identifies the entry within the file and is echoed back
	// on the matching result line (see ParseBatchResults).
	CustomID string
	// Body is the endpoint request payload (e.g. ChatCompletionRequest).
	Body any
}

type batchInputLine struct {
	CustomID string `json:"custom_id"`
	Body     any    `json:"body"`
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

// OCREntryOption configures the optional fields of an OCR batch entry body.
// The With* helpers below cover the common fields; custom options can set any
// OCROptions field directly.
type OCREntryOption func(*OCROptions)

// WithOCRPages limits OCR to the given zero-based page indices.
func WithOCRPages(pages ...int) OCREntryOption {
	return func(o *OCROptions) { o.Pages = pages }
}

// WithOCRTableFormat sets the table output format ("markdown" or "html").
func WithOCRTableFormat(format string) OCREntryOption {
	return func(o *OCROptions) { o.TableFormat = format }
}

// WithOCRImageBase64 requests base64 image payloads in the response.
func WithOCRImageBase64(include bool) OCREntryOption {
	return func(o *OCROptions) { o.IncludeImageBase64 = new(include) }
}

// WithOCRExtractHeader toggles document header extraction.
func WithOCRExtractHeader(extract bool) OCREntryOption {
	return func(o *OCROptions) { o.ExtractHeader = new(extract) }
}

// WithOCRExtractFooter toggles document footer extraction.
func WithOCRExtractFooter(extract bool) OCREntryOption {
	return func(o *OCROptions) { o.ExtractFooter = new(extract) }
}

// WithOCRDocumentAnnotation sets a structured-extraction prompt and its required
// response format (see JSONSchemaFormat).
func WithOCRDocumentAnnotation(prompt string, format *ResponseFormat) OCREntryOption {
	return func(o *OCROptions) {
		o.DocumentAnnotationPrompt = prompt
		o.DocumentAnnotationFormat = format
	}
}

func ocrEntryOptions(opts []OCREntryOption) OCROptions {
	var o OCROptions
	for _, opt := range opts {
		opt(&o)
	}
	return o
}

// OCREntry builds a batch entry for the /v1/ocr endpoint. Unlike the synchronous
// OCR call, a batch OCR body references a file that has already been uploaded
// (see UploadFile), so pass its file id rather than raw content. model defaults
// to DefaultOCRModel when empty.
func OCREntry(customID, model, fileID string, opts ...OCREntryOption) BatchEntry {
	doc := ocrDocument{Type: documentTypeFile, FileID: fileID}
	return BatchEntry{CustomID: customID, Body: ocrBody(model, doc, ocrEntryOptions(opts))}
}

// OCRURLEntry builds a batch entry for the /v1/ocr endpoint that references a
// document by URL instead of an uploaded file id. model defaults to
// DefaultOCRModel when empty.
func OCRURLEntry(customID, model, documentURL string, opts ...OCREntryOption) BatchEntry {
	doc := ocrDocument{Type: documentTypeDocumentURL, DocumentURL: documentURL}
	return BatchEntry{CustomID: customID, Body: ocrBody(model, doc, ocrEntryOptions(opts))}
}

// BuildBatchInputJSONL serializes entries into newline-delimited JSON (one
// {"custom_id":...,"body":...} object per line). It validates that there is at
// least one entry, that every custom id is non-empty and unique, and that no
// body is nil.
func BuildBatchInputJSONL(entries []BatchEntry) ([]byte, error) {
	if len(entries) == 0 {
		return nil, fmt.Errorf("%w: at least one batch entry is required", ErrInvalidRequest)
	}
	seen := make(map[string]struct{}, len(entries))
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf) // Encode appends a newline, yielding valid JSONL
	for i, e := range entries {
		if strings.TrimSpace(e.CustomID) == "" {
			return nil, fmt.Errorf("%w: batch entry %d: custom_id is required", ErrInvalidRequest, i)
		}
		if _, dup := seen[e.CustomID]; dup {
			return nil, fmt.Errorf("%w: duplicate custom_id %q", ErrInvalidRequest, e.CustomID)
		}
		seen[e.CustomID] = struct{}{}
		if e.Body == nil {
			return nil, fmt.Errorf("%w: batch entry %q: body is required", ErrInvalidRequest, e.CustomID)
		}
		if err := enc.Encode(batchInputLine{CustomID: e.CustomID, Body: e.Body}); err != nil {
			return nil, fmt.Errorf("mistral: encode batch entry %q: %w", e.CustomID, err)
		}
	}
	return buf.Bytes(), nil
}

// UploadBatchInput builds a JSONL input file from entries and uploads it with
// purpose "batch", returning the file id for CreateBatchJobRequest.InputFiles.
func (c *Client) UploadBatchInput(ctx context.Context, filename string, entries []BatchEntry) (string, error) {
	if strings.TrimSpace(filename) == "" {
		return "", fmt.Errorf("%w: filename is required", ErrInvalidRequest)
	}
	jsonl, err := BuildBatchInputJSONL(entries)
	if err != nil {
		return "", err
	}
	return c.UploadFile(ctx, UploadFileRequest{
		Filename:    filename,
		Content:     bytes.NewReader(jsonl),
		ContentType: "application/jsonl",
		Purpose:     FilePurposeBatch,
	})
}
