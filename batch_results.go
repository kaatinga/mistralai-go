package mistralai

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
)

// BatchResult is one parsed line of a batch output (or error) file, keyed by the
// entry's custom id. Body is populated only when StatusCode is 200; otherwise it
// is the zero value of T and Error/StatusCode describe the failure.
type BatchResult[T any] struct {
	ID         string
	CustomID   string
	StatusCode int
	Body       T
	// Error is the raw error payload of a failed entry (nil on success);
	// unmarshal it into your own type as needed.
	Error json.RawMessage
}

type batchResultLine struct {
	ID       string `json:"id"`
	CustomID string `json:"custom_id"`
	Response *struct {
		StatusCode int             `json:"status_code"`
		Body       json.RawMessage `json:"body"`
	} `json:"response"`
	Error json.RawMessage `json:"error"`
}

// DownloadFile streams raw file content and transfers body ownership to the
// caller. Always close the returned body.
func (c *Client) DownloadFile(ctx context.Context, fileID string) (io.ReadCloser, error) {
	id, err := pathID("file id", fileID)
	if err != nil {
		return nil, err
	}
	resp, err := c.doStream(ctx, http.MethodGet, pathFiles+"/"+id+"/content", nil, true)
	if err != nil {
		return nil, fmt.Errorf("mistral: download file: %w", err)
	}
	return resp.Body, nil
}

// DefaultMaxBatchRecordBytes bounds a single JSONL record read by
// DecodeBatchResults. It is far above any real result line and exists so a
// truncated or malformed output file cannot be read whole into memory.
const DefaultMaxBatchRecordBytes int64 = 64 << 20

// BatchResultsOptions tunes DecodeBatchResultsWithOptions.
type BatchResultsOptions struct {
	// MaxRecordBytes bounds one JSONL record, delimiter included. Zero uses
	// DefaultMaxBatchRecordBytes; positive values override the default.
	MaxRecordBytes int64
}

// DecodeBatchResults consumes batch output JSONL one result at a time, with
// DefaultMaxBatchRecordBytes per record. Memory stays bounded by the current
// record and the consumer's work; the complete file is never materialized.
func DecodeBatchResults[T any](r io.Reader, consume func(BatchResult[T]) error) error {
	return DecodeBatchResultsWithOptions(r, BatchResultsOptions{}, consume)
}

// DecodeBatchResultsWithOptions is DecodeBatchResults with an explicit record
// size bound, for output files whose lines are known to be unusually large.
func DecodeBatchResultsWithOptions[T any](r io.Reader, opts BatchResultsOptions, consume func(BatchResult[T]) error) error {
	if r == nil {
		return fmt.Errorf("%w: batch results reader is required", ErrInvalidRequest)
	}
	if consume == nil {
		return fmt.Errorf("%w: batch results consumer is required", ErrInvalidRequest)
	}
	if opts.MaxRecordBytes < 0 {
		return fmt.Errorf("%w: max record bytes must not be negative", ErrInvalidRequest)
	}
	limit := opts.MaxRecordBytes
	if limit == 0 {
		limit = DefaultMaxBatchRecordBytes
	}
	reader := bufio.NewReader(r)
	var lineNumber int
	for {
		line, readErr := readBoundedLine(reader, limit)
		if errors.Is(readErr, errReadLimitExceeded) {
			return fmt.Errorf("mistral: batch result line %d exceeds %d bytes", lineNumber+1, limit)
		}
		if trimmed := bytes.TrimSpace(line); len(trimmed) > 0 {
			lineNumber++
			res, err := parseBatchResultLine[T](trimmed)
			if err != nil {
				return fmt.Errorf("mistral: parse batch result line %d (custom_id unknown): %w", lineNumber, err)
			}
			if err := consume(res); err != nil {
				return fmt.Errorf("mistral: consume batch result line %d (custom_id %q): %w", lineNumber, res.CustomID, err)
			}
		}
		if readErr != nil {
			if readErr == io.EOF {
				return nil
			}
			return fmt.Errorf("mistral: read batch results after line %d: %w", lineNumber, readErr)
		}
	}
}

func parseBatchResultLine[T any](line []byte) (BatchResult[T], error) {
	var raw batchResultLine
	if err := json.Unmarshal(line, &raw); err != nil {
		return BatchResult[T]{}, err
	}
	res := BatchResult[T]{ID: raw.ID, CustomID: raw.CustomID}
	if len(raw.Error) > 0 && string(raw.Error) != "null" {
		res.Error = raw.Error
	}
	if raw.Response != nil {
		res.StatusCode = raw.Response.StatusCode
		if raw.Response.StatusCode == http.StatusOK && len(raw.Response.Body) > 0 {
			if err := json.Unmarshal(raw.Response.Body, &res.Body); err != nil {
				return BatchResult[T]{}, fmt.Errorf("parse result body (custom_id %q): %w", raw.CustomID, err)
			}
		}
	}
	return res, nil
}

// ResultsByCustomID indexes parsed batch results by their custom id. When custom
// ids repeat, the last occurrence wins.
func ResultsByCustomID[T any](results []BatchResult[T]) map[string]BatchResult[T] {
	m := make(map[string]BatchResult[T], len(results))
	for _, r := range results {
		m[r.CustomID] = r
	}
	return m
}
