package mistralai

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"strings"
)

// maxBatchResultLine bounds a single JSONL result line. OCR responses can be
// large, so allow up to 16 MiB per line.
const maxBatchResultLine = 16 * 1024 * 1024

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

// DownloadFile downloads raw file content (GET /v1/files/{file_id}/content).
// Use it to fetch a batch job's OutputFile or ErrorFile, then parse the bytes
// with ParseBatchResults.
func (c *Client) DownloadFile(ctx context.Context, fileID string) ([]byte, error) {
	if strings.TrimSpace(fileID) == "" {
		return nil, fmt.Errorf("%w: file id is required", ErrInvalidRequest)
	}
	body, err := c.getRaw(ctx, "/v1/files/"+url.PathEscape(fileID)+"/content")
	if err != nil {
		return nil, fmt.Errorf("mistral: download file: %w", err)
	}
	return body, nil
}

// ParseBatchResults parses batch output JSONL (from DownloadFile) into typed
// results, preserving file order. T is the endpoint's response type (e.g.
// ChatCompletionResponse, OCRResponse). For each line with status_code 200 and
// no error, response.body is unmarshaled into T; other lines keep their
// StatusCode and Error with a zero Body. Blank lines are skipped.
func ParseBatchResults[T any](jsonl []byte) ([]BatchResult[T], error) {
	var out []BatchResult[T]
	sc := bufio.NewScanner(bytes.NewReader(jsonl))
	sc.Buffer(make([]byte, 0, 64*1024), maxBatchResultLine)
	for sc.Scan() {
		line := bytes.TrimSpace(sc.Bytes())
		if len(line) == 0 {
			continue
		}
		var raw batchResultLine
		if err := json.Unmarshal(line, &raw); err != nil {
			return nil, fmt.Errorf("mistral: parse batch result line: %w", err)
		}
		res := BatchResult[T]{ID: raw.ID, CustomID: raw.CustomID}
		if len(raw.Error) > 0 && string(raw.Error) != "null" {
			res.Error = raw.Error
		}
		if raw.Response != nil {
			res.StatusCode = raw.Response.StatusCode
			if raw.Response.StatusCode == http.StatusOK && len(raw.Response.Body) > 0 {
				body, err := ParseJSON[T](string(raw.Response.Body))
				if err != nil {
					return nil, fmt.Errorf("mistral: parse result body (custom_id %q): %w", raw.CustomID, err)
				}
				res.Body = body
			}
		}
		out = append(out, res)
	}
	if err := sc.Err(); err != nil {
		return nil, fmt.Errorf("mistral: scan batch results: %w", err)
	}
	return out, nil
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
