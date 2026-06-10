package mistralai

import (
	"context"
	"fmt"
	"strings"
)

// OCRFileRequest describes OCR of a file already uploaded via UploadFile.
type OCRFileRequest struct {
	// FileID is the uploaded file's id (required).
	FileID string
	// Model overrides DefaultOCRModel when non-empty.
	Model string
	// Pages limits processing to zero-based page indices.
	Pages []int
	// TableFormat is "markdown" or "html" when set.
	TableFormat string
	// IncludeImageBase64 requests base64 image payloads in the response when true.
	IncludeImageBase64 *bool
	ExtractHeader      *bool
	ExtractFooter      *bool
	// ID is an optional client-side correlation id forwarded to the API.
	ID string
	// DocumentAnnotationPrompt guides structured extraction for the whole document.
	// DocumentAnnotationFormat must be set when using a prompt.
	DocumentAnnotationPrompt string
	DocumentAnnotationFormat *ResponseFormat
}

func (r OCRFileRequest) options() OCROptions {
	return OCROptions{
		Pages:                    r.Pages,
		TableFormat:              r.TableFormat,
		IncludeImageBase64:       r.IncludeImageBase64,
		ExtractHeader:            r.ExtractHeader,
		ExtractFooter:            r.ExtractFooter,
		ID:                       r.ID,
		DocumentAnnotationPrompt: r.DocumentAnnotationPrompt,
		DocumentAnnotationFormat: r.DocumentAnnotationFormat,
	}
}

// OCRByFileID runs POST /v1/ocr for a file already uploaded via UploadFile.
// The caller is responsible for deleting the file when finished.
func (c *Client) OCRByFileID(ctx context.Context, req OCRFileRequest) (OCRResponse, error) {
	fileID := strings.TrimSpace(req.FileID)
	if fileID == "" {
		return OCRResponse{}, fmt.Errorf("%w: file id is required", ErrInvalidRequest)
	}
	opts := req.options()
	if err := opts.validate(); err != nil {
		return OCRResponse{}, err
	}
	body := ocrBody(req.Model, ocrDocument{Type: documentTypeFile, FileID: fileID}, opts)

	var resp OCRResponse
	if err := c.postJSON(ctx, "/v1/ocr", body, &resp); err != nil {
		return OCRResponse{}, fmt.Errorf("mistral: ocr: %w", err)
	}
	return resp, nil
}
