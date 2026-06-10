package mistralai

import (
	"context"
	"fmt"
	"strings"
)

// OCRURLRequest describes a document to OCR by URL instead of file upload.
// Exactly one of DocumentURL or ImageURL is required.
type OCRURLRequest struct {
	// DocumentURL references a document (e.g. a PDF) by URL.
	DocumentURL string
	// DocumentName optionally names the document; only valid with DocumentURL.
	DocumentName string
	// ImageURL references an image by https URL or data: URI.
	ImageURL string
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

// normalize trims the URL fields so validate and document agree on what is set.
func (r *OCRURLRequest) normalize() {
	r.DocumentURL = strings.TrimSpace(r.DocumentURL)
	r.ImageURL = strings.TrimSpace(r.ImageURL)
	r.DocumentName = strings.TrimSpace(r.DocumentName)
}

func (r OCRURLRequest) validate() error {
	hasDocument := r.DocumentURL != ""
	hasImage := r.ImageURL != ""
	if hasDocument == hasImage {
		return fmt.Errorf("%w: exactly one of DocumentURL or ImageURL is required", ErrInvalidRequest)
	}
	if r.DocumentName != "" && !hasDocument {
		return fmt.Errorf("%w: DocumentName requires DocumentURL", ErrInvalidRequest)
	}
	if r.DocumentAnnotationPrompt != "" && r.DocumentAnnotationFormat == nil {
		return fmt.Errorf("%w: document_annotation_format is required with document_annotation_prompt", ErrInvalidRequest)
	}
	return nil
}

func (r OCRURLRequest) document() ocrDocument {
	if r.ImageURL != "" {
		return ocrDocument{Type: documentTypeImageURL, ImageURL: r.ImageURL}
	}
	return ocrDocument{
		Type:         documentTypeDocumentURL,
		DocumentURL:  r.DocumentURL,
		DocumentName: r.DocumentName,
	}
}

// OCRByURL runs POST /v1/ocr on a document or image referenced by URL,
// skipping the file upload round-trip.
func (c *Client) OCRByURL(ctx context.Context, req OCRURLRequest) (OCRResponse, error) {
	req.normalize()
	if err := req.validate(); err != nil {
		return OCRResponse{}, err
	}

	model := req.Model
	if model == "" {
		model = DefaultOCRModel
	}

	body := ocrRequestBody{
		Model:                    model,
		Document:                 req.document(),
		Pages:                    req.Pages,
		ID:                       req.ID,
		TableFmt:                 req.TableFormat,
		Include:                  req.IncludeImageBase64,
		ExtractH:                 req.ExtractHeader,
		ExtractF:                 req.ExtractFooter,
		DocumentAnnotationFormat: req.DocumentAnnotationFormat,
	}
	if req.DocumentAnnotationPrompt != "" {
		body.DocumentAnnotationPrompt = new(req.DocumentAnnotationPrompt)
	}

	var resp OCRResponse
	if err := c.postJSON(ctx, "/v1/ocr", body, &resp); err != nil {
		return OCRResponse{}, fmt.Errorf("mistral: ocr: %w", err)
	}
	return resp, nil
}
