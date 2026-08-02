package mistralai

import (
	"context"
	"errors"
	"fmt"
	"io"
	"strings"
	"time"
)

const (
	documentTypeFile        = "file"
	documentTypeDocumentURL = "document_url"
	documentTypeImageURL    = "image_url"
)

// UploadFileRequest describes a streamed file upload.
type UploadFileRequest struct {
	Filename    string
	Content     io.Reader
	ContentType string
	Purpose     string
}

// UploadFile uploads a file without buffering its content and returns the
// complete file metadata. Uploads are deliberately never retried.
func (c *Client) UploadFile(ctx context.Context, req UploadFileRequest) (File, error) {
	if strings.TrimSpace(req.Filename) == "" {
		return File{}, fmt.Errorf("%w: filename is required", ErrInvalidRequest)
	}
	if req.Content == nil {
		return File{}, fmt.Errorf("%w: content is required", ErrInvalidRequest)
	}
	purpose := strings.TrimSpace(req.Purpose)
	if purpose == "" {
		purpose = FilePurposeOCR
	}
	file, err := c.uploadFile(ctx, req.Filename, req.Content, req.ContentType, purpose)
	if err != nil {
		return File{}, fmt.Errorf("mistral: upload file: %w", err)
	}
	return file, nil
}

// OCRSource is the closed source union accepted by OCR. Use one of the
// exported source types; LocalFile is uploaded and cleaned up by OCR.
type OCRSource interface{ ocrSource() }

type UploadedFile struct{ FileID string }

func (UploadedFile) ocrSource() {}

type DocumentURL struct {
	URL  string
	Name string
}

func (DocumentURL) ocrSource() {}

type ImageURL struct{ URL string }

func (ImageURL) ocrSource() {}

type LocalFile struct {
	Name        string
	ContentType string
	Reader      io.Reader
}

func (LocalFile) ocrSource() {}

// OCRRequest describes one OCR operation and its source.
type OCRRequest struct {
	Model  string
	Source OCRSource
	Pages  []int

	TableFormat                 string
	IncludeImageBase64          *bool
	ImageLimit                  *int
	ImageMinSize                *int
	BBoxAnnotationFormat        *ResponseFormat
	IncludeBlocks               *bool
	ConfidenceScoresGranularity string
	ExtractHeader               *bool
	ExtractFooter               *bool
	ID                          string
	DocumentAnnotationPrompt    string
	DocumentAnnotationFormat    *ResponseFormat
}

func (r OCRRequest) validateOptions() error {
	if r.DocumentAnnotationPrompt != "" && r.DocumentAnnotationFormat == nil {
		return fmt.Errorf("%w: document_annotation_format is required with document_annotation_prompt", ErrInvalidRequest)
	}
	if r.ImageLimit != nil && *r.ImageLimit < 0 {
		return fmt.Errorf("%w: image_limit must not be negative", ErrInvalidRequest)
	}
	if r.ImageMinSize != nil && *r.ImageMinSize < 0 {
		return fmt.Errorf("%w: image_min_size must not be negative", ErrInvalidRequest)
	}
	return nil
}

func (r OCRRequest) validate() error {
	if r.Source == nil {
		return fmt.Errorf("%w: source is required", ErrInvalidRequest)
	}
	if err := r.validateOptions(); err != nil {
		return err
	}
	switch source := r.Source.(type) {
	case UploadedFile:
		if strings.TrimSpace(source.FileID) == "" {
			return fmt.Errorf("%w: uploaded file id is required", ErrInvalidRequest)
		}
	case DocumentURL:
		if strings.TrimSpace(source.URL) == "" {
			return fmt.Errorf("%w: document URL is required", ErrInvalidRequest)
		}
	case ImageURL:
		if strings.TrimSpace(source.URL) == "" {
			return fmt.Errorf("%w: image URL is required", ErrInvalidRequest)
		}
	case LocalFile:
		if strings.TrimSpace(source.Name) == "" {
			return fmt.Errorf("%w: local file name is required", ErrInvalidRequest)
		}
		if source.Reader == nil {
			return fmt.Errorf("%w: local file reader is required", ErrInvalidRequest)
		}
	default:
		return fmt.Errorf("%w: unsupported OCR source %T", ErrInvalidRequest, r.Source)
	}
	return nil
}

func (r OCRRequest) document(fileID string) ocrDocument {
	switch source := r.Source.(type) {
	case UploadedFile:
		return ocrDocument{Type: documentTypeFile, FileID: source.FileID}
	case DocumentURL:
		return ocrDocument{Type: documentTypeDocumentURL, DocumentURL: source.URL, DocumentName: source.Name}
	case ImageURL:
		return ocrDocument{Type: documentTypeImageURL, ImageURL: source.URL}
	default:
		return ocrDocument{Type: documentTypeFile, FileID: fileID}
	}
}

type OCRCleanupError struct {
	FileID string
	Err    error
}

func (e *OCRCleanupError) Error() string {
	return fmt.Sprintf("mistral: cleanup OCR file %q: %v", e.FileID, e.Err)
}

func (e *OCRCleanupError) Unwrap() error { return e.Err }

// OCR runs OCR for any supported source. Local files are uploaded, processed,
// and deleted; uploaded-file sources remain owned by the caller.
func (c *Client) OCR(ctx context.Context, req OCRRequest) (OCRResponse, error) {
	if err := req.validate(); err != nil {
		return OCRResponse{}, err
	}
	fileID := ""
	if source, ok := req.Source.(LocalFile); ok {
		file, err := c.UploadFile(ctx, UploadFileRequest{
			Filename: source.Name, Content: source.Reader, ContentType: source.ContentType, Purpose: FilePurposeOCR,
		})
		if err != nil {
			return OCRResponse{}, err
		}
		fileID = file.ID
	}

	body := ocrBody(req, req.document(fileID))
	var response OCRResponse
	ocrErr := c.postJSON(ctx, pathOCR, body, &response)

	if fileID == "" {
		if ocrErr != nil {
			return OCRResponse{}, fmt.Errorf("mistral: ocr: %w", ocrErr)
		}
		return response, nil
	}

	cleanupCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), 30*time.Second)
	cleanupErr := c.DeleteFile(cleanupCtx, fileID)
	cancel()
	if cleanupErr != nil {
		cleanupErr = &OCRCleanupError{FileID: fileID, Err: cleanupErr}
	}
	if ocrErr != nil {
		primary := fmt.Errorf("mistral: ocr: %w", ocrErr)
		if cleanupErr != nil {
			return OCRResponse{}, errors.Join(primary, cleanupErr)
		}
		return OCRResponse{}, primary
	}
	if cleanupErr != nil {
		return response, cleanupErr
	}
	return response, nil
}

// ocrBody maps a validated request onto the wire body. doc is passed separately
// because a LocalFile source resolves to a file id only after it is uploaded.
func ocrBody(r OCRRequest, doc ocrDocument) ocrRequestBody {
	model := r.Model
	if model == "" {
		model = DefaultOCRModel
	}
	body := ocrRequestBody{
		Model: model, Document: doc, Pages: r.Pages, ID: r.ID,
		TableFmt: r.TableFormat, Include: r.IncludeImageBase64, ImageLimit: r.ImageLimit,
		ImageMinSize: r.ImageMinSize, BBoxAnnotationFormat: r.BBoxAnnotationFormat,
		IncludeBlocks: r.IncludeBlocks, ConfidenceScoresGranularity: r.ConfidenceScoresGranularity,
		ExtractH: r.ExtractHeader, ExtractF: r.ExtractFooter,
		DocumentAnnotationFormat: r.DocumentAnnotationFormat,
	}
	if r.DocumentAnnotationPrompt != "" {
		body.DocumentAnnotationPrompt = new(r.DocumentAnnotationPrompt)
	}
	return body
}
