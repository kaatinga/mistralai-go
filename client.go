package mistralai

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"
)

const (
	// DefaultBaseURL is the Mistral cloud API origin.
	DefaultBaseURL = "https://api.mistral.ai"
	// DefaultOCRModel is used when OCRRequest.Model is empty.
	DefaultOCRModel         = "mistral-ocr-latest"
	filePurposeOCR          = "ocr"
	documentTypeFile        = "file"
	documentTypeDocumentURL = "document_url"
	documentTypeImageURL    = "image_url"
)

// Client is a synchronous HTTP client for the Mistral API. Construct it with
// NewClient. Methods are safe for concurrent use.
//
// Client is intentionally a concrete struct, not an interface, so new API
// endpoints can be added without breaking consumers. For dependency inversion
// or mocking, define a narrow interface with just the methods you use:
//
//	type ocrClient interface {
//		OCR(ctx context.Context, req mistralai.OCRRequest) (mistralai.OCRResponse, error)
//	}
//
// Helpers in this package that take a client accept such role interfaces
// (ChatCompleter, BatchJobGetter, OCRRunner), all satisfied by *Client.
type Client struct {
	apiKey      string
	baseURL     string
	http        *http.Client
	maxAttempts int
}

// ChatCompleter runs chat completions; satisfied by *Client. It is the
// dependency of ChatCompletionWithTools and ChatStructured.
type ChatCompleter interface {
	ChatCompletion(ctx context.Context, req ChatCompletionRequest) (ChatCompletionResponse, error)
}

// BatchJobGetter fetches batch jobs; satisfied by *Client. It is the
// dependency of WaitForBatchJob.
type BatchJobGetter interface {
	GetBatchJob(ctx context.Context, jobID string) (BatchJob, error)
}

// OCRRunner runs upload-based OCR; satisfied by *Client. It is the
// dependency of OCRStructured.
type OCRRunner interface {
	OCR(ctx context.Context, req OCRRequest) (OCRResponse, error)
}

// OCRRequest describes a document to OCR via file upload (not a URL).
// The client performs POST /v1/files (multipart, purpose=ocr), then POST /v1/ocr with the returned file id.
type OCRRequest struct {
	// Filename is sent in multipart upload (e.g. "invoice.pdf").
	Filename string
	// Content is the raw file bytes.
	Content io.Reader
	// ContentType is optional; when empty, application/octet-stream is used.
	ContentType string
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

// UploadFileRequest uploads a file to Mistral files API.
type UploadFileRequest struct {
	Filename    string
	Content     io.Reader
	ContentType string
	Purpose     string
}

// ClientOption configures optional NewClient settings. API key is always required separately.
type ClientOption func(*clientOptions)

type clientOptions struct {
	baseURL    string
	httpClient *http.Client
	maxRetries *int
}

// WithBaseURL overrides DefaultBaseURL (for tests).
func WithBaseURL(baseURL string) ClientOption {
	return func(o *clientOptions) { o.baseURL = baseURL }
}

// WithHTTPClient sets the HTTP client used for Mistral API calls.
func WithHTTPClient(httpClient *http.Client) ClientOption {
	return func(o *clientOptions) { o.httpClient = httpClient }
}

// WithMaxRetries sets how many times a failed request is retried after the
// initial attempt, for retryable failures (429/5xx and transport errors).
// 0 disables retries; negative values are treated as 0. Default is 4 retries
// (5 attempts in total).
func WithMaxRetries(n int) ClientOption {
	return func(o *clientOptions) { o.maxRetries = new(max(n, 0)) }
}

// NewClient returns a synchronous HTTP client for the Mistral API.
// apiKey is required; an empty key returns an error.
func NewClient(apiKey string, opts ...ClientOption) (*Client, error) {
	apiKey = strings.TrimSpace(apiKey)
	if apiKey == "" {
		return nil, errors.New("mistral: API key is required")
	}
	var o clientOptions
	for _, opt := range opts {
		opt(&o)
	}
	baseURL := o.baseURL
	if baseURL == "" {
		baseURL = DefaultBaseURL
	}
	httpClient := o.httpClient
	if httpClient == nil {
		httpClient = &http.Client{Timeout: 10 * time.Minute}
	}
	maxAttempts := defaultMaxAttempts
	if o.maxRetries != nil {
		maxAttempts = *o.maxRetries + 1
	}

	return &Client{
		apiKey:      apiKey,
		baseURL:     baseURL,
		http:        httpClient,
		maxAttempts: maxAttempts,
	}, nil
}

// Close releases resources. Safe to call more than once.
func (c *Client) Close() error {
	return nil
}

// OCR uploads the document via POST /v1/files, then runs POST /v1/ocr on the
// uploaded file id; the file is deleted afterwards (best effort).
func (c *Client) OCR(ctx context.Context, req OCRRequest) (OCRResponse, error) {
	if err := req.validate(); err != nil {
		return OCRResponse{}, err
	}
	resp, err := c.processOCR(ctx, req)
	if err != nil {
		return OCRResponse{}, err
	}
	return *resp, nil
}

// UploadFile uploads file bytes (POST /v1/files) and returns the API file id.
// Purpose defaults to "ocr".
func (c *Client) UploadFile(ctx context.Context, req UploadFileRequest) (string, error) {
	if strings.TrimSpace(req.Filename) == "" {
		return "", errors.New("mistral: filename is required")
	}
	if req.Content == nil {
		return "", errors.New("mistral: content is required")
	}
	purpose := strings.TrimSpace(req.Purpose)
	if purpose == "" {
		purpose = filePurposeOCR
	}
	return c.uploadFile(ctx, req.Filename, req.Content, req.ContentType, purpose)
}

// DeleteFile removes an uploaded file (DELETE /v1/files/{file_id}).
func (c *Client) DeleteFile(ctx context.Context, fileID string) error {
	if strings.TrimSpace(fileID) == "" {
		return errors.New("mistral: file id is required")
	}
	return c.doJSON(ctx, http.MethodDelete, "/v1/files/"+url.PathEscape(fileID), nil, nil, nil)
}

func (r OCRRequest) validate() error {
	if strings.TrimSpace(r.Filename) == "" {
		return errors.New("mistral: filename is required")
	}
	if r.Content == nil {
		return errors.New("mistral: content is required")
	}
	if r.DocumentAnnotationPrompt != "" && r.DocumentAnnotationFormat == nil {
		return errors.New("mistral: document_annotation_format is required with document_annotation_prompt")
	}
	return nil
}

func (c *Client) processOCR(ctx context.Context, req OCRRequest) (*OCRResponse, error) {
	fileID, err := c.uploadFile(ctx, req.Filename, req.Content, req.ContentType, filePurposeOCR)
	if err != nil {
		return nil, fmt.Errorf("mistral: upload file: %w", err)
	}
	defer func() {
		cctx, cancel := context.WithTimeout(context.WithoutCancel(ctx), 30*time.Second)
		defer cancel()
		_ = c.DeleteFile(cctx, fileID)
	}()

	model := req.Model
	if model == "" {
		model = DefaultOCRModel
	}

	body := ocrRequestBody{
		Model: model,
		Document: ocrDocument{
			Type:   documentTypeFile,
			FileID: fileID,
		},
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
	if err = c.postJSON(ctx, "/v1/ocr", body, &resp); err != nil {
		return nil, fmt.Errorf("mistral: ocr: %w", err)
	}
	return &resp, nil
}

// WithTimeout returns a child context with timeout unless ctx already has a sooner deadline.
func WithTimeout(ctx context.Context, timeout time.Duration) (context.Context, context.CancelFunc) {
	if deadline, ok := ctx.Deadline(); ok && time.Until(deadline) < timeout {
		return context.WithCancel(ctx)
	}
	return context.WithTimeout(ctx, timeout)
}
