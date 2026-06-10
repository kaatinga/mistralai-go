package mistralai

import (
	"context"
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
	// DefaultOCRModel is used when OCRRequest.Model is empty. It is also the
	// job-level model CreateBatchJob defaults to for the OCR batch endpoint.
	DefaultOCRModel = "mistral-ocr-latest"

	// FilePurposeOCR and FilePurposeBatch are the purposes accepted by
	// UploadFile (UploadFileRequest.Purpose).
	FilePurposeOCR   = "ocr"
	FilePurposeBatch = "batch"

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

// WithBaseURL overrides DefaultBaseURL, e.g. to route requests through a
// proxy or API gateway, or to point tests at a local server.
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
		return nil, fmt.Errorf("%w: API key is required", ErrInvalidRequest)
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
// Purpose defaults to FilePurposeOCR; use FilePurposeBatch for batch inputs.
func (c *Client) UploadFile(ctx context.Context, req UploadFileRequest) (string, error) {
	if strings.TrimSpace(req.Filename) == "" {
		return "", fmt.Errorf("%w: filename is required", ErrInvalidRequest)
	}
	if req.Content == nil {
		return "", fmt.Errorf("%w: content is required", ErrInvalidRequest)
	}
	purpose := strings.TrimSpace(req.Purpose)
	if purpose == "" {
		purpose = FilePurposeOCR
	}
	return c.uploadFile(ctx, req.Filename, req.Content, req.ContentType, purpose)
}

// DeleteFile removes an uploaded file (DELETE /v1/files/{file_id}).
func (c *Client) DeleteFile(ctx context.Context, fileID string) error {
	if strings.TrimSpace(fileID) == "" {
		return fmt.Errorf("%w: file id is required", ErrInvalidRequest)
	}
	return c.doJSON(ctx, http.MethodDelete, "/v1/files/"+url.PathEscape(fileID), nil, nil, nil)
}

func (r OCRRequest) validate() error {
	if strings.TrimSpace(r.Filename) == "" {
		return fmt.Errorf("%w: filename is required", ErrInvalidRequest)
	}
	if r.Content == nil {
		return fmt.Errorf("%w: content is required", ErrInvalidRequest)
	}
	return r.options().validate()
}

func (r OCRRequest) options() OCROptions {
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

func (c *Client) processOCR(ctx context.Context, req OCRRequest) (*OCRResponse, error) {
	fileID, err := c.uploadFile(ctx, req.Filename, req.Content, req.ContentType, FilePurposeOCR)
	if err != nil {
		return nil, fmt.Errorf("mistral: upload file: %w", err)
	}
	defer func() {
		cctx, cancel := context.WithTimeout(context.WithoutCancel(ctx), 30*time.Second)
		defer cancel()
		_ = c.DeleteFile(cctx, fileID)
	}()

	body := ocrBody(req.Model, ocrDocument{Type: documentTypeFile, FileID: fileID}, req.options())

	var resp OCRResponse
	if err = c.postJSON(ctx, "/v1/ocr", body, &resp); err != nil {
		return nil, fmt.Errorf("mistral: ocr: %w", err)
	}
	return &resp, nil
}
