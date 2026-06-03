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
	DefaultOCRModel  = "mistral-ocr-latest"
	filePurposeOCR   = "ocr"
	documentTypeFile = "file"
)

// Client is the port for Mistral cloud API access (dependency inversion).
type Client interface {
	// OCR uploads the document via POST /v1/files, then POST /v1/ocr; blocks until 200 or error.
	OCR(ctx context.Context, req OCRRequest) (OCRResponse, error)
	// Chat runs POST /v1/chat/completions with a single user turn; blocks until 200 or error.
	Chat(ctx context.Context, req ChatRequest) (ChatResponse, error)
	// ChatCompletion runs POST /v1/chat/completions with full message control.
	ChatCompletion(ctx context.Context, req ChatCompletionRequest) (ChatCompletionResponse, error)
	// Embeddings runs POST /v1/embeddings.
	Embeddings(ctx context.Context, req EmbeddingRequest) (EmbeddingResponse, error)
	// ListModels returns models available to the API key (GET /v1/models).
	ListModels(ctx context.Context) (ModelList, error)
	// UploadFile uploads file bytes and returns the API file id.
	UploadFile(ctx context.Context, req UploadFileRequest) (string, error)
	// ListFiles returns uploaded files for the API key (GET /v1/files).
	ListFiles(ctx context.Context, req ListFilesRequest) (FileList, error)
	// DeleteFile removes an uploaded file (DELETE /v1/files/{file_id}).
	DeleteFile(ctx context.Context, fileID string) error
	// DownloadFile downloads raw file content (GET /v1/files/{file_id}/content),
	// e.g. a batch job's output or error JSONL file.
	DownloadFile(ctx context.Context, fileID string) ([]byte, error)
	// UploadBatchInput builds a JSONL input file from entries and uploads it with
	// purpose "batch"; returns the file id for CreateBatchJobRequest.InputFiles.
	UploadBatchInput(ctx context.Context, filename string, entries []BatchEntry) (string, error)
	// CreateBatchJob creates an async batch job (POST /v1/batch/jobs).
	CreateBatchJob(ctx context.Context, req CreateBatchJobRequest) (BatchJob, error)
	// ListBatchJobs lists batch jobs for the API key (GET /v1/batch/jobs).
	ListBatchJobs(ctx context.Context, req ListBatchJobsRequest) (BatchJobList, error)
	// GetBatchJob fetches one batch job (GET /v1/batch/jobs/{job_id}).
	GetBatchJob(ctx context.Context, jobID string) (BatchJob, error)
	// CancelBatchJob requests cancellation (POST /v1/batch/jobs/{job_id}/cancel).
	CancelBatchJob(ctx context.Context, jobID string) (BatchJob, error)
	// Close releases resources. Safe to call more than once.
	Close() error
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
	maxRetries int
}

// WithBaseURL overrides DefaultBaseURL (for tests).
func WithBaseURL(baseURL string) ClientOption {
	return func(o *clientOptions) { o.baseURL = baseURL }
}

// WithHTTPClient sets the HTTP client used for Mistral API calls.
func WithHTTPClient(httpClient *http.Client) ClientOption {
	return func(o *clientOptions) { o.httpClient = httpClient }
}

// WithMaxRetries sets retry attempts for retryable HTTP status codes (default 5).
func WithMaxRetries(n int) ClientOption {
	return func(o *clientOptions) { o.maxRetries = n }
}

type client struct {
	apiKey     string
	baseURL    string
	http       *http.Client
	maxRetries int
}

// NewClient returns a synchronous HTTP client for the Mistral API.
// apiKey is required; an empty key returns an error.
func NewClient(apiKey string, opts ...ClientOption) (Client, error) {
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
	maxRetries := o.maxRetries
	if maxRetries <= 0 {
		maxRetries = defaultMaxRetries
	}

	return &client{
		apiKey:     apiKey,
		baseURL:    baseURL,
		http:       httpClient,
		maxRetries: maxRetries,
	}, nil
}

func (c *client) Close() error {
	return nil
}

func (c *client) OCR(ctx context.Context, req OCRRequest) (OCRResponse, error) {
	if err := req.validate(); err != nil {
		return OCRResponse{}, err
	}
	resp, err := c.processOCR(ctx, req)
	if err != nil {
		return OCRResponse{}, err
	}
	return *resp, nil
}

func (c *client) UploadFile(ctx context.Context, req UploadFileRequest) (string, error) {
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

func (c *client) DeleteFile(ctx context.Context, fileID string) error {
	if strings.TrimSpace(fileID) == "" {
		return errors.New("mistral: file id is required")
	}
	return c.doJSON(ctx, http.MethodDelete, "/v1/files/"+url.PathEscape(fileID), nil, nil)
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

func (c *client) processOCR(ctx context.Context, req OCRRequest) (*OCRResponse, error) {
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
		Document: fileDocument{
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
