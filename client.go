package mistralai

import (
	"context"
	"fmt"
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
	apiKey         string
	baseURL        *url.URL
	http           *http.Client
	retryPolicy    RetryPolicy
	userAgentValue string
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

// ClientOption configures optional NewClient settings. API key is always required separately.
type ClientOption func(*clientOptions)

type clientOptions struct {
	baseURL     string
	httpClient  *http.Client
	retryPolicy *RetryPolicy
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

// WithMaxRetries sets how many times a replay-safe request is retried after
// the initial attempt, for retryable failures (429/5xx and transport errors).
// Unsafe requests remain single-attempt. 0 disables retries; negative values
// are treated as 0. Default is 4 retries (5 attempts in total).
func WithMaxRetries(n int) ClientOption {
	return func(o *clientOptions) {
		policy := defaultRetryPolicy()
		policy.MaxAttempts = max(n, 0) + 1
		o.retryPolicy = &policy
	}
}

// WithRetryPolicy configures retries for replay-safe requests. Unsafe requests
// are never retried by the default transport policy.
func WithRetryPolicy(policy RetryPolicy) ClientOption {
	return func(o *clientOptions) { o.retryPolicy = &policy }
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
	parsedBaseURL, err := url.Parse(baseURL)
	if err != nil || parsedBaseURL.Scheme == "" || parsedBaseURL.Host == "" {
		return nil, fmt.Errorf("%w: invalid base URL %q", ErrInvalidRequest, baseURL)
	}
	httpClient := o.httpClient
	if httpClient == nil {
		httpClient = &http.Client{Timeout: 10 * time.Minute}
	}
	retryPolicy := defaultRetryPolicy()
	if o.retryPolicy != nil {
		retryPolicy = *o.retryPolicy
	}
	if err := retryPolicy.validate(); err != nil {
		return nil, err
	}

	return &Client{
		apiKey:         apiKey,
		baseURL:        parsedBaseURL,
		http:           httpClient,
		retryPolicy:    retryPolicy,
		userAgentValue: "mistralai-go/" + moduleVersion(),
	}, nil
}

// DeleteFile removes an uploaded file (DELETE /v1/files/{file_id}).
func (c *Client) DeleteFile(ctx context.Context, fileID string) error {
	id, err := pathID("file id", fileID)
	if err != nil {
		return err
	}
	return c.doJSON(ctx, http.MethodDelete, pathFiles+"/"+id, nil, nil, nil)
}
