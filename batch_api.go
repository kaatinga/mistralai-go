package mistralai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/url"
	"strconv"
	"strings"
	"time"
)

const filePurposeBatch = "batch"

// Batch endpoints accepted by CreateBatchJobRequest.Endpoint. Any string is
// allowed (forward compatibility); these constants document the set supported
// by the Mistral Batch API.
const (
	BatchEndpointChatCompletions     = "/v1/chat/completions"
	BatchEndpointEmbeddings          = "/v1/embeddings"
	BatchEndpointFIMCompletions      = "/v1/fim/completions"
	BatchEndpointModerations         = "/v1/moderations"
	BatchEndpointChatModerations     = "/v1/chat/moderations"
	BatchEndpointOCR                 = "/v1/ocr"
	BatchEndpointClassifications     = "/v1/classifications"
	BatchEndpointChatClassifications = "/v1/chat/classifications"
	BatchEndpointConversations       = "/v1/conversations"
	BatchEndpointAudioTranscriptions = "/v1/audio/transcriptions"
)

// BatchStatus is a batch job lifecycle state.
type BatchStatus string

const (
	BatchStatusQueued                BatchStatus = "QUEUED"
	BatchStatusRunning               BatchStatus = "RUNNING"
	BatchStatusSuccess               BatchStatus = "SUCCESS"
	BatchStatusFailed                BatchStatus = "FAILED"
	BatchStatusTimeoutExceeded       BatchStatus = "TIMEOUT_EXCEEDED"
	BatchStatusCancellationRequested BatchStatus = "CANCELLATION_REQUESTED"
	BatchStatusCancelled             BatchStatus = "CANCELLED"
)

// IsTerminal reports whether no further status transitions are expected.
// CANCELLATION_REQUESTED is non-terminal: it transitions to CANCELLED.
func (s BatchStatus) IsTerminal() bool {
	switch s {
	case BatchStatusSuccess, BatchStatusFailed, BatchStatusTimeoutExceeded, BatchStatusCancelled:
		return true
	default:
		return false
	}
}

// BatchJob is a batch job record (Mistral BatchJobOut).
type BatchJob struct {
	ID                string            `json:"id"`
	Object            string            `json:"object"` // "batch"
	Status            BatchStatus       `json:"status"`
	Endpoint          string            `json:"endpoint"`
	Model             *string           `json:"model"`
	InputFiles        []string          `json:"input_files"`
	Metadata          map[string]string `json:"metadata,omitempty"`
	CreatedAt         int64             `json:"created_at"`
	StartedAt         *int64            `json:"started_at"`
	CompletedAt       *int64            `json:"completed_at"`
	TotalRequests     int               `json:"total_requests"`
	SucceededRequests int               `json:"succeeded_requests"`
	FailedRequests    int               `json:"failed_requests"`
	CompletedRequests int               `json:"completed_requests"`
	Errors            []BatchJobError   `json:"errors,omitempty"`
	OutputFile        *string           `json:"output_file"`
	ErrorFile         *string           `json:"error_file"`
	Outputs           json.RawMessage   `json:"outputs,omitempty"`
	AgentID           *string           `json:"agent_id,omitempty"`
}

// BatchJobError is one job-level error reported on a batch job.
type BatchJobError struct {
	Message string `json:"message"`
}

// BatchJobList is the list batch jobs API response.
type BatchJobList struct {
	Object string     `json:"object"` // "list"
	Data   []BatchJob `json:"data"`
	Total  int        `json:"total"`
}

// CreateBatchJobRequest is the body for POST /v1/batch/jobs.
type CreateBatchJobRequest struct {
	// Endpoint is the target API endpoint (required). Use a BatchEndpoint* constant.
	Endpoint string
	// InputFiles are JSONL file ids (see UploadBatchInput). At least one is required.
	InputFiles []string
	// Model is required by the Batch API at job level (per-entry body model is optional then).
	// When empty, CreateBatchJob defaults by endpoint (OCR → DefaultOCRModel, chat → DefaultChatModel).
	Model string
	// Metadata is arbitrary string metadata stored on the job.
	Metadata map[string]string
	// TimeoutHours bounds job runtime; 0 omits the field (API default is 24).
	TimeoutHours int
}

type createBatchJobBody struct {
	Endpoint     string            `json:"endpoint"`
	InputFiles   []string          `json:"input_files,omitempty"`
	Model        string            `json:"model,omitempty"`
	Metadata     map[string]string `json:"metadata,omitempty"`
	TimeoutHours int               `json:"timeout_hours,omitempty"`
}

func (r CreateBatchJobRequest) validate() error {
	if strings.TrimSpace(r.Endpoint) == "" {
		return errors.New("mistral: endpoint is required")
	}
	if len(r.InputFiles) == 0 {
		return errors.New("mistral: at least one input file is required")
	}
	for i, f := range r.InputFiles {
		if strings.TrimSpace(f) == "" {
			return fmt.Errorf("mistral: input file %d is empty", i)
		}
	}
	if r.TimeoutHours < 0 {
		return errors.New("mistral: timeout_hours must not be negative")
	}
	if _, err := r.jobModel(); err != nil {
		return err
	}
	return nil
}

func defaultBatchJobModel(endpoint string) string {
	switch endpoint {
	case BatchEndpointOCR:
		return DefaultOCRModel
	case BatchEndpointChatCompletions:
		return DefaultChatModel
	default:
		return ""
	}
}

func (r CreateBatchJobRequest) jobModel() (string, error) {
	if m := strings.TrimSpace(r.Model); m != "" {
		return m, nil
	}
	if m := defaultBatchJobModel(r.Endpoint); m != "" {
		return m, nil
	}
	return "", errors.New("mistral: model is required")
}

// ListBatchJobsRequest filters and paginates GET /v1/batch/jobs.
type ListBatchJobsRequest struct {
	Page        *int
	PageSize    *int
	CreatedByMe *bool
}

func (r ListBatchJobsRequest) queryValues() url.Values {
	q := url.Values{}
	if r.Page != nil {
		q.Set("page", strconv.Itoa(*r.Page))
	}
	if r.PageSize != nil {
		q.Set("page_size", strconv.Itoa(*r.PageSize))
	}
	if r.CreatedByMe != nil {
		q.Set("created_by_me", strconv.FormatBool(*r.CreatedByMe))
	}
	return q
}

// CreateBatchJob creates an async batch job (POST /v1/batch/jobs).
func (c *Client) CreateBatchJob(ctx context.Context, req CreateBatchJobRequest) (BatchJob, error) {
	if err := req.validate(); err != nil {
		return BatchJob{}, err
	}
	model, err := req.jobModel()
	if err != nil {
		return BatchJob{}, err
	}
	body := createBatchJobBody{
		Endpoint:     req.Endpoint,
		InputFiles:   req.InputFiles,
		Model:        model,
		Metadata:     req.Metadata,
		TimeoutHours: req.TimeoutHours,
	}
	var job BatchJob
	if err := c.postJSON(ctx, "/v1/batch/jobs", body, &job); err != nil {
		return BatchJob{}, fmt.Errorf("mistral: create batch job: %w", err)
	}
	return job, nil
}

// ListBatchJobs lists batch jobs for the API key (GET /v1/batch/jobs).
func (c *Client) ListBatchJobs(ctx context.Context, req ListBatchJobsRequest) (BatchJobList, error) {
	var list BatchJobList
	if err := c.getJSON(ctx, "/v1/batch/jobs", req.queryValues(), &list); err != nil {
		return BatchJobList{}, fmt.Errorf("mistral: list batch jobs: %w", err)
	}
	return list, nil
}

// GetBatchJob fetches one batch job (GET /v1/batch/jobs/{job_id}).
func (c *Client) GetBatchJob(ctx context.Context, jobID string) (BatchJob, error) {
	if strings.TrimSpace(jobID) == "" {
		return BatchJob{}, errors.New("mistral: job id is required")
	}
	var job BatchJob
	if err := c.getJSON(ctx, "/v1/batch/jobs/"+url.PathEscape(jobID), nil, &job); err != nil {
		return BatchJob{}, fmt.Errorf("mistral: get batch job: %w", err)
	}
	return job, nil
}

// CancelBatchJob requests cancellation (POST /v1/batch/jobs/{job_id}/cancel).
func (c *Client) CancelBatchJob(ctx context.Context, jobID string) (BatchJob, error) {
	if strings.TrimSpace(jobID) == "" {
		return BatchJob{}, errors.New("mistral: job id is required")
	}
	var job BatchJob
	if err := c.postJSON(ctx, "/v1/batch/jobs/"+url.PathEscape(jobID)+"/cancel", nil, &job); err != nil {
		return BatchJob{}, fmt.Errorf("mistral: cancel batch job: %w", err)
	}
	return job, nil
}

// WaitForBatchJob polls GetBatchJob every pollInterval (default 5s) until the
// job reaches a terminal status (SUCCESS, FAILED, TIMEOUT_EXCEEDED, CANCELLED)
// or ctx is done. It polls once immediately, then on each tick. Wrap ctx with
// WithTimeout to bound the total wait.
//
// A terminal-but-failed job (FAILED, TIMEOUT_EXCEEDED) is returned without a Go
// error; inspect BatchJob.Errors and download BatchJob.ErrorFile to handle it.
func WaitForBatchJob(ctx context.Context, c BatchJobGetter, jobID string, pollInterval time.Duration) (BatchJob, error) {
	if pollInterval <= 0 {
		pollInterval = 5 * time.Second
	}
	ticker := time.NewTicker(pollInterval)
	defer ticker.Stop()
	for {
		job, err := c.GetBatchJob(ctx, jobID)
		if err != nil {
			return BatchJob{}, err
		}
		if job.Status.IsTerminal() {
			return job, nil
		}
		select {
		case <-ctx.Done():
			return BatchJob{}, ctx.Err()
		case <-ticker.C:
		}
	}
}
