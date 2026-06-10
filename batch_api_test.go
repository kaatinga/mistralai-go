package mistralai

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"
	"time"
)

func TestCreateBatchJob(t *testing.T) {
	const wantJobID = "b1f2c3d4-0000-1111-2222-333344445555"

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/batch/jobs" || r.Method != http.MethodPost {
			http.NotFound(w, r)
			return
		}
		if r.Header.Get("Authorization") != "Bearer test-key" {
			t.Errorf("authorization %q", r.Header.Get("Authorization"))
		}
		var body createBatchJobBody
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatal(err)
		}
		if body.Endpoint != BatchEndpointChatCompletions {
			t.Errorf("endpoint = %q", body.Endpoint)
		}
		if !reflect.DeepEqual(body.InputFiles, []string{"file-abc"}) {
			t.Errorf("input_files = %v", body.InputFiles)
		}
		if body.TimeoutHours != 12 {
			t.Errorf("timeout_hours = %d", body.TimeoutHours)
		}
		if body.Metadata["job"] != "test" {
			t.Errorf("metadata = %v", body.Metadata)
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"id":                 wantJobID,
			"object":             "batch",
			"status":             "QUEUED",
			"endpoint":           BatchEndpointChatCompletions,
			"input_files":        []string{"file-abc"},
			"created_at":         1700000000,
			"total_requests":     2,
			"succeeded_requests": 0,
			"failed_requests":    0,
			"completed_requests": 0,
			"output_file":        nil,
			"started_at":         nil,
		})
	}))
	defer srv.Close()

	cl, err := NewClient("test-key", WithBaseURL(srv.URL))
	if err != nil {
		t.Fatal(err)
	}

	job, err := cl.CreateBatchJob(context.Background(), CreateBatchJobRequest{
		Endpoint:     BatchEndpointChatCompletions,
		InputFiles:   []string{"file-abc"},
		TimeoutHours: 12,
		Metadata:     map[string]string{"job": "test"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if job.ID != wantJobID {
		t.Errorf("id = %q", job.ID)
	}
	if job.Status != BatchStatusQueued {
		t.Errorf("status = %q", job.Status)
	}
	if job.TotalRequests != 2 {
		t.Errorf("total_requests = %d", job.TotalRequests)
	}
	if job.OutputFile != nil || job.StartedAt != nil {
		t.Errorf("expected nil output_file/started_at, got %v / %v", job.OutputFile, job.StartedAt)
	}
}

func TestCreateBatchJob_OCR_defaultsModel(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/batch/jobs" || r.Method != http.MethodPost {
			http.NotFound(w, r)
			return
		}
		var body createBatchJobBody
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatal(err)
		}
		if body.Endpoint != BatchEndpointOCR {
			t.Errorf("endpoint = %q", body.Endpoint)
		}
		if body.Model != DefaultOCRModel {
			t.Errorf("model = %q, want %q", body.Model, DefaultOCRModel)
		}
		_ = json.NewEncoder(w).Encode(BatchJob{ID: "job-ocr", Status: BatchStatusQueued, Endpoint: BatchEndpointOCR})
	}))
	defer srv.Close()

	cl, err := NewClient("test-key", WithBaseURL(srv.URL))
	if err != nil {
		t.Fatal(err)
	}

	if _, err := cl.CreateBatchJob(context.Background(), CreateBatchJobRequest{
		Endpoint:   BatchEndpointOCR,
		InputFiles: []string{"file-abc"},
	}); err != nil {
		t.Fatal(err)
	}
}

func TestCreateBatchJob_validate(t *testing.T) {
	cl, err := NewClient("test-key", WithBaseURL("http://example.invalid"))
	if err != nil {
		t.Fatal(err)
	}

	cases := map[string]CreateBatchJobRequest{
		"missing endpoint": {InputFiles: []string{"f"}},
		"no input files":   {Endpoint: BatchEndpointChatCompletions},
		"empty input file": {Endpoint: BatchEndpointChatCompletions, InputFiles: []string{" "}},
		"negative timeout": {Endpoint: BatchEndpointChatCompletions, InputFiles: []string{"f"}, TimeoutHours: -1},
		"missing model":    {Endpoint: BatchEndpointEmbeddings, InputFiles: []string{"f"}},
	}
	for name, req := range cases {
		t.Run(name, func(t *testing.T) {
			if _, err := cl.CreateBatchJob(context.Background(), req); err == nil {
				t.Fatal("expected validation error")
			}
		})
	}
}

func TestListBatchJobs(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/batch/jobs" || r.Method != http.MethodGet {
			http.NotFound(w, r)
			return
		}
		q := r.URL.Query()
		if q.Get("page") != "1" || q.Get("page_size") != "50" || q.Get("created_by_me") != "true" {
			t.Errorf("query = %v", q)
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"object": "list",
			"total":  1,
			"data": []map[string]any{{
				"id":          "job-1",
				"object":      "batch",
				"status":      "RUNNING",
				"endpoint":    BatchEndpointChatCompletions,
				"input_files": []string{"file-abc"},
				"created_at":  1700000000,
			}},
		})
	}))
	defer srv.Close()

	cl, err := NewClient("test-key", WithBaseURL(srv.URL))
	if err != nil {
		t.Fatal(err)
	}

	list, err := cl.ListBatchJobs(context.Background(), ListBatchJobsRequest{
		Page:        new(1),
		PageSize:    new(50),
		CreatedByMe: new(true),
	})
	if err != nil {
		t.Fatal(err)
	}
	if list.Total != 1 || len(list.Data) != 1 || list.Data[0].ID != "job-1" {
		t.Fatalf("list = %+v", list)
	}
}

func TestListBatchJobsRequest_queryValues(t *testing.T) {
	if q := (ListBatchJobsRequest{}).queryValues(); len(q) != 0 {
		t.Errorf("empty request produced %v", q)
	}
	q := ListBatchJobsRequest{Page: new(2), PageSize: new(10), CreatedByMe: new(false)}.queryValues()
	if q.Get("page") != "2" || q.Get("page_size") != "10" || q.Get("created_by_me") != "false" {
		t.Errorf("query = %v", q)
	}
}

func TestGetBatchJob(t *testing.T) {
	const wantJobID = "job with spaces"
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/batch/jobs/"+wantJobID || r.Method != http.MethodGet {
			http.NotFound(w, r)
			return
		}
		out := "file-out"
		_ = json.NewEncoder(w).Encode(BatchJob{
			ID:            wantJobID,
			Object:        "batch",
			Status:        BatchStatusSuccess,
			Endpoint:      BatchEndpointChatCompletions,
			InputFiles:    []string{"file-abc"},
			OutputFile:    &out,
			TotalRequests: 2,
		})
	}))
	defer srv.Close()

	cl, err := NewClient("test-key", WithBaseURL(srv.URL))
	if err != nil {
		t.Fatal(err)
	}

	job, err := cl.GetBatchJob(context.Background(), wantJobID)
	if err != nil {
		t.Fatal(err)
	}
	if job.Status != BatchStatusSuccess || job.OutputFile == nil || *job.OutputFile != "file-out" {
		t.Fatalf("job = %+v", job)
	}

	if _, err := cl.GetBatchJob(context.Background(), "  "); err == nil {
		t.Fatal("expected validation error for empty job id")
	}
}

func TestCancelBatchJob(t *testing.T) {
	const wantJobID = "job-1"
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/batch/jobs/"+wantJobID+"/cancel" || r.Method != http.MethodPost {
			http.NotFound(w, r)
			return
		}
		_ = json.NewEncoder(w).Encode(BatchJob{
			ID:     wantJobID,
			Status: BatchStatusCancellationRequested,
		})
	}))
	defer srv.Close()

	cl, err := NewClient("test-key", WithBaseURL(srv.URL))
	if err != nil {
		t.Fatal(err)
	}

	job, err := cl.CancelBatchJob(context.Background(), wantJobID)
	if err != nil {
		t.Fatal(err)
	}
	if job.Status != BatchStatusCancellationRequested {
		t.Errorf("status = %q", job.Status)
	}

	if _, err := cl.CancelBatchJob(context.Background(), ""); err == nil {
		t.Fatal("expected validation error for empty job id")
	}
}

func TestBatchStatus_IsTerminal(t *testing.T) {
	terminal := []BatchStatus{BatchStatusSuccess, BatchStatusFailed, BatchStatusTimeoutExceeded, BatchStatusCancelled}
	for _, s := range terminal {
		if !s.IsTerminal() {
			t.Errorf("%q should be terminal", s)
		}
	}
	nonTerminal := []BatchStatus{BatchStatusQueued, BatchStatusRunning, BatchStatusCancellationRequested}
	for _, s := range nonTerminal {
		if s.IsTerminal() {
			t.Errorf("%q should not be terminal", s)
		}
	}
}

func TestWaitForBatchJob(t *testing.T) {
	t.Run("polls until terminal", func(t *testing.T) {
		calls := 0
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			calls++
			status := BatchStatusRunning
			if calls >= 3 {
				status = BatchStatusSuccess
			}
			_ = json.NewEncoder(w).Encode(BatchJob{ID: "job-1", Status: status})
		}))
		defer srv.Close()

		cl, err := NewClient("test-key", WithBaseURL(srv.URL))
		if err != nil {
			t.Fatal(err)
		}

		job, err := WaitForBatchJob(context.Background(), cl, "job-1", time.Millisecond)
		if err != nil {
			t.Fatal(err)
		}
		if job.Status != BatchStatusSuccess {
			t.Errorf("status = %q", job.Status)
		}
		if calls < 3 {
			t.Errorf("expected at least 3 polls, got %d", calls)
		}
	})

	t.Run("already terminal returns on first poll", func(t *testing.T) {
		calls := 0
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			calls++
			_ = json.NewEncoder(w).Encode(BatchJob{ID: "job-1", Status: BatchStatusFailed})
		}))
		defer srv.Close()

		cl, err := NewClient("test-key", WithBaseURL(srv.URL))
		if err != nil {
			t.Fatal(err)
		}

		job, err := WaitForBatchJob(context.Background(), cl, "job-1", time.Hour)
		if err != nil {
			t.Fatalf("failed terminal status should not error: %v", err)
		}
		if job.Status != BatchStatusFailed || calls != 1 {
			t.Errorf("status = %q calls = %d", job.Status, calls)
		}
	})

	t.Run("respects context cancellation", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			_ = json.NewEncoder(w).Encode(BatchJob{ID: "job-1", Status: BatchStatusRunning})
		}))
		defer srv.Close()

		cl, err := NewClient("test-key", WithBaseURL(srv.URL))
		if err != nil {
			t.Fatal(err)
		}

		ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
		defer cancel()
		if _, err := WaitForBatchJob(ctx, cl, "job-1", 5*time.Millisecond); err == nil {
			t.Fatal("expected context error")
		}
	})
}
