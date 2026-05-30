# mistralai-go

Synchronous Go client for the [Mistral API](https://docs.mistral.ai/api). Each call blocks until Mistral returns HTTP 200 with the full JSON body (no job polling or background workers).

## API surface

| Method | HTTP | Use when |
|--------|------|----------|
| `OCR` | `POST /v1/files`, then `POST /v1/ocr` | Document OCR via file upload |
| `Chat` | `POST /v1/chat/completions` | Single user turn with optional system prompt and output format helpers |
| `ChatCompletion` | `POST /v1/chat/completions` | Full control: message list, temperature, `response_format`, etc. |
| `ListModels` | `GET /v1/models` | List models available to your API key |
| `UploadFile` | `POST /v1/files` | Upload a file; returns file id |
| `ListFiles` | `GET /v1/files` | List uploaded files (optional pagination and filters) |
| `DeleteFile` | `DELETE /v1/files/{file_id}` | Remove an uploaded file |
| `DownloadFile` | `GET /v1/files/{file_id}/content` | Download raw file content (e.g. batch results) |
| `UploadBatchInput` | `POST /v1/files` | Build a JSONL input file from typed entries and upload it (`purpose=batch`) |
| `CreateBatchJob` | `POST /v1/batch/jobs` | Create an async batch job over an uploaded input file |
| `ListBatchJobs` | `GET /v1/batch/jobs` | List batch jobs (optional pagination, `created_by_me`) |
| `GetBatchJob` | `GET /v1/batch/jobs/{job_id}` | Fetch one batch job |
| `CancelBatchJob` | `POST /v1/batch/jobs/{job_id}/cancel` | Request cancellation of a batch job |

JSON API calls (`Chat`, `ChatCompletion`, `ListModels`, `ListFiles`, `DeleteFile`, `DownloadFile`, the batch job calls, OCR after upload) retry on **429** and **5xx** with exponential backoff (default **5** attempts, context-aware). Configure with `WithMaxRetries`.

## Install

```bash
go get github.com/kaatinga/mistralai-go
```

## Usage

```go
package main

import (
	"context"
	"bytes"
	"log"
	"net/http"
	"os"
	"time"

	mistralai "github.com/kaatinga/mistralai-go"
)

func main() {
	cl, err := mistralai.NewClient(
		os.Getenv("MISTRAL_API_KEY"),
		mistralai.WithHTTPClient(&http.Client{Timeout: 120 * time.Second}),
	)
	if err != nil {
		log.Fatal(err)
	}
	defer cl.Close()

	ctx := context.Background()

	// OCR: POST /v1/files, then POST /v1/ocr
	pdf, _ := os.ReadFile("document.pdf")
	ocr, err := cl.OCR(ctx, mistralai.OCRRequest{
		Filename:    "document.pdf",
		Content:     bytes.NewReader(pdf),
		ContentType: "application/pdf",
	})
	if err != nil {
		log.Fatal(err)
	}
	log.Println(ocr.Pages[0].Markdown)

	// Chat — one system + one user message; optional markdown/json formatting
	chat, err := cl.Chat(ctx, mistralai.ChatRequest{
		Input:  "Summarize the document in one sentence.",
		Format: mistralai.OutputText,
	})
	if err != nil {
		log.Fatal(err)
	}
	log.Println(chat.Content)

	// ChatCompletion — multi-turn or custom parameters
	resp, err := cl.ChatCompletion(ctx, mistralai.ChatCompletionRequest{
		Model: "mistral-small-latest",
		Messages: []mistralai.ChatMessage{
			{Role: "system", Content: "You are concise."},
			{Role: "user", Content: "Hello"},
		},
		Temperature: new(0.7), // pointer so temperature 0 is distinguishable from unset
	})
	if err != nil {
		log.Fatal(err)
	}
	log.Println(resp.FirstChoiceContent())

	// ListModels
	models, err := cl.ListModels(ctx)
	if err != nil {
		log.Fatal(err)
	}
	for _, m := range models.Data {
		log.Println(m.ID)
	}
}
```

### High-level `Chat` vs `ChatCompletion`

- **`Chat`** builds a short message list from `Input` and optional `System`, and can request text, markdown, or JSON output via `Format` / `ResponseFormat`.
- **`ChatCompletion`** maps directly to the REST request body: any number of messages, `temperature`, `max_tokens`, `top_p`, and `response_format`. Use this for conversation history or app-specific control.

`Chat` is implemented on top of `ChatCompletion` internally.

### Client options

- `WithHTTPClient` — custom `http.Client` (default timeout **10 minutes**, suitable for OCR).
- `WithMaxRetries` — retries for retryable status codes (default **5**).
- `WithBaseURL` — override API origin (tests or proxies).

### JSON output with `Chat`

```go
resp, err := cl.Chat(ctx, mistralai.ChatRequest{
	Input:  `{"task":"translate","text":"hello"}`,
	Format: mistralai.OutputJSON,
})
var out map[string]string
if err = resp.JSON(&out); err != nil {
	log.Fatal(err)
}
```

For `json_schema`, set `ResponseFormat` on `ChatRequest`, `ChatCompletionRequest`, or `OCRRequest.DocumentAnnotationFormat`. Unmarshal OCR output with `OCRStructured[T]` or `DocumentAnnotationInto[T](resp)`.

### Ergonomic helpers

- `new(v)` (Go 1.26 built-in) — set optional pointer fields (`Temperature`, `TopP`, `ExtractHeader`, `IncludeImageBase64`, …) inline; `new(0.0)` is an explicit zero, a nil pointer is "unset".
- `JSONSchemaFormat(name, schema)` — build a strict `json_schema` `*ResponseFormat` without hand-writing the `ResponseFormat`/`JSONSchema` nesting.
- `TextMessage(role, text)` / `MultipartMessage(role, parts...)` and `TextPart`, `FilePart`, `ImageURLPart`, `DocumentURLPart` — build messages and multimodal content without magic strings.
- `RoleSystem`, `RoleUser`, `RoleAssistant` and `ResponseFormatText`, `ResponseFormatJSONObject`, `ResponseFormatJSONSchema` constants for the role and `response_format` type fields.

```go
req := mistralai.ChatCompletionRequest{
	Model: mistralai.ChatModelPixtralLargeLatest,
	Messages: []mistralai.ChatMessage{
		mistralai.TextMessage(mistralai.RoleSystem, "Classify the document."),
		mistralai.MultipartMessage(mistralai.RoleUser,
			mistralai.TextPart("What kind of document is this?"),
			mistralai.FilePart(fileID), // from cl.UploadFile
		),
	},
	Temperature:    new(0.0),
	ResponseFormat: mistralai.JSONSchemaFormat("doc_type", schema),
}
```

### Batch API

The Batch API runs one endpoint over many requests asynchronously (roughly half
the per-token cost, no per-request rate limits) and returns results as a JSONL
file. It is **not** for latency-sensitive calls — a job completes within its
`timeout_hours` (default 24), not immediately.

Build the input from the same typed request values you already use:
`ChatCompletionEntry` for `/v1/chat/completions`, `OCREntry` for `/v1/ocr`
(referencing an already-uploaded `file_id`), or `Entry(customID, body)` for any
other endpoint. Parse results back into the matching typed response with
`ParseBatchResults[T]`.

```go
entries := []mistralai.BatchEntry{
	mistralai.ChatCompletionEntry("0", mistralai.ChatCompletionRequest{
		Model:    mistralai.ChatModelMistralSmallLatest,
		Messages: []mistralai.ChatMessage{mistralai.TextMessage(mistralai.RoleUser, "Hello")},
	}),
	mistralai.ChatCompletionEntry("1", mistralai.ChatCompletionRequest{
		Model:    mistralai.ChatModelMistralSmallLatest,
		Messages: []mistralai.ChatMessage{mistralai.TextMessage(mistralai.RoleUser, "Bonjour")},
	}),
}

inputFileID, err := cl.UploadBatchInput(ctx, "batch.jsonl", entries)
if err != nil {
	log.Fatal(err)
}

job, err := cl.CreateBatchJob(ctx, mistralai.CreateBatchJobRequest{
	Endpoint:   mistralai.BatchEndpointChatCompletions,
	InputFiles: []string{inputFileID},
})
if err != nil {
	log.Fatal(err)
}

// Poll until terminal (SUCCESS / FAILED / TIMEOUT_EXCEEDED / CANCELLED).
waitCtx, cancel := mistralai.WithTimeout(ctx, time.Hour)
defer cancel()
job, err = mistralai.WaitForBatchJob(waitCtx, cl, job.ID, 10*time.Second)
if err != nil {
	log.Fatal(err)
}

if job.Status == mistralai.BatchStatusSuccess && job.OutputFile != nil {
	raw, err := cl.DownloadFile(ctx, *job.OutputFile)
	if err != nil {
		log.Fatal(err)
	}
	results, err := mistralai.ParseBatchResults[mistralai.ChatCompletionResponse](raw)
	if err != nil {
		log.Fatal(err)
	}
	for _, r := range results {
		if r.StatusCode == 200 {
			content, _ := r.Body.FirstChoiceContent()
			log.Println(r.CustomID, content)
		}
	}
}
```

`WaitForBatchJob` returns terminal-but-failed jobs **without** a Go error — inspect
`job.Errors` and download `job.ErrorFile` (also JSONL, parse with
`ParseBatchResults[T]`) to see per-request failures. The input file is not
deleted automatically; use `DeleteFile` when you no longer need it.

### Error handling

Non-200 responses return a typed `*APIError`. Inspect it with `errors.As` to
branch on the HTTP status, error type, or whether the client already retried it:

```go
resp, err := cl.Chat(ctx, req)
var apiErr *mistralai.APIError
if errors.As(err, &apiErr) {
	switch apiErr.StatusCode {
	case http.StatusUnauthorized:
		log.Fatal("bad API key")
	case http.StatusTooManyRequests:
		// apiErr.Retryable() == true; the client already exhausted WithMaxRetries
	}
}
```

## Comparison with other Go Mistral clients

There is no official Go SDK for Mistral. The table below compares `mistralai-go`
with the three most widely used community clients (state as of May 2026):

- [`gage-technologies/mistral-go`](https://github.com/Gage-Technologies/mistral-go) — the most popular and most-referenced client (e.g. used by `langchaingo`); MIT, last release v1.1.0 (Jun 2024).
- [`robertjkeck2/mistral-go`](https://github.com/robertjkeck2/mistral-go) — an early, minimal client; MIT, last release v0.1.1 (Jan 2024).
- [`onkyou/go-mistral`](https://pkg.go.dev/github.com/onkyou/go-mistral/mistral) — a newer, broad client with a service-oriented API; MIT, untagged.

| Capability | **mistralai-go** | gage-technologies | robertjkeck2 | onkyou/go-mistral |
|---|:---:|:---:|:---:|:---:|
| Chat completions | ✅ | ✅ | ✅ | ✅ |
| Streaming chat | ❌ | ✅ | ✅ | ✅ |
| Embeddings | ❌ | ✅ | ✅ | ✅ |
| FIM / code completion | ❌ | ✅ | ❌ | ❌ |
| Function / tool calling | ❌ | ✅ | ❌ | ✅ |
| Moderation / classification | ❌ | ❌ | ❌ | ✅ |
| **OCR + document annotation** | ✅ | ❌ | ❌ | ❌ |
| **File API (upload / list / delete / download)** | ✅ | ❌ | ❌ | ❌ |
| **Batch API** (typed entries + typed results) | ✅ | ❌ | ❌ | ❌ |
| List models | ✅ | ✅ | ✅ | ❌ |
| `response_format: json_object` | ✅ | ✅ | ❌ | ✅ |
| `response_format: json_schema` (strict) | ✅ | ❌ | ❌ | ✅ |
| Typed structured-output helpers (generics) | ✅ | ❌ | ❌ | partial |
| Multimodal content parts (file / image / document URL) | ✅ | ❌ | ❌ | ❌ |
| Built-in retries (429 / 5xx, backoff) | ✅ | ✅ | ❌ | ❌ |
| Typed API error (`errors.As`) | ✅ | partial | partial | ✅ |

### What `mistralai-go` does that the others do not

- **OCR** (`mistral-ocr-latest`) with the upload→OCR→cleanup flow handled for you, plus **document annotation** (`json_schema`-driven structured extraction) — no other client implements the OCR endpoint.
- A first-class **Files API** (`UploadFile`, `ListFiles` with pagination/filters, `DeleteFile`, `DownloadFile`).
- The **Batch API** with an ergonomic typed contract: build input from `ChatCompletionRequest`/OCR/arbitrary bodies (`UploadBatchInput`), create/poll/cancel jobs (`WaitForBatchJob`), and parse results straight into `ChatCompletionResponse`/`OCRResponse` (`ParseBatchResults[T]`) — no hand-rolled JSONL.
- Strict **`json_schema`** structured output with generic helpers (`OCRStructured[T]`, `ChatStructured[T]`, `DocumentAnnotationInto[T]`).
- **Multimodal** message parts (`FilePart`, `ImageURLPart`, `DocumentURLPart`) for vision/document chat.

### What `mistralai-go` does NOT do (yet)

If you need any of the following, one of the clients above will serve you better today:

- **Streaming responses** — `mistralai-go` is fully synchronous (blocks until the complete JSON body returns). All three other clients support streamed chat. The OCR-first design assumes a single blocking call.
- **Embeddings** — not implemented. Available in all three other clients (`mistral-embed`).
- **FIM / code completion** (Codestral) — available in `gage-technologies`.
- **Function / tool calling** — available in `gage-technologies` and `onkyou/go-mistral`.
- **Moderation and classification** — available only in `onkyou/go-mistral`.
- **Agents and fine-tuning** — not implemented by any of these clients, including this one. (The **batch API** *is* now implemented here — see the Batch API section above.)

In short: choose `mistralai-go` for **OCR, document/file workflows, and strict structured output**; reach for `gage-technologies/mistral-go` for the broadest inference feature set (streaming, embeddings, FIM, tools) and `onkyou/go-mistral` if you specifically need moderation/classification or schema-validated tool calling.

## Testing

```bash
go test ./...
```

### Live OCR test

```bash
MISTRAL_API_KEY=... go test -tags=mistral_test ./...
```

## License

See repository for license terms.
