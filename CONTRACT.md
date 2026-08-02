# `mistralai-go` v1 wire contract

This document freezes the public and wire contract used by stages 2–6 of the
v1 roadmap. It is a design artifact; stage 1 does not change runtime code.

## Source of truth

- OpenAPI repository: `mistralai/platform-docs-public`
- File: `openapi.yaml`
- Main commit checked on 2026-08-02: `6e06c6cfc14e66e45ddf1f06445146314cff5393`
- API origin: `https://api.mistral.ai`

The OpenAPI file is authoritative for field names, HTTP methods, status codes,
and nullable values. A live request is only used to resolve an OpenAPI/live
discrepancy; live checks must be bounded and must not log keys or response
content.

## Public surface

`NewClient(apiKey string, opts ...ClientOption) (*Client, error)` remains the
only constructor and `Client` remains concrete. Consumers define narrow
interfaces when they need dependency inversion. The v1 surface is:

| Operation | Go method | Notes |
|---|---|---|
| Chat | `ChatCompletion` | One complete response |
| Chat stream | `ChatCompletionStream` | SSE, introduced in stage 3 |
| FIM | `FIMCompletion`, `FIMCompletionStream` | Buffered and SSE |
| OCR | `OCR` | One `OCRRequest` with a closed source union |
| Files | `UploadFile`, `GetFile`, `ListFiles`, `DeleteFile`, `GetFileSignedURL`, `DownloadFile` | Upload returns `File`; download returns `io.ReadCloser` |
| Embeddings | `Embeddings` | `EmbeddingRequest.Input` is `[]string` |
| Batch | `CreateBatchJob`, `ListBatchJobs`, `GetBatchJob`, `CancelBatchJob` | JSONL helpers are streaming |
| Models | `ListModels`, `GetModel` | Read-only |
| Moderation/classification | `Moderate`, `Classify` | Stable endpoints only |

The beta Conversations/Agents, libraries, fine-tuning, and admin APIs are
explicitly out of scope.

## Endpoint matrix

| Method and path | Request → response | Streaming | Retry class |
|---|---|---|---|
| `GET /v1/models` | query → `ModelList` | no | read |
| `GET /v1/models/{model_id}` | path → `Model` | no | read |
| `POST /v1/chat/completions` | `ChatCompletionRequest` → `ChatCompletionResponse` | SSE when `stream=true` | paid POST: no default retry |
| `POST /v1/fim/completions` | `FIMCompletionRequest` → `FIMCompletionResponse` | SSE when `stream=true` | paid POST: no default retry |
| `POST /v1/ocr` | `OCRRequest` → `OCRResponse` | no | paid POST: no default retry |
| `POST /v1/files` | multipart → `File` | no | creates entity: no default retry |
| `GET /v1/files` | query → `FileList` | no | read |
| `GET /v1/files/{file_id}` | path → `File` | no | read |
| `DELETE /v1/files/{file_id}` | path → delete response | no | idempotent |
| `GET /v1/files/{file_id}/content` | path → `io.ReadCloser` | body stream | read |
| `GET /v1/files/{file_id}/url` | path + optional `expiry` (hours, API default 24) → signed URL response | no | read |
| `POST /v1/embeddings` | `EmbeddingRequest` → `EmbeddingResponse` | no | paid POST: no default retry |
| `POST /v1/batch/jobs` | `CreateBatchJobRequest` → `BatchJob` | no | creates entity: no default retry |
| `GET /v1/batch/jobs` | query → `BatchJobList` | no | read |
| `GET /v1/batch/jobs/{job_id}` | path → `BatchJob` | no | read |
| `POST /v1/batch/jobs/{job_id}/cancel` | path → `BatchJob` | no | state-changing POST: no default retry |
| `POST /v1/moderations` | `ModerationRequest` → `ModerationResponse` | no | paid POST: no default retry |
| `POST /v1/classifications` | `ClassificationRequest` → `ClassificationResponse` | no | paid POST: no default retry |

Read operations and explicitly idempotent deletes may retry transport failures,
429, and 5xx responses. Paid or entity-creating POSTs do not retry by default.
Streaming requests are never replayed after bytes have been received. A response
that exceeds the client's read limit is deterministic, not transient, and is
never retried.

Identifiers interpolated into a request path are validated before the request is
built: `.` and `..` are rejected rather than silently resolving to a different
resource once the path is cleaned.

## Bounded reads

No API response is read into memory without a bound:

- JSON responses: 10 MiB; error bodies: 64 KiB (`ErrResponseTooLarge` beyond).
- Chat/FIM SSE: 10 MiB per line and per combined event payload
  (`ErrResponseTooLarge` beyond).
- Batch output JSONL: one record at a time, `DefaultMaxBatchRecordBytes`
  (64 MiB) per record, overridable with a positive limit via
  `DecodeBatchResultsWithOptions`; negative limits are invalid.
- File downloads and multipart uploads stream; the caller owns the reader.

## Required, optional, nullable, and zero values

- Required request fields are validated before transport and cannot be omitted
  by a zero value.
- Optional scalar fields use pointers when zero is meaningful (`*bool`,
  `*int`, `*float64`); nil means omitted.
- Optional slices and maps use `omitempty`; nil and empty have the same wire
  meaning unless the OpenAPI schema says otherwise.
- Nullable response fields use pointers or `json.RawMessage` where the API can
  return either a value or null. A missing field and explicit null are not
  conflated in request unions.
- Server enums are represented by named string types, but unknown values
  decode without failure for forward compatibility.

## Closed unions

The following unions must be represented by constructors and custom JSON
marshalers (or validation that rejects impossible struct literals):

1. Chat message content: text, or a non-empty list of text/file/image-URL/
   document-URL parts.
2. Tool choice: a mode string or a forced function `{name}`. `auto`, `none` and
   `any` are the modes documented at the pinned commit, but the mode set is
   open: an unrecognised mode marshals and unmarshals unchanged, per the
   forward-compatibility rule above. Only structure is validated.
3. OCR source: uploaded file id, document URL, image URL, or local file.
   `LocalFile` is encoded as upload + file-id OCR and is never sent as a
   fabricated wire discriminator.
4. Embedding representation: request input is `[]string`; response vectors
   are float arrays or explicitly decoded dtype/base64 payloads.

Fixtures in `contract/fixtures/` are the canonical JSON examples for these
variants and for optional fields.

## Target examples

The target API examples are kept in `contract/target_api.go.txt` so they do
not become buildable before the implementation stages add the types. They are
compile fixtures for the later stages, not compatibility wrappers.
