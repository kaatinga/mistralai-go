# mistralai-go

`mistralai-go` is a community-maintained Go client for the Mistral API. It is
not an official Mistral SDK.

The v1 API covers Chat Completions and streaming, structured output, tools,
FIM, OCR, Files, Embeddings, Batch, Models, moderation, and classification.
Beta Conversations/Agents, libraries, fine-tuning management, and admin APIs
are intentionally outside the v1 scope.

## Install

```bash
go get github.com/kaatinga/mistralai-go@main
```

The module requires Go 1.26.

> **Release candidate.** `v1.0.0` is not tagged yet. The API described here is
> the intended v1 surface, but it is only frozen once the tag exists. The
> command above resolves `main` to a commit-based pseudo-version in `go.mod`;
> pin that version until the release. See `RELEASE_NOTES.md` for the remaining
> gates.

## Client, errors, and retries

```go
client, err := mistralai.NewClient(
	os.Getenv("MISTRAL_API_KEY"),
	mistralai.WithHTTPClient(&http.Client{Timeout: 2 * time.Minute}),
)
```

`NewClient` returns a concrete `*Client`; consumers should define narrow local
interfaces when needed. API failures wrap `*mistralai.APIError`, which exposes
the status, provider message, request ID, bounded response fragment, and
retryability. Transport failures never produce an `*APIError` — there is no
status to report — so match those with `errors.Is` against the underlying cause
(`context.DeadlineExceeded` and friends) instead.

The default transport retries only replay-safe `GET`, `HEAD`, and `DELETE`
operations. Paid or entity-creating POST operations—including Chat, FIM, OCR,
Embeddings, uploads, moderation/classification, and Batch creation/cancel—are
sent exactly once. `WithMaxRetries` and `WithRetryPolicy` tune retries for safe
operations only. Backoff and `Retry-After` waits stop immediately when the
context is cancelled.

## Chat, structured output, and tools

```go
response, err := client.ChatCompletion(ctx, mistralai.ChatCompletionRequest{
	Model: mistralai.ChatModelMistralSmallLatest,
	Messages: []mistralai.Message{
		{Role: mistralai.RoleUser, Content: mistralai.TextContent("Hello")},
	},
})
text, err := response.FirstText()
```

For multimodal input, use `MultipartMessage` with `TextPart`, `FilePart`,
`ImageURLPart`, or `DocumentURLPart`. `ChatCompletionRequest` has no streaming
flag: call `ChatCompletionStream`, repeatedly call `Recv`, and always `Close`
the stream — including after `Recv` reports `io.EOF`. `Accumulate` is available
when collecting the full text is desired; it returns `ErrIncompleteStream`, with
the text read so far, if the body ends before the terminating `[DONE]` event.
Individual SSE lines and combined event payloads are bounded at 10 MiB;
oversized events return `ErrResponseTooLarge`.

`ChatStructured[T]` accepts the complete request, calls `ChatCompletion`, and
returns both the decoded value and original response/usage. Set a schema with
`JSONSchemaFormat`:

```go
value, response, err := mistralai.ChatStructured[Result](ctx, client,
	mistralai.ChatCompletionRequest{
		Model: mistralai.ChatModelMistralSmallLatest,
		Messages: []mistralai.Message{
			mistralai.TextMessage(mistralai.RoleUser, "Return the answer as JSON"),
		},
		ResponseFormat: mistralai.JSONSchemaFormat("result", schema),
	})
```

`ChatCompletionWithTools` runs a validated tool loop. A forced named tool choice
applies to the first round only; later rounds use `auto`, allowing normal final
text. `ChatCompletionWithToolsOptions` can deliberately force every round.
Unknown tools, duplicate call IDs, malformed arguments, empty results, and
round-limit exhaustion return errors.

`maxRounds` bounds how many rounds of tool calls are executed, so the loop sends
at most `maxRounds+1` completions. The answer that follows the last executed
round is always inspected and returned; exhaustion is reported only when the
model is still asking for tools at that point.

## FIM

`FIMCompletion` returns one buffered `FIMCompletionResponse`.
`FIMCompletionStream` selects streaming without a request flag and returns a
stream owned by the caller; close it on every path.

## OCR and Files

One `OCR` method accepts the closed source union:

- `UploadedFile` — caller owns the remote file; OCR does not delete it.
- `DocumentURL` and `ImageURL` — no upload lifecycle.
- `LocalFile` — the SDK streams upload → OCR → delete.

```go
file, err := os.Open("invoice.pdf")
if err != nil { /* handle */ }
defer file.Close() // the caller owns the input reader

response, err := client.OCR(ctx, mistralai.OCRRequest{
	Model: mistralai.DefaultOCRModel,
	Source: mistralai.LocalFile{
		Name: "invoice.pdf", ContentType: "application/pdf", Reader: file,
	},
})
```

`LocalFile.Reader` remains caller-owned and is never closed by the SDK. A delete
failure after successful OCR returns the response together with
`*OCRCleanupError`; simultaneous OCR and cleanup failures are joined.

`UploadFile` streams the caller-owned `io.Reader` and returns complete `File`
metadata. `DownloadFile` returns an `io.ReadCloser`; the caller must close it.
The SDK closes all non-success response bodies itself.

## Embeddings

```go
response, err := client.Embeddings(ctx, mistralai.EmbeddingRequest{
	Model: mistralai.EmbeddingModelMistralEmbed,
	Input: []string{"first", "second"},
	OutputDType: mistralai.OutputDTypeInt8,
})
vectors, err := response.Int8Vectors()
```

The effective encoding and dtype are retained in `EmbeddingResponse`. Use the
matching float32/float64, int8, uint8, binary, or ubinary decoder. A mismatched
decoder returns `*ErrEmbeddingType`; vector helpers validate response indexes
and restore input order.

## Batch JSONL

`EncodeBatchEntries` writes JSONL to an `io.Writer`; `UploadBatchInput` encodes
directly into a streamed multipart upload. Build typed entries with
`ChatCompletionEntry`, `EmbeddingEntry`, `OCREntry`, `ModerationEntry`, or
`ClassificationEntry`. `OCREntry` returns an error rather than a malformed
entry: it validates the request like synchronous `OCR` and rejects `LocalFile`,
which cannot carry an upload lifecycle into a batch job.

Batch output is also streaming, one record at a time and bounded by
`DefaultMaxBatchRecordBytes` per line (override it with
`DecodeBatchResultsWithOptions`; positive limits override the default and
negative limits are invalid):

```go
body, err := client.DownloadFile(ctx, outputFileID)
if err != nil { /* handle */ }
defer body.Close()

err = mistralai.DecodeBatchResults[mistralai.ChatCompletionResponse](body,
	func(result mistralai.BatchResult[mistralai.ChatCompletionResponse]) error {
		// Consume one result without retaining the whole JSONL file.
		return nil
	})
```

`WaitForBatchJob` polls through a narrow `BatchJobGetter`, honors context
cancellation, and returns terminal job data for inspection.

## Models, moderation, and classification

`ListModels` and `GetModel` return lifecycle, ownership, aliases, deprecation,
default temperature, and capability data. Use `ModelList.FilterByCapability`;
never infer Chat support from a model name. Unknown future capability fields are
preserved during JSON round trips and can be queried with `Supports`.

`Moderate` and `Classify` accept `[]string` inputs, making request/response
cardinality and Batch encoding explicit. Provider-defined category and target
names are maps so new server values decode without an SDK release.

See [MIGRATING.md](MIGRATING.md) for v0.19.0 changes and the package examples
for complete compilable call patterns.
