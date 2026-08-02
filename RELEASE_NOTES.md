# v1.0.0 release candidate

Date prepared: 2026-08-02

Contract source: Mistral `platform-docs-public/openapi.yaml` at verified commit
`6e06c6cfc14e66e45ddf1f06445146314cff5393`.

## Scope

- One Chat Completions API with typed content/tool-choice unions, SSE streaming,
  structured output, and a validated forced-once tool loop.
- Buffered and streaming FIM.
- Streaming Files plus one OCR API for uploaded files, document/image URLs, and
  local upload/OCR/cleanup lifecycles.
- Dtype-aware Embeddings and streaming Batch JSONL input/output.
- Complete read-only Model cards with capability-based filtering and preserved
  future capability fields.
- Stable text moderation and classification, including typed Batch entries.

Compatibility wrappers from v0.19.0 are not included. See `MIGRATING.md`.

## Post-RC review fixes

A review of the RC found and fixed the following before the tag:

- **Tool loop off-by-one.** `ChatCompletionWithTools` evaluated the response of
  the last permitted round only after the budget check, so with `maxRounds = N`
  the model got `N-1` usable rounds and the final answer was billed and then
  discarded. The response is now always inspected first.
- **`ToolChoice` forward compatibility.** The union rejected any mode outside
  `auto`/`none`/`any`, including while *decoding*, contradicting the package
  policy that enum membership is never validated. Only structure is checked now,
  so a mode the API adds later round-trips through a released SDK.
- **Unbounded batch record reads.** `DecodeBatchResults` accumulated a whole
  JSONL line with no cap. Records are now bounded by
  `DefaultMaxBatchRecordBytes`, overridable with a positive limit through
  `DecodeBatchResultsWithOptions`; negative limits are rejected rather than
  restoring an unbounded read.
- **Unbounded SSE events.** Chat/FIM stream lines and combined event payloads
  are now bounded at 10 MiB and report `ErrResponseTooLarge` beyond that limit.
- **Truncated streams.** `Accumulate` returned a partial completion with a nil
  error when the body ended before `[DONE]`; it now reports
  `ErrIncompleteStream` alongside the text read so far.
- **`APIError.Err`.** The field was never assigned, so `Unwrap` always returned
  nil while documenting otherwise. Both are removed; transport failures were
  never `*APIError` and stay matchable with `errors.Is` on the cause.
- **`GetFileSignedURL` expiry unit.** Documented as seconds, but the pinned
  OpenAPI defines `expiry` in hours (default 24). The parameter is now
  `expiryHours` and non-positive values are rejected.
- **Oversized responses were retried.** A body over the read limit fails
  identically on every attempt; it is now `ErrResponseTooLarge` and single-shot.
- **Upload error-body limits.** Multipart upload errors now use the documented
  64 KiB error-body cap instead of the 10 MiB success-response cap.
- **Path identifiers.** `.` and `..` are rejected instead of resolving, after
  cleaning, to a different endpoint.
- **`OCREntry`.** Returned a nil-bodied entry for a `LocalFile` or invalid
  request, surfacing much later as a generic "body is required". It now returns
  `(BatchEntry, error)` and validates like synchronous `OCR`.
- Cleanups: the `ocrOptions` clone of `OCRRequest` is gone, the two duplicated
  retry loops share one implementation, request paths are named constants,
  `FIMCompletionStream` reuses `validate`, and batch result bodies decode
  without a string round trip.

## Verification matrix

| Gate | Result |
|---|---|
| `gofmt` / `git diff --check` | pass |
| `go test ./...` | pass |
| `go test -race ./...` | pass |
| `go vet ./...` | pass |
| `staticcheck 2026.1 (v0.7.0) ./...` | pass; packages matched |
| `go list -m -json`, `go list ./...`, `go mod verify` | pass |
| Package examples | compile in the normal test suite |
| Fresh temporary copy, without `.git` or workspace state | pass |
| Bounded live smoke with `MISTRAL_API_KEY` | pass; post-review rerun 2026-08-02 |

The post-review live smoke covered Models list/get, short Chat, stream
cancellation, JSON-schema structured output, forced tool once followed by final
text, a token-dependent two-round tool chain whose answer arrived after the
last permitted round, FIM, float and int8 embeddings, and tiny local-file OCR
with guaranteed cleanup. It used short contexts and did not log keys, prompts,
documents, or responses.

Large real file transfers and real Batch job creation were not run. They remain
opt-in because of transfer size/cost. Offline tests cover streaming reader
errors, multipart filename handling, cleanup failure semantics, a JSONL record
larger than 16 MiB, record-size-limit enforcement, SSE line/event limits,
upload error-body limits, and single-attempt unsafe POST behavior.

## Remaining release gates

- Migrate and test `3lines.club` against this candidate through `go.work`.
- Migrate and test `services/bcm` against this candidate through `go.work`.
- Build both consumers without local workspace/`replace` directives after an RC
  version is published.
- Create and publish tag `v1.0.0` only after those consumer gates pass.

No release tag has been created by this preparation session.
