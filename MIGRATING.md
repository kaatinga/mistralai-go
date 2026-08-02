# Migrating from v0.19.0 to v1.0.0

v1 intentionally has no compatibility layer. The module path remains
`github.com/kaatinga/mistralai-go`; update call sites directly.

| v0.19.0 | v1.0.0 |
|---|---|
| `Client.Chat`, `ChatRequest`, `ChatResponse`, `OutputFormat` | Removed; use `ChatCompletion` and `ChatCompletionRequest` |
| `ChatMessage.Content string` assumptions | Use `TextContent`, `TextMessage`, or multipart content constructors |
| `AllChoicesContent`, `FirstChoiceContent` | Use `FirstChoice` or `FirstText` |
| buffered request `Stream` field | Use `ChatCompletionStream` or `FIMCompletionStream` |
| `OCRByFileID`, `OCRByURL`, old OCR options | Use one `OCRRequest` with `UploadedFile`, `DocumentURL`, `ImageURL`, or `LocalFile` |
| upload-based `OCRRequest` fields | Move reader/name/content type into `LocalFile` |
| `UploadFile(...) (string, error)` | `UploadFile(UploadFileRequest) (File, error)`; read `.ID` |
| `DownloadFile(...) ([]byte, error)` | `DownloadFile(...) (io.ReadCloser, error)`; consume and close it |
| hidden `EmbeddingInput` constructors | Set `EmbeddingRequest.Input` to `[]string` |
| dtype guessed while decoding | Call the decoder matching `EmbeddingResponse` dtype |
| buffered Batch JSONL builders/parsers | Use `EncodeBatchEntries` and `DecodeBatchResults` streams |
| batch OCR entry builders | `OCREntry(customID, OCRRequest) (BatchEntry, error)`; check the error, `LocalFile` is rejected |
| signed URL expiry treated as seconds | `GetFileSignedURL(ctx, fileID, expiryHours)` — the API unit is hours |
| mutable package `Version` | Removed; User-Agent derives from build metadata |
| package-wide client interfaces | Define narrow interfaces in the consuming package |

## Chat

Before:

```go
response, err := client.Chat(ctx, mistralai.ChatRequest{
	Input: "hello",
	Format: mistralai.OutputText,
})
fmt.Println(response.Content)
```

After:

```go
response, err := client.ChatCompletion(ctx, mistralai.ChatCompletionRequest{
	Model: mistralai.ChatModelMistralSmallLatest,
	Messages: []mistralai.Message{
		{Role: mistralai.RoleUser, Content: mistralai.TextContent("hello")},
	},
})
text, err := response.FirstText()
```

For tools, `ChatCompletionWithTools` now validates the complete loop and changes
a forced named choice to `auto` after the first round. Use
`ChatCompletionWithToolsOptions` with `ForceToolChoiceEachRound: true` only when
repeated forcing is deliberate. `maxRounds` counts executed tool rounds, so the
loop sends at most `maxRounds+1` completions and always returns the answer that
follows the last one.

## OCR and file ownership

Before:

```go
response, err := client.OCR(ctx, mistralai.OCRRequest{
	Filename: "invoice.pdf",
	Content: file,
	ContentType: "application/pdf",
})
```

After:

```go
response, err := client.OCR(ctx, mistralai.OCRRequest{
	Model: mistralai.DefaultOCRModel,
	Source: mistralai.LocalFile{
		Name: "invoice.pdf", ContentType: "application/pdf", Reader: file,
	},
})
```

The caller still owns `file` and closes it. `LocalFile` uploads and deletes its
temporary remote file. For a previously uploaded file, use
`UploadedFile{FileID: id}`; that remote file remains caller-owned.

Before buffered download:

```go
data, err := client.DownloadFile(ctx, fileID)
```

After streaming download:

```go
body, err := client.DownloadFile(ctx, fileID)
if err != nil { /* handle */ }
defer body.Close()
_, err = io.Copy(destination, body)
```

## Embeddings and Batch

Before:

```go
request.Input = mistralai.EmbeddingInputStrings("one", "two")
results, err := mistralai.ParseBatchResults[Response](data)
```

After:

```go
request.Input = []string{"one", "two"}
err := mistralai.DecodeBatchResults[Response](reader, func(result mistralai.BatchResult[Response]) error {
	return consume(result)
})
```

Quantized embeddings must be decoded with `Int8Vectors`, `Uint8Vectors`,
`BinaryVectors`, or `UBinaryVectors` as appropriate. A mismatch is an error.
