# mistralai-go

Synchronous Go client for the [Mistral API](https://docs.mistral.ai/api). Each call blocks until Mistral returns HTTP 200 with the full JSON body (no job polling or background workers).

## API surface

| Method | HTTP | Use when |
|--------|------|----------|
| `OCR` | `POST /v1/files`, then `POST /v1/ocr` | Document OCR via file upload |
| `Chat` | `POST /v1/chat/completions` | Single user turn with optional system prompt and output format helpers |
| `ChatCompletion` | `POST /v1/chat/completions` | Full control: message list, temperature, `response_format`, etc. |
| `ListModels` | `GET /v1/models` | List models available to your API key |

JSON API calls (`Chat`, `ChatCompletion`, `ListModels`, OCR after upload) retry on **429** and **5xx** with exponential backoff (default **5** attempts, context-aware). Configure with `WithMaxRetries`.

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
		Temperature: 0.7,
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

For `json_schema`, set `ResponseFormat` on `ChatRequest`, `ChatCompletionRequest`, or `OCRRequest.DocumentAnnotationFormat`. Unmarshal OCR output with `OCRStructured[T]` or `OCRResponse.DocumentAnnotationInto[T]()`.

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
