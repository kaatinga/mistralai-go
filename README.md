# mistralai-go

Synchronous Go client for the [Mistral API](https://docs.mistral.ai/api): OCR (file upload + `/v1/ocr`) and chat completions (`/v1/chat/completions`). Each call blocks until Mistral returns HTTP 200 with the full JSON body (no job polling or channel-based workers).

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
	"os"

	"github.com/kaatinga/mistralai-go"
)

func main() {
	cl, err := mistralai.NewClient(os.Getenv("MISTRAL_API_KEY"))
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

	// Chat: POST /v1/chat/completions
	chat, err := cl.Chat(ctx, mistralai.ChatRequest{
		Input:  "Summarize OCR in one sentence.",
		Format: mistralai.OutputText,
	})
	if err != nil {
		log.Fatal(err)
	}
	log.Println(chat.Content)
}
```

For long OCR runs, pass a context with timeout or deadline. The default HTTP client timeout is 10 minutes.

## Live OCR test

```bash
MISTRAL_API_KEY=... go test -tags=mistral_test ./...
```

## License

See repository for license terms.
