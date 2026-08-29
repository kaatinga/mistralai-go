package mistralai_test

import (
	"context"
	"io"
	"os"
	"strings"

	mistralai "github.com/kaatinga/mistralai-go"
)

func ExampleClient_ChatCompletion() {
	client, _ := mistralai.NewClient("api-key")
	response, err := client.ChatCompletion(context.Background(), mistralai.ChatCompletionRequest{
		Model: mistralai.ChatModelMistralSmallLatest,
		Messages: []mistralai.Message{
			{Role: mistralai.RoleUser, Content: mistralai.TextContent("Hello")},
		},
	})
	if err != nil {
		return
	}
	_, _ = response.FirstText()
}

func ExampleClient_ChatCompletionStream() {
	client, _ := mistralai.NewClient("api-key")
	stream, err := client.ChatCompletionStream(context.Background(), mistralai.ChatCompletionRequest{
		Model:    mistralai.ChatModelMistralSmallLatest,
		Messages: []mistralai.Message{mistralai.TextMessage(mistralai.RoleUser, "Hello")},
	})
	if err != nil {
		return
	}
	defer func() { _ = stream.Close() }()
	for {
		_, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			return
		}
	}
}

func ExampleClient_OCR() {
	client, _ := mistralai.NewClient("api-key")
	source := strings.NewReader("document bytes")
	_, _ = client.OCR(context.Background(), mistralai.OCRRequest{
		Model: mistralai.DefaultOCRModel,
		Source: mistralai.LocalFile{
			Name: "document.pdf", ContentType: "application/pdf", Reader: source,
		},
	})
}

func ExampleClient_DownloadFile() {
	client, _ := mistralai.NewClient("api-key")
	body, err := client.DownloadFile(context.Background(), "file-id")
	if err != nil {
		return
	}
	defer func() { _ = body.Close() }()
	_, _ = io.Copy(os.Stdout, body)
}

func ExampleDecodeBatchResults() {
	jsonl := strings.NewReader(`{"custom_id":"one","response":{"status_code":200,"body":{"id":"mod-1","model":"mistral-moderation-latest","results":[]}}}`)
	_ = mistralai.DecodeBatchResults[mistralai.ModerationResponse](jsonl,
		func(result mistralai.BatchResult[mistralai.ModerationResponse]) error {
			return nil
		})
}
