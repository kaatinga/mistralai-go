package mistralai

import (
	"bufio"
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestChatCompletionStream_fragmentedEvents(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"hel"))
		_, _ = w.Write([]byte("lo\"}}]}\n\n"))
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\" world\"},\"finish_reason\":\"stop\"}]}\n\n"))
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer srv.Close()

	client, err := NewClient("key", WithBaseURL(srv.URL))
	if err != nil {
		t.Fatal(err)
	}
	stream, err := client.ChatCompletionStream(context.Background(), ChatCompletionRequest{
		Model: "mistral-small-latest", Messages: []ChatMessage{TextMessage(RoleUser, "hi")},
	})
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	text, last, err := stream.Accumulate()
	if err != nil {
		t.Fatal(err)
	}
	if text != "hello world" || last.Choices[0].FinishReason != "stop" {
		t.Fatalf("text=%q last=%+v", text, last)
	}
	if _, err := stream.Recv(); !errors.Is(err, io.EOF) {
		t.Fatalf("after done err=%v", err)
	}
}

func TestChatCompletionStreamProviderError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"error\":{\"message\":\"bad stream\"}}\n\n"))
	}))
	defer srv.Close()
	client, err := NewClient("key", WithBaseURL(srv.URL))
	if err != nil {
		t.Fatal(err)
	}
	stream, err := client.ChatCompletionStream(context.Background(), ChatCompletionRequest{
		Model: "m", Messages: []ChatMessage{TextMessage(RoleUser, "hi")},
	})
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	_, err = stream.Recv()
	var streamErr *StreamError
	if !errors.As(err, &streamErr) || streamErr.Message != "bad stream" {
		t.Fatalf("err=%v", err)
	}
}

// A body that stops before [DONE] is a truncated completion, not a short one:
// Accumulate must say so instead of returning the prefix with a nil error.
func TestChatCompletionStream_truncatedWithoutDone(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"par\"}}]}\n\n"))
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"tial\"}}]}\n\n"))
	}))
	defer srv.Close()

	client, err := NewClient("key", WithBaseURL(srv.URL))
	if err != nil {
		t.Fatal(err)
	}
	stream, err := client.ChatCompletionStream(context.Background(), ChatCompletionRequest{
		Model: "m", Messages: []ChatMessage{TextMessage(RoleUser, "hi")},
	})
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()

	text, _, err := stream.Accumulate()
	if !errors.Is(err, ErrIncompleteStream) {
		t.Fatalf("err = %v", err)
	}
	if text != "partial" {
		t.Fatalf("text = %q, want the prefix read so far", text)
	}
}

func TestChatCompletionStream_rejectsOversizedEvents(t *testing.T) {
	newStream := func(body string) *ChatCompletionStream {
		reader := io.NopCloser(strings.NewReader(body))
		return &ChatCompletionStream{body: reader, reader: bufio.NewReader(reader)}
	}

	t.Run("single line", func(t *testing.T) {
		stream := newStream("data: " + strings.Repeat("x", maxStreamEventBytes) + "\n\n")
		defer stream.Close()
		if _, err := stream.Recv(); !errors.Is(err, ErrResponseTooLarge) {
			t.Fatalf("err = %v", err)
		}
	})

	t.Run("combined data", func(t *testing.T) {
		half := maxStreamEventBytes / 2
		stream := newStream(
			"data: " + strings.Repeat("x", half) + "\n" +
				"data: " + strings.Repeat("y", half) + "\n\n",
		)
		defer stream.Close()
		if _, err := stream.Recv(); !errors.Is(err, ErrResponseTooLarge) {
			t.Fatalf("err = %v", err)
		}
	})
}
