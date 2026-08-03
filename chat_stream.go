package mistralai

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
)

const maxStreamEventBytes = maxJSONResponse

// ChatCompletionStreamChoice is one incremental chat choice.
type ChatCompletionStreamChoice struct {
	Index        int             `json:"index"`
	Delta        ChatStreamDelta `json:"delta"`
	FinishReason string          `json:"finish_reason"`
}

// ChatStreamDelta accepts the object form used by Chat Completions and the
// string form used by some FIM stream responses.
type ChatStreamDelta struct {
	Role      string     `json:"role,omitempty"`
	Content   any        `json:"content,omitempty"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
}

func (d *ChatStreamDelta) UnmarshalJSON(data []byte) error {
	var text string
	if err := json.Unmarshal(data, &text); err == nil {
		d.Content = text
		return nil
	}
	type plain ChatStreamDelta
	var value plain
	if err := json.Unmarshal(data, &value); err != nil {
		return err
	}
	*d = ChatStreamDelta(value)
	return nil
}

// ChatCompletionStreamEvent is one server-sent chat completion event.
type ChatCompletionStreamEvent struct {
	ID      string                       `json:"id"`
	Model   string                       `json:"model"`
	Choices []ChatCompletionStreamChoice `json:"choices"`
	Usage   *UsageInfo                   `json:"usage,omitempty"`
}

// StreamError is a provider error delivered inside an otherwise successful
// SSE response.
type StreamError struct {
	Message string `json:"message"`
	Type    string `json:"type,omitempty"`
}

func (e *StreamError) Error() string {
	return "mistral: stream error: " + e.Message
}

// ChatCompletionStream owns the response body of a streaming chat request.
// The caller must always close it, including after Recv reports io.EOF, so the
// connection is released back to the pool.
type ChatCompletionStream struct {
	body   io.ReadCloser
	reader *bufio.Reader
	closed bool
	done   bool
}

// ChatCompletionStream starts an SSE chat completion. Streaming is selected by
// this method, never by changing the return type of ChatCompletion.
func (c *Client) ChatCompletionStream(ctx context.Context, req ChatCompletionRequest) (*ChatCompletionStream, error) {
	if err := req.validate(); err != nil {
		return nil, err
	}
	payload := struct {
		ChatCompletionRequest
		Stream bool `json:"stream"`
	}{ChatCompletionRequest: req, Stream: true}
	stream, err := c.newJSONStream(ctx, pathChatCompletions, payload)
	if err != nil {
		return nil, fmt.Errorf("mistral: chat completion stream: %w", err)
	}
	return stream, nil
}

func (c *Client) newJSONStream(ctx context.Context, endpoint string, payload any) (*ChatCompletionStream, error) {
	data, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}
	resp, err := c.doStreamRequest(ctx, endpoint, data)
	if err != nil {
		return nil, err
	}
	return &ChatCompletionStream{body: resp.Body, reader: bufio.NewReader(resp.Body)}, nil
}

// Recv returns the next SSE event, or io.EOF after [DONE].
func (s *ChatCompletionStream) Recv() (ChatCompletionStreamEvent, error) {
	var event ChatCompletionStreamEvent
	if s == nil || s.closed {
		return event, errors.New("mistral: stream is closed")
	}
	if s.done {
		return event, io.EOF
	}
	var data bytes.Buffer
	for {
		lineBytes, err := readBoundedLine(s.reader, maxStreamEventBytes)
		if errors.Is(err, errReadLimitExceeded) {
			return event, fmt.Errorf("%w of %d bytes in SSE line", ErrResponseTooLarge, maxStreamEventBytes)
		}
		line := string(lineBytes)
		if err != nil && len(line) == 0 {
			if err == io.EOF && data.Len() > 0 {
				break
			}
			return event, err
		}
		line = strings.TrimSuffix(strings.TrimSuffix(line, "\n"), "\r")
		if line == "" {
			if data.Len() == 0 {
				if err != nil {
					return event, err
				}
				continue
			}
			break
		}
		if strings.HasPrefix(line, ":") {
			continue
		}
		if after, ok := strings.CutPrefix(line, "data:"); ok {
			value := strings.TrimSpace(after)
			additional := len(value)
			if data.Len() > 0 {
				additional++
			}
			if int64(data.Len())+int64(additional) > maxStreamEventBytes {
				return event, fmt.Errorf("%w of %d bytes in SSE event", ErrResponseTooLarge, maxStreamEventBytes)
			}
			if data.Len() > 0 {
				data.WriteByte('\n')
			}
			data.WriteString(value)
		}
		if err != nil {
			break
		}
	}
	raw := bytes.TrimSpace(data.Bytes())
	if bytes.Equal(raw, []byte("[DONE]")) {
		s.done = true
		return event, io.EOF
	}
	if len(raw) == 0 {
		return event, errors.New("mistral: empty SSE event")
	}
	var streamError StreamError
	if raw[0] == '{' {
		var envelope struct {
			Error json.RawMessage `json:"error"`
		}
		if json.Unmarshal(raw, &envelope) == nil && len(envelope.Error) > 0 && string(envelope.Error) != "null" {
			if err := json.Unmarshal(envelope.Error, &streamError); err != nil {
				streamError.Message = string(envelope.Error)
			}
			return event, &streamError
		}
	}
	if err := json.Unmarshal(raw, &event); err != nil {
		return event, fmt.Errorf("mistral: decode SSE event: %w", err)
	}
	return event, nil
}

// Close releases the response body and unblocks a pending Recv.
func (s *ChatCompletionStream) Close() error {
	if s == nil || s.closed {
		return nil
	}
	s.closed = true
	return s.body.Close()
}

// ErrIncompleteStream reports a stream whose body ended before the terminating
// [DONE] event, so the accumulated text is a prefix of the real completion.
var ErrIncompleteStream = errors.New("mistral: stream ended before [DONE]")

// Accumulate reads the stream and returns the complete text and last event.
// A body that ends without [DONE] yields ErrIncompleteStream rather than a
// silent partial result; the text read so far is still returned alongside it.
func (s *ChatCompletionStream) Accumulate() (string, ChatCompletionStreamEvent, error) {
	var text strings.Builder
	var last ChatCompletionStreamEvent
	for {
		event, err := s.Recv()
		if errors.Is(err, io.EOF) {
			if !s.done {
				return text.String(), last, ErrIncompleteStream
			}
			return text.String(), last, nil
		}
		if err != nil {
			return "", last, err
		}
		last = event
		for _, choice := range event.Choices {
			if content, ok := choice.Delta.Content.(string); ok {
				text.WriteString(content)
			}
		}
	}
}
