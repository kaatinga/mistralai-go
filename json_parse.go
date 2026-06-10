package mistralai

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
)

// ParseJSON unmarshals raw JSON into T.
func ParseJSON[T any](raw string) (T, error) {
	var out T
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return out, fmt.Errorf("mistral: empty json")
	}
	if err := json.Unmarshal([]byte(raw), &out); err != nil {
		return out, fmt.Errorf("mistral: parse json: %w", err)
	}
	return out, nil
}

// DocumentAnnotationInto unmarshals document_annotation from an OCR response into T.
func DocumentAnnotationInto[T any](r OCRResponse) (T, error) {
	var zero T
	if r.DocumentAnnotation == nil || strings.TrimSpace(*r.DocumentAnnotation) == "" {
		return zero, fmt.Errorf("mistral: missing document_annotation")
	}
	return ParseJSON[T](*r.DocumentAnnotation)
}

// OCRStructured runs OCR and unmarshals document_annotation into T.
func OCRStructured[T any](ctx context.Context, c OCRRunner, req OCRRequest) (T, OCRResponse, error) {
	var zero T
	resp, err := c.OCR(ctx, req)
	if err != nil {
		return zero, resp, err
	}
	out, err := DocumentAnnotationInto[T](resp)
	return out, resp, err
}

// ChatStructured runs ChatCompletion and unmarshals the first choice content into T.
func ChatStructured[T any](ctx context.Context, c ChatCompleter, req ChatCompletionRequest) (T, ChatCompletionResponse, error) {
	var zero T
	resp, err := c.ChatCompletion(ctx, req)
	if err != nil {
		return zero, resp, err
	}
	content, err := resp.FirstChoiceContent()
	if err != nil {
		return zero, resp, err
	}
	out, err := ParseJSON[T](content)
	return out, resp, err
}
