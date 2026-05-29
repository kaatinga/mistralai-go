package mistralai

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
)

// APIError is returned when the Mistral API responds with a non-200 status.
// Inspect it with errors.As:
//
//	var apiErr *mistralai.APIError
//	if errors.As(err, &apiErr) && apiErr.StatusCode == http.StatusUnauthorized {
//		// handle bad API key
//	}
type APIError struct {
	// StatusCode is the HTTP status code of the response.
	StatusCode int
	// Message is the human-readable message from the API, when present.
	Message string
	// Type is the Mistral error type (e.g. "invalid_request_error"), when present.
	Type string
	// Code is the Mistral error code, when present.
	Code string
	// Body is the raw response body.
	Body []byte
}

func (e *APIError) Error() string {
	msg := e.Message
	if msg == "" {
		msg = strings.TrimSpace(string(e.Body))
	}
	if msg == "" {
		msg = http.StatusText(e.StatusCode)
	}
	return fmt.Sprintf("mistral api status %d: %s", e.StatusCode, msg)
}

// Retryable reports whether the client retries this status (429 and 5xx).
func (e *APIError) Retryable() bool {
	return isRetryStatusCode(e.StatusCode)
}

func apiError(status int, body []byte) *APIError {
	e := &APIError{StatusCode: status, Body: body}
	var parsed apiErrorResponse
	if err := json.Unmarshal(body, &parsed); err == nil {
		e.Message = parsed.Message
		e.Type = parsed.Type
		if parsed.Code != nil {
			e.Code = fmt.Sprintf("%v", parsed.Code)
		}
	}
	return e
}
