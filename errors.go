package mistralai

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
)

// ErrInvalidRequest wraps every client-side (pre-flight) validation error, so
// callers can detect them with errors.Is(err, ErrInvalidRequest) instead of
// matching message text. It is never returned for API-side rejections; those
// surface as *APIError.
var ErrInvalidRequest = errors.New("mistral: invalid request")

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
	return fmt.Sprintf("mistral: api status %d: %s", e.StatusCode, msg)
}

// Retryable reports whether the client retries this status (429 and 5xx).
func (e *APIError) Retryable() bool {
	return isRetryStatusCode(e.StatusCode)
}

func apiError(status int, body []byte) *APIError {
	e := &APIError{StatusCode: status, Body: body}
	var parsed apiErrorResponse
	if err := json.Unmarshal(body, &parsed); err == nil {
		e.Message = parsed.messageText()
		e.Type = parsed.Type
		if parsed.Code != nil {
			e.Code = fmt.Sprintf("%v", parsed.Code)
		}
	}
	return e
}

// messageText extracts a human-readable message. Mistral uses two shapes:
// {"message": "..."} and validation errors as {"detail": [{"loc":..., "msg":...}]}
// (sometimes nested under "message").
func (r apiErrorResponse) messageText() string {
	if len(r.Message) > 0 {
		var s string
		if err := json.Unmarshal(r.Message, &s); err == nil {
			return s
		}
		var nested struct {
			Detail json.RawMessage `json:"detail"`
		}
		if err := json.Unmarshal(r.Message, &nested); err == nil {
			if msg := detailText(nested.Detail); msg != "" {
				return msg
			}
		}
	}
	return detailText(r.Detail)
}

func detailText(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return s
	}
	var items []struct {
		Loc []any  `json:"loc"`
		Msg string `json:"msg"`
	}
	if err := json.Unmarshal(raw, &items); err != nil {
		return ""
	}
	parts := make([]string, 0, len(items))
	for _, it := range items {
		msg := it.Msg
		if len(it.Loc) > 0 {
			locs := make([]string, len(it.Loc))
			for i, l := range it.Loc {
				locs[i] = fmt.Sprintf("%v", l)
			}
			msg = strings.Join(locs, ".") + ": " + msg
		}
		if msg != "" {
			parts = append(parts, msg)
		}
	}
	return strings.Join(parts, "; ")
}
