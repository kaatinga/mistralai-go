package mistralai

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math/rand/v2"
	"mime/multipart"
	"net/http"
	"net/textproto"
	"net/url"
	"path"
	"strconv"
	"time"
)

const (
	// defaultMaxAttempts is the total number of tries per request: the initial
	// attempt plus 4 default retries.
	defaultMaxAttempts = 5
	retryInitialDelay  = time.Second
	retryMaxDelay      = time.Minute
)

func isRetryStatusCode(code int) bool {
	return code == http.StatusTooManyRequests ||
		code == http.StatusInternalServerError ||
		code == http.StatusBadGateway ||
		code == http.StatusServiceUnavailable ||
		code == http.StatusGatewayTimeout
}

func isSuccessStatusCode(code int) bool {
	return code >= 200 && code < 300
}

// endpointURL joins the API origin with a cleaned endpoint path and an optional
// query string. The query is appended after cleaning so its values are never
// path-mangled.
func (c *Client) endpointURL(endpoint string, query url.Values) string {
	u := c.baseURL + path.Clean("/"+endpoint)
	if len(query) > 0 {
		u += "?" + query.Encode()
	}
	return u
}

// retryAfter parses a Retry-After response header (delay-seconds or HTTP-date).
func retryAfter(h http.Header) (time.Duration, bool) {
	v := h.Get("Retry-After")
	if v == "" {
		return 0, false
	}
	if secs, err := strconv.Atoi(v); err == nil && secs >= 0 {
		return time.Duration(secs) * time.Second, true
	}
	if at, err := http.ParseTime(v); err == nil {
		if d := time.Until(at); d > 0 {
			return d, true
		}
		return 0, true
	}
	return 0, false
}

// jittered adds up to 25% random jitter to a backoff delay.
func jittered(d time.Duration) time.Duration {
	if d <= 0 {
		return d
	}
	return d + rand.N(d/4+1)
}

// doRetry sends requests built by makeReq until a non-retryable outcome,
// retrying 429/5xx and transport errors with exponential backoff (jittered,
// honoring Retry-After, capped at retryMaxDelay). Authorization and User-Agent
// headers are set here. It returns the response body of a 2xx response.
func (c *Client) doRetry(ctx context.Context, makeReq func() (*http.Request, error)) ([]byte, error) {
	delay := retryInitialDelay
	var lastErr error

	for attempt := 0; attempt < c.maxAttempts; attempt++ {
		if attempt > 0 {
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(jittered(delay)):
			}
			delay = min(delay*2, retryMaxDelay)
		}

		req, err := makeReq()
		if err != nil {
			return nil, err
		}
		req.Header.Set("Authorization", bearerToken(c.apiKey))
		req.Header.Set("User-Agent", userAgent)

		resp, err := c.http.Do(req)
		if err != nil {
			lastErr = err
			continue
		}

		body, err := io.ReadAll(resp.Body)
		resp.Body.Close()
		if err != nil {
			lastErr = err
			continue
		}

		if isRetryStatusCode(resp.StatusCode) {
			lastErr = apiError(resp.StatusCode, body)
			if d, ok := retryAfter(resp.Header); ok {
				delay = min(max(d, retryInitialDelay), retryMaxDelay)
			}
			continue
		}
		if !isSuccessStatusCode(resp.StatusCode) {
			return nil, apiError(resp.StatusCode, body)
		}
		return body, nil
	}

	if lastErr == nil {
		lastErr = errors.New("mistral: request retries exhausted")
	}
	return nil, lastErr
}

func (c *Client) doJSON(ctx context.Context, method, endpoint string, query url.Values, payload, dest any) error {
	var payloadBytes []byte
	if payload != nil {
		var err error
		payloadBytes, err = json.Marshal(payload)
		if err != nil {
			return err
		}
	}

	u := c.endpointURL(endpoint, query)
	body, err := c.doRetry(ctx, func() (*http.Request, error) {
		var bodyReader io.Reader
		if payload != nil {
			bodyReader = bytes.NewReader(payloadBytes)
		}
		req, err := http.NewRequestWithContext(ctx, method, u, bodyReader)
		if err != nil {
			return nil, err
		}
		if payload != nil {
			req.Header.Set("Content-Type", "application/json")
		}
		return req, nil
	})
	if err != nil {
		return err
	}
	if dest == nil {
		return nil
	}
	if err = json.Unmarshal(body, dest); err != nil {
		return fmt.Errorf("mistral: decode response: %w", err)
	}
	return nil
}

func (c *Client) postJSON(ctx context.Context, endpoint string, payload, dest any) error {
	return c.doJSON(ctx, http.MethodPost, endpoint, nil, payload, dest)
}

func (c *Client) getJSON(ctx context.Context, endpoint string, query url.Values, dest any) error {
	return c.doJSON(ctx, http.MethodGet, endpoint, query, nil, dest)
}

// getRaw issues a GET and returns the raw 2xx response body without
// unmarshaling, for endpoints returning non-JSON-object payloads (e.g. JSONL
// file content). It shares doRetry's retry-on-429/5xx behavior.
func (c *Client) getRaw(ctx context.Context, endpoint string) ([]byte, error) {
	u := c.endpointURL(endpoint, nil)
	return c.doRetry(ctx, func() (*http.Request, error) {
		return http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	})
}

func (c *Client) uploadFile(ctx context.Context, filename string, content io.Reader, contentType, purpose string) (string, error) {
	var buf bytes.Buffer
	w := multipart.NewWriter(&buf)

	if err := w.WriteField("purpose", purpose); err != nil {
		return "", err
	}

	if contentType == "" {
		contentType = "application/octet-stream"
	}
	h := make(textproto.MIMEHeader)
	h.Set("Content-Disposition", fmt.Sprintf(`form-data; name="file"; filename="%s"`, filename))
	h.Set("Content-Type", contentType)

	part, err := w.CreatePart(h)
	if err != nil {
		return "", err
	}
	if _, err = io.Copy(part, content); err != nil {
		return "", err
	}
	if err = w.Close(); err != nil {
		return "", err
	}

	// The multipart body is fully buffered, so each retry attempt can replay it.
	data := buf.Bytes()
	u := c.endpointURL("/v1/files", nil)
	body, err := c.doRetry(ctx, func() (*http.Request, error) {
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, u, bytes.NewReader(data))
		if err != nil {
			return nil, err
		}
		req.Header.Set("Content-Type", w.FormDataContentType())
		return req, nil
	})
	if err != nil {
		return "", err
	}

	var uploaded uploadFileResponse
	if err = json.Unmarshal(body, &uploaded); err != nil {
		return "", fmt.Errorf("mistral: decode upload response: %w", err)
	}
	if uploaded.ID == "" {
		return "", fmt.Errorf("mistral: upload response missing file id")
	}
	return uploaded.ID, nil
}

func bearerToken(apiKey string) string {
	return "Bearer " + apiKey
}
