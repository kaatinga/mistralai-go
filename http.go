package mistralai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"net/textproto"
	"path"
	"strings"
	"time"
)

const (
	defaultMaxRetries = 5
	retryInitialDelay = time.Second
)

func isRetryStatusCode(code int) bool {
	return code == http.StatusTooManyRequests ||
		code == http.StatusInternalServerError ||
		code == http.StatusBadGateway ||
		code == http.StatusServiceUnavailable ||
		code == http.StatusGatewayTimeout
}

func (c *client) doJSON(ctx context.Context, method, endpoint string, payload, dest any) error {
	var bodyReader io.Reader
	if payload != nil {
		bts, err := json.Marshal(payload)
		if err != nil {
			return err
		}
		bodyReader = bytes.NewReader(bts)
	}

	url := c.baseURL + path.Clean("/"+endpoint)
	delay := retryInitialDelay
	var lastErr error

	for attempt := 0; attempt < c.maxRetries; attempt++ {
		if attempt > 0 {
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(delay):
			}
			delay *= 2
		}

		req, err := http.NewRequestWithContext(ctx, method, url, bodyReader)
		if err != nil {
			return err
		}
		if payload != nil {
			req.Header.Set("Content-Type", "application/json")
		}
		req.Header.Set("Authorization", bearerToken(c.apiKey))

		resp, err := c.http.Do(req)
		if err != nil {
			lastErr = err
			if payload != nil {
				if bts, mErr := json.Marshal(payload); mErr == nil {
					bodyReader = bytes.NewReader(bts)
				}
			}
			continue
		}

		body, err := io.ReadAll(resp.Body)
		resp.Body.Close()
		if err != nil {
			lastErr = err
			if payload != nil {
				if bts, mErr := json.Marshal(payload); mErr == nil {
					bodyReader = bytes.NewReader(bts)
				}
			}
			continue
		}

		if isRetryStatusCode(resp.StatusCode) {
			lastErr = apiError(resp.StatusCode, body)
			if payload != nil {
				if bts, mErr := json.Marshal(payload); mErr == nil {
					bodyReader = bytes.NewReader(bts)
				}
			}
			continue
		}

		if resp.StatusCode != http.StatusOK {
			return apiError(resp.StatusCode, body)
		}
		if dest == nil {
			return nil
		}
		if err = json.Unmarshal(body, dest); err != nil {
			return fmt.Errorf("decode response: %w", err)
		}
		return nil
	}

	if lastErr == nil {
		lastErr = fmt.Errorf("mistral: request retries exhausted")
	}
	return lastErr
}

func (c *client) postJSON(ctx context.Context, endpoint string, payload, dest any) error {
	return c.doJSON(ctx, http.MethodPost, endpoint, payload, dest)
}

func (c *client) getJSON(ctx context.Context, endpoint string, dest any) error {
	return c.doJSON(ctx, http.MethodGet, endpoint, nil, dest)
}

func (c *client) uploadFile(ctx context.Context, filename string, content io.Reader, contentType string) (string, error) {
	var buf bytes.Buffer
	w := multipart.NewWriter(&buf)

	if err := w.WriteField("purpose", filePurposeOCR); err != nil {
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

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/v1/files", &buf)
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", w.FormDataContentType())
	req.Header.Set("Authorization", bearerToken(c.apiKey))

	resp, err := c.http.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}
	if resp.StatusCode != http.StatusOK {
		return "", apiError(resp.StatusCode, body)
	}

	var uploaded uploadFileResponse
	if err = json.Unmarshal(body, &uploaded); err != nil {
		return "", fmt.Errorf("decode upload response: %w", err)
	}
	if uploaded.ID == "" {
		return "", fmt.Errorf("upload response missing file id")
	}
	return uploaded.ID, nil
}

func bearerToken(apiKey string) string {
	return "Bearer " + apiKey
}

func apiError(status int, body []byte) error {
	var e apiErrorResponse
	if err := json.Unmarshal(body, &e); err == nil && e.Message != "" {
		return fmt.Errorf("mistral api status %d: %s", status, e.Message)
	}
	msg := strings.TrimSpace(string(body))
	if msg == "" {
		msg = http.StatusText(status)
	}
	return fmt.Errorf("mistral api status %d: %s", status, msg)
}
