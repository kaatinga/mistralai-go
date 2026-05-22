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
)

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

func (c *client) postJSON(ctx context.Context, endpoint string, payload, dest any) error {
	bts, err := json.Marshal(payload)
	if err != nil {
		return err
	}

	url := c.baseURL + path.Clean("/"+endpoint)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(bts))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", bearerToken(c.apiKey))

	resp, err := c.http.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return err
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
