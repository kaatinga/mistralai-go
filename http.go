package mistralai

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math/rand/v2"
	"mime"
	"mime/multipart"
	"net/http"
	"net/textproto"
	"net/url"
	"path"
	"strconv"
	"strings"
	"time"
)

// Request paths. The BatchEndpoint* constants name several of the same paths,
// but as batch job targets rather than as paths this client requests directly;
// they are defined in terms of these so the two never drift apart.
const (
	pathChatCompletions = "/v1/chat/completions"
	pathFIMCompletions  = "/v1/fim/completions"
	pathEmbeddings      = "/v1/embeddings"
	pathOCR             = "/v1/ocr"
	pathModerations     = "/v1/moderations"
	pathClassifications = "/v1/classifications"
	pathFiles           = "/v1/files"
	pathBatchJobs       = "/v1/batch/jobs"
	pathModels          = "/v1/models"
)

const (
	defaultMaxAttempts = 5
	retryInitialDelay  = time.Second
	retryMaxDelay      = time.Minute
	maxJSONResponse    = 10 * 1024 * 1024
	maxErrorBody       = 64 * 1024
)

// RetryPolicy controls retries for replay-safe requests.
type RetryPolicy struct {
	MaxAttempts  int
	InitialDelay time.Duration
	MaxDelay     time.Duration
	Jitter       float64
}

func defaultRetryPolicy() RetryPolicy {
	return RetryPolicy{
		MaxAttempts:  defaultMaxAttempts,
		InitialDelay: retryInitialDelay,
		MaxDelay:     retryMaxDelay,
		Jitter:       0.25,
	}
}

func (p RetryPolicy) validate() error {
	if p.MaxAttempts < 1 || p.InitialDelay < 0 || p.MaxDelay < p.InitialDelay ||
		p.Jitter < 0 || p.Jitter > 1 {
		return fmt.Errorf("%w: invalid retry policy", ErrInvalidRequest)
	}
	return nil
}

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
	u := *c.baseURL
	basePath := strings.TrimRight(u.EscapedPath(), "/")
	endpointPath := path.Clean("/" + endpoint)
	joined := path.Join(basePath, endpointPath)
	if strings.HasSuffix(endpoint, "/") && !strings.HasSuffix(joined, "/") {
		joined += "/"
	}
	u.Path, _ = url.PathUnescape(joined)
	u.RawPath = joined
	mergedQuery := u.Query()
	for key, values := range query {
		mergedQuery[key] = append([]string(nil), values...)
	}
	u.RawQuery = mergedQuery.Encode()
	return u.String()
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

// jittered adds up to the configured fraction of random jitter.
func jittered(d time.Duration, fractions ...float64) time.Duration {
	if d <= 0 {
		return d
	}
	fraction := 0.25
	if len(fractions) > 0 {
		fraction = fractions[0]
	}
	return d + time.Duration(float64(d)*fraction*rand.Float64())
}

// attemptResult tells retryLoop what one attempt produced. The zero value means
// success: stop and report no error.
type attemptResult struct {
	// retry asks for another attempt; err is then only reported if the budget
	// runs out first.
	retry bool
	err   error
	// after carries a Retry-After hint, applied only when hasAfter is set so a
	// header naming a time already past still overrides the computed backoff.
	after    time.Duration
	hasAfter bool
}

// retryLoop applies the client's backoff policy around attempt. Retry
// eligibility is decided per attempt, so an operation that is unsafe to replay
// simply never asks for one.
func (c *Client) retryLoop(ctx context.Context, attempt func() attemptResult) error {
	policy := c.retryPolicy
	delay := policy.InitialDelay
	var lastErr error
	for i := range policy.MaxAttempts {
		if i > 0 {
			if err := waitBackoff(ctx, jittered(delay, policy.Jitter)); err != nil {
				return err
			}
			delay = min(delay*2, policy.MaxDelay)
		}
		res := attempt()
		if !res.retry {
			return res.err
		}
		lastErr = res.err
		if res.hasAfter {
			delay = min(max(res.after, policy.InitialDelay), policy.MaxDelay)
		}
	}
	if lastErr == nil {
		lastErr = errors.New("mistral: request retries exhausted")
	}
	return lastErr
}

// errorAttempt classifies a non-2xx response. It owns closing resp.Body.
func errorAttempt(resp *http.Response, retryable bool) attemptResult {
	body, readErr := readLimited(resp.Body, maxErrorBody)
	resp.Body.Close()
	var err error
	if readErr != nil {
		err = readErr
	} else {
		err = apiError(resp.StatusCode, body, resp.Header)
	}
	// A body over the limit fails identically on every attempt, so replaying
	// only re-downloads it.
	if !retryable || !isRetryStatusCode(resp.StatusCode) || errors.Is(readErr, ErrResponseTooLarge) {
		return attemptResult{err: err}
	}
	after, ok := retryAfter(resp.Header)
	return attemptResult{retry: true, err: err, after: after, hasAfter: ok}
}

// doRetry sends requests built by makeReq. Retry eligibility is explicit:
// callers must opt in only when replaying the operation is safe.
func (c *Client) doRetry(ctx context.Context, retryable bool, makeReq func() (*http.Request, error)) ([]byte, error) {
	var body []byte
	err := c.retryLoop(ctx, func() attemptResult {
		req, err := makeReq()
		if err != nil {
			return attemptResult{err: err}
		}
		c.authorize(req)

		resp, err := c.http.Do(req)
		if err != nil {
			return attemptResult{retry: retryable, err: err}
		}
		if !isSuccessStatusCode(resp.StatusCode) {
			return errorAttempt(resp, retryable)
		}
		payload, err := readLimited(resp.Body, maxJSONResponse)
		resp.Body.Close()
		if err != nil {
			return attemptResult{retry: retryable && !errors.Is(err, ErrResponseTooLarge), err: err}
		}
		body = payload
		return attemptResult{}
	})
	if err != nil {
		return nil, err
	}
	return body, nil
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
	body, err := c.doRetry(ctx, method == http.MethodGet || method == http.MethodHead || method == http.MethodDelete, func() (*http.Request, error) {
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

// doStream returns ownership of a successful response body to the caller.
// Error and retry response bodies are always closed before returning/retrying.
func (c *Client) doStream(ctx context.Context, method, endpoint string, query url.Values, retryable bool) (*http.Response, error) {
	u := c.endpointURL(endpoint, query)
	var stream *http.Response
	err := c.retryLoop(ctx, func() attemptResult {
		req, err := http.NewRequestWithContext(ctx, method, u, nil)
		if err != nil {
			return attemptResult{err: err}
		}
		c.authorize(req)
		resp, err := c.http.Do(req)
		if err != nil {
			return attemptResult{retry: retryable, err: err}
		}
		if !isSuccessStatusCode(resp.StatusCode) {
			return errorAttempt(resp, retryable)
		}
		stream = resp
		return attemptResult{}
	})
	if err != nil {
		return nil, err
	}
	return stream, nil
}

// doStreamRequest opens an SSE response for a paid POST. It is deliberately
// single-attempt: a streamed completion must never be silently replayed.
func (c *Client) doStreamRequest(ctx context.Context, endpoint string, payload []byte) (*http.Response, error) {
	u := c.endpointURL(endpoint, nil)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, u, bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}
	c.authorize(req)
	req.Header.Set("Content-Type", "application/json")
	resp, err := c.http.Do(req)
	if err != nil {
		return nil, err
	}
	if !isSuccessStatusCode(resp.StatusCode) {
		return nil, errorAttempt(resp, false).err
	}
	return resp, nil
}

func (c *Client) uploadFile(ctx context.Context, filename string, content io.Reader, contentType, purpose string) (File, error) {
	if strings.TrimSpace(filename) == "" {
		return File{}, fmt.Errorf("%w: filename is required", ErrInvalidRequest)
	}
	if content == nil {
		return File{}, fmt.Errorf("%w: content is required", ErrInvalidRequest)
	}
	if contentType == "" {
		contentType = "application/octet-stream"
	}

	reader, writer := io.Pipe()
	multipartWriter := multipart.NewWriter(writer)
	producerDone := make(chan error, 1)
	go func() {
		var err error
		// CloseWithError(nil) closes the pipe cleanly, so this single deferred
		// close covers every exit path and the reader can never be left hanging.
		defer func() {
			_ = writer.CloseWithError(err)
			producerDone <- err
		}()

		if err = multipartWriter.WriteField("purpose", purpose); err != nil {
			return
		}
		disposition := mime.FormatMediaType("form-data", map[string]string{
			"name": "file", "filename": filename,
		})
		if disposition == "" {
			err = fmt.Errorf("mistral: encode multipart filename %q", filename)
			return
		}
		header := make(textproto.MIMEHeader)
		header.Set("Content-Disposition", disposition)
		header.Set("Content-Type", contentType)
		part, err := multipartWriter.CreatePart(header)
		if err != nil {
			return
		}
		if _, err = io.Copy(part, content); err != nil {
			return
		}
		err = multipartWriter.Close()
	}()

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.endpointURL(pathFiles, nil), reader)
	if err != nil {
		_ = reader.CloseWithError(err)
		return File{}, err
	}
	req.Header.Set("Content-Type", multipartWriter.FormDataContentType())
	c.authorize(req)
	resp, err := c.http.Do(req)
	if err != nil {
		_ = reader.CloseWithError(err)
		<-producerDone
		return File{}, err
	}
	limit := int64(maxJSONResponse)
	if !isSuccessStatusCode(resp.StatusCode) {
		limit = maxErrorBody
	}
	body, readErr := readLimited(resp.Body, limit)
	resp.Body.Close()
	producerErr := <-producerDone
	if readErr != nil {
		return File{}, readErr
	}
	if producerErr != nil {
		return File{}, producerErr
	}
	if !isSuccessStatusCode(resp.StatusCode) {
		return File{}, apiError(resp.StatusCode, body, resp.Header)
	}
	var uploaded File
	if err := json.Unmarshal(body, &uploaded); err != nil {
		return File{}, fmt.Errorf("mistral: decode upload response: %w", err)
	}
	if uploaded.ID == "" {
		return File{}, fmt.Errorf("mistral: upload response missing file id")
	}
	return uploaded, nil
}

// pathID validates a caller-supplied identifier that is interpolated into a
// request path. url.PathEscape leaves "." untouched, and endpointURL cleans the
// result, so "." and ".." would silently address a different resource instead
// of failing. Slashes are escaped and therefore harmless.
func pathID(kind, id string) (string, error) {
	id = strings.TrimSpace(id)
	switch id {
	case "":
		return "", fmt.Errorf("%w: %s is required", ErrInvalidRequest, kind)
	case ".", "..":
		return "", fmt.Errorf("%w: %s %q is not a valid identifier", ErrInvalidRequest, kind, id)
	}
	return url.PathEscape(id), nil
}

// authorize sets the headers every request to the API carries.
func (c *Client) authorize(req *http.Request) {
	req.Header.Set("Authorization", "Bearer "+c.apiKey)
	req.Header.Set("User-Agent", c.userAgentValue)
}

// ErrResponseTooLarge reports a response body over the client's read limit. It
// is deterministic, so it is never retried.
var ErrResponseTooLarge = errors.New("mistral: response body exceeds read limit")

func readLimited(r io.Reader, limit int64) ([]byte, error) {
	body, err := io.ReadAll(io.LimitReader(r, limit+1))
	if err != nil {
		return nil, err
	}
	if int64(len(body)) > limit {
		return nil, fmt.Errorf("%w of %d bytes", ErrResponseTooLarge, limit)
	}
	return body, nil
}

func waitBackoff(ctx context.Context, delay time.Duration) error {
	if delay <= 0 {
		return nil
	}
	timer := time.NewTimer(delay)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}
