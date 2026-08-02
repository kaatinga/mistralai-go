package mistralai

import (
	"context"
	"fmt"
	"net/url"
	"strconv"
)

// File is an uploaded file metadata record (Mistral FileSchema).
type File struct {
	ID         string  `json:"id"`
	Object     string  `json:"object"`
	Bytes      *int64  `json:"bytes"`
	CreatedAt  int64   `json:"created_at"`
	Filename   string  `json:"filename"`
	Purpose    string  `json:"purpose"`
	SampleType string  `json:"sample_type"`
	NumLines   *int    `json:"num_lines,omitempty"`
	Mimetype   *string `json:"mimetype,omitempty"`
	Source     string  `json:"source"`
	Signature  *string `json:"signature,omitempty"`
	ExpiresAt  *int64  `json:"expires_at,omitempty"`
	Visibility *string `json:"visibility,omitempty"`
}

// FileList is the list files API response (Mistral ListFilesOut).
type FileList struct {
	Object string `json:"object"`
	Data   []File `json:"data"`
	Total  *int   `json:"total,omitempty"`
}

// ListFilesRequest filters and pagination for GET /v1/files.
// Zero value lists files with API defaults (page=0, page_size=100, include_total=true).
type ListFilesRequest struct {
	Page         *int
	PageSize     *int
	IncludeTotal *bool
	SampleType   []string
	Source       []string
	Search       string
	Purpose      string
	Mimetypes    []string
}

func (r ListFilesRequest) queryValues() url.Values {
	q := url.Values{}
	if r.Page != nil {
		q.Set("page", strconv.Itoa(*r.Page))
	}
	if r.PageSize != nil {
		q.Set("page_size", strconv.Itoa(*r.PageSize))
	}
	if r.IncludeTotal != nil {
		q.Set("include_total", strconv.FormatBool(*r.IncludeTotal))
	}
	for _, v := range r.SampleType {
		q.Add("sample_type", v)
	}
	for _, v := range r.Source {
		q.Add("source", v)
	}
	if s := r.Search; s != "" {
		q.Set("search", s)
	}
	if p := r.Purpose; p != "" {
		q.Set("purpose", p)
	}
	for _, v := range r.Mimetypes {
		q.Add("mimetypes", v)
	}
	return q
}

// ListFiles returns files for the API key (GET /v1/files).
func (c *Client) ListFiles(ctx context.Context, req ListFilesRequest) (FileList, error) {
	var list FileList
	if err := c.getJSON(ctx, pathFiles, req.queryValues(), &list); err != nil {
		return FileList{}, fmt.Errorf("mistral: list files: %w", err)
	}
	return list, nil
}

// GetFile fetches file metadata (GET /v1/files/{file_id}).
func (c *Client) GetFile(ctx context.Context, fileID string) (File, error) {
	id, err := pathID("file id", fileID)
	if err != nil {
		return File{}, err
	}
	var file File
	if err := c.getJSON(ctx, pathFiles+"/"+id, nil, &file); err != nil {
		return File{}, fmt.Errorf("mistral: get file: %w", err)
	}
	return file, nil
}

// FileSignedURL is the response from GET /v1/files/{file_id}/url.
type FileSignedURL struct {
	URL string `json:"url"`
}

// GetFileSignedURL returns a temporary signed download URL. expiryHours is the
// number of hours before the URL becomes invalid; nil lets the API apply its
// own default of 24 hours.
func (c *Client) GetFileSignedURL(ctx context.Context, fileID string, expiryHours *int) (FileSignedURL, error) {
	id, err := pathID("file id", fileID)
	if err != nil {
		return FileSignedURL{}, err
	}
	if expiryHours != nil && *expiryHours <= 0 {
		return FileSignedURL{}, fmt.Errorf("%w: expiry hours must be positive", ErrInvalidRequest)
	}
	query := url.Values{}
	if expiryHours != nil {
		query.Set("expiry", strconv.Itoa(*expiryHours))
	}
	var signed FileSignedURL
	if err := c.getJSON(ctx, pathFiles+"/"+id+"/url", query, &signed); err != nil {
		return FileSignedURL{}, fmt.Errorf("mistral: get file signed url: %w", err)
	}
	if signed.URL == "" {
		return FileSignedURL{}, fmt.Errorf("mistral: signed URL response missing url")
	}
	return signed, nil
}
