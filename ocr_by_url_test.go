package mistralai

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestOCRByURL_documentURL(t *testing.T) {
	var body ocrRequestBody
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/ocr" || r.Method != http.MethodPost {
			http.NotFound(w, r)
			return
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatal(err)
		}
		_ = json.NewEncoder(w).Encode(OCRResponse{
			Pages:     []OCRPage{{Index: 0, Markdown: "# Doc"}},
			UsageInfo: OCRUsageInfo{PagesProcessed: 1},
		})
	}))
	defer srv.Close()

	cl, err := NewClient("k", WithBaseURL(srv.URL))
	if err != nil {
		t.Fatal(err)
	}
	defer cl.Close()

	resp, err := cl.OCRByURL(context.Background(), OCRURLRequest{
		DocumentURL:  "https://example.com/a.pdf",
		DocumentName: "a.pdf",
		Pages:        []int{0, 1},
		TableFormat:  "html",
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Pages) != 1 || resp.Pages[0].Markdown != "# Doc" {
		t.Fatalf("resp = %+v", resp)
	}

	doc := body.Document
	if doc.Type != "document_url" || doc.DocumentURL != "https://example.com/a.pdf" || doc.DocumentName != "a.pdf" {
		t.Fatalf("document = %+v", doc)
	}
	if doc.FileID != "" || doc.ImageURL != "" {
		t.Fatalf("unexpected union fields set: %+v", doc)
	}
	if body.Model != DefaultOCRModel {
		t.Fatalf("model = %q", body.Model)
	}
	if len(body.Pages) != 2 || body.TableFmt != "html" {
		t.Fatalf("body = %+v", body)
	}
}

func TestOCRByURL_imageURL(t *testing.T) {
	var body ocrRequestBody
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatal(err)
		}
		_ = json.NewEncoder(w).Encode(OCRResponse{UsageInfo: OCRUsageInfo{PagesProcessed: 1}})
	}))
	defer srv.Close()

	cl, err := NewClient("k", WithBaseURL(srv.URL))
	if err != nil {
		t.Fatal(err)
	}
	defer cl.Close()

	if _, err := cl.OCRByURL(context.Background(), OCRURLRequest{
		ImageURL: "data:image/png;base64,iVBORw0",
		Model:    "custom-ocr",
	}); err != nil {
		t.Fatal(err)
	}

	doc := body.Document
	if doc.Type != "image_url" || doc.ImageURL != "data:image/png;base64,iVBORw0" {
		t.Fatalf("document = %+v", doc)
	}
	if doc.DocumentURL != "" || doc.DocumentName != "" || doc.FileID != "" {
		t.Fatalf("unexpected union fields set: %+v", doc)
	}
	if body.Model != "custom-ocr" {
		t.Fatalf("model = %q", body.Model)
	}
}

func TestOCRByURL_validation(t *testing.T) {
	cl, err := NewClient("k", WithBaseURL("http://127.0.0.1:1"))
	if err != nil {
		t.Fatal(err)
	}
	defer cl.Close()

	cases := []struct {
		name    string
		req     OCRURLRequest
		wantErr string
	}{
		{"neither_url", OCRURLRequest{}, "exactly one of DocumentURL or ImageURL"},
		{"both_urls", OCRURLRequest{DocumentURL: "https://a", ImageURL: "https://b"}, "exactly one of DocumentURL or ImageURL"},
		{"name_without_document", OCRURLRequest{ImageURL: "https://b", DocumentName: "x"}, "DocumentName requires DocumentURL"},
		{"prompt_without_format", OCRURLRequest{DocumentURL: "https://a", DocumentAnnotationPrompt: "extract"}, "document_annotation_format is required"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := cl.OCRByURL(context.Background(), tc.req)
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("err = %v", err)
			}
		})
	}
}

func TestOCRURLEntry(t *testing.T) {
	e := OCRURLEntry("doc-1", "", "https://example.com/a.pdf", WithOCRTableFormat("markdown"))
	if e.CustomID != "doc-1" {
		t.Fatalf("custom id = %q", e.CustomID)
	}
	body, ok := e.Body.(ocrRequestBody)
	if !ok {
		t.Fatalf("body type %T", e.Body)
	}
	if body.Model != DefaultOCRModel {
		t.Fatalf("model = %q", body.Model)
	}
	if body.Document.Type != "document_url" || body.Document.DocumentURL != "https://example.com/a.pdf" {
		t.Fatalf("document = %+v", body.Document)
	}
	if body.TableFmt != "markdown" {
		t.Fatalf("table format = %q", body.TableFmt)
	}
}
