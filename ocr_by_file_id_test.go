package mistralai

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestOCRByFileID(t *testing.T) {
	const wantFileID = "file-ocr-by-id"
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/ocr" || r.Method != http.MethodPost {
			http.NotFound(w, r)
			return
		}
		var body ocrRequestBody
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatal(err)
		}
		if body.Document.FileID != wantFileID {
			t.Errorf("file_id = %q", body.Document.FileID)
		}
		if body.Model != DefaultOCRModel {
			t.Errorf("model = %q", body.Model)
		}
		if len(body.Pages) != 2 || body.TableFmt != "html" {
			t.Errorf("options not forwarded: %+v", body)
		}
		_ = json.NewEncoder(w).Encode(OCRResponse{
			Pages:     []OCRPage{{Index: 0, Markdown: "# Title"}},
			UsageInfo: OCRUsageInfo{PagesProcessed: 1},
		})
	}))
	defer srv.Close()

	cl, err := NewClient("test-key", WithBaseURL(srv.URL))
	if err != nil {
		t.Fatal(err)
	}
	resp, err := cl.OCRByFileID(context.Background(), OCRFileRequest{
		FileID:      wantFileID,
		Pages:       []int{0, 1},
		TableFormat: "html",
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Pages) != 1 || resp.Pages[0].Markdown != "# Title" {
		t.Fatalf("response: %+v", resp)
	}
}
