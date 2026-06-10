package mistralai

import (
	"context"
	"errors"
	"fmt"
	"strings"
)

// OCRByFileID runs POST /v1/ocr for a file already uploaded via UploadFile.
// The caller is responsible for deleting the file when finished.
func (c *Client) OCRByFileID(ctx context.Context, fileID string, model string) (OCRResponse, error) {
	fileID = strings.TrimSpace(fileID)
	if fileID == "" {
		return OCRResponse{}, errors.New("mistral: file id is required")
	}
	if model == "" {
		model = DefaultOCRModel
	}
	body := ocrRequestBody{
		Model: model,
		Document: ocrDocument{
			Type:   documentTypeFile,
			FileID: fileID,
		},
	}
	var resp OCRResponse
	if err := c.postJSON(ctx, "/v1/ocr", body, &resp); err != nil {
		return OCRResponse{}, fmt.Errorf("mistral: ocr: %w", err)
	}
	return resp, nil
}
