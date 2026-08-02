package mistralai

import (
	"context"
	"fmt"
	"strings"
)

const DefaultModerationModel = "mistral-moderation-latest"

// ModerationRequest is the batch-compatible request for POST /v1/moderations.
// Input must contain at least one text; one input is represented as a one-item
// slice so the request and response cardinalities are always explicit.
type ModerationRequest struct {
	Model    string         `json:"model"`
	Metadata map[string]any `json:"metadata,omitempty"`
	Input    []string       `json:"input"`
}

// ModerationResult contains provider-defined category names. Maps preserve new
// categories without requiring an SDK update.
type ModerationResult struct {
	Categories     map[string]bool    `json:"categories"`
	CategoryScores map[string]float64 `json:"category_scores"`
}

type ModerationResponse struct {
	ID      string             `json:"id"`
	Model   string             `json:"model"`
	Results []ModerationResult `json:"results"`
}

// ClassificationRequest is the batch-compatible request for POST
// /v1/classifications.
type ClassificationRequest struct {
	Model    string         `json:"model"`
	Metadata map[string]any `json:"metadata,omitempty"`
	Input    []string       `json:"input"`
}

type ClassificationTargetResult struct {
	Scores map[string]float64 `json:"scores"`
}

type ClassificationResponse struct {
	ID      string                                  `json:"id"`
	Model   string                                  `json:"model"`
	Results []map[string]ClassificationTargetResult `json:"results"`
}

func validateClassifierRequest(model string, input []string) error {
	if strings.TrimSpace(model) == "" {
		return fmt.Errorf("%w: model is required", ErrInvalidRequest)
	}
	if len(input) == 0 {
		return fmt.Errorf("%w: input is required", ErrInvalidRequest)
	}
	for index, value := range input {
		if strings.TrimSpace(value) == "" {
			return fmt.Errorf("%w: input %d is empty", ErrInvalidRequest, index)
		}
	}
	return nil
}

// Moderate classifies text against provider-defined safety categories. The
// paid POST is sent once and is never retried automatically.
func (c *Client) Moderate(ctx context.Context, req ModerationRequest) (ModerationResponse, error) {
	if err := validateClassifierRequest(req.Model, req.Input); err != nil {
		return ModerationResponse{}, err
	}
	var response ModerationResponse
	if err := c.postJSON(ctx, pathModerations, req, &response); err != nil {
		return ModerationResponse{}, fmt.Errorf("mistral: moderate: %w", err)
	}
	return response, nil
}

// Classify applies a stable classifier model to one or more texts. The paid
// POST is sent once and is never retried automatically.
func (c *Client) Classify(ctx context.Context, req ClassificationRequest) (ClassificationResponse, error) {
	if err := validateClassifierRequest(req.Model, req.Input); err != nil {
		return ClassificationResponse{}, err
	}
	var response ClassificationResponse
	if err := c.postJSON(ctx, pathClassifications, req, &response); err != nil {
		return ClassificationResponse{}, fmt.Errorf("mistral: classify: %w", err)
	}
	return response, nil
}
