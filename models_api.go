package mistralai

import (
	"context"
	"fmt"
)

// ModelCard describes a Mistral model from GET /v1/models.
type ModelCard struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
}

// ModelList is the list models API response.
type ModelList struct {
	Object string      `json:"object"`
	Data   []ModelCard `json:"data"`
}

// ListModels returns models available to the API key (GET /v1/models).
func (c *client) ListModels(ctx context.Context) (ModelList, error) {
	var list ModelList
	if err := c.getJSON(ctx, "/v1/models", &list); err != nil {
		return ModelList{}, fmt.Errorf("mistral: list models: %w", err)
	}
	return list, nil
}
