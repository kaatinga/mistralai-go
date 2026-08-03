package mistralai

import (
	"context"
	"encoding/json"
	"fmt"
	"maps"
	"time"
)

// ModelCapability is a capability field returned by the Models API. Unknown
// future capability names can also be passed to ModelCapabilities.Supports.
type ModelCapability string

const (
	ModelCapabilityChat               ModelCapability = "completion_chat"
	ModelCapabilityFunctionCalling    ModelCapability = "function_calling"
	ModelCapabilityFIM                ModelCapability = "completion_fim"
	ModelCapabilityFineTuning         ModelCapability = "fine_tuning"
	ModelCapabilityVision             ModelCapability = "vision"
	ModelCapabilityOCR                ModelCapability = "ocr"
	ModelCapabilityClassification     ModelCapability = "classification"
	ModelCapabilityModeration         ModelCapability = "moderation"
	ModelCapabilityAudio              ModelCapability = "audio"
	ModelCapabilityAudioTranscription ModelCapability = "audio_transcription"
)

// ModelCapabilities describes operations a model actually supports. Additional
// preserves unknown future capability fields during JSON round trips.
type ModelCapabilities struct {
	CompletionChat     bool
	FunctionCalling    bool
	CompletionFIM      bool
	FineTuning         bool
	Vision             bool
	OCR                bool
	Classification     bool
	Moderation         bool
	Audio              bool
	AudioTranscription bool
	Additional         map[string]json.RawMessage
}

var knownModelCapabilities = map[ModelCapability]struct{}{
	ModelCapabilityChat: {}, ModelCapabilityFunctionCalling: {}, ModelCapabilityFIM: {},
	ModelCapabilityFineTuning: {}, ModelCapabilityVision: {}, ModelCapabilityOCR: {},
	ModelCapabilityClassification: {}, ModelCapabilityModeration: {}, ModelCapabilityAudio: {},
	ModelCapabilityAudioTranscription: {},
}

func (c *ModelCapabilities) UnmarshalJSON(data []byte) error {
	type wire struct {
		CompletionChat     bool `json:"completion_chat"`
		FunctionCalling    bool `json:"function_calling"`
		CompletionFIM      bool `json:"completion_fim"`
		FineTuning         bool `json:"fine_tuning"`
		Vision             bool `json:"vision"`
		OCR                bool `json:"ocr"`
		Classification     bool `json:"classification"`
		Moderation         bool `json:"moderation"`
		Audio              bool `json:"audio"`
		AudioTranscription bool `json:"audio_transcription"`
	}
	var decoded wire
	if err := json.Unmarshal(data, &decoded); err != nil {
		return err
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(data, &fields); err != nil {
		return err
	}
	for capability := range knownModelCapabilities {
		delete(fields, string(capability))
	}
	*c = ModelCapabilities{
		CompletionChat: decoded.CompletionChat, FunctionCalling: decoded.FunctionCalling,
		CompletionFIM: decoded.CompletionFIM, FineTuning: decoded.FineTuning,
		Vision: decoded.Vision, OCR: decoded.OCR, Classification: decoded.Classification,
		Moderation: decoded.Moderation, Audio: decoded.Audio,
		AudioTranscription: decoded.AudioTranscription, Additional: fields,
	}
	return nil
}

func (c ModelCapabilities) MarshalJSON() ([]byte, error) {
	fields := make(map[string]json.RawMessage, len(c.Additional)+len(knownModelCapabilities))
	maps.Copy(fields, c.Additional)
	known := map[ModelCapability]bool{
		ModelCapabilityChat: c.CompletionChat, ModelCapabilityFunctionCalling: c.FunctionCalling,
		ModelCapabilityFIM: c.CompletionFIM, ModelCapabilityFineTuning: c.FineTuning,
		ModelCapabilityVision: c.Vision, ModelCapabilityOCR: c.OCR,
		ModelCapabilityClassification: c.Classification, ModelCapabilityModeration: c.Moderation,
		ModelCapabilityAudio: c.Audio, ModelCapabilityAudioTranscription: c.AudioTranscription,
	}
	for name, enabled := range known {
		encoded, err := json.Marshal(enabled)
		if err != nil {
			return nil, err
		}
		fields[string(name)] = encoded
	}
	return json.Marshal(fields)
}

// Supports reports the boolean value of a known or future capability field.
// Non-boolean future fields are not treated as supported.
func (c ModelCapabilities) Supports(capability ModelCapability) bool {
	switch capability {
	case ModelCapabilityChat:
		return c.CompletionChat
	case ModelCapabilityFunctionCalling:
		return c.FunctionCalling
	case ModelCapabilityFIM:
		return c.CompletionFIM
	case ModelCapabilityFineTuning:
		return c.FineTuning
	case ModelCapabilityVision:
		return c.Vision
	case ModelCapabilityOCR:
		return c.OCR
	case ModelCapabilityClassification:
		return c.Classification
	case ModelCapabilityModeration:
		return c.Moderation
	case ModelCapabilityAudio:
		return c.Audio
	case ModelCapabilityAudioTranscription:
		return c.AudioTranscription
	default:
		var enabled bool
		return json.Unmarshal(c.Additional[string(capability)], &enabled) == nil && enabled
	}
}

// ModelCard describes either a base or fine-tuned model returned by /v1/models.
type ModelCard struct {
	ID                          string            `json:"id"`
	Object                      string            `json:"object"`
	Created                     int64             `json:"created"`
	OwnedBy                     string            `json:"owned_by"`
	Capabilities                ModelCapabilities `json:"capabilities"`
	Name                        *string           `json:"name"`
	Description                 *string           `json:"description"`
	MaxContextLength            int               `json:"max_context_length"`
	Aliases                     []string          `json:"aliases"`
	Deprecation                 *time.Time        `json:"deprecation"`
	DeprecationReplacementModel *string           `json:"deprecation_replacement_model"`
	DefaultModelTemperature     *float64          `json:"default_model_temperature"`
	Type                        string            `json:"type"`
	Job                         string            `json:"job,omitempty"`
	Root                        string            `json:"root,omitempty"`
	Archived                    bool              `json:"archived,omitempty"`
}

// ModelList is the list models API response.
type ModelList struct {
	Object string      `json:"object"`
	Data   []ModelCard `json:"data"`
}

// FilterByCapability returns model cards that explicitly advertise capability.
// It never infers support from a model name.
func (l ModelList) FilterByCapability(capability ModelCapability) []ModelCard {
	models := make([]ModelCard, 0, len(l.Data))
	for _, model := range l.Data {
		if model.Capabilities.Supports(capability) {
			models = append(models, model)
		}
	}
	return models
}

// ListModels returns models available to the API key (GET /v1/models).
func (c *Client) ListModels(ctx context.Context) (ModelList, error) {
	var list ModelList
	if err := c.getJSON(ctx, pathModels, nil, &list); err != nil {
		return ModelList{}, fmt.Errorf("mistral: list models: %w", err)
	}
	return list, nil
}

// GetModel returns one model card (GET /v1/models/{model_id}).
func (c *Client) GetModel(ctx context.Context, modelID string) (ModelCard, error) {
	id, err := pathID("model id", modelID)
	if err != nil {
		return ModelCard{}, err
	}
	var model ModelCard
	if err := c.getJSON(ctx, pathModels+"/"+id, nil, &model); err != nil {
		return ModelCard{}, fmt.Errorf("mistral: get model: %w", err)
	}
	return model, nil
}
