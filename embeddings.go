package mistralai

import (
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"strings"
)

const (
	EmbeddingModelMistralEmbed = "mistral-embed"

	// DefaultEmbeddingModel is used when EmbeddingRequest.Model is empty.
	DefaultEmbeddingModel = EmbeddingModelMistralEmbed

	EncodingFormatFloat  = "float"
	EncodingFormatBase64 = "base64"
	OutputDTypeFloat     = "float"
	OutputDTypeInt8      = "int8"
	OutputDTypeUint8     = "uint8"
	OutputDTypeBinary    = "binary"
	OutputDTypeUBinary   = "ubinary"
)

// EmbeddingInput is the request "input" field: one string or a batch of strings.
type EmbeddingInput struct {
	value any
}

// EmbeddingInputString embeds a single string.
func EmbeddingInputString(s string) EmbeddingInput {
	return EmbeddingInput{value: s}
}

// EmbeddingInputStrings embeds multiple strings in one request.
func EmbeddingInputStrings(ss ...string) EmbeddingInput {
	return EmbeddingInput{value: ss}
}

// MarshalJSON encodes a string or []string per the Mistral embeddings API.
func (in EmbeddingInput) MarshalJSON() ([]byte, error) {
	if in.value == nil {
		return nil, fmt.Errorf("%w: embedding input is required", ErrInvalidRequest)
	}
	return json.Marshal(in.value)
}

// EmbeddingRequest is the body for POST /v1/embeddings.
type EmbeddingRequest struct {
	Model           string         `json:"model"`
	Input           EmbeddingInput `json:"input"`
	EncodingFormat  string         `json:"encoding_format,omitempty"`
	OutputDimension *int           `json:"output_dimension,omitempty"`
	OutputDType     string         `json:"output_dtype,omitempty"`
	Metadata        map[string]any `json:"metadata,omitempty"`
}

// EmbeddingData is one vector in an embeddings response.
type EmbeddingData struct {
	Index     int             `json:"index"`
	Object    string          `json:"object"`
	Embedding json.RawMessage `json:"embedding"`
}

// Float64s decodes the embedding as a float slice. For encoding_format "float"
// this is a direct JSON array; for "base64" it decodes little-endian float32 bytes.
func (d EmbeddingData) Float64s() ([]float64, error) {
	if len(d.Embedding) == 0 {
		return nil, errors.New("mistral: embedding is empty")
	}
	var floats []float64
	if err := json.Unmarshal(d.Embedding, &floats); err == nil {
		return floats, nil
	}
	var encoded string
	if err := json.Unmarshal(d.Embedding, &encoded); err != nil {
		return nil, fmt.Errorf("mistral: decode embedding: %w", err)
	}
	f32, err := decodeBase64Float32s(encoded)
	if err != nil {
		return nil, err
	}
	out := make([]float64, len(f32))
	for i, v := range f32 {
		out[i] = float64(v)
	}
	return out, nil
}

// Float32s decodes the embedding as float32 values (base64 payloads are decoded
// as float32; JSON float arrays are narrowed).
func (d EmbeddingData) Float32s() ([]float32, error) {
	f64, err := d.Float64s()
	if err != nil {
		return nil, err
	}
	out := make([]float32, len(f64))
	for i, v := range f64 {
		out[i] = float32(v)
	}
	return out, nil
}

// EmbeddingResponse is the successful embeddings API payload.
type EmbeddingResponse struct {
	ID     string          `json:"id"`
	Object string          `json:"object"`
	Model  string          `json:"model"`
	Data   []EmbeddingData `json:"data"`
	Usage  UsageInfo       `json:"usage"`
}

// Float64Vectors returns decoded vectors for every data item, in index order.
func (r EmbeddingResponse) Float64Vectors() ([][]float64, error) {
	if len(r.Data) == 0 {
		return nil, errors.New("mistral: embeddings response has no data")
	}
	out := make([][]float64, len(r.Data))
	for i, d := range r.Data {
		vec, err := d.Float64s()
		if err != nil {
			return nil, fmt.Errorf("mistral: embedding index %d: %w", d.Index, err)
		}
		out[i] = vec
	}
	return out, nil
}

// validate checks only structural requirements; enum-like fields such as
// EncodingFormat and OutputDType are passed through so new server-side values
// never require an SDK release (see the package doc validation policy).
func (r EmbeddingRequest) validate() error {
	if strings.TrimSpace(r.Model) == "" {
		return fmt.Errorf("%w: model is required", ErrInvalidRequest)
	}
	switch v := r.Input.value.(type) {
	case string:
		if strings.TrimSpace(v) == "" {
			return fmt.Errorf("%w: input is required", ErrInvalidRequest)
		}
	case []string:
		if len(v) == 0 {
			return fmt.Errorf("%w: input is required", ErrInvalidRequest)
		}
	case nil:
		return fmt.Errorf("%w: input is required", ErrInvalidRequest)
	default:
		return fmt.Errorf("%w: input must be a string or []string", ErrInvalidRequest)
	}
	return nil
}

// Embeddings runs POST /v1/embeddings. Model defaults to DefaultEmbeddingModel.
func (c *Client) Embeddings(ctx context.Context, req EmbeddingRequest) (EmbeddingResponse, error) {
	if req.Model == "" {
		req.Model = DefaultEmbeddingModel
	}
	if err := req.validate(); err != nil {
		return EmbeddingResponse{}, err
	}

	var resp EmbeddingResponse
	if err := c.postJSON(ctx, "/v1/embeddings", req, &resp); err != nil {
		return EmbeddingResponse{}, fmt.Errorf("mistral: embeddings: %w", err)
	}
	return resp, nil
}

func decodeBase64Float32s(encoded string) ([]float32, error) {
	raw, err := base64.StdEncoding.DecodeString(encoded)
	if err != nil {
		return nil, fmt.Errorf("mistral: decode base64 embedding: %w", err)
	}
	if len(raw)%4 != 0 {
		return nil, fmt.Errorf("mistral: base64 embedding length %d is not a multiple of 4", len(raw))
	}
	n := len(raw) / 4
	out := make([]float32, n)
	for i := range n {
		bits := binary.LittleEndian.Uint32(raw[i*4 : (i+1)*4])
		out[i] = math.Float32frombits(bits)
	}
	return out, nil
}
