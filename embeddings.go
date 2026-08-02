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
)

type EmbeddingEncoding string
type EmbeddingDType string

const (
	EncodingFormatFloat  EmbeddingEncoding = "float"
	EncodingFormatBase64 EmbeddingEncoding = "base64"
	OutputDTypeFloat     EmbeddingDType    = "float"
	OutputDTypeInt8      EmbeddingDType    = "int8"
	OutputDTypeUint8     EmbeddingDType    = "uint8"
	OutputDTypeBinary    EmbeddingDType    = "binary"
	OutputDTypeUBinary   EmbeddingDType    = "ubinary"
)

// EmbeddingRequest is the body for POST /v1/embeddings.
type EmbeddingRequest struct {
	Model           string            `json:"model"`
	Input           []string          `json:"input"`
	EncodingFormat  EmbeddingEncoding `json:"encoding_format,omitempty"`
	OutputDimension *int              `json:"output_dimension,omitempty"`
	OutputDType     EmbeddingDType    `json:"output_dtype,omitempty"`
	Metadata        map[string]any    `json:"metadata,omitempty"`
}

// EmbeddingData is one vector in an embeddings response.
type EmbeddingData struct {
	Index     int             `json:"index"`
	Object    string          `json:"object"`
	Embedding json.RawMessage `json:"embedding"`
}

// ErrEmbeddingType reports a decoder/request dtype mismatch.
type ErrEmbeddingType struct {
	Want EmbeddingDType
	Got  EmbeddingDType
}

func (e *ErrEmbeddingType) Error() string {
	return fmt.Sprintf("mistral: embedding dtype %q cannot be decoded as %q", e.Got, e.Want)
}

func embeddingTypeError(want, got EmbeddingDType) error {
	return &ErrEmbeddingType{Want: want, Got: got}
}

// Float32s decodes a float embedding. The response dtype is authoritative;
// callers must use the decoder matching EmbeddingResponse.OutputDType.
func (d EmbeddingData) Float32s() ([]float32, error) {
	return decodeFloat32JSON(d.Embedding)
}

func decodeFloat32JSON(raw json.RawMessage) ([]float32, error) {
	if len(raw) == 0 {
		return nil, errors.New("mistral: embedding is empty")
	}
	var floats []float32
	if err := json.Unmarshal(raw, &floats); err == nil {
		return floats, nil
	}
	var encoded string
	if err := json.Unmarshal(raw, &encoded); err != nil {
		return nil, fmt.Errorf("mistral: decode embedding: %w", err)
	}
	f32, err := decodeBase64Float32s(encoded)
	return f32, err
}

func (d EmbeddingData) Float64s() ([]float64, error) {
	var values []float64
	if err := json.Unmarshal(d.Embedding, &values); err == nil {
		return values, nil
	}
	encoded, err := embeddingString(d.Embedding)
	if err != nil {
		return nil, err
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

func (d EmbeddingData) Int8s() ([]int8, error) {
	var values []int8
	if err := json.Unmarshal(d.Embedding, &values); err == nil {
		return values, nil
	}
	encoded, err := embeddingString(d.Embedding)
	if err != nil {
		return nil, err
	}
	raw, err := base64.StdEncoding.DecodeString(encoded)
	if err != nil {
		return nil, fmt.Errorf("mistral: decode int8 embedding: %w", err)
	}
	out := make([]int8, len(raw))
	for i, v := range raw {
		out[i] = int8(v)
	}
	return out, nil
}

func (d EmbeddingData) Uint8s() ([]uint8, error) {
	var values []uint8
	if err := json.Unmarshal(d.Embedding, &values); err == nil {
		return values, nil
	}
	encoded, err := embeddingString(d.Embedding)
	if err != nil {
		return nil, err
	}
	raw, err := base64.StdEncoding.DecodeString(encoded)
	if err != nil {
		return nil, fmt.Errorf("mistral: decode uint8 embedding: %w", err)
	}
	return raw, nil
}

func (d EmbeddingData) Binary() ([]byte, error) {
	encoded, err := embeddingString(d.Embedding)
	if err != nil {
		return nil, err
	}
	raw, err := base64.StdEncoding.DecodeString(encoded)
	if err != nil {
		return nil, fmt.Errorf("mistral: decode binary embedding: %w", err)
	}
	return raw, nil
}

func (d EmbeddingData) UBinary() ([]byte, error) { return d.Binary() }

func embeddingString(raw json.RawMessage) (string, error) {
	if len(raw) == 0 {
		return "", errors.New("mistral: embedding is empty")
	}
	var encoded string
	if err := json.Unmarshal(raw, &encoded); err != nil {
		return "", fmt.Errorf("mistral: embedding is not base64: %w", err)
	}
	return encoded, nil
}

// EmbeddingResponse is the successful embeddings API payload.
type EmbeddingResponse struct {
	ID              string            `json:"id"`
	Object          string            `json:"object"`
	Model           string            `json:"model"`
	Data            []EmbeddingData   `json:"data"`
	Usage           UsageInfo         `json:"usage"`
	EncodingFormat  EmbeddingEncoding `json:"-"`
	OutputDType     EmbeddingDType    `json:"-"`
	OutputDimension *int              `json:"-"`
}

// Float32Vectors returns decoded float vectors for every data item, in input
// index order. It rejects duplicate, missing, or out-of-range indexes.
func (r EmbeddingResponse) Float32Vectors() ([][]float32, error) {
	if r.OutputDType != "" && r.OutputDType != OutputDTypeFloat {
		return nil, embeddingTypeError(OutputDTypeFloat, r.OutputDType)
	}
	return reorderVectors(r.Data, func(d EmbeddingData) ([]float32, error) {
		return d.Float32s()
	})
}

// Float64Vectors returns decoded vectors for every data item, in index order.
func (r EmbeddingResponse) Float64Vectors() ([][]float64, error) {
	if r.OutputDType != "" && r.OutputDType != OutputDTypeFloat {
		return nil, embeddingTypeError(OutputDTypeFloat, r.OutputDType)
	}
	return reorderVectors(r.Data, func(d EmbeddingData) ([]float64, error) {
		return d.Float64s()
	})
}

func (r EmbeddingResponse) Int8Vectors() ([][]int8, error) {
	if r.OutputDType != "" && r.OutputDType != OutputDTypeInt8 {
		return nil, embeddingTypeError(OutputDTypeInt8, r.OutputDType)
	}
	return reorderVectors(r.Data, func(d EmbeddingData) ([]int8, error) { return d.Int8s() })
}

func (r EmbeddingResponse) Uint8Vectors() ([][]uint8, error) {
	if r.OutputDType != "" && r.OutputDType != OutputDTypeUint8 {
		return nil, embeddingTypeError(OutputDTypeUint8, r.OutputDType)
	}
	return reorderVectors(r.Data, func(d EmbeddingData) ([]uint8, error) { return d.Uint8s() })
}

func (r EmbeddingResponse) BinaryVectors() ([][]byte, error) {
	if r.OutputDType != "" && r.OutputDType != OutputDTypeBinary {
		return nil, embeddingTypeError(OutputDTypeBinary, r.OutputDType)
	}
	return reorderVectors(r.Data, func(d EmbeddingData) ([]byte, error) { return d.Binary() })
}

func (r EmbeddingResponse) UBinaryVectors() ([][]byte, error) {
	if r.OutputDType != "" && r.OutputDType != OutputDTypeUBinary {
		return nil, embeddingTypeError(OutputDTypeUBinary, r.OutputDType)
	}
	return reorderVectors(r.Data, func(d EmbeddingData) ([]byte, error) { return d.UBinary() })
}

func reorderVectors[T any](data []EmbeddingData, decode func(EmbeddingData) (T, error)) ([]T, error) {
	if len(data) == 0 {
		return nil, errors.New("mistral: embeddings response has no data")
	}
	out := make([]T, len(data))
	seen := make([]bool, len(data))
	for _, d := range data {
		if d.Index < 0 || d.Index >= len(data) {
			return nil, fmt.Errorf("mistral: embedding index %d out of range", d.Index)
		}
		if seen[d.Index] {
			return nil, fmt.Errorf("mistral: duplicate embedding index %d", d.Index)
		}
		seen[d.Index] = true
		value, err := decode(d)
		if err != nil {
			return nil, fmt.Errorf("mistral: embedding index %d: %w", d.Index, err)
		}
		out[d.Index] = value
	}
	for i, ok := range seen {
		if !ok {
			return nil, fmt.Errorf("mistral: missing embedding index %d", i)
		}
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
	if len(r.Input) == 0 {
		return fmt.Errorf("%w: input is required", ErrInvalidRequest)
	}
	for i, input := range r.Input {
		if strings.TrimSpace(input) == "" {
			return fmt.Errorf("%w: input %d is required", ErrInvalidRequest, i)
		}
	}
	return nil
}

// Embeddings runs POST /v1/embeddings. Model is required (e.g.
// EmbeddingModelMistralEmbed); like the other raw-request methods, this one
// never defaults it.
func (c *Client) Embeddings(ctx context.Context, req EmbeddingRequest) (EmbeddingResponse, error) {
	if err := req.validate(); err != nil {
		return EmbeddingResponse{}, err
	}

	var resp EmbeddingResponse
	if err := c.postJSON(ctx, pathEmbeddings, req, &resp); err != nil {
		return EmbeddingResponse{}, fmt.Errorf("mistral: embeddings: %w", err)
	}
	resp.EncodingFormat = req.EncodingFormat
	if resp.EncodingFormat == "" {
		resp.EncodingFormat = EncodingFormatFloat
	}
	resp.OutputDType = req.OutputDType
	if resp.OutputDType == "" {
		resp.OutputDType = OutputDTypeFloat
	}
	resp.OutputDimension = req.OutputDimension
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
