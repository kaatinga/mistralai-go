package mistralai

import "encoding/json"

// API DTOs aligned with https://github.com/mistralai/platform-docs-public OpenAPI.

type ocrRequestBody struct {
	Model                       string          `json:"model"`
	Document                    ocrDocument     `json:"document"`
	Pages                       []int           `json:"pages,omitempty"`
	ID                          string          `json:"id,omitempty"`
	TableFmt                    string          `json:"table_format,omitempty"`
	Include                     *bool           `json:"include_image_base64,omitempty"`
	ImageLimit                  *int            `json:"image_limit,omitempty"`
	ImageMinSize                *int            `json:"image_min_size,omitempty"`
	BBoxAnnotationFormat        *ResponseFormat `json:"bbox_annotation_format,omitempty"`
	IncludeBlocks               *bool           `json:"include_blocks,omitempty"`
	ConfidenceScoresGranularity string          `json:"confidence_scores_granularity,omitempty"`
	ExtractH                    *bool           `json:"extract_header,omitempty"`
	ExtractF                    *bool           `json:"extract_footer,omitempty"`
	DocumentAnnotationFormat    *ResponseFormat `json:"document_annotation_format,omitempty"`
	DocumentAnnotationPrompt    *string         `json:"document_annotation_prompt,omitempty"`
}

// Response format type values for ResponseFormat.Type.
const (
	ResponseFormatText       = "text"
	ResponseFormatJSONObject = "json_object"
	ResponseFormatJSONSchema = "json_schema"
)

// ResponseFormat selects structured output for chat completions
// (ChatCompletionRequest.ResponseFormat) and OCR document annotations.
type ResponseFormat struct {
	Type       string      `json:"type"`
	JSONSchema *JSONSchema `json:"json_schema,omitempty"`
}

// JSONSchemaFormat builds a strict json_schema response_format from a schema
// name and a JSON Schema document. Use it for ChatCompletionRequest.ResponseFormat
// and OCRRequest.DocumentAnnotationFormat.
func JSONSchemaFormat(name string, schema map[string]any) *ResponseFormat {
	return &ResponseFormat{
		Type: ResponseFormatJSONSchema,
		JSONSchema: &JSONSchema{
			Name:   name,
			Schema: schema,
			Strict: true,
		},
	}
}

// JSONSchema is the schema wrapper for document_annotation_format.
type JSONSchema struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Schema      map[string]any `json:"schema"`
	Strict      bool           `json:"strict,omitempty"`
}

// ocrDocument is the OCR request "document" union: a file chunk (file_id),
// a document_url chunk, or an image_url chunk, discriminated by Type.
type ocrDocument struct {
	Type         string `json:"type"`
	FileID       string `json:"file_id,omitempty"`
	DocumentURL  string `json:"document_url,omitempty"`
	DocumentName string `json:"document_name,omitempty"`
	ImageURL     string `json:"image_url,omitempty"`
}

type apiErrorResponse struct {
	Object string `json:"object"`
	// Message is usually a string, but validation errors may nest a detail list.
	Message json.RawMessage `json:"message"`
	Type    string          `json:"type"`
	Code    any             `json:"code"`
	// Detail carries FastAPI-style validation errors on 422 responses.
	Detail json.RawMessage `json:"detail"`
}

// OCRPage is one page from a Mistral OCR response.
type OCRPage struct {
	Index           int                `json:"index"`
	Markdown        string             `json:"markdown"`
	Images          []OCRImage         `json:"images,omitempty"`
	Tables          []OCRTable         `json:"tables,omitempty"`
	Hyperlinks      []string           `json:"hyperlinks,omitempty"`
	Header          *string            `json:"header,omitempty"`
	Footer          *string            `json:"footer,omitempty"`
	Dimensions      *OCRPageDimensions `json:"dimensions,omitempty"`
	Blocks          []OCRBlock         `json:"blocks,omitempty"`
	ConfidenceScore *float64           `json:"confidence_score,omitempty"`
}

// OCRImage is an extracted image on an OCR page.
type OCRImage struct {
	ID              string   `json:"id"`
	TopLeftX        int      `json:"top_left_x"`
	TopLeftY        int      `json:"top_left_y"`
	BottomRightX    int      `json:"bottom_right_x"`
	BottomRightY    int      `json:"bottom_right_y"`
	ImageBase64     string   `json:"image_base64,omitempty"`
	ConfidenceScore *float64 `json:"confidence_score,omitempty"`
}

// OCRBlock is a bounding-box annotated OCR block.
type OCRBlock struct {
	ID              string     `json:"id"`
	Type            string     `json:"type"`
	Text            string     `json:"text"`
	BoundingBox     []OCRPoint `json:"bounding_box,omitempty"`
	ConfidenceScore *float64   `json:"confidence_score,omitempty"`
}

type OCRPoint struct {
	X float64 `json:"x"`
	Y float64 `json:"y"`
}

// OCRTable is an extracted table on an OCR page.
type OCRTable struct {
	ID              string   `json:"id"`
	Content         string   `json:"content"`
	Format          string   `json:"format"`
	ConfidenceScore *float64 `json:"confidence_score,omitempty"`
}

// OCRPageDimensions holds page size metadata from OCR.
type OCRPageDimensions struct {
	DPI    int `json:"dpi"`
	Height int `json:"height"`
	Width  int `json:"width"`
}

// OCRUsageInfo reports billing-related OCR usage.
type OCRUsageInfo struct {
	PagesProcessed int    `json:"pages_processed"`
	DocSizeBytes   *int64 `json:"doc_size_bytes"`
}

// OCRResponse is the successful OCR API payload.
type OCRResponse struct {
	Pages              []OCRPage    `json:"pages"`
	Model              string       `json:"model"`
	DocumentAnnotation *string      `json:"document_annotation,omitempty"`
	UsageInfo          OCRUsageInfo `json:"usage_info"`
}
