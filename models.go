package mistralai

// API DTOs aligned with https://github.com/mistralai/platform-docs-public OpenAPI.

type uploadFileResponse struct {
	ID       string `json:"id"`
	Object   string `json:"object"`
	Bytes    int64  `json:"bytes"`
	Filename string `json:"filename"`
	Purpose  string `json:"purpose"`
}

type ocrRequestBody struct {
	Model                    string          `json:"model"`
	Document                 fileDocument    `json:"document"`
	Pages                    []int           `json:"pages,omitempty"`
	ID                       string          `json:"id,omitempty"`
	TableFmt                 string          `json:"table_format,omitempty"`
	Include                  *bool           `json:"include_image_base64,omitempty"`
	ExtractH                 *bool           `json:"extract_header,omitempty"`
	ExtractF                 *bool           `json:"extract_footer,omitempty"`
	DocumentAnnotationFormat *ResponseFormat `json:"document_annotation_format,omitempty"`
	DocumentAnnotationPrompt *string         `json:"document_annotation_prompt,omitempty"`
}

// ResponseFormat selects structured OCR output (see Mistral OCR API).
type ResponseFormat struct {
	Type       string      `json:"type"`
	JSONSchema *JSONSchema `json:"json_schema,omitempty"`
}

// JSONSchema is the schema wrapper for document_annotation_format.
type JSONSchema struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Schema      map[string]any `json:"schema"`
	Strict      bool           `json:"strict,omitempty"`
}

type fileDocument struct {
	Type   string `json:"type"`
	FileID string `json:"file_id"`
}

type apiErrorResponse struct {
	Object  string `json:"object"`
	Message string `json:"message"`
	Type    string `json:"type"`
	Code    any    `json:"code"`
}

// OCRPage is one page from a Mistral OCR response.
type OCRPage struct {
	Index      int                `json:"index"`
	Markdown   string             `json:"markdown"`
	Images     []OCRImage         `json:"images,omitempty"`
	Tables     []OCRTable         `json:"tables,omitempty"`
	Hyperlinks []string           `json:"hyperlinks,omitempty"`
	Header     *string            `json:"header,omitempty"`
	Footer     *string            `json:"footer,omitempty"`
	Dimensions *OCRPageDimensions `json:"dimensions,omitempty"`
}

// OCRImage is an extracted image on an OCR page.
type OCRImage struct {
	ID           string `json:"id"`
	TopLeftX     int    `json:"top_left_x"`
	TopLeftY     int    `json:"top_left_y"`
	BottomRightX int    `json:"bottom_right_x"`
	BottomRightY int    `json:"bottom_right_y"`
	ImageBase64  string `json:"image_base64,omitempty"`
}

// OCRTable is an extracted table on an OCR page.
type OCRTable struct {
	ID      string `json:"id"`
	Content string `json:"content"`
	Format  string `json:"format"`
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
