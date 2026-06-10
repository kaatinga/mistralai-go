package mistralai

import "fmt"

// OCROptions holds the optional processing fields shared by every OCR entry
// point. The synchronous requests (OCRRequest, OCRURLRequest, OCRFileRequest)
// expose the same fields flat; batch entries set them via OCREntryOption.
type OCROptions struct {
	// Pages limits processing to zero-based page indices.
	Pages []int
	// TableFormat is "markdown" or "html" when set.
	TableFormat string
	// IncludeImageBase64 requests base64 image payloads in the response when true.
	IncludeImageBase64 *bool
	ExtractHeader      *bool
	ExtractFooter      *bool
	// ID is an optional client-side correlation id forwarded to the API.
	ID string
	// DocumentAnnotationPrompt guides structured extraction for the whole document.
	// DocumentAnnotationFormat must be set when using a prompt.
	DocumentAnnotationPrompt string
	DocumentAnnotationFormat *ResponseFormat
}

func (o OCROptions) validate() error {
	if o.DocumentAnnotationPrompt != "" && o.DocumentAnnotationFormat == nil {
		return fmt.Errorf("%w: document_annotation_format is required with document_annotation_prompt", ErrInvalidRequest)
	}
	return nil
}

// ocrBody maps the shared OCR fields onto the wire body, defaulting the model
// to DefaultOCRModel when empty.
func ocrBody(model string, doc ocrDocument, o OCROptions) ocrRequestBody {
	if model == "" {
		model = DefaultOCRModel
	}
	b := ocrRequestBody{
		Model:                    model,
		Document:                 doc,
		Pages:                    o.Pages,
		ID:                       o.ID,
		TableFmt:                 o.TableFormat,
		Include:                  o.IncludeImageBase64,
		ExtractH:                 o.ExtractHeader,
		ExtractF:                 o.ExtractFooter,
		DocumentAnnotationFormat: o.DocumentAnnotationFormat,
	}
	if o.DocumentAnnotationPrompt != "" {
		b.DocumentAnnotationPrompt = new(o.DocumentAnnotationPrompt)
	}
	return b
}
