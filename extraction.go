package mistralai

// DefaultBuildingExtractionPrompt guides OCR document annotation for property documents.
const DefaultBuildingExtractionPrompt = `Extract building data from this document.
Return JSON with:
- building: object with address (street, city, postal code, country — use fields present in the document)
- apartments: array of apartments/units (name, number, or identifier when available)
- rooms: array of rooms (name/number and apartment or unit they belong to when available)
Use empty strings and empty arrays when information is missing. Do not invent data.`

// BuildingDocumentJSONSchema is the document_annotation_format for building/apartment/room extraction.
func BuildingDocumentJSONSchema() ResponseFormat {
	return ResponseFormat{
		Type: "json_schema",
		JSONSchema: &JSONSchema{
			Name:   "building_document",
			Strict: true,
			Schema: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"building": map[string]any{
						"type": "object",
						"properties": map[string]any{
							"address": map[string]any{
								"type": "object",
								"properties": map[string]any{
									"street":      map[string]any{"type": "string"},
									"city":        map[string]any{"type": "string"},
									"postal_code": map[string]any{"type": "string"},
									"country":     map[string]any{"type": "string"},
									"full":        map[string]any{"type": "string"},
								},
								"additionalProperties": false,
							},
						},
						"required":             []any{"address"},
						"additionalProperties": false,
					},
					"apartments": map[string]any{
						"type": "array",
						"items": map[string]any{
							"type": "object",
							"properties": map[string]any{
								"name":       map[string]any{"type": "string"},
								"number":     map[string]any{"type": "string"},
								"identifier": map[string]any{"type": "string"},
							},
							"additionalProperties": false,
						},
					},
					"rooms": map[string]any{
						"type": "array",
						"items": map[string]any{
							"type": "object",
							"properties": map[string]any{
								"name":      map[string]any{"type": "string"},
								"number":    map[string]any{"type": "string"},
								"apartment": map[string]any{"type": "string"},
							},
							"additionalProperties": false,
						},
					},
				},
				"required":             []any{"building", "apartments", "rooms"},
				"additionalProperties": false,
			},
		},
	}
}

// ExtractedBuildingDocument is the expected JSON shape in document_annotation.
type ExtractedBuildingDocument struct {
	Building struct {
		Address struct {
			Street     string `json:"street"`
			City       string `json:"city"`
			PostalCode string `json:"postal_code"`
			Country    string `json:"country"`
			Full       string `json:"full"`
		} `json:"address"`
	} `json:"building"`
	Apartments []struct {
		Name       string `json:"name"`
		Number     string `json:"number"`
		Identifier string `json:"identifier"`
	} `json:"apartments"`
	Rooms []struct {
		Name      string `json:"name"`
		Number    string `json:"number"`
		Apartment string `json:"apartment"`
	} `json:"rooms"`
}
