package mistralai

// ChatMessageContentPart is one multimodal content item for Chat Completions.
type ChatMessageContentPart struct {
	Type        string `json:"type"`
	Text        string `json:"text,omitempty"`
	FileID      string `json:"file_id,omitempty"`
	FileURL     string `json:"file_url,omitempty"`
	DocumentURL string `json:"document_url,omitempty"`
	ImageURL    string `json:"image_url,omitempty"`
}
