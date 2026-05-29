package mistralai

// Ptr returns a pointer to v. It is a convenience for the optional pointer
// fields in this package — ChatCompletionRequest.Temperature, TopP,
// OCRRequest.ExtractHeader, IncludeImageBase64, and so on — where a nil pointer
// means "unset" and a pointer to a zero value (e.g. Ptr(0.0)) means
// "explicitly zero".
//
//	mistralai.ChatCompletionRequest{Temperature: mistralai.Ptr(0.0)}
func Ptr[T any](v T) *T {
	return &v
}
