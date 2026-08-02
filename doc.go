// Package mistralai is a community-maintained, third-party client for the
// Mistral AI platform API. It is not an official Mistral SDK.
//
// # Field conventions
//
// Request fields are pointers if and only if an explicit zero value differs
// from leaving the field unset (e.g. *bool toggles, *int page numbers).
// Everything else is a plain value, so requests can be built with flat
// composite literals.
//
// # Validation policy
//
// Client-side validation covers only structurally required fields and mutual
// exclusivity — never enum membership, so new server-side values (models,
// formats, dtypes) work without an SDK release. Validation failures wrap
// ErrInvalidRequest; API-side rejections surface as *APIError.
package mistralai
