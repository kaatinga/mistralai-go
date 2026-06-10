package mistralai

import (
	"encoding/json"
	"fmt"
)

// Tool and finish-reason constants aligned with Mistral Chat Completions API.
const (
	RoleTool = "tool"

	ToolTypeFunction = "function"

	FinishReasonStop      = "stop"
	FinishReasonToolCalls = "tool_calls"

	ToolChoiceAuto     = "auto"
	ToolChoiceNone     = "none"
	ToolChoiceAny      = "any"
	ToolChoiceRequired = "required"
)

// Tool is one callable function exposed to the model.
type Tool struct {
	Type     string      `json:"type"`
	Function FunctionDef `json:"function"`
}

// FunctionDef describes a function the model may invoke.
type FunctionDef struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters"`
	Strict      bool           `json:"strict,omitempty"`
}

// ToolCall is one function invocation requested by the model.
type ToolCall struct {
	ID       string       `json:"id"`
	Type     string       `json:"type"`
	Index    *int         `json:"index,omitempty"`
	Function FunctionCall `json:"function"`
}

// FunctionCall holds the function name and JSON-encoded arguments string.
type FunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// NamedToolChoice forces the model to call one function by name.
type NamedToolChoice struct {
	Type     string              `json:"type"`
	Function NamedToolChoiceFunc `json:"function"`
}

// NamedToolChoiceFunc is the function name inside a named tool_choice.
type NamedToolChoiceFunc struct {
	Name string `json:"name"`
}

// ToolChoice is the chat tool_choice union: a mode string or a forced named
// function. Build it with ToolChoiceMode or ToolChoiceFunction; the zero value
// omits the field from the request.
type ToolChoice struct {
	value any
}

// ToolChoiceMode selects a tool-choice mode: ToolChoiceAuto, ToolChoiceNone,
// ToolChoiceAny, or ToolChoiceRequired.
func ToolChoiceMode(mode string) ToolChoice {
	return ToolChoice{value: mode}
}

// ToolChoiceFunction forces the model to call the named function.
func ToolChoiceFunction(name string) ToolChoice {
	return ToolChoice{value: NamedToolChoice{
		Type:     ToolTypeFunction,
		Function: NamedToolChoiceFunc{Name: name},
	}}
}

// IsZero reports whether the value is unset, so `json:"...,omitzero"` drops it.
func (t ToolChoice) IsZero() bool {
	return t.value == nil
}

// MarshalJSON encodes a mode string or named-function object per the Mistral
// chat completions API.
func (t ToolChoice) MarshalJSON() ([]byte, error) {
	if t.value == nil {
		return []byte("null"), nil
	}
	return json.Marshal(t.value)
}

// UnmarshalJSON decodes either union shape, so request bodies round-trip
// (e.g. through batch JSONL).
func (t *ToolChoice) UnmarshalJSON(data []byte) error {
	var mode string
	if err := json.Unmarshal(data, &mode); err == nil {
		t.value = mode
		return nil
	}
	var named NamedToolChoice
	if err := json.Unmarshal(data, &named); err != nil {
		return fmt.Errorf("mistral: tool_choice must be a string or named function: %w", err)
	}
	t.value = named
	return nil
}

// FunctionTool builds a function tool definition. A nil parameters map defaults
// to an empty object schema so the request never sends "parameters": null.
func FunctionTool(name, description string, parameters map[string]any) Tool {
	if parameters == nil {
		parameters = map[string]any{"type": "object"}
	}
	return Tool{
		Type: ToolTypeFunction,
		Function: FunctionDef{
			Name:        name,
			Description: description,
			Parameters:  parameters,
		},
	}
}
