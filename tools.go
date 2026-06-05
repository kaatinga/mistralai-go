package mistralai

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
}

// ToolCall is one function invocation requested by the model.
type ToolCall struct {
	ID       string       `json:"id"`
	Type     string       `json:"type"`
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

// FunctionTool builds a function tool definition.
func FunctionTool(name, description string, parameters map[string]any) Tool {
	return Tool{
		Type: ToolTypeFunction,
		Function: FunctionDef{
			Name:        name,
			Description: description,
			Parameters:  parameters,
		},
	}
}

// ToolChoiceNamed returns a tool_choice value that forces one function.
func ToolChoiceNamed(name string) NamedToolChoice {
	return NamedToolChoice{
		Type: ToolTypeFunction,
		Function: NamedToolChoiceFunc{
			Name: name,
		},
	}
}
