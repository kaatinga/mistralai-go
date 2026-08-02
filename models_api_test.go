package mistralai

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestModelCapabilities_unknownFieldsRoundTrip(t *testing.T) {
	input := []byte(`{"completion_chat":true,"vision":false,"future_reasoning":true,"future_config":{"level":2}}`)
	var capabilities ModelCapabilities
	if err := json.Unmarshal(input, &capabilities); err != nil {
		t.Fatal(err)
	}
	if !capabilities.Supports(ModelCapabilityChat) || !capabilities.Supports("future_reasoning") {
		t.Fatalf("capabilities = %+v", capabilities)
	}
	encoded, err := json.Marshal(capabilities)
	if err != nil {
		t.Fatal(err)
	}
	var roundTrip map[string]json.RawMessage
	if err := json.Unmarshal(encoded, &roundTrip); err != nil {
		t.Fatal(err)
	}
	if string(roundTrip["future_config"]) != `{"level":2}` {
		t.Fatalf("future_config = %s", roundTrip["future_config"])
	}
}

func TestModelListFilterByCapability(t *testing.T) {
	models := ModelList{Data: []ModelCard{
		{ID: "oddly-named", Capabilities: ModelCapabilities{CompletionChat: true}},
		{ID: "mistral-embed", Capabilities: ModelCapabilities{}},
	}}
	filtered := models.FilterByCapability(ModelCapabilityChat)
	if len(filtered) != 1 || filtered[0].ID != "oddly-named" {
		t.Fatalf("filtered = %+v", filtered)
	}
}

func TestGetModel(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.EscapedPath() != "/v1/models/team%2Fmodel" || r.Method != http.MethodGet {
			t.Fatalf("request = %s %s", r.Method, r.URL.EscapedPath())
		}
		_ = json.NewEncoder(w).Encode(ModelCard{
			ID: "team/model", Capabilities: ModelCapabilities{CompletionChat: true},
		})
	}))
	defer srv.Close()

	client, err := NewClient("key", WithBaseURL(srv.URL))
	if err != nil {
		t.Fatal(err)
	}
	model, err := client.GetModel(context.Background(), "team/model")
	if err != nil {
		t.Fatal(err)
	}
	if model.ID != "team/model" || !model.Capabilities.CompletionChat {
		t.Fatalf("model = %+v", model)
	}
	if _, err := client.GetModel(context.Background(), " "); err == nil || !strings.Contains(err.Error(), "model id") {
		t.Fatalf("err = %v", err)
	}
}
