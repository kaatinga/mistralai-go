package mistralai

import (
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"math"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
)

func TestEmbeddings_singleAndBatch(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/embeddings" || r.Method != http.MethodPost {
			http.NotFound(w, r)
			return
		}
		if r.Header.Get("Authorization") != "Bearer test-key" {
			t.Errorf("authorization %q", r.Header.Get("Authorization"))
		}
		var body struct {
			Model           string         `json:"model"`
			Input           any            `json:"input"`
			EncodingFormat  string         `json:"encoding_format"`
			OutputDimension *int           `json:"output_dimension"`
			OutputDType     string         `json:"output_dtype"`
			Metadata        map[string]any `json:"metadata"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatal(err)
		}
		if body.Model != EmbeddingModelMistralEmbed {
			t.Errorf("model = %q", body.Model)
		}
		if body.EncodingFormat != EncodingFormatFloat {
			t.Errorf("encoding_format = %q", body.EncodingFormat)
		}
		if body.OutputDimension == nil || *body.OutputDimension != 512 {
			t.Errorf("output_dimension = %v", body.OutputDimension)
		}
		if body.OutputDType != OutputDTypeFloat {
			t.Errorf("output_dtype = %q", body.OutputDType)
		}
		if body.Metadata["source"] != "test" {
			t.Errorf("metadata = %v", body.Metadata)
		}

		inputs, ok := body.Input.([]any)
		if !ok || len(inputs) != 2 {
			t.Fatalf("input = %#v", body.Input)
		}

		_ = json.NewEncoder(w).Encode(EmbeddingResponse{
			ID:     "emb-1",
			Object: "list",
			Model:  body.Model,
			Data: []EmbeddingData{
				{Index: 0, Object: "embedding", Embedding: mustJSON(t, []float64{0.1, 0.2})},
				{Index: 1, Object: "embedding", Embedding: mustJSON(t, []float64{0.3, 0.4})},
			},
			Usage: UsageInfo{PromptTokens: 15, TotalTokens: 15},
		})
	}))
	defer srv.Close()

	cl, err := NewClient("test-key", WithBaseURL(srv.URL))
	if err != nil {
		t.Fatal(err)
	}

	dim := 512
	resp, err := cl.Embeddings(context.Background(), EmbeddingRequest{
		Model:           EmbeddingModelMistralEmbed,
		Input:           EmbeddingInputStrings("first", "second"),
		EncodingFormat:  EncodingFormatFloat,
		OutputDimension: &dim,
		OutputDType:     OutputDTypeFloat,
		Metadata:        map[string]any{"source": "test"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if resp.ID != "emb-1" || resp.Model != EmbeddingModelMistralEmbed {
		t.Fatalf("response = %+v", resp)
	}
	vecs, err := resp.Float64Vectors()
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(vecs, [][]float64{{0.1, 0.2}, {0.3, 0.4}}) {
		t.Fatalf("vectors = %v", vecs)
	}
}

func TestEmbeddings_singleInput(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Model string `json:"model"`
			Input string `json:"input"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatal(err)
		}
		if body.Model != EmbeddingModelMistralEmbed {
			t.Errorf("model = %q", body.Model)
		}
		if body.Input != "hello" {
			t.Errorf("input = %q", body.Input)
		}
		_ = json.NewEncoder(w).Encode(EmbeddingResponse{
			Object: "list",
			Model:  body.Model,
			Data: []EmbeddingData{
				{Index: 0, Object: "embedding", Embedding: mustJSON(t, []float64{1})},
			},
		})
	}))
	defer srv.Close()

	cl, err := NewClient("test-key", WithBaseURL(srv.URL))
	if err != nil {
		t.Fatal(err)
	}

	resp, err := cl.Embeddings(context.Background(), EmbeddingRequest{
		Model: EmbeddingModelMistralEmbed,
		Input: EmbeddingInputString("hello"),
	})
	if err != nil {
		t.Fatal(err)
	}
	vec, err := resp.Data[0].Float64s()
	if err != nil || len(vec) != 1 || vec[0] != 1 {
		t.Fatalf("vec = %v err = %v", vec, err)
	}
}

func TestEmbeddings_base64Encoding(t *testing.T) {
	raw := make([]byte, 8)
	binary.LittleEndian.PutUint32(raw[0:4], math.Float32bits(0.5))
	binary.LittleEndian.PutUint32(raw[4:8], math.Float32bits(-1.25))
	encoded := base64.StdEncoding.EncodeToString(raw)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(EmbeddingResponse{
			Object: "list",
			Model:  EmbeddingModelMistralEmbed,
			Data: []EmbeddingData{
				{Index: 0, Object: "embedding", Embedding: mustJSON(t, encoded)},
			},
		})
	}))
	defer srv.Close()

	cl, err := NewClient("test-key", WithBaseURL(srv.URL))
	if err != nil {
		t.Fatal(err)
	}

	resp, err := cl.Embeddings(context.Background(), EmbeddingRequest{
		Model:          EmbeddingModelMistralEmbed,
		Input:          EmbeddingInputString("x"),
		EncodingFormat: EncodingFormatBase64,
	})
	if err != nil {
		t.Fatal(err)
	}
	f32, err := resp.Data[0].Float32s()
	if err != nil {
		t.Fatal(err)
	}
	if len(f32) != 2 || f32[0] != 0.5 || f32[1] != -1.25 {
		t.Fatalf("float32s = %v", f32)
	}
}

func TestEmbeddings_validation(t *testing.T) {
	cl, err := NewClient("k", WithBaseURL("http://127.0.0.1:1"))
	if err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name string
		req  EmbeddingRequest
		want string
	}{
		{"missing model", EmbeddingRequest{Input: EmbeddingInputString("x")}, "model is required"},
		{"missing input", EmbeddingRequest{Model: "mistral-embed"}, "input is required"},
		{"empty string", EmbeddingRequest{Model: "mistral-embed", Input: EmbeddingInputString("  ")}, "input is required"},
		{"empty batch", EmbeddingRequest{Model: "mistral-embed", Input: EmbeddingInputStrings()}, "input is required"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := cl.Embeddings(context.Background(), tc.req)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("err = %v want substring %q", err, tc.want)
			}
			if !errors.Is(err, ErrInvalidRequest) {
				t.Fatalf("err = %v want errors.Is ErrInvalidRequest", err)
			}
		})
	}
}

func TestEmbeddingEntry_batchJSONL(t *testing.T) {
	jsonl, err := BuildBatchInputJSONL([]BatchEntry{
		EmbeddingEntry("e1", EmbeddingRequest{
			Model: EmbeddingModelMistralEmbed,
			Input: EmbeddingInputString("doc"),
		}),
	})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(jsonl), `"custom_id":"e1"`) || !strings.Contains(string(jsonl), `"model":"mistral-embed"`) {
		t.Fatalf("jsonl = %s", jsonl)
	}
}

func mustJSON(t *testing.T, v any) json.RawMessage {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	return b
}
