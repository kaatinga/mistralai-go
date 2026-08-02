package mistralai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"image"
	"image/png"
	"os"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func TestLiveReleaseSmoke(t *testing.T) {
	if os.Getenv("MISTRAL_LIVE") != "1" {
		t.Skip("set MISTRAL_LIVE=1 to run bounded live release smoke")
	}
	key := strings.TrimSpace(os.Getenv("MISTRAL_API_KEY"))
	if key == "" {
		t.Skip("MISTRAL_API_KEY is not set")
	}
	client, err := NewClient(key)
	if err != nil {
		t.Fatal("create client:", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()

	models, err := client.ListModels(ctx)
	if err != nil {
		t.Fatal("list models:", err)
	}
	chatModels := models.FilterByCapability(ModelCapabilityChat)
	if len(chatModels) == 0 {
		t.Fatal("models response has no chat-capable model")
	}
	chatModel := preferredLiveModel(chatModels, ChatModelMistralSmallLatest)
	if _, err := client.GetModel(ctx, chatModel); err != nil {
		t.Fatal("get model:", err)
	}

	t.Run("chat", func(t *testing.T) {
		callCtx, callCancel := context.WithTimeout(ctx, 30*time.Second)
		defer callCancel()
		response, err := client.ChatCompletion(callCtx, ChatCompletionRequest{
			Model: chatModel, MaxTokens: 8,
			Messages: []ChatMessage{TextMessage(RoleUser, "Reply with one word.")},
		})
		if err != nil {
			t.Fatal("chat completion:", err)
		}
		if _, err := response.FirstText(); err != nil {
			t.Fatal("chat response shape:", err)
		}
	})

	t.Run("stream_cancellation", func(t *testing.T) {
		callCtx, callCancel := context.WithCancel(ctx)
		stream, err := client.ChatCompletionStream(callCtx, ChatCompletionRequest{
			Model: chatModel, MaxTokens: 32,
			Messages: []ChatMessage{TextMessage(RoleUser, "Count slowly from one to ten.")},
		})
		if err != nil {
			callCancel()
			t.Fatal("start chat stream:", err)
		}
		callCancel()
		var recvErr error
		for range 16 {
			_, recvErr = stream.Recv()
			if recvErr != nil {
				break
			}
		}
		closeErr := stream.Close()
		if recvErr == nil {
			t.Fatal("stream receive succeeded after cancellation")
		}
		if closeErr != nil {
			t.Fatal("close stream:", closeErr)
		}
	})

	t.Run("structured_output", func(t *testing.T) {
		type result struct {
			OK bool `json:"ok"`
		}
		schema := map[string]any{
			"type": "object", "additionalProperties": false,
			"properties": map[string]any{"ok": map[string]any{"type": "boolean"}},
			"required":   []string{"ok"},
		}
		callCtx, callCancel := context.WithTimeout(ctx, 30*time.Second)
		defer callCancel()
		_, response, err := ChatStructured[result](callCtx, client, ChatCompletionRequest{
			Model: chatModel, MaxTokens: 16,
			Messages:       []ChatMessage{TextMessage(RoleUser, "Return true.")},
			ResponseFormat: JSONSchemaFormat("result", schema),
		})
		if err != nil {
			t.Fatal("structured chat:", err)
		}
		if len(response.Choices) == 0 {
			t.Fatal("structured chat returned no choices")
		}
	})

	toolModels := models.FilterByCapability(ModelCapabilityFunctionCalling)
	if len(toolModels) == 0 {
		t.Fatal("models response has no function-calling model")
	}
	t.Run("forced_tool_once", func(t *testing.T) {
		var calls atomic.Int32
		callCtx, callCancel := context.WithTimeout(ctx, 45*time.Second)
		defer callCancel()
		response, err := ChatCompletionWithTools(callCtx, client, ChatCompletionRequest{
			Model: preferredLiveModel(toolModels, ChatModelMistralSmallLatest), MaxTokens: 32,
			Messages:   []ChatMessage{TextMessage(RoleUser, "Call ping once, then give a short final answer.")},
			Tools:      []Tool{FunctionTool("ping", "Return an availability signal", nil)},
			ToolChoice: ToolChoiceFunction("ping"),
		}, func(context.Context, ToolCall) (string, error) {
			calls.Add(1)
			return `{"ok":true}`, nil
		}, 3)
		if err != nil {
			t.Fatal("forced tool loop:", err)
		}
		if calls.Load() != 1 {
			t.Fatalf("tool calls = %d, want 1", calls.Load())
		}
		if _, err := response.FirstText(); err != nil {
			t.Fatal("tool loop final response:", err)
		}
	})

	t.Run("chained_tools", func(t *testing.T) {
		const nextToken = "mistral-live-chain-second-step"
		var firstCalls atomic.Int32
		var secondCalls atomic.Int32
		callCtx, callCancel := context.WithTimeout(ctx, 45*time.Second)
		defer callCancel()
		response, err := ChatCompletionWithTools(callCtx, client, ChatCompletionRequest{
			Model: preferredLiveModel(toolModels, ChatModelMistralSmallLatest), MaxTokens: 64,
			Messages: []ChatMessage{TextMessage(RoleUser,
				"First call first_step. Read next_token from its result, then call second_step with that exact token. Only then give a short final answer.")},
			Tools: []Tool{
				FunctionTool("first_step", "Return the token required for the second step", nil),
				FunctionTool("second_step", "Complete the chain using the token from first_step", map[string]any{
					"type": "object", "additionalProperties": false,
					"properties": map[string]any{"token": map[string]any{"type": "string"}},
					"required":   []string{"token"},
				}),
			},
			ToolChoice: ToolChoiceFunction("first_step"),
		}, func(_ context.Context, call ToolCall) (string, error) {
			switch call.Function.Name {
			case "first_step":
				firstCalls.Add(1)
				return `{"next_token":"` + nextToken + `"}`, nil
			case "second_step":
				var args struct {
					Token string `json:"token"`
				}
				if err := json.Unmarshal([]byte(call.Function.Arguments), &args); err != nil {
					return "", err
				}
				if args.Token != nextToken {
					return "", fmt.Errorf("second_step received the wrong chain token")
				}
				secondCalls.Add(1)
				return `{"done":true}`, nil
			default:
				return "", fmt.Errorf("unexpected tool %q", call.Function.Name)
			}
		}, 2)
		if err != nil {
			t.Fatal("chained tool loop:", err)
		}
		if firstCalls.Load() != 1 || secondCalls.Load() != 1 {
			t.Fatalf("tool calls: first=%d second=%d, want one each", firstCalls.Load(), secondCalls.Load())
		}
		if _, err := response.FirstText(); err != nil {
			t.Fatal("chained tool final response:", err)
		}
	})

	fimModels := models.FilterByCapability(ModelCapabilityFIM)
	if len(fimModels) == 0 {
		t.Fatal("models response has no FIM-capable model")
	}
	t.Run("fim", func(t *testing.T) {
		callCtx, callCancel := context.WithTimeout(ctx, 30*time.Second)
		defer callCancel()
		response, err := client.FIMCompletion(callCtx, FIMCompletionRequest{
			Model:  preferredLiveModel(fimModels, "codestral-latest"),
			Prompt: "func add(a, b int) int {", Suffix: "}", MaxTokens: 8,
		})
		if err != nil {
			t.Fatal("FIM completion:", err)
		}
		if len(response.Choices) == 0 {
			t.Fatal("FIM returned no choices")
		}
	})

	t.Run("embeddings", func(t *testing.T) {
		for _, testCase := range []struct {
			model string
			dtype EmbeddingDType
		}{
			{model: EmbeddingModelMistralEmbed, dtype: OutputDTypeFloat},
			{model: "codestral-embed", dtype: OutputDTypeInt8},
		} {
			callCtx, callCancel := context.WithTimeout(ctx, 30*time.Second)
			response, err := client.Embeddings(callCtx, EmbeddingRequest{
				Model: testCase.model, Input: []string{"small input"}, OutputDType: testCase.dtype,
			})
			callCancel()
			if err != nil {
				t.Fatalf("embeddings dtype %s: %v", testCase.dtype, err)
			}
			if testCase.dtype == OutputDTypeFloat {
				_, err = response.Float64Vectors()
			} else {
				_, err = response.Int8Vectors()
			}
			if err != nil {
				t.Fatalf("decode embeddings dtype %s: %v", testCase.dtype, err)
			}
		}
	})

	t.Run("tiny_ocr", func(t *testing.T) {
		var imageBytes bytes.Buffer
		img := image.NewGray(image.Rect(0, 0, 64, 64))
		for index := range img.Pix {
			img.Pix[index] = 0xff
		}
		if err := png.Encode(&imageBytes, img); err != nil {
			t.Fatal("encode tiny PNG:", err)
		}
		callCtx, callCancel := context.WithTimeout(ctx, 60*time.Second)
		defer callCancel()
		response, err := client.OCR(callCtx, OCRRequest{
			Model:         DefaultOCRModel,
			Source:        LocalFile{Name: "smoke.png", ContentType: "image/png", Reader: bytes.NewReader(imageBytes.Bytes())},
			IncludeBlocks: new(true),
		})
		if err != nil {
			t.Fatal("tiny OCR:", err)
		}
		if len(response.Pages) == 0 {
			t.Fatal("tiny OCR returned no pages")
		}
	})
}

func preferredLiveModel(models []ModelCard, preferred string) string {
	for _, model := range models {
		if model.ID == preferred {
			return model.ID
		}
		for _, alias := range model.Aliases {
			if alias == preferred {
				return alias
			}
		}
	}
	return models[0].ID
}
