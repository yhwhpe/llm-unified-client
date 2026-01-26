package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// qwenClient implements Client for Alibaba Qwen
type qwenClient struct {
	config     Config
	httpClient *http.Client
}

// newQwenClient creates a new Qwen client
func newQwenClient(config Config) (*qwenClient, error) {
	if config.APIKey == "" {
		return nil, fmt.Errorf("API key is required")
	}

	if config.BaseURL == "" {
		config.BaseURL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
	}

	if config.Timeout == 0 {
		config.Timeout = 30 * time.Second
	}

	if config.DefaultModel == "" {
		config.DefaultModel = "qwen3-next-80b-a3b-instruct"
	}

	httpClient := &http.Client{
		Timeout: config.Timeout,
	}

	return &qwenClient{
		config:     config,
		httpClient: httpClient,
	}, nil
}

// Generate sends a request to Qwen and returns the response
func (c *qwenClient) Generate(ctx context.Context, request Request) (*Response, error) {
	startTime := time.Now()

	// Prepare the request payload
	payload := c.buildPayload(request)

	jsonPayload, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Create HTTP request (use OpenAI-compatible endpoint)
	req, err := http.NewRequestWithContext(ctx, "POST", c.config.BaseURL+"/chat/completions", bytes.NewBuffer(jsonPayload))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.config.APIKey)
	req.Header.Set("X-DashScope-SSE", "disable") // Disable SSE for simplicity

	// Send request
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	// Read response
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("Qwen API error %d: %s", resp.StatusCode, string(body))
	}

	// Parse response (OpenAI-compatible format for compatible-mode)
	var apiResp struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
				Role    string `json:"role"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
		Usage struct {
			PromptTokens     int `json:"prompt_tokens"`
			CompletionTokens int `json:"completion_tokens"`
			TotalTokens      int `json:"total_tokens"`
		} `json:"usage"`
	}

	if err := json.Unmarshal(body, &apiResp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	if len(apiResp.Choices) == 0 {
		return nil, fmt.Errorf("no choices in response")
	}

	responseTime := time.Since(startTime)

	return &Response{
		Content:      apiResp.Choices[0].Message.Content,
		Role:         RoleAssistant,
		TokensUsed:   apiResp.Usage.TotalTokens,
		ResponseTime: responseTime,
	}, nil
}

// GenerateWithHistory generates a response using chat history
func (c *qwenClient) GenerateWithHistory(ctx context.Context, history ChatHistory, userMessage string, systemPrompt string) (*Response, error) {
	request := BuildChatRequest(history.GetMessages(), userMessage)
	if systemPrompt != "" {
		request.AddSystemMessage(systemPrompt)
	}
	return c.Generate(ctx, request)
}

// Close closes the client
func (c *qwenClient) Close() error {
	return nil
}

// GetConfig returns the client configuration
func (c *qwenClient) GetConfig() Config {
	return c.config
}

// buildPayload builds the request payload for Qwen API (OpenAI-compatible)
func (c *qwenClient) buildPayload(request Request) map[string]interface{} {
	// Convert messages to OpenAI format
	var messages []map[string]interface{}
	for _, msg := range request.Messages {
		messages = append(messages, map[string]interface{}{
			"role":    string(msg.Role),
			"content": msg.Content,
		})
	}

	payload := map[string]interface{}{
		"model":      c.getModel(request.Model),
		"messages":   messages,
		"max_tokens": c.getMaxTokens(request.MaxTokens),
	}

	// Add temperature if set
	if request.Temperature != nil {
		payload["temperature"] = *request.Temperature
	} else if c.config.DefaultTemperature != nil {
		payload["temperature"] = *c.config.DefaultTemperature
	}

	// Add top_p if set
	if request.TopP != nil {
		payload["top_p"] = *request.TopP
	} else if c.config.DefaultTopP != nil {
		payload["top_p"] = *c.config.DefaultTopP
	}

	// Add top_k if set (Qwen-specific parameter)
	if request.TopK != nil {
		payload["top_k"] = *request.TopK
	} else if c.config.DefaultTopK != nil {
		payload["top_k"] = *c.config.DefaultTopK
	}

	// Add any extra parameters (e.g., enable_thinking for models that support it)
	// Users can pass enable_thinking via request.ExtraParams if needed
	for key, value := range request.ExtraParams {
		payload[key] = value
	}

	return payload
}

// buildPromptFromMessages is no longer needed for OpenAI-compatible format
// Messages are sent as array in buildPayload
func (c *qwenClient) buildPromptFromMessages(messages []Message) string {
	// This method is kept for backward compatibility but not used
	return ""
}

// getModel returns the model to use for the request
func (c *qwenClient) getModel(override *string) string {
	if override != nil {
		return *override
	}
	return c.config.DefaultModel
}

// getMaxTokens returns the max tokens to use
func (c *qwenClient) getMaxTokens(override *int) int {
	if override != nil {
		return *override
	}
	if c.config.DefaultMaxTokens != nil {
		return *c.config.DefaultMaxTokens
	}
	return 1500 // Default for Qwen
}
