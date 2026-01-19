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
		config.BaseURL = "https://dashscope.aliyuncs.com/api/v1"
	}

	if config.Timeout == 0 {
		config.Timeout = 30 * time.Second
	}

	if config.DefaultModel == "" {
		config.DefaultModel = "qwen-turbo"
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

	// Create HTTP request
	req, err := http.NewRequestWithContext(ctx, "POST", c.config.BaseURL+"/services/aigc/text-generation/generation", bytes.NewBuffer(jsonPayload))
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

	// Parse response (Qwen has a different response format)
	var apiResp struct {
		Output struct {
			Text string `json:"text"`
		} `json:"output"`
		Usage struct {
			TotalTokens int `json:"total_tokens"`
		} `json:"usage"`
		RequestId string `json:"request_id"`
	}

	if err := json.Unmarshal(body, &apiResp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	responseTime := time.Since(startTime)

	return &Response{
		Content:      apiResp.Output.Text,
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

// buildPayload builds the request payload for Qwen API
func (c *qwenClient) buildPayload(request Request) map[string]interface{} {
	// Convert messages to Qwen format
	var prompt string
	if len(request.Messages) > 0 {
		// For Qwen, we need to build a single prompt from messages
		prompt = c.buildPromptFromMessages(request.Messages)
	}

	payload := map[string]interface{}{
		"model": c.getModel(request.Model),
		"input": map[string]interface{}{
			"prompt": prompt,
		},
		"parameters": map[string]interface{}{
			"max_tokens": c.getMaxTokens(request.MaxTokens),
		},
	}

	// Add temperature if set
	if request.Temperature != nil {
		payload["parameters"].(map[string]interface{})["temperature"] = *request.Temperature
	} else if c.config.DefaultTemperature != nil {
		payload["parameters"].(map[string]interface{})["temperature"] = *c.config.DefaultTemperature
	}

	// Add top_p if set
	if request.TopP != nil {
		payload["parameters"].(map[string]interface{})["top_p"] = *request.TopP
	} else if c.config.DefaultTopP != nil {
		payload["parameters"].(map[string]interface{})["top_p"] = *c.config.DefaultTopP
	}

	// Add top_k if set
	if request.TopK != nil {
		payload["parameters"].(map[string]interface{})["top_k"] = *request.TopK
	} else if c.config.DefaultTopK != nil {
		payload["parameters"].(map[string]interface{})["top_k"] = *c.config.DefaultTopK
	}

	return payload
}

// buildPromptFromMessages converts chat messages to a single prompt for Qwen
func (c *qwenClient) buildPromptFromMessages(messages []Message) string {
	var prompt string

	for _, msg := range messages {
		switch msg.Role {
		case RoleSystem:
			prompt += "System: " + msg.Content + "\n\n"
		case RoleUser:
			prompt += "User: " + msg.Content + "\n\n"
		case RoleAssistant:
			prompt += "Assistant: " + msg.Content + "\n\n"
		}
	}

	prompt += "Assistant:"
	return prompt
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