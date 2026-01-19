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

// openAIClient implements Client for OpenAI-compatible APIs
type openAIClient struct {
	config     Config
	httpClient *http.Client
}

// newOpenAIClient creates a new OpenAI-compatible client
func newOpenAIClient(config Config) (*openAIClient, error) {
	if config.APIKey == "" {
		return nil, fmt.Errorf("API key is required")
	}

	if config.BaseURL == "" {
		// Set default URLs based on provider
		switch config.Provider {
		case ProviderOpenAI:
			config.BaseURL = "https://api.openai.com/v1"
		case ProviderDeepSeek:
			config.BaseURL = "https://api.deepseek.com"
		default:
			config.BaseURL = "https://api.openai.com/v1"
		}
	}

	if config.Timeout == 0 {
		config.Timeout = 30 * time.Second
	}

	if config.DefaultModel == "" {
		// Set default models based on provider
		switch config.Provider {
		case ProviderOpenAI:
			config.DefaultModel = "gpt-3.5-turbo"
		case ProviderDeepSeek:
			config.DefaultModel = "deepseek-chat"
		default:
			config.DefaultModel = "gpt-3.5-turbo"
		}
	}

	httpClient := &http.Client{
		Timeout: config.Timeout,
	}

	return &openAIClient{
		config:     config,
		httpClient: httpClient,
	}, nil
}

// Generate sends a request to the LLM and returns the response
func (c *openAIClient) Generate(ctx context.Context, request Request) (*Response, error) {
	startTime := time.Now()

	// Prepare the request payload
	payload := c.buildPayload(request)

	jsonPayload, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Create HTTP request
	req, err := http.NewRequestWithContext(ctx, "POST", c.config.BaseURL+"/chat/completions", bytes.NewBuffer(jsonPayload))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.config.APIKey)

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
		return nil, fmt.Errorf("LLM API error %d: %s", resp.StatusCode, string(body))
	}

	// Parse response
	var apiResp struct {
		Choices []struct {
			Message struct {
				Role    string `json:"role"`
				Content string `json:"content"`
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
		return nil, fmt.Errorf("no choices in LLM response")
	}

	responseTime := time.Since(startTime)

	return &Response{
		Content:      apiResp.Choices[0].Message.Content,
		Role:         MessageRole(apiResp.Choices[0].Message.Role),
		TokensUsed:   apiResp.Usage.TotalTokens,
		ResponseTime: responseTime,
		FinishReason: apiResp.Choices[0].FinishReason,
	}, nil
}

// GenerateWithHistory generates a response using chat history
func (c *openAIClient) GenerateWithHistory(ctx context.Context, history ChatHistory, userMessage string, systemPrompt string) (*Response, error) {
	request := BuildChatRequest(history.GetMessages(), userMessage)
	if systemPrompt != "" {
		request.AddSystemMessage(systemPrompt)
	}
	return c.Generate(ctx, request)
}

// Close closes the client
func (c *openAIClient) Close() error {
	// HTTP client doesn't need explicit closing
	return nil
}

// GetConfig returns the client configuration
func (c *openAIClient) GetConfig() Config {
	return c.config
}

// buildPayload builds the request payload for OpenAI API
func (c *openAIClient) buildPayload(request Request) map[string]interface{} {
	payload := map[string]interface{}{
		"model": c.getModel(request.Model),
		"messages": c.convertMessages(request.Messages),
		"stream": request.Stream,
	}

	// Add parameters if set
	if request.Temperature != nil {
		payload["temperature"] = *request.Temperature
	} else if c.config.DefaultTemperature != nil {
		payload["temperature"] = *c.config.DefaultTemperature
	}

	if request.MaxTokens != nil {
		payload["max_tokens"] = *request.MaxTokens
	} else if c.config.DefaultMaxTokens != nil {
		payload["max_tokens"] = *c.config.DefaultMaxTokens
	}

	if request.TopP != nil {
		payload["top_p"] = *request.TopP
	} else if c.config.DefaultTopP != nil {
		payload["top_p"] = *c.config.DefaultTopP
	}

	// Add extra parameters
	for k, v := range request.ExtraParams {
		payload[k] = v
	}

	return payload
}

// getModel returns the model to use for the request
func (c *openAIClient) getModel(override *string) string {
	if override != nil {
		return *override
	}
	return c.config.DefaultModel
}

// convertMessages converts internal Message format to OpenAI format
func (c *openAIClient) convertMessages(messages []Message) []map[string]interface{} {
	result := make([]map[string]interface{}, len(messages))
	for i, msg := range messages {
		result[i] = map[string]interface{}{
			"role":    string(msg.Role),
			"content": msg.Content,
		}
		if msg.Name != "" {
			result[i]["name"] = msg.Name
		}
	}
	return result
}