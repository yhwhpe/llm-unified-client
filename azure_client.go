package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// azureClient implements Client for Azure OpenAI
type azureClient struct {
	config     Config
	httpClient *http.Client
}

// newAzureClient creates a new Azure OpenAI client
func newAzureClient(config Config) (*azureClient, error) {
	if config.APIKey == "" {
		return nil, fmt.Errorf("API key is required")
	}

	if config.BaseURL == "" {
		return nil, fmt.Errorf("base URL is required for Azure OpenAI")
	}

	// Azure OpenAI requires the URL to end with the deployment name
	// Format: https://<resource-name>.openai.azure.com/openai/deployments/<deployment-name>
	if !strings.Contains(config.BaseURL, "/deployments/") {
		return nil, fmt.Errorf("Azure OpenAI URL must include deployment name: /deployments/<deployment-name>")
	}

	if config.Timeout == 0 {
		config.Timeout = 30 * time.Second
	}

	if config.DefaultModel == "" {
		// For Azure, model is usually specified in the deployment
		config.DefaultModel = "gpt-35-turbo" // Default Azure deployment name
	}

	httpClient := &http.Client{
		Timeout: config.Timeout,
	}

	return &azureClient{
		config:     config,
		httpClient: httpClient,
	}, nil
}

// Generate sends a request to Azure OpenAI and returns the response
func (c *azureClient) Generate(ctx context.Context, request Request) (*Response, error) {
	startTime := time.Now()

	// Prepare the request payload (same as OpenAI)
	payload := c.buildPayload(request)

	jsonPayload, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Azure OpenAI uses a different endpoint format
	url := c.config.BaseURL + "/chat/completions?api-version=2023-12-01-preview"

	// Create HTTP request
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonPayload))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("api-key", c.config.APIKey) // Azure uses api-key header instead of Authorization

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
		return nil, fmt.Errorf("Azure OpenAI API error %d: %s", resp.StatusCode, string(body))
	}

	// Parse response (same format as OpenAI)
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
		return nil, fmt.Errorf("no choices in Azure OpenAI response")
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
func (c *azureClient) GenerateWithHistory(ctx context.Context, history ChatHistory, userMessage string, systemPrompt string) (*Response, error) {
	request := BuildChatRequest(history.GetMessages(), userMessage)
	if systemPrompt != "" {
		request.AddSystemMessage(systemPrompt)
	}
	return c.Generate(ctx, request)
}

// Close closes the client
func (c *azureClient) Close() error {
	return nil
}

// GetConfig returns the client configuration
func (c *azureClient) GetConfig() Config {
	return c.config
}

// CreateEmbedding generates embeddings for the given text(s)
func (c *azureClient) CreateEmbedding(ctx context.Context, request EmbeddingRequest) (*EmbeddingResponse, error) {
	return nil, fmt.Errorf("embeddings not supported for Azure provider yet")
}

// buildPayload builds the request payload for Azure OpenAI API (same as OpenAI)
func (c *azureClient) buildPayload(request Request) map[string]interface{} {
	payload := map[string]interface{}{
		"messages": c.convertMessages(request.Messages),
		"stream":   request.Stream,
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

// convertMessages converts internal Message format to OpenAI format
func (c *azureClient) convertMessages(messages []Message) []map[string]interface{} {
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
