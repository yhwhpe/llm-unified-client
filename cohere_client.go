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

// cohereClient implements Client for Cohere AI
type cohereClient struct {
	config     Config
	httpClient *http.Client
}

// newCohereClient creates a new Cohere client
func newCohereClient(config Config) (*cohereClient, error) {
	if config.APIKey == "" {
		return nil, fmt.Errorf("API key is required")
	}

	if config.BaseURL == "" {
		config.BaseURL = "https://api.cohere.ai/v1"
	}

	if config.Timeout == 0 {
		config.Timeout = 30 * time.Second
	}

	if config.DefaultModel == "" {
		// Default to multilingual v3 for embeddings
		config.DefaultModel = "embed-multilingual-v3.0"
	}

	httpClient := &http.Client{
		Timeout: config.Timeout,
	}

	return &cohereClient{
		config:     config,
		httpClient: httpClient,
	}, nil
}

// Generate sends a request to Cohere and returns the response
func (c *cohereClient) Generate(ctx context.Context, request Request) (*Response, error) {
	startTime := time.Now()

	// Prepare the request payload
	payload := c.buildPayload(request)

	jsonPayload, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Create HTTP request
	req, err := http.NewRequestWithContext(ctx, "POST", c.config.BaseURL+"/chat", bytes.NewBuffer(jsonPayload))
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
		return nil, fmt.Errorf("Cohere API error %d: %s", resp.StatusCode, string(body))
	}

	// Parse response
	var apiResp struct {
		Text string `json:"text"`
		Meta struct {
			BilledUnits struct {
				InputTokens  int `json:"input_tokens"`
				OutputTokens int `json:"output_tokens"`
			} `json:"billed_units"`
		} `json:"meta"`
		FinishReason string `json:"finish_reason"`
	}

	if err := json.Unmarshal(body, &apiResp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	responseTime := time.Since(startTime)

	return &Response{
		Content:      apiResp.Text,
		Role:         RoleAssistant,
		TokensUsed:   apiResp.Meta.BilledUnits.InputTokens + apiResp.Meta.BilledUnits.OutputTokens,
		ResponseTime: responseTime,
		FinishReason: apiResp.FinishReason,
	}, nil
}

// GenerateWithHistory generates a response using chat history
func (c *cohereClient) GenerateWithHistory(ctx context.Context, history ChatHistory, userMessage string, systemPrompt string) (*Response, error) {
	request := BuildChatRequest(history.GetMessages(), userMessage)
	if systemPrompt != "" {
		request.AddSystemMessage(systemPrompt)
	}
	return c.Generate(ctx, request)
}

// CreateEmbedding generates embeddings for the given text(s)
func (c *cohereClient) CreateEmbedding(ctx context.Context, request EmbeddingRequest) (*EmbeddingResponse, error) {
	startTime := time.Now()

	// Determine embedding model
	embeddingModel := "embed-multilingual-v3.0"
	if request.Model != nil {
		embeddingModel = *request.Model
	} else if c.config.DefaultModel != "" {
		embeddingModel = c.config.DefaultModel
	}

	// Prepare the request payload for Cohere embed API
	payload := map[string]interface{}{
		"model":      embeddingModel,
		"texts":      request.Input,
		"input_type": "search_document", // or "search_query", "classification", "clustering"
	}

	jsonPayload, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal embedding request: %w", err)
	}

	// Create HTTP request
	req, err := http.NewRequestWithContext(ctx, "POST", c.config.BaseURL+"/embed", bytes.NewBuffer(jsonPayload))
	if err != nil {
		return nil, fmt.Errorf("failed to create embedding request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.config.APIKey)

	// Send request
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send embedding request: %w", err)
	}
	defer resp.Body.Close()

	// Read response
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read embedding response: %w", err)
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("Cohere Embedding API error %d: %s", resp.StatusCode, string(body))
	}

	// Parse response
	var apiResp struct {
		Embeddings [][]float64 `json:"embeddings"`
		ID         string      `json:"id"`
		Meta       struct {
			BilledUnits struct {
				InputTokens int `json:"input_tokens"`
			} `json:"billed_units"`
		} `json:"meta"`
	}

	if err := json.Unmarshal(body, &apiResp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal embedding response: %w", err)
	}

	if len(apiResp.Embeddings) == 0 {
		return nil, fmt.Errorf("no embeddings in response")
	}

	responseTime := time.Since(startTime)

	return &EmbeddingResponse{
		Embeddings:   apiResp.Embeddings,
		Model:        embeddingModel,
		TokensUsed:   apiResp.Meta.BilledUnits.InputTokens,
		ResponseTime: responseTime,
	}, nil
}

// Close closes the client
func (c *cohereClient) Close() error {
	return nil
}

// GetConfig returns the client configuration
func (c *cohereClient) GetConfig() Config {
	return c.config
}

// buildPayload builds the request payload for Cohere Chat API
func (c *cohereClient) buildPayload(request Request) map[string]interface{} {
	// Convert messages to Cohere format
	var message string
	var chatHistory []map[string]interface{}

	for i, msg := range request.Messages {
		if msg.Role == RoleSystem {
			// Cohere doesn't have system role, prepend to first user message
			continue
		}
		if msg.Role == RoleUser {
			if i == len(request.Messages)-1 {
				// Last user message is the main message
				message = msg.Content
			} else {
				chatHistory = append(chatHistory, map[string]interface{}{
					"role":    "USER",
					"message": msg.Content,
				})
			}
		} else if msg.Role == RoleAssistant {
			chatHistory = append(chatHistory, map[string]interface{}{
				"role":    "CHATBOT",
				"message": msg.Content,
			})
		}
	}

	payload := map[string]interface{}{
		"message": message,
		"model":   c.getModel(request.Model),
	}

	if len(chatHistory) > 0 {
		payload["chat_history"] = chatHistory
	}

	// Add temperature if set
	if request.Temperature != nil {
		payload["temperature"] = *request.Temperature
	} else if c.config.DefaultTemperature != nil {
		payload["temperature"] = *c.config.DefaultTemperature
	}

	// Add max_tokens if set
	if request.MaxTokens != nil {
		payload["max_tokens"] = *request.MaxTokens
	} else if c.config.DefaultMaxTokens != nil {
		payload["max_tokens"] = *c.config.DefaultMaxTokens
	}

	// Add top_p if set
	if request.TopP != nil {
		payload["p"] = *request.TopP
	} else if c.config.DefaultTopP != nil {
		payload["p"] = *c.config.DefaultTopP
	}

	// Add top_k if set
	if request.TopK != nil {
		payload["k"] = *request.TopK
	} else if c.config.DefaultTopK != nil {
		payload["k"] = *c.config.DefaultTopK
	}

	// Add any extra parameters
	for key, value := range request.ExtraParams {
		payload[key] = value
	}

	return payload
}

// getModel returns the model to use for the request
func (c *cohereClient) getModel(override *string) string {
	if override != nil {
		return *override
	}
	// For chat, use command model if not specified
	if c.config.DefaultModel == "" || c.config.DefaultModel == "embed-multilingual-v3.0" {
		return "command-r-plus"
	}
	return c.config.DefaultModel
}
