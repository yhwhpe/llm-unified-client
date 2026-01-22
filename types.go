package llm

import (
	"context"
	"time"
)

// Provider represents supported LLM providers
type Provider string

const (
	ProviderOpenAI   Provider = "openai"
	ProviderDeepSeek Provider = "deepseek"
	ProviderQwen     Provider = "qwen"
	ProviderAzure    Provider = "azure"
)

// Message represents a chat message
type Message struct {
	Role    MessageRole `json:"role"`
	Content string      `json:"content"`
	Name    string      `json:"name,omitempty"` // For function calls
}

// MessageRole defines the role of a message
type MessageRole string

const (
	RoleSystem    MessageRole = "system"
	RoleUser      MessageRole = "user"
	RoleAssistant MessageRole = "assistant"
	RoleFunction  MessageRole = "function"
)

// ChatHistory represents a conversation history
type ChatHistory struct {
	Messages []Message `json:"messages"`
}

// Request represents a request to the LLM
type Request struct {
	// Basic parameters
	Messages     []Message `json:"messages"`
	SystemPrompt string    `json:"system_prompt,omitempty"`
	Temperature  *float64  `json:"temperature,omitempty"`
	MaxTokens    *int      `json:"max_tokens,omitempty"`
	TopP         *float64  `json:"top_p,omitempty"`
	TopK         *int      `json:"top_k,omitempty"`
	Stream       bool      `json:"stream,omitempty"`

	// Provider-specific parameters
	ExtraParams map[string]interface{} `json:"extra_params,omitempty"`

	// Model configuration override
	Model *string `json:"model,omitempty"`
}

// Response represents a response from the LLM
type Response struct {
	Content      string        `json:"content"`
	Role         MessageRole   `json:"role,omitempty"`
	TokensUsed   int           `json:"tokens_used,omitempty"`
	ResponseTime time.Duration `json:"response_time"`
	FinishReason string        `json:"finish_reason,omitempty"`

	// Streaming support
	Stream chan StreamChunk `json:"-"` // For streaming responses
}

// StreamChunk represents a chunk of streaming response
type StreamChunk struct {
	Content      string `json:"content"`
	FinishReason string `json:"finish_reason,omitempty"`
	Done         bool   `json:"done"`
}

// Config holds configuration for LLM clients
type Config struct {
	// Provider settings
	Provider Provider      `json:"provider"`
	APIKey   string        `json:"api_key"`
	BaseURL  string        `json:"base_url,omitempty"`
	Timeout  time.Duration `json:"timeout"`

	// Model settings
	DefaultModel string `json:"default_model"`

	// Default parameters
	DefaultTemperature *float64 `json:"default_temperature,omitempty"`
	DefaultMaxTokens   *int     `json:"default_max_tokens,omitempty"`
	DefaultTopP        *float64 `json:"default_top_p,omitempty"`
	DefaultTopK        *int     `json:"default_top_k,omitempty"`

	// Provider-specific settings
	ExtraConfig map[string]interface{} `json:"extra_config,omitempty"`
}

// Client defines the interface for LLM operations
type Client interface {
	// Generate generates a response from the LLM
	Generate(ctx context.Context, request Request) (*Response, error)

	// GenerateWithHistory generates a response using chat history
	GenerateWithHistory(ctx context.Context, history ChatHistory, userMessage string, systemPrompt string) (*Response, error)

	// Close closes the client and cleans up resources
	Close() error

	// GetConfig returns the client configuration
	GetConfig() Config
}
