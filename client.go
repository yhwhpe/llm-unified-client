package llm

import (
	"context"
	"fmt"
)

// NewClient creates a new LLM client based on the provider
func NewClient(config Config) (Client, error) {
	switch config.Provider {
	case ProviderOpenAI, ProviderDeepSeek:
		return NewOpenAICompatibleClient(config)
	case ProviderQwen:
		return NewQwenClient(config)
	case ProviderAzure:
		return NewAzureClient(config)
	default:
		return nil, fmt.Errorf("unsupported LLM provider: %s", config.Provider)
	}
}

// NewOpenAICompatibleClient creates a client for OpenAI-compatible APIs (OpenAI, DeepSeek, etc.)
func NewOpenAICompatibleClient(config Config) (Client, error) {
	return newOpenAIClient(config)
}

// NewQwenClient creates a client for Alibaba Qwen
func NewQwenClient(config Config) (Client, error) {
	return newQwenClient(config)
}

// NewAzureClient creates a client for Azure OpenAI
func NewAzureClient(config Config) (Client, error) {
	return newAzureClient(config)
}

// Helper functions for building requests

// BuildSimpleRequest creates a simple request with a single user message
func BuildSimpleRequest(message string) Request {
	return Request{
		Messages: []Message{
			{Role: RoleUser, Content: message},
		},
	}
}

// BuildChatRequest creates a request with chat history
func BuildChatRequest(history []Message, userMessage string) Request {
	messages := make([]Message, 0, len(history)+1)
	messages = append(messages, history...)
	messages = append(messages, Message{Role: RoleUser, Content: userMessage})

	return Request{
		Messages: messages,
	}
}

// BuildRequestWithSystemPrompt creates a request with system prompt and user message
func BuildRequestWithSystemPrompt(systemPrompt, userMessage string) Request {
	return Request{
		Messages: []Message{
			{Role: RoleSystem, Content: systemPrompt},
			{Role: RoleUser, Content: userMessage},
		},
	}
}

// AddSystemMessage adds a system message to the request
func (r *Request) AddSystemMessage(content string) {
	r.Messages = append([]Message{{Role: RoleSystem, Content: content}}, r.Messages...)
}

// AddUserMessage adds a user message to the request
func (r *Request) AddUserMessage(content string) {
	r.Messages = append(r.Messages, Message{Role: RoleUser, Content: content})
}

// AddAssistantMessage adds an assistant message to the request
func (r *Request) AddAssistantMessage(content string) {
	r.Messages = append(r.Messages, Message{Role: RoleAssistant, Content: content})
}

// SetTemperature sets the temperature parameter
func (r *Request) SetTemperature(temp float64) {
	r.Temperature = &temp
}

// SetMaxTokens sets the max tokens parameter
func (r *Request) SetMaxTokens(tokens int) {
	r.MaxTokens = &tokens
}

// SetTopP sets the top-p parameter
func (r *Request) SetTopP(topP float64) {
	r.TopP = &topP
}

// SetTopK sets the top-k parameter
func (r *Request) SetTopK(topK int) {
	r.TopK = &topK
}

// SetModel sets the model override
func (r *Request) SetModel(model string) {
	r.Model = &model
}

// SetStreaming enables or disables streaming
func (r *Request) SetStreaming(stream bool) {
	r.Stream = stream
}

// ChatHistory methods

// AddMessage adds a message to the chat history
func (h *ChatHistory) AddMessage(role MessageRole, content string) {
	h.Messages = append(h.Messages, Message{
		Role:    role,
		Content: content,
	})
}

// AddSystemMessage adds a system message to history
func (h *ChatHistory) AddSystemMessage(content string) {
	h.AddMessage(RoleSystem, content)
}

// AddUserMessage adds a user message to history
func (h *ChatHistory) AddUserMessage(content string) {
	h.AddMessage(RoleUser, content)
}

// AddAssistantMessage adds an assistant message to history
func (h *ChatHistory) AddAssistantMessage(content string) {
	h.AddMessage(RoleAssistant, content)
}

// GetMessages returns all messages in the history
func (h *ChatHistory) GetMessages() []Message {
	return h.Messages
}

// Clear clears the chat history
func (h *ChatHistory) Clear() {
	h.Messages = nil
}

// GetLastMessage returns the last message in history
func (h *ChatHistory) GetLastMessage() *Message {
	if len(h.Messages) == 0 {
		return nil
	}
	return &h.Messages[len(h.Messages)-1]
}

// Truncate truncates history to keep only the last n messages
func (h *ChatHistory) Truncate(n int) {
	if len(h.Messages) > n {
		h.Messages = h.Messages[len(h.Messages)-n:]
	}
}

// Convenience functions for common operations

// GenerateSimple generates a response for a simple text prompt
func GenerateSimple(ctx context.Context, client Client, prompt string) (*Response, error) {
	req := BuildSimpleRequest(prompt)
	return client.Generate(ctx, req)
}

// GenerateWithHistory generates a response using chat history
func GenerateWithHistory(ctx context.Context, client Client, history ChatHistory, userMessage, systemPrompt string) (*Response, error) {
	req := BuildChatRequest(history.GetMessages(), userMessage)
	if systemPrompt != "" {
		req.AddSystemMessage(systemPrompt)
	}
	return client.Generate(ctx, req)
}

// GenerateWithSystemPrompt generates a response with system prompt
func GenerateWithSystemPrompt(ctx context.Context, client Client, systemPrompt, userMessage string) (*Response, error) {
	req := BuildRequestWithSystemPrompt(systemPrompt, userMessage)
	return client.Generate(ctx, req)
}
