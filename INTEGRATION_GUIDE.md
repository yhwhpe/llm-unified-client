# LLM Unified Client - Integration Guide

This guide provides comprehensive instructions for integrating the LLM Unified Client into your Go applications.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [Advanced Features](#advanced-features)
- [Error Handling](#error-handling)
- [Testing](#testing)
- [Migration Guide](#migration-guide)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Installation

### Method 1: Go Modules (Recommended)

```bash
go get github.com/yhwhpe/llm-unified-client
```

### Method 2: Git Clone

```bash
git clone git@github.com:yhwhpe/llm-unified-client.git
cd llm-unified-client
go mod tidy
```

### Requirements

- Go 1.24+
- Valid API keys for your chosen LLM provider(s)

## Quick Start

### Basic Setup

```go
package main

import (
    "context"
    "log"
    "time"

    llm "github.com/yhwhpe/llm-unified-client"
)

func main() {
    // Create configuration
    config := llm.Config{
        Provider:     llm.ProviderDeepSeek,
        APIKey:       "your-api-key-here",
        BaseURL:      "https://api.deepseek.com",
        DefaultModel: "deepseek-chat",
        Timeout:      30 * time.Second,
    }

    // Create client
    client, err := llm.NewClient(config)
    if err != nil {
        log.Fatal(err)
    }
    defer client.Close()

    // Generate response
    ctx := context.Background()
    response, err := llm.GenerateSimple(ctx, client, "Hello, world!")
    if err != nil {
        log.Fatal(err)
    }

    log.Printf("Response: %s", response.Content)
}
```

### Environment Variables

```bash
# DeepSeek
export DEEPSEEK_API_KEY="sk-your-key-here"

# OpenAI
export OPENAI_API_KEY="sk-your-key-here"

# Qwen (Alibaba Cloud)
export QWEN_API_KEY="sk-your-key-here"

# Azure OpenAI
export AZURE_OPENAI_API_KEY="your-key-here"
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"
```

## Configuration

### DeepSeek Configuration

```go
config := llm.Config{
    Provider:   llm.ProviderDeepSeek,
    APIKey:     os.Getenv("DEEPSEEK_API_KEY"),
    BaseURL:    "https://api.deepseek.com", // Optional, uses default
    DefaultModel: "deepseek-chat",           // Optional, uses default
    Timeout:    30 * time.Second,

    // Optional: Default parameters
    DefaultTemperature: &[]float64{0.7}[0],
    DefaultMaxTokens:   &[]int{1000}[0],
    DefaultTopP:        &[]float64{0.9}[0],
}
```

### OpenAI Configuration

```go
config := llm.Config{
    Provider:   llm.ProviderOpenAI,
    APIKey:     os.Getenv("OPENAI_API_KEY"),
    BaseURL:    "https://api.openai.com/v1", // Optional
    DefaultModel: "gpt-4",                     // Optional
    Timeout:    30 * time.Second,
}
```

### Qwen (Alibaba Cloud) Configuration

```go
config := llm.Config{
    Provider:   llm.ProviderQwen,
    APIKey:     os.Getenv("QWEN_API_KEY"),
    BaseURL:    "https://dashscope.aliyuncs.com/api/v1", // Optional
    DefaultModel: "qwen-turbo",                           // Optional
    Timeout:    30 * time.Second,
}
```

### Azure OpenAI Configuration

```go
config := llm.Config{
    Provider:   llm.ProviderAzure,
    APIKey:     os.Getenv("AZURE_OPENAI_API_KEY"),
    BaseURL:    "https://your-resource.openai.azure.com/openai/deployments/your-deployment",
    Timeout:    30 * time.Second,
}
```

### Advanced Configuration

```go
config := llm.Config{
    Provider:   llm.ProviderDeepSeek,
    APIKey:     os.Getenv("DEEPSEEK_API_KEY"),

    // Model settings
    DefaultModel: "deepseek-chat",

    // Request parameters
    DefaultTemperature: &[]float64{0.7}[0],
    DefaultMaxTokens:   &[]int{1000}[0],
    DefaultTopP:        &[]float64{0.9}[0],
    DefaultTopK:        &[]int{40}[0],

    // Network settings
    Timeout: 30 * time.Second,

    // Provider-specific settings
    ExtraConfig: map[string]interface{}{
        "custom_param": "value",
        "retry_attempts": 3,
    },
}
```

## Usage Examples

### Simple Text Generation

```go
// Method 1: Using helper function
response, err := llm.GenerateSimple(ctx, client, "Explain quantum computing simply")

// Method 2: Using direct API
request := llm.BuildSimpleRequest("Explain quantum computing simply")
response, err := client.Generate(ctx, request)
```

### System Prompts

```go
// Method 1: Helper function
response, err := llm.GenerateWithSystemPrompt(
    ctx,
    client,
    "You are a helpful coding assistant.",
    "Write a Go function to reverse a string",
)

// Method 2: Builder pattern
request := llm.BuildRequestWithSystemPrompt(
    "You are a helpful coding assistant.",
    "Write a Go function to reverse a string",
)
response, err := client.Generate(ctx, request)
```

### Chat Conversations

```go
// Create chat history
history := llm.ChatHistory{}
history.AddSystemMessage("You are a helpful assistant")
history.AddUserMessage("Hello!")
history.AddAssistantMessage("Hi! How can I help you?")

// Generate with history
response, err := client.GenerateWithHistory(ctx, history, "Tell me about Go", "")

// Add new messages to history
history.AddUserMessage("Tell me about Go")
history.AddAssistantMessage(response.Content)
```

### Advanced Request Building

```go
// Create custom request
request := llm.Request{
    Messages: []llm.Message{
        {Role: llm.RoleSystem, Content: "You are an expert in mathematics."},
        {Role: llm.RoleUser, Content: "Solve: 2x + 3 = 7"},
    },
}

// Set parameters
request.SetTemperature(0.1)  // More deterministic
request.SetMaxTokens(200)    // Limit response length
request.SetTopP(0.8)         // Nucleus sampling
request.SetModel("gpt-4")    // Override default model

// Optional: Enable streaming (future feature)
// request.SetStreaming(true)

response, err := client.Generate(ctx, request)
```

### Builder Pattern Examples

```go
// Simple request
simpleReq := llm.BuildSimpleRequest("Hello!")

// System + User
sysReq := llm.BuildRequestWithSystemPrompt("Be concise", "What is AI?")

// Manual building
manualReq := llm.Request{}
manualReq.AddSystemMessage("You are a chef")
manualReq.AddUserMessage("Suggest a recipe")
manualReq.AddAssistantMessage("I'd be happy to help!")
manualReq.AddUserMessage("Make it vegetarian")
manualReq.SetTemperature(0.8)
```

## Advanced Features

### Chat History Management

```go
history := llm.ChatHistory{}

// Add messages
history.AddSystemMessage("You are a helpful assistant")
history.AddUserMessage("Hello")
history.AddAssistantMessage("Hi there!")

// Get messages
messages := history.GetMessages()

// Get last message
lastMsg := history.GetLastMessage()

// Truncate to keep recent messages
history.Truncate(10)

// Clear history
history.Clear()

// Check if empty
if len(history.GetMessages()) == 0 {
    fmt.Println("History is empty")
}
```

### Custom Request Parameters

```go
request := llm.BuildSimpleRequest("Generate a story")

// Temperature (0.0 - 2.0)
request.SetTemperature(0.8)

// Max tokens
request.SetMaxTokens(500)

// Top-p sampling (0.0 - 1.0)
request.SetTopP(0.9)

// Top-k sampling
request.SetTopK(40)

// Model override
request.SetModel("gpt-4-turbo")

// Streaming (prepared for future)
request.SetStreaming(false)
```

### Provider-Specific Features

```go
// DeepSeek specific parameters
deepSeekReq := llm.BuildSimpleRequest("Analyze this code")
deepSeekReq.ExtraParams = map[string]interface{}{
    "frequency_penalty": 0.1,
    "presence_penalty": 0.1,
}

// Qwen specific parameters
qwenReq := llm.BuildSimpleRequest("Translate to Chinese")
qwenReq.ExtraParams = map[string]interface{}{
    "repetition_penalty": 1.1,
}
```

### Response Analysis

```go
response, err := client.Generate(ctx, request)
if err != nil {
    // Handle error
}

// Analyze response
fmt.Printf("Content: %s\n", response.Content)
fmt.Printf("Role: %s\n", response.Role)
fmt.Printf("Tokens used: %d\n", response.TokensUsed)
fmt.Printf("Response time: %v\n", response.ResponseTime)
fmt.Printf("Finish reason: %s\n", response.FinishReason)

// Check for truncation
if response.FinishReason == "length" {
    fmt.Println("Response was truncated due to token limit")
}
```

## Error Handling

### Basic Error Handling

```go
response, err := client.Generate(ctx, request)
if err != nil {
    log.Printf("LLM request failed: %v", err)

    // Check for specific error types
    switch {
    case strings.Contains(err.Error(), "timeout"):
        // Handle timeout
        return retryWithBackoff()
    case strings.Contains(err.Error(), "rate limit"):
        // Handle rate limiting
        return waitAndRetry()
    case strings.Contains(err.Error(), "authentication"):
        // Handle auth errors
        return refreshCredentials()
    default:
        // Handle other errors
        return err
    }
}
```

### Advanced Error Handling

```go
func generateWithRetry(ctx context.Context, client llm.Client, request llm.Request, maxRetries int) (*llm.Response, error) {
    var lastErr error

    for attempt := 1; attempt <= maxRetries; attempt++ {
        response, err := client.Generate(ctx, request)
        if err == nil {
            return response, nil
        }

        lastErr = err

        // Check if error is retryable
        if isRetryableError(err) && attempt < maxRetries {
            backoff := time.Duration(attempt) * time.Second
            log.Printf("Attempt %d failed, retrying in %v: %v", attempt, backoff, err)
            time.Sleep(backoff)
            continue
        }

        break
    }

    return nil, fmt.Errorf("failed after %d attempts: %w", maxRetries, lastErr)
}

func isRetryableError(err error) bool {
    errStr := err.Error()
    return strings.Contains(errStr, "timeout") ||
           strings.Contains(errStr, "rate limit") ||
           strings.Contains(errStr, "server error") ||
           strings.Contains(errStr, "network")
}
```

### Timeout Handling

```go
// Set request timeout
ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
defer cancel()

response, err := client.Generate(ctx, request)
if err != nil {
    if ctx.Err() == context.DeadlineExceeded {
        log.Println("Request timed out")
        return
    }
    log.Printf("Request failed: %v", err)
    return
}
```

## Testing

### Unit Tests

```go
func TestClientConfiguration(t *testing.T) {
    config := llm.Config{
        Provider:     llm.ProviderDeepSeek,
        APIKey:       "test-key",
        DefaultModel: "test-model",
        Timeout:      5 * time.Second,
    }

    client, err := llm.NewClient(config)
    assert.NoError(t, err)
    defer client.Close()

    clientConfig := client.GetConfig()
    assert.Equal(t, llm.ProviderDeepSeek, clientConfig.Provider)
    assert.Equal(t, "test-key", clientConfig.APIKey)
}
```

### Integration Tests

```go
func TestDeepSeekIntegration(t *testing.T) {
    apiKey := os.Getenv("DEEPSEEK_API_KEY")
    if apiKey == "" {
        t.Skip("DEEPSEEK_API_KEY not set")
    }

    config := llm.Config{
        Provider:     llm.ProviderDeepSeek,
        APIKey:       apiKey,
        DefaultModel: "deepseek-chat",
        Timeout:      30 * time.Second,
    }

    client, err := llm.NewClient(config)
    assert.NoError(t, err)
    defer client.Close()

    response, err := llm.GenerateSimple(context.Background(), client, "Say 'test' and nothing else")
    assert.NoError(t, err)
    assert.Contains(t, response.Content, "test")
    assert.Greater(t, response.TokensUsed, 0)
}
```

### Mock Testing

```go
type mockClient struct {
    response *llm.Response
    err      error
}

func (m *mockClient) Generate(ctx context.Context, request llm.Request) (*llm.Response, error) {
    return m.response, m.err
}

func (m *mockClient) GenerateWithHistory(ctx context.Context, history llm.ChatHistory, userMessage, systemPrompt string) (*llm.Response, error) {
    return m.response, m.err
}

func (m *mockClient) Close() error {
    return nil
}

func (m *mockClient) GetConfig() llm.Config {
    return llm.Config{Provider: llm.ProviderDeepSeek}
}

// Usage in tests
func TestWithMock(t *testing.T) {
    mockResponse := &llm.Response{
        Content:      "Mock response",
        TokensUsed:   10,
        ResponseTime: time.Second,
    }

    mockClient := &mockClient{response: mockResponse}
    response, err := llm.GenerateSimple(context.Background(), mockClient, "test")

    assert.NoError(t, err)
    assert.Equal(t, "Mock response", response.Content)
}
```

## Migration Guide

### From profile-builder LLM package

**Old code:**
```go
import "agents/profile-builder/internal/llm"

client, err := llm.NewClient(config.LLMConfig{})
response, err := client.Generate(ctx, llm.Request{
    Prompt: "Hello",
    Temperature: 0.7,
})
```

**New code:**
```go
import llm "github.com/yhwhpe/llm-unified-client"

config := llm.Config{
    Provider:   llm.ProviderDeepSeek,
    APIKey:     "your-key",
    DefaultModel: "deepseek-chat",
}
client, err := llm.NewClient(config)
response, err := llm.GenerateSimple(ctx, client, "Hello")
```

### Migration Checklist

1. ✅ Update import path
2. ✅ Change config structure (`LLMConfig` → `llm.Config`)
3. ✅ Update provider constants (string → enum)
4. ✅ Replace `client.Generate()` with helper functions
5. ✅ Add `defer client.Close()`
6. ✅ Update error handling
7. ✅ Use `ChatHistory` for conversations

### Breaking Changes

- **Provider**: String → Enum (`llm.ProviderDeepSeek`)
- **Config**: `LLMConfig` → `llm.Config`
- **Request**: `Request{Prompt: "..."}` → `BuildSimpleRequest("...")`
- **Resource Management**: Must call `client.Close()`

## Best Practices

### Client Lifecycle

```go
// ✅ Good: Proper lifecycle management
client, err := llm.NewClient(config)
if err != nil {
    return err
}
defer client.Close() // Always close

// Use client...
```

### Context Management

```go
// ✅ Good: Use context with timeout
ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
defer cancel()

response, err := client.Generate(ctx, request)
```

### Error Handling

```go
// ✅ Good: Comprehensive error handling
response, err := client.Generate(ctx, request)
if err != nil {
    switch {
    case strings.Contains(err.Error(), "timeout"):
        return retryWithBackoff()
    case strings.Contains(err.Error(), "rate limit"):
        return waitForRateLimit()
    default:
        return fmt.Errorf("LLM request failed: %w", err)
    }
}
```

### Configuration Management

```go
// ✅ Good: Environment-based configuration
func loadLLMConfig() llm.Config {
    provider := llm.ProviderDeepSeek
    if p := os.Getenv("LLM_PROVIDER"); p != "" {
        provider = llm.Provider(p)
    }

    return llm.Config{
        Provider:   provider,
        APIKey:     getRequiredEnv("LLM_API_KEY"),
        BaseURL:    getOptionalEnv("LLM_BASE_URL", ""),
        DefaultModel: getOptionalEnv("LLM_MODEL", "deepseek-chat"),
        Timeout:    getDurationEnv("LLM_TIMEOUT", 30*time.Second),
    }
}
```

### Chat History Management

```go
// ✅ Good: Limit history size
const maxHistorySize = 20

func addToHistory(history *llm.ChatHistory, role llm.MessageRole, content string) {
    history.AddMessage(role, content)

    // Keep only recent messages
    if len(history.GetMessages()) > maxHistorySize {
        history.Truncate(maxHistorySize)
    }
}
```

### Performance Optimization

```go
// ✅ Good: Reuse clients
type LLMService struct {
    client llm.Client
}

func NewLLMService() (*LLMService, error) {
    config := loadLLMConfig()
    client, err := llm.NewClient(config)
    if err != nil {
        return nil, err
    }

    return &LLMService{client: client}, nil
}

func (s *LLMService) Close() error {
    return s.client.Close()
}

// Reuse for multiple requests
func (s *LLMService) GenerateText(prompt string) (string, error) {
    response, err := llm.GenerateSimple(context.Background(), s.client, prompt)
    if err != nil {
        return "", err
    }
    return response.Content, nil
}
```

## Troubleshooting

### Common Issues

#### 1. Authentication Errors

**Error:** `API key is required`
```go
// Solution: Set API key
config := llm.Config{
    Provider: llm.ProviderDeepSeek,
    APIKey:   os.Getenv("DEEPSEEK_API_KEY"), // Make sure this is set
}
```

#### 2. Timeout Errors

**Error:** `context deadline exceeded`
```go
// Solution: Increase timeout or use shorter prompts
config := llm.Config{
    Timeout: 60 * time.Second, // Increase timeout
}

// Or use shorter context
request := llm.BuildSimpleRequest("Short prompt")
request.SetMaxTokens(100) // Reduce max tokens
```

#### 3. Rate Limiting

**Error:** `rate limit exceeded`
```go
// Solution: Implement exponential backoff
func generateWithBackoff(ctx context.Context, client llm.Client, request llm.Request) (*llm.Response, error) {
    backoff := time.Second

    for attempt := 1; attempt <= 5; attempt++ {
        response, err := client.Generate(ctx, request)
        if err == nil {
            return response, nil
        }

        if !strings.Contains(err.Error(), "rate limit") {
            return nil, err
        }

        log.Printf("Rate limited, attempt %d, waiting %v", attempt, backoff)
        time.Sleep(backoff)
        backoff *= 2
    }

    return nil, errors.New("rate limit exceeded, max retries reached")
}
```

#### 4. Invalid Model

**Error:** `model not found`
```go
// Solution: Check available models for your provider
// DeepSeek: "deepseek-chat", "deepseek-coder"
// OpenAI: "gpt-4", "gpt-3.5-turbo"
// Qwen: "qwen-turbo", "qwen-plus"
config := llm.Config{
    DefaultModel: "deepseek-chat", // Use valid model
}
```

### Debug Logging

```go
// Enable detailed logging
import "log"

log.SetFlags(log.LstdFlags | log.Lshortfile)

// The client will log API requests and responses
client, _ := llm.NewClient(config)
// Now you'll see detailed logs for debugging
```

### Health Checks

```go
// Implement health check
func (s *LLMService) HealthCheck(ctx context.Context) error {
    // Simple health check request
    _, err := llm.GenerateSimple(ctx, s.client, "test")
    return err
}
```

---

## Support

For issues and questions:
- Check this integration guide
- Review the examples in `cmd/demo/`
- Check the test files for usage patterns
- Open an issue on GitHub

## License

This integration guide is part of the LLM Unified Client project.