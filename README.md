# LLM Unified Client

A unified Go client library for interacting with various Large Language Model providers including OpenAI, DeepSeek, Qwen (Alibaba Cloud), and Azure OpenAI.

## Features

- **Multiple Provider Support**: OpenAI, DeepSeek, Qwen, Azure OpenAI
- **Unified Interface**: Single API for all providers
- **Chat History Management**: Built-in support for conversation history
- **Streaming Support**: Ready for streaming responses (future implementation)
- **Flexible Configuration**: Extensive configuration options
- **Error Handling**: Comprehensive error handling with detailed messages

## Installation

```bash
go get github.com/yhwhpe/llm-unified-client
```

## Quick Start

```go
package main

import (
    "context"
    "fmt"
    "log"

    llm "github.com/yhwhpe/llm-unified-client"
)

func main() {
    // Create a client for DeepSeek
    config := llm.Config{
        Provider:   llm.ProviderDeepSeek,
        APIKey:     "your-api-key",
        BaseURL:    "https://api.deepseek.com",
        DefaultModel: "deepseek-chat",
        Timeout:    30 * time.Second,
    }

    client, err := llm.NewClient(config)
    if err != nil {
        log.Fatal(err)
    }
    defer client.Close()

    // Simple text generation
    response, err := llm.GenerateSimple(context.Background(), client, "Hello, how are you?")
    if err != nil {
        log.Fatal(err)
    }

    fmt.Println("Response:", response.Content)
    fmt.Printf("Tokens used: %d, Response time: %v\n", response.TokensUsed, response.ResponseTime)
}
```

## Configuration

### DeepSeek Configuration

```go
config := llm.Config{
    Provider:   llm.ProviderDeepSeek,
    APIKey:     "your-deepseek-api-key",
    BaseURL:    "https://api.deepseek.com",
    DefaultModel: "deepseek-chat",
    Timeout:    30 * time.Second,
    DefaultTemperature: &temperature,
    DefaultMaxTokens:   &maxTokens,
}
```

### OpenAI Configuration

```go
config := llm.Config{
    Provider:   llm.ProviderOpenAI,
    APIKey:     "your-openai-api-key",
    BaseURL:    "https://api.openai.com/v1",
    DefaultModel: "gpt-4",
    Timeout:    30 * time.Second,
}
```

### Qwen (Alibaba Cloud) Configuration

```go
config := llm.Config{
    Provider:   llm.ProviderQwen,
    APIKey:     "your-qwen-api-key",
    BaseURL:    "https://dashscope.aliyuncs.com/api/v1",
    DefaultModel: "qwen-turbo",
    Timeout:    30 * time.Second,
}
```

### Azure OpenAI Configuration

```go
config := llm.Config{
    Provider:   llm.ProviderAzure,
    APIKey:     "your-azure-api-key",
    BaseURL:    "https://your-resource.openai.azure.com/openai/deployments/your-deployment",
    Timeout:    30 * time.Second,
}
```

## Usage Examples

### Simple Text Generation

```go
response, err := llm.GenerateSimple(ctx, client, "Explain quantum computing in simple terms")
if err != nil {
    log.Fatal(err)
}
fmt.Println(response.Content)
```

### Chat with History

```go
// Create chat history
history := llm.ChatHistory{}
history.AddUserMessage("Hello!")
history.AddAssistantMessage("Hi there! How can I help you?")

// Generate response with history
response, err := client.GenerateWithHistory(ctx, history, "Tell me about Go programming", "")
if err != nil {
    log.Fatal(err)
}

fmt.Println(response.Content)

// Add new messages to history
history.AddUserMessage("Tell me about Go programming")
history.AddAssistantMessage(response.Content)
```

### Advanced Request Building

```go
// Create custom request
request := llm.Request{
    Messages: []llm.Message{
        {Role: llm.RoleSystem, Content: "You are a helpful assistant."},
        {Role: llm.RoleUser, Content: "What's the weather like?"},
    },
}

// Set parameters
request.SetTemperature(0.7)
request.SetMaxTokens(150)
request.SetTopP(0.9)

// Generate response
response, err := client.Generate(ctx, request)
```

### Using Builder Pattern

```go
request := llm.BuildRequestWithSystemPrompt(
    "You are a coding assistant.",
    "Write a Go function to reverse a string",
)
request.SetTemperature(0.3)
request.SetModel("gpt-4")

response, err := client.Generate(ctx, request)
```

## Chat History Management

```go
history := llm.ChatHistory{}

// Add messages
history.AddSystemMessage("You are a helpful assistant.")
history.AddUserMessage("Hello!")
history.AddAssistantMessage("Hi! How can I help?")

// Get messages
messages := history.GetMessages()

// Truncate to keep recent messages
history.Truncate(10)

// Clear history
history.Clear()
```

## Error Handling

The library provides detailed error messages:

```go
response, err := client.Generate(ctx, request)
if err != nil {
    // Handle different types of errors
    switch {
    case strings.Contains(err.Error(), "API key"):
        // Authentication error
    case strings.Contains(err.Error(), "timeout"):
        // Timeout error
    case strings.Contains(err.Error(), "rate limit"):
        // Rate limiting
    default:
        // Other errors
    }
    log.Fatal(err)
}
```

## Provider-Specific Features

### OpenAI/DeepSeek Features
- Full OpenAI API compatibility
- Streaming support (planned)
- Function calling support (planned)
- All standard parameters supported

### Qwen Features
- Alibaba Cloud integration
- Optimized for Chinese language models
- Cost-effective for certain use cases

### Azure OpenAI Features
- Enterprise-grade security
- Azure authentication integration
- Custom deployment support
- Azure monitoring integration

## Testing

```bash
go test ./...
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Migration from profile-builder LLM package

If you're migrating from the old `agents/profile-builder/internal/llm` package:

### Old API
```go
// Old way
client, err := llm.NewClient(config.LLMConfig{})
response, err := client.Generate(ctx, llm.Request{
    Prompt: "Hello",
    Temperature: 0.7,
})
```

### New API
```go
// New way
config := llm.Config{
    Provider:   llm.ProviderDeepSeek,
    APIKey:     "your-key",
    BaseURL:    "https://api.deepseek.com",
    DefaultModel: "deepseek-chat",
}
client, err := llm.NewClient(config)
response, err := llm.GenerateSimple(ctx, client, "Hello")
```

The new API is more flexible and supports multiple providers with a unified interface.