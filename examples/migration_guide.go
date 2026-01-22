// Migration guide from agents/profile-builder/internal/llm to llm-unified-client
//
// This file shows how to migrate from the old profile-builder LLM package
// to the new unified LLM client library.

package main

import (
	"context"
	"fmt"
	"log"
	"strings"
	"time"

	// Old import (to be replaced)
	// oldLLM "agents/profile-builder/internal/llm"

	// New import
	llm "github.com/yhwhpe/llm-unified-client"
)

// OLD WAY - profile-builder internal LLM package
/*
func oldLLMUsage() {
	// Old config structure
	config := config.LLMConfig{
		Provider:   "deepseek",
		APIKey:     "your-key",
		BaseURL:    "https://api.deepseek.com",
		Model:      "deepseek-chat",
		Timeout:    30 * time.Second,
	}

	// Old client creation
	client, err := oldLLM.NewClient(config)
	if err != nil {
		log.Fatal(err)
	}

	// Old request structure
	request := oldLLM.Request{
		Prompt:      "Hello, world!",
		Temperature: 0.7,
		MaxTokens:   100,
	}

	// Old generation
	response, err := client.Generate(context.Background(), request)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Response: %s\n", response.Content)
}
*/

// NEW WAY - Unified LLM Client
func newLLMUsage() {
	// New config structure
	config := llm.Config{
		Provider:           llm.ProviderDeepSeek,
		APIKey:             "your-key",
		BaseURL:            "https://api.deepseek.com",
		DefaultModel:       "deepseek-chat",
		Timeout:            30 * time.Second,
		DefaultTemperature: func() *float64 { t := 0.7; return &t }(),
		DefaultMaxTokens:   func() *int { m := 100; return &m }(),
	}

	// New client creation
	client, err := llm.NewClient(config)
	if err != nil {
		log.Fatal(err)
	}
	defer client.Close()

	// New simple generation (replaces old Request structure)
	response, err := llm.GenerateSimple(context.Background(), client, "Hello, world!")
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Response: %s\n", response.Content)
	fmt.Printf("Tokens: %d, Time: %v\n", response.TokensUsed, response.ResponseTime)
}

// Migration examples for different use cases

// Example 1: Simple text generation
func migrateSimpleGeneration() {
	// OLD
	/*
		request := oldLLM.Request{
			Prompt: "Explain quantum computing",
			Temperature: 0.5,
			MaxTokens: 200,
		}
		response, _ := client.Generate(ctx, request)
	*/

	// NEW
	client, _ := llm.NewClient(llm.Config{Provider: llm.ProviderDeepSeek, APIKey: "key"})
	request := llm.BuildSimpleRequest("Explain quantum computing")
	request.SetTemperature(0.5)
	request.SetMaxTokens(200)
	response, _ := client.Generate(context.Background(), request)
	fmt.Println(response.Content)
}

// Example 2: Chat-based generation with history
func migrateChatGeneration() {
	// OLD - didn't have chat history support
	/*
		prompt := "System: You are helpful\nUser: Hello\nAssistant: Hi!\nUser: How are you?"
		request := oldLLM.Request{Prompt: prompt}
		response, _ := client.Generate(ctx, request)
	*/

	// NEW
	client, _ := llm.NewClient(llm.Config{Provider: llm.ProviderDeepSeek, APIKey: "key"})

	history := llm.ChatHistory{}
	history.AddSystemMessage("You are helpful")
	history.AddUserMessage("Hello")
	history.AddAssistantMessage("Hi!")
	// history.AddUserMessage("How are you?") // This would be the new user message

	response, _ := client.GenerateWithHistory(
		context.Background(),
		history,
		"How are you?", // New user message
		"",             // System prompt (already in history)
	)
	fmt.Println(response.Content)
}

// Example 3: System prompt usage
func migrateSystemPrompt() {
	// OLD
	/*
		prompt := "You are a coding assistant. Write a function to reverse a string."
		request := oldLLM.Request{
			Prompt: prompt,
			Temperature: 0.3,
		}
	*/

	// NEW
	client, _ := llm.NewClient(llm.Config{Provider: llm.ProviderOpenAI, APIKey: "key"})
	response, _ := llm.GenerateWithSystemPrompt(
		context.Background(),
		client,
		"You are a coding assistant.",
		"Write a function to reverse a string.",
	)
	fmt.Println(response.Content)
}

// Example 4: Configuration migration
func migrateConfiguration() {
	// OLD config (from profile-builder)
	/*
		type LLMConfig struct {
			Provider string
			APIKey   string
			BaseURL  string
			Model    string
			Timeout  time.Duration
		}
	*/

	// NEW config example:
	// config := llm.Config{
	//     Provider:   llm.ProviderDeepSeek,  // enum instead of string
	//     APIKey:     "your-api-key",
	//     BaseURL:    "https://api.deepseek.com",
	//     DefaultModel: "deepseek-chat",     // renamed from Model
	//     Timeout:    30 * time.Second,
	//     // New features
	//     DefaultTemperature: &[]float64{0.7}[0],
	//     DefaultMaxTokens:   &[]int{1000}[0],
	//     DefaultTopP:        &[]float64{0.9}[0],
	//     DefaultTopK:        &[]int{40}[0],
	//     // Provider-specific settings
	//     ExtraConfig: map[string]interface{}{
	//         "custom_param": "value",
	//     },
	// }
}

// Example 5: Error handling improvements
func migrateErrorHandling() {
	client, _ := llm.NewClient(llm.Config{Provider: llm.ProviderDeepSeek, APIKey: "key"})

	response, err := llm.GenerateSimple(context.Background(), client, "Hello")
	if err != nil {
		// NEW: More specific error handling
		switch {
		case strings.Contains(err.Error(), "API key"):
			log.Println("Authentication error")
		case strings.Contains(err.Error(), "timeout"):
			log.Println("Request timeout")
		case strings.Contains(err.Error(), "rate limit"):
			log.Println("Rate limit exceeded")
		default:
			log.Printf("LLM error: %v", err)
		}
		return
	}

	// NEW: More detailed response information
	fmt.Printf("Content: %s\n", response.Content)
	fmt.Printf("Tokens used: %d\n", response.TokensUsed)
	fmt.Printf("Response time: %v\n", response.ResponseTime)
	fmt.Printf("Finish reason: %s\n", response.FinishReason)
}

// Migration checklist for profile-builder:
/*
1. Replace import: agents/profile-builder/internal/llm → github.com/yhwhpe/llm-unified-client
2. Update config: LLMConfig → llm.Config with Provider enum
3. Change client creation: NewClient(config) → llm.NewClient(config)
4. Update request building: Request{Prompt: "..."} → BuildSimpleRequest("...")
5. Add defer client.Close() for proper resource management
6. Update error handling to use new response fields
7. Use GenerateWithHistory for chat-based interactions
8. Use GenerateWithSystemPrompt for system prompt scenarios
9. Consider using ChatHistory for conversation management
10. Update model names and parameters as needed
*/
