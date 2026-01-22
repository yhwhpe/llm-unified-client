package main

import (
	"context"
	"fmt"
	"log"
	"time"

	llm "github.com/yhwhpe/llm-unified-client"
)

func main() {
	// Example 1: DeepSeek client (replaces old profile-builder LLM)
	fmt.Println("=== DeepSeek Example ===")
	deepSeekConfig := llm.Config{
		Provider:           llm.ProviderDeepSeek,
		APIKey:             "your-deepseek-api-key", // from env
		BaseURL:            "https://api.deepseek.com",
		DefaultModel:       "deepseek-chat",
		Timeout:            30 * time.Second,
		DefaultTemperature: func() *float64 { t := 0.7; return &t }(),
		DefaultMaxTokens:   func() *int { m := 1000; return &m }(),
	}

	deepSeekClient, err := llm.NewClient(deepSeekConfig)
	if err != nil {
		log.Printf("Failed to create DeepSeek client: %v", err)
	} else {
		// Simple generation (replaces old Generate method)
		response, err := llm.GenerateSimple(context.Background(), deepSeekClient, "Hello from the new unified client!")
		if err != nil {
			log.Printf("DeepSeek generation failed: %v", err)
		} else {
			fmt.Printf("DeepSeek Response: %s\n", response.Content)
			fmt.Printf("Tokens: %d, Time: %v\n", response.TokensUsed, response.ResponseTime)
		}
		deepSeekClient.Close()
	}

	// Example 2: OpenAI client
	fmt.Println("\n=== OpenAI Example ===")
	openAIConfig := llm.Config{
		Provider:     llm.ProviderOpenAI,
		APIKey:       "your-openai-api-key",
		BaseURL:      "https://api.openai.com/v1",
		DefaultModel: "gpt-4",
		Timeout:      30 * time.Second,
	}

	openAIClient, err := llm.NewClient(openAIConfig)
	if err != nil {
		log.Printf("Failed to create OpenAI client: %v", err)
	} else {
		// Using system prompt (replaces old prompt-based generation)
		response, err := llm.GenerateWithSystemPrompt(
			context.Background(),
			openAIClient,
			"You are a helpful coding assistant.",
			"Write a Go function to calculate fibonacci numbers.",
		)
		if err != nil {
			log.Printf("OpenAI generation failed: %v", err)
		} else {
			fmt.Printf("OpenAI Response: %s\n", response.Content[:200]+"...")
		}
		openAIClient.Close()
	}

	// Example 3: Chat history usage
	fmt.Println("\n=== Chat History Example ===")
	chatClient, _ := llm.NewClient(deepSeekConfig)

	history := llm.ChatHistory{}
	history.AddSystemMessage("You are a helpful assistant.")
	history.AddUserMessage("What's the capital of France?")
	history.AddAssistantMessage("The capital of France is Paris.")

	response, err := chatClient.GenerateWithHistory(
		context.Background(),
		history,
		"What's the population of that city?",
		"", // system prompt already in history
	)
	if err != nil {
		log.Printf("Chat generation failed: %v", err)
	} else {
		fmt.Printf("Chat Response: %s\n", response.Content)
	}

	// Example 4: Advanced request building
	fmt.Println("\n=== Advanced Request Example ===")
	request := llm.BuildRequestWithSystemPrompt(
		"You are an expert in Go programming.",
		"Explain goroutines and channels.",
	)
	request.SetTemperature(0.3) // More focused
	request.SetMaxTokens(500)   // Limit response length
	request.SetModel("gpt-4")   // Override default model

	if openAIClient != nil {
		response, err := openAIClient.Generate(context.Background(), request)
		if err != nil {
			log.Printf("Advanced request failed: %v", err)
		} else {
			fmt.Printf("Advanced Response: %s\n", response.Content[:200]+"...")
		}
	}

	fmt.Println("\n=== Migration Complete ===")
	fmt.Println("Old profile-builder LLM package replaced with unified client!")
}
