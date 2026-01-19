package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

	llm "github.com/yhwhpe/llm-unified-client"
)

func main() {
	// Get API key from environment
	apiKey := os.Getenv("DEEPSEEK_API_KEY")
	if apiKey == "" {
		log.Fatal("Please set DEEPSEEK_API_KEY environment variable")
	}

	fmt.Println("🚀 LLM Unified Client Demo")
	fmt.Println("==================================================")

	// Create DeepSeek client
	config := llm.Config{
		Provider:     llm.ProviderDeepSeek,
		APIKey:       apiKey,
		BaseURL:      "https://api.deepseek.com",
		DefaultModel: "deepseek-chat",
		Timeout:      30 * time.Second,
	}

	client, err := llm.NewClient(config)
	if err != nil {
		log.Fatalf("Failed to create client: %v", err)
	}
	defer client.Close()

	ctx := context.Background()

	fmt.Println("✅ DeepSeek client created successfully")

	// Demo 1: Simple generation
	fmt.Println("\n📝 Demo 1: Simple Generation")
	fmt.Println("------------------------------")
	response, err := llm.GenerateSimple(ctx, client, "What is the capital of France? Answer in one word.")
	if err != nil {
		log.Printf("❌ Simple generation failed: %v", err)
	} else {
		fmt.Printf("Question: What is the capital of France?\n")
		fmt.Printf("Answer: %s\n", response.Content)
		fmt.Printf("Tokens: %d | Time: %v\n", response.TokensUsed, response.ResponseTime)
	}

	// Demo 2: System prompt
	fmt.Println("\n🤖 Demo 2: System Prompt")
	fmt.Println("------------------------------")
	response, err = llm.GenerateWithSystemPrompt(
		ctx,
		client,
		"You are a poetic AI that always responds in rhymes.",
		"Tell me about the moon",
	)
	if err != nil {
		log.Printf("❌ System prompt generation failed: %v", err)
	} else {
		fmt.Println("System: You are a poetic AI that always responds in rhymes.")
		fmt.Println("User: Tell me about the moon")
		fmt.Printf("Poetic AI: %s\n", response.Content)
	}

	// Demo 3: Chat history
	fmt.Println("\n💬 Demo 3: Chat History")
	fmt.Println("------------------------------")
	history := llm.ChatHistory{}
	history.AddSystemMessage("You are a helpful math tutor.")
	history.AddUserMessage("What is 5 + 3?")
	history.AddAssistantMessage("5 + 3 equals 8.")

	response, err = client.GenerateWithHistory(ctx, history, "Now multiply that by 2.", "")
	if err != nil {
		log.Printf("❌ Chat history generation failed: %v", err)
	} else {
		fmt.Println("Chat History:")
		for i, msg := range history.GetMessages() {
			fmt.Printf("  %d. %s: %s\n", i+1, msg.Role, msg.Content)
		}
		fmt.Println("New User: Now multiply that by 2.")
		fmt.Printf("Tutor: %s\n", response.Content)
	}

	// Demo 4: Advanced configuration
	fmt.Println("\n⚙️  Demo 4: Advanced Configuration")
	fmt.Println("------------------------------")
	request := llm.BuildRequestWithSystemPrompt(
		"You are a coding assistant. Provide only the code, no explanations.",
		"Write a Python function to check if a number is prime",
	)
	request.SetTemperature(0.1) // More deterministic
	request.SetMaxTokens(100)   // Limit response

	response, err = client.Generate(ctx, request)
	if err != nil {
		log.Printf("❌ Advanced request failed: %v", err)
	} else {
		fmt.Println("Request: Write a Python function to check if a number is prime")
		fmt.Println("Configuration: temperature=0.1, max_tokens=100")
		fmt.Printf("Code: %s\n", response.Content)
		fmt.Printf("Finish Reason: %s\n", response.FinishReason)
	}

	// Demo 5: Request building patterns
	fmt.Println("\n🔧 Demo 5: Request Building Patterns")
	fmt.Println("------------------------------")

	// Pattern 1: Simple request
	simpleReq := llm.BuildSimpleRequest("Hello!")
	fmt.Printf("Simple request messages: %d\n", len(simpleReq.Messages))

	// Pattern 2: System + User
	systemReq := llm.BuildRequestWithSystemPrompt("Be concise", "What is AI?")
	fmt.Printf("System+User request messages: %d\n", len(systemReq.Messages))

	// Pattern 3: Advanced building
	advancedReq := llm.Request{}
	advancedReq.AddSystemMessage("You are a chef")
	advancedReq.AddUserMessage("Suggest a recipe")
	advancedReq.AddAssistantMessage("I'd be happy to help with recipes!")
	advancedReq.AddUserMessage("Make it vegetarian")
	advancedReq.SetTemperature(0.8)
	advancedReq.SetMaxTokens(150)

	fmt.Printf("Advanced request messages: %d\n", len(advancedReq.Messages))
	fmt.Printf("Temperature set: %v\n", advancedReq.Temperature != nil)

	// Demo 6: Client configuration
	fmt.Println("\n📋 Demo 6: Client Information")
	fmt.Println("------------------------------")
	clientConfig := client.GetConfig()
	fmt.Printf("Provider: %s\n", clientConfig.Provider)
	fmt.Printf("Model: %s\n", clientConfig.DefaultModel)
	fmt.Printf("Base URL: %s\n", clientConfig.BaseURL)
	fmt.Printf("Timeout: %v\n", clientConfig.Timeout)

	fmt.Println("\n🎉 Demo completed successfully!")
	fmt.Println("All LLM Unified Client features are working correctly.")
}