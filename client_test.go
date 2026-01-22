package llm

import (
	"context"
	"os"
	"testing"
	"time"
)

func TestDeepSeekClient(t *testing.T) {
	apiKey := os.Getenv("DEEPSEEK_API_KEY")
	if apiKey == "" {
		t.Skip("DEEPSEEK_API_KEY not set, skipping integration test")
	}

	config := Config{
		Provider:     ProviderDeepSeek,
		APIKey:       apiKey,
		BaseURL:      "https://api.deepseek.com",
		DefaultModel: "deepseek-chat",
		Timeout:      30 * time.Second,
	}

	client, err := NewClient(config)
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}
	defer client.Close()

	ctx := context.Background()

	t.Run("Simple Generation", func(t *testing.T) {
		response, err := GenerateSimple(ctx, client, "Say 'Hello from DeepSeek!' and nothing else.")
		if err != nil {
			t.Fatalf("Failed to generate response: %v", err)
		}

		if response.Content == "" {
			t.Error("Response content is empty")
		}

		if response.TokensUsed == 0 {
			t.Error("Tokens used should be greater than 0")
		}

		t.Logf("Response: %s", response.Content)
		t.Logf("Tokens used: %d", response.TokensUsed)
		t.Logf("Response time: %v", response.ResponseTime)
	})

	t.Run("System Prompt Generation", func(t *testing.T) {
		response, err := GenerateWithSystemPrompt(
			ctx,
			client,
			"You are a helpful assistant that always responds in exactly 3 words.",
			"How are you?",
		)
		if err != nil {
			t.Fatalf("Failed to generate with system prompt: %v", err)
		}

		if response.Content == "" {
			t.Error("Response content is empty")
		}

		t.Logf("System prompt response: %s", response.Content)
	})

	t.Run("Chat History", func(t *testing.T) {
		history := ChatHistory{}
		history.AddSystemMessage("You are a helpful assistant.")
		history.AddUserMessage("What is 2+2?")
		history.AddAssistantMessage("2+2 equals 4.")

		response, err := client.GenerateWithHistory(ctx, history, "Now multiply that by 3.", "")
		if err != nil {
			t.Fatalf("Failed to generate with history: %v", err)
		}

		if response.Content == "" {
			t.Error("Response content is empty")
		}

		t.Logf("Chat history response: %s", response.Content)
	})

	t.Run("Advanced Request", func(t *testing.T) {
		request := BuildRequestWithSystemPrompt(
			"You are a programming expert.",
			"Write a simple Go hello world function.",
		)
		request.SetTemperature(0.1) // More deterministic
		request.SetMaxTokens(200)

		response, err := client.Generate(ctx, request)
		if err != nil {
			t.Fatalf("Failed to generate advanced request: %v", err)
		}

		if response.Content == "" {
			t.Error("Response content is empty")
		}

		t.Logf("Advanced request response: %s", response.Content)
		t.Logf("Finish reason: %s", response.FinishReason)
	})
}

func TestClientConfiguration(t *testing.T) {
	config := Config{
		Provider:     ProviderDeepSeek,
		APIKey:       "test-key",
		DefaultModel: "deepseek-chat",
		Timeout:      10 * time.Second,
	}

	client, err := NewClient(config)
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}
	defer client.Close()

	clientConfig := client.GetConfig()
	if clientConfig.Provider != ProviderDeepSeek {
		t.Errorf("Expected provider %s, got %s", ProviderDeepSeek, clientConfig.Provider)
	}

	if clientConfig.APIKey != "test-key" {
		t.Error("API key not set correctly")
	}
}

func TestChatHistory(t *testing.T) {
	history := ChatHistory{}

	// Test adding messages
	history.AddSystemMessage("You are helpful")
	history.AddUserMessage("Hello")
	history.AddAssistantMessage("Hi!")

	messages := history.GetMessages()
	if len(messages) != 3 {
		t.Errorf("Expected 3 messages, got %d", len(messages))
	}

	if messages[0].Role != RoleSystem {
		t.Error("First message should be system")
	}

	if messages[1].Role != RoleUser {
		t.Error("Second message should be user")
	}

	if messages[2].Role != RoleAssistant {
		t.Error("Third message should be assistant")
	}

	// Test truncation
	history.Truncate(2)
	messages = history.GetMessages()
	if len(messages) != 2 {
		t.Errorf("Expected 2 messages after truncation, got %d", len(messages))
	}

	// Test clearing
	history.Clear()
	if len(history.GetMessages()) != 0 {
		t.Error("History should be empty after clear")
	}
}

func TestRequestBuilding(t *testing.T) {
	// Test simple request
	request := BuildSimpleRequest("Hello")
	if len(request.Messages) != 1 {
		t.Errorf("Expected 1 message, got %d", len(request.Messages))
	}
	if request.Messages[0].Content != "Hello" {
		t.Error("Message content not set correctly")
	}

	// Test system prompt request
	request = BuildRequestWithSystemPrompt("You are helpful", "Hello")
	if len(request.Messages) != 2 {
		t.Errorf("Expected 2 messages, got %d", len(request.Messages))
	}
	if request.Messages[0].Role != RoleSystem {
		t.Error("First message should be system")
	}
	if request.Messages[1].Role != RoleUser {
		t.Error("Second message should be user")
	}

	// Test parameter setting
	request.SetTemperature(0.5)
	if request.Temperature == nil || *request.Temperature != 0.5 {
		t.Error("Temperature not set correctly")
	}

	request.SetMaxTokens(100)
	if request.MaxTokens == nil || *request.MaxTokens != 100 {
		t.Error("MaxTokens not set correctly")
	}

	request.SetModel("test-model")
	if request.Model == nil || *request.Model != "test-model" {
		t.Error("Model not set correctly")
	}
}
