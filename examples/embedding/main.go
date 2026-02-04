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
	fmt.Println("=== LLM Unified Client - Embedding Examples ===\n")

	// Example 1: OpenAI Embeddings
	runOpenAIEmbeddingExample()

	// Example 2: Cohere Embeddings
	runCohereEmbeddingExample()

	// Example 3: Batch Embeddings
	runBatchEmbeddingExample()
}

// runOpenAIEmbeddingExample demonstrates OpenAI embedding generation
func runOpenAIEmbeddingExample() {
	fmt.Println("1. OpenAI Embedding Example")
	fmt.Println("----------------------------")

	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		fmt.Println("⚠️  OPENAI_API_KEY not set, skipping example\n")
		return
	}

	// Create OpenAI client
	client, err := llm.NewClient(llm.Config{
		Provider:     llm.ProviderOpenAI,
		APIKey:       apiKey,
		BaseURL:      "https://api.openai.com/v1",
		DefaultModel: "text-embedding-3-small",
		Timeout:      30 * time.Second,
	})
	if err != nil {
		log.Fatalf("Failed to create OpenAI client: %v", err)
	}
	defer client.Close()

	// Generate embedding for a single text
	ctx := context.Background()
	text := "The quick brown fox jumps over the lazy dog"

	embeddingReq := llm.EmbeddingRequest{
		Input: []string{text},
	}

	resp, err := client.CreateEmbedding(ctx, embeddingReq)
	if err != nil {
		log.Fatalf("Failed to create embedding: %v", err)
	}

	fmt.Printf("✅ Text: %s\n", text)
	fmt.Printf("✅ Model: %s\n", resp.Model)
	fmt.Printf("✅ Embedding dimension: %d\n", len(resp.Embeddings[0]))
	fmt.Printf("✅ First 5 values: %.4f, %.4f, %.4f, %.4f, %.4f\n",
		resp.Embeddings[0][0],
		resp.Embeddings[0][1],
		resp.Embeddings[0][2],
		resp.Embeddings[0][3],
		resp.Embeddings[0][4])
	fmt.Printf("✅ Tokens used: %d\n", resp.TokensUsed)
	fmt.Printf("✅ Response time: %v\n\n", resp.ResponseTime)
}

// runCohereEmbeddingExample demonstrates Cohere embedding generation
func runCohereEmbeddingExample() {
	fmt.Println("2. Cohere Embedding Example")
	fmt.Println("---------------------------")

	apiKey := os.Getenv("COHERE_API_KEY")
	if apiKey == "" {
		fmt.Println("⚠️  COHERE_API_KEY not set, skipping example\n")
		return
	}

	// Create Cohere client
	client, err := llm.NewClient(llm.Config{
		Provider:     llm.ProviderCohere,
		APIKey:       apiKey,
		BaseURL:      "https://api.cohere.ai/v1",
		DefaultModel: "embed-multilingual-v3.0",
		Timeout:      30 * time.Second,
	})
	if err != nil {
		log.Fatalf("Failed to create Cohere client: %v", err)
	}
	defer client.Close()

	// Generate embedding for multilingual text
	ctx := context.Background()
	text := "Быстрая коричневая лиса перепрыгивает через ленивую собаку"

	embeddingReq := llm.EmbeddingRequest{
		Input: []string{text},
	}

	resp, err := client.CreateEmbedding(ctx, embeddingReq)
	if err != nil {
		log.Fatalf("Failed to create embedding: %v", err)
	}

	fmt.Printf("✅ Text: %s\n", text)
	fmt.Printf("✅ Model: %s\n", resp.Model)
	fmt.Printf("✅ Embedding dimension: %d\n", len(resp.Embeddings[0]))
	fmt.Printf("✅ First 5 values: %.4f, %.4f, %.4f, %.4f, %.4f\n",
		resp.Embeddings[0][0],
		resp.Embeddings[0][1],
		resp.Embeddings[0][2],
		resp.Embeddings[0][3],
		resp.Embeddings[0][4])
	fmt.Printf("✅ Tokens used: %d\n", resp.TokensUsed)
	fmt.Printf("✅ Response time: %v\n\n", resp.ResponseTime)
}

// runBatchEmbeddingExample demonstrates batch embedding generation
func runBatchEmbeddingExample() {
	fmt.Println("3. Batch Embedding Example")
	fmt.Println("--------------------------")

	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		fmt.Println("⚠️  OPENAI_API_KEY not set, skipping example\n")
		return
	}

	// Create OpenAI client
	client, err := llm.NewClient(llm.Config{
		Provider:     llm.ProviderOpenAI,
		APIKey:       apiKey,
		DefaultModel: "text-embedding-3-small",
		Timeout:      30 * time.Second,
	})
	if err != nil {
		log.Fatalf("Failed to create OpenAI client: %v", err)
	}
	defer client.Close()

	// Generate embeddings for multiple texts
	ctx := context.Background()
	texts := []string{
		"I want to find a therapist for burnout",
		"Looking for career coaching",
		"Need help with anxiety and stress",
	}

	embeddingReq := llm.EmbeddingRequest{
		Input: texts,
	}

	resp, err := client.CreateEmbedding(ctx, embeddingReq)
	if err != nil {
		log.Fatalf("Failed to create embeddings: %v", err)
	}

	fmt.Printf("✅ Generated embeddings for %d texts\n", len(texts))
	fmt.Printf("✅ Model: %s\n", resp.Model)
	fmt.Printf("✅ Total tokens used: %d\n", resp.TokensUsed)
	fmt.Printf("✅ Response time: %v\n\n", resp.ResponseTime)

	for i, text := range texts {
		fmt.Printf("Text %d: %s\n", i+1, text)
		fmt.Printf("  Dimension: %d\n", len(resp.Embeddings[i]))
		fmt.Printf("  First 3 values: %.4f, %.4f, %.4f\n\n",
			resp.Embeddings[i][0],
			resp.Embeddings[i][1],
			resp.Embeddings[i][2])
	}

	// Calculate cosine similarity between first two embeddings
	similarity := cosineSimilarity(resp.Embeddings[0], resp.Embeddings[1])
	fmt.Printf("✅ Cosine similarity between text 1 and 2: %.4f\n", similarity)
}

// cosineSimilarity calculates cosine similarity between two vectors
func cosineSimilarity(a, b []float64) float64 {
	if len(a) != len(b) {
		return 0
	}

	var dotProduct, normA, normB float64
	for i := 0; i < len(a); i++ {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	return dotProduct / (sqrt(normA) * sqrt(normB))
}

// sqrt calculates square root
func sqrt(x float64) float64 {
	if x < 0 {
		return 0
	}
	z := 1.0
	for i := 0; i < 10; i++ {
		z = z - (z*z-x)/(2*z)
	}
	return z
}
