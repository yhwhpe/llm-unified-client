package llm

import (
	"context"
	"os"
	"testing"
	"time"
)

// TestEmbeddingRequest tests the EmbeddingRequest structure
func TestEmbeddingRequest(t *testing.T) {
	tests := []struct {
		name  string
		input []string
		model *string
	}{
		{
			name:  "single text",
			input: []string{"Hello world"},
			model: nil,
		},
		{
			name:  "multiple texts",
			input: []string{"Hello", "World", "Test"},
			model: nil,
		},
		{
			name:  "with custom model",
			input: []string{"Test"},
			model: stringPtr("text-embedding-3-small"),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := EmbeddingRequest{
				Input: tt.input,
				Model: tt.model,
			}

			if len(req.Input) != len(tt.input) {
				t.Errorf("Expected %d inputs, got %d", len(tt.input), len(req.Input))
			}

			if tt.model != nil && req.Model == nil {
				t.Error("Expected model to be set")
			}
		})
	}
}

// TestOpenAIEmbedding tests OpenAI embedding generation (integration test)
func TestOpenAIEmbedding(t *testing.T) {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		t.Skip("OPENAI_API_KEY not set, skipping integration test")
	}

	client, err := NewClient(Config{
		Provider:     ProviderOpenAI,
		APIKey:       apiKey,
		BaseURL:      "https://api.openai.com/v1",
		DefaultModel: "text-embedding-3-small",
		Timeout:      30 * time.Second,
	})
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}
	defer client.Close()

	ctx := context.Background()

	t.Run("single embedding", func(t *testing.T) {
		resp, err := client.CreateEmbedding(ctx, EmbeddingRequest{
			Input: []string{"The quick brown fox jumps over the lazy dog"},
		})

		if err != nil {
			t.Fatalf("Failed to create embedding: %v", err)
		}

		if len(resp.Embeddings) != 1 {
			t.Errorf("Expected 1 embedding, got %d", len(resp.Embeddings))
		}

		if len(resp.Embeddings[0]) == 0 {
			t.Error("Expected non-empty embedding vector")
		}

		if resp.Model == "" {
			t.Error("Expected model name in response")
		}

		if resp.TokensUsed == 0 {
			t.Error("Expected non-zero tokens used")
		}

		t.Logf("Embedding dimension: %d, tokens: %d, time: %v",
			len(resp.Embeddings[0]), resp.TokensUsed, resp.ResponseTime)
	})

	t.Run("batch embeddings", func(t *testing.T) {
		texts := []string{
			"Hello world",
			"OpenAI embeddings",
			"Machine learning",
		}

		resp, err := client.CreateEmbedding(ctx, EmbeddingRequest{
			Input: texts,
		})

		if err != nil {
			t.Fatalf("Failed to create embeddings: %v", err)
		}

		if len(resp.Embeddings) != len(texts) {
			t.Errorf("Expected %d embeddings, got %d", len(texts), len(resp.Embeddings))
		}

		for i, emb := range resp.Embeddings {
			if len(emb) == 0 {
				t.Errorf("Embedding %d is empty", i)
			}
		}

		t.Logf("Generated %d embeddings in %v", len(resp.Embeddings), resp.ResponseTime)
	})
}

// TestCohereEmbedding tests Cohere embedding generation (integration test)
func TestCohereEmbedding(t *testing.T) {
	apiKey := os.Getenv("COHERE_API_KEY")
	if apiKey == "" {
		t.Skip("COHERE_API_KEY not set, skipping integration test")
	}

	client, err := NewClient(Config{
		Provider:     ProviderCohere,
		APIKey:       apiKey,
		BaseURL:      "https://api.cohere.ai/v1",
		DefaultModel: "embed-multilingual-v3.0",
		Timeout:      30 * time.Second,
	})
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}
	defer client.Close()

	ctx := context.Background()

	t.Run("multilingual embedding", func(t *testing.T) {
		texts := []string{
			"Hello world",
			"Привет мир",
			"你好世界",
		}

		resp, err := client.CreateEmbedding(ctx, EmbeddingRequest{
			Input: texts,
		})

		if err != nil {
			t.Fatalf("Failed to create embeddings: %v", err)
		}

		if len(resp.Embeddings) != len(texts) {
			t.Errorf("Expected %d embeddings, got %d", len(texts), len(resp.Embeddings))
		}

		// Cohere embeddings are typically 1024 dimensions for v3 models
		for i, emb := range resp.Embeddings {
			if len(emb) == 0 {
				t.Errorf("Embedding %d is empty", i)
			}
			t.Logf("Embedding %d dimension: %d", i, len(emb))
		}

		t.Logf("Generated %d multilingual embeddings in %v", len(resp.Embeddings), resp.ResponseTime)
	})
}

// TestCosineSimilarity tests cosine similarity calculation
func TestCosineSimilarity(t *testing.T) {
	tests := []struct {
		name     string
		a        []float64
		b        []float64
		expected float64
		delta    float64
	}{
		{
			name:     "identical vectors",
			a:        []float64{1, 0, 0},
			b:        []float64{1, 0, 0},
			expected: 1.0,
			delta:    0.001,
		},
		{
			name:     "orthogonal vectors",
			a:        []float64{1, 0},
			b:        []float64{0, 1},
			expected: 0.0,
			delta:    0.001,
		},
		{
			name:     "opposite vectors",
			a:        []float64{1, 0},
			b:        []float64{-1, 0},
			expected: -1.0,
			delta:    0.001,
		},
		{
			name:     "similar vectors",
			a:        []float64{1, 2, 3},
			b:        []float64{1, 2, 3},
			expected: 1.0,
			delta:    0.001,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			similarity := cosineSimilarity(tt.a, tt.b)
			if abs(similarity-tt.expected) > tt.delta {
				t.Errorf("Expected similarity ~%.3f, got %.3f", tt.expected, similarity)
			}
		})
	}
}

// Helper functions

func stringPtr(s string) *string {
	return &s
}

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

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}
