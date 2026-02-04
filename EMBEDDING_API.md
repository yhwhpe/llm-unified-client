# Embedding API Guide

Complete guide for text embedding generation using the LLM Unified Client.

## Overview

The embedding API allows you to convert text into numerical vector representations (embeddings) that capture semantic meaning. These embeddings are essential for:

- **Semantic search** - Find documents similar to a query
- **Clustering** - Group related content together
- **Classification** - Categorize text by similarity
- **Recommendation** - Suggest related items
- **Matchmaking** - Match entities based on semantic similarity

## Supported Providers

| Provider | Model | Dimensions | Languages | Notes |
|----------|-------|------------|-----------|-------|
| OpenAI | text-embedding-3-small | 1536 | English-focused | Fast, cost-effective |
| OpenAI | text-embedding-3-large | 3072 | English-focused | Higher quality |
| OpenAI | text-embedding-ada-002 | 1536 | English-focused | Legacy model |
| Cohere | embed-multilingual-v3.0 | 1024 | 100+ languages | Best for multilingual |
| Cohere | embed-english-v3.0 | 1024 | English | Optimized for English |
| Cohere | embed-multilingual-light-v3.0 | 384 | 100+ languages | Smaller, faster |

## Quick Start

### OpenAI Embedding

```go
package main

import (
    "context"
    "fmt"
    "log"

    llm "github.com/yhwhpe/llm-unified-client"
)

func main() {
    // Create client
    client, err := llm.NewClient(llm.Config{
        Provider:     llm.ProviderOpenAI,
        APIKey:       "your-openai-api-key",
        DefaultModel: "text-embedding-3-small",
    })
    if err != nil {
        log.Fatal(err)
    }
    defer client.Close()

    // Generate embedding
    ctx := context.Background()
    resp, err := client.CreateEmbedding(ctx, llm.EmbeddingRequest{
        Input: []string{"Hello, world!"},
    })
    if err != nil {
        log.Fatal(err)
    }

    fmt.Printf("Embedding dimension: %d\n", len(resp.Embeddings[0]))
    fmt.Printf("Tokens used: %d\n", resp.TokensUsed)
}
```

### Cohere Multilingual Embedding

```go
client, err := llm.NewClient(llm.Config{
    Provider:     llm.ProviderCohere,
    APIKey:       "your-cohere-api-key",
    DefaultModel: "embed-multilingual-v3.0",
})

// Works with any language
texts := []string{
    "Hello world",           // English
    "Привет мир",           // Russian
    "你好世界",              // Chinese
    "مرحبا بالعالم",         // Arabic
}

resp, err := client.CreateEmbedding(ctx, llm.EmbeddingRequest{
    Input: texts,
})
```

## API Reference

### EmbeddingRequest

```go
type EmbeddingRequest struct {
    // Input texts to embed (required)
    Input []string

    // Model override (optional)
    // If not set, uses client's DefaultModel
    Model *string
}
```

### EmbeddingResponse

```go
type EmbeddingResponse struct {
    // Array of embedding vectors
    // Length equals len(request.Input)
    Embeddings [][]float64

    // Model used for generation
    Model string

    // Total tokens consumed
    TokensUsed int

    // API response time
    ResponseTime time.Duration
}
```

### Client Interface

```go
type Client interface {
    // CreateEmbedding generates embeddings for the given text(s)
    CreateEmbedding(ctx context.Context, request EmbeddingRequest) (*EmbeddingResponse, error)

    // ... other methods
}
```

## Usage Examples

### Single Text

```go
resp, err := client.CreateEmbedding(ctx, llm.EmbeddingRequest{
    Input: []string{"The quick brown fox jumps over the lazy dog"},
})

embedding := resp.Embeddings[0] // []float64
```

### Batch Processing

```go
// Process multiple texts in one API call
texts := []string{
    "I need therapy for burnout",
    "Looking for career coaching",
    "Help with anxiety",
}

resp, err := client.CreateEmbedding(ctx, llm.EmbeddingRequest{
    Input: texts,
})

// resp.Embeddings[i] corresponds to texts[i]
for i, emb := range resp.Embeddings {
    fmt.Printf("Text %d: %d dimensions\n", i, len(emb))
}
```

### Custom Model

```go
model := "text-embedding-3-large"
resp, err := client.CreateEmbedding(ctx, llm.EmbeddingRequest{
    Input: []string{"High quality embedding needed"},
    Model: &model,
})
```

## Integration Examples

### Mesa Agent Goal Vector Generation

```go
// In Mesa processor after LLM analysis
func (p *Processor) generateGoalVector(ctx context.Context, msCore string) ([]float64, error) {
    if msCore == "" {
        return nil, fmt.Errorf("ms_core is empty")
    }

    resp, err := p.llmClient.CreateEmbedding(ctx, llm.EmbeddingRequest{
        Input: []string{msCore},
    })
    if err != nil {
        return nil, fmt.Errorf("failed to create embedding: %w", err)
    }

    if len(resp.Embeddings) == 0 {
        return nil, fmt.Errorf("no embeddings returned")
    }

    return resp.Embeddings[0], nil
}
```

### Semantic Search with Weaviate

```go
// 1. Generate query embedding
queryResp, _ := client.CreateEmbedding(ctx, llm.EmbeddingRequest{
    Input: []string{userQuery},
})
queryVector := queryResp.Embeddings[0]

// 2. Search in Weaviate
results, _ := weaviateClient.SearchMastersBySimilarity(
    ctx,
    queryVector,
    0.7,  // certainty threshold
    10,   // top K results
)

// 3. Use results for matchmaking
for _, result := range results {
    fmt.Printf("Master: %s, Similarity: %.2f\n",
        result.MasterID, result.Similarity)
}
```

### Cosine Similarity Calculation

```go
// Calculate similarity between two embeddings
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

    return dotProduct / (math.Sqrt(normA) * math.Sqrt(normB))
}

// Usage
emb1 := resp.Embeddings[0]
emb2 := resp.Embeddings[1]
similarity := cosineSimilarity(emb1, emb2)
fmt.Printf("Similarity: %.4f\n", similarity)
```

## Best Practices

### 1. Batch When Possible

```go
// ✅ Good - single API call
resp, _ := client.CreateEmbedding(ctx, llm.EmbeddingRequest{
    Input: []string{"text1", "text2", "text3"},
})

// ❌ Avoid - multiple API calls
for _, text := range texts {
    resp, _ := client.CreateEmbedding(ctx, llm.EmbeddingRequest{
        Input: []string{text},
    })
}
```

### 2. Cache Embeddings

Embeddings for the same text are deterministic - cache them to save costs.

```go
// Pseudo-code
func getEmbedding(text string) []float64 {
    // Check cache first
    if cached, exists := cache.Get(text); exists {
        return cached
    }

    // Generate and cache
    resp, _ := client.CreateEmbedding(ctx, llm.EmbeddingRequest{
        Input: []string{text},
    })
    embedding := resp.Embeddings[0]
    cache.Set(text, embedding)

    return embedding
}
```

### 3. Choose Right Model

- **OpenAI text-embedding-3-small**: Best for English, cost-effective
- **Cohere embed-multilingual-v3.0**: Best for multilingual content
- **OpenAI text-embedding-3-large**: When you need highest quality

### 4. Normalize Vectors

For cosine similarity, vectors don't need normalization. But if using dot product or Euclidean distance, normalize first.

```go
func normalize(v []float64) []float64 {
    var norm float64
    for _, val := range v {
        norm += val * val
    }
    norm = math.Sqrt(norm)

    normalized := make([]float64, len(v))
    for i, val := range v {
        normalized[i] = val / norm
    }
    return normalized
}
```

### 5. Error Handling

```go
resp, err := client.CreateEmbedding(ctx, llm.EmbeddingRequest{
    Input: texts,
})
if err != nil {
    if strings.Contains(err.Error(), "rate limit") {
        // Implement backoff retry
        time.Sleep(time.Second)
        return retry(ctx, texts)
    }
    return nil, fmt.Errorf("embedding failed: %w", err)
}
```

## Performance Considerations

### OpenAI

- **text-embedding-3-small**: ~10ms per text
- **text-embedding-3-large**: ~20ms per text
- Rate limit: 3000 RPM (requests per minute)
- Batch size: Up to 2048 texts per request

### Cohere

- **embed-multilingual-v3.0**: ~50ms per batch
- Rate limit: Varies by plan
- Batch size: Recommended 96 texts per request

## Cost Optimization

### OpenAI Pricing (as of 2024)

- text-embedding-3-small: $0.02 / 1M tokens
- text-embedding-3-large: $0.13 / 1M tokens

### Cohere Pricing

- Varies by plan, generally competitive with OpenAI
- Free tier available for testing

### Tips

1. **Batch texts** to reduce API overhead
2. **Cache embeddings** for repeated texts
3. **Use smaller models** when appropriate (small vs large)
4. **Monitor token usage** with `resp.TokensUsed`

## Troubleshooting

### Issue: Empty embeddings returned

```go
if len(resp.Embeddings) == 0 {
    return nil, fmt.Errorf("no embeddings in response")
}
```

Check that input texts are not empty.

### Issue: Dimension mismatch

Ensure you're using the same model for all embeddings you want to compare.

```go
// ✅ Good - same model
allResp, _ := client.CreateEmbedding(ctx, llm.EmbeddingRequest{
    Input: []string{"text1", "text2"},
})

// ❌ Avoid - different models
resp1, _ := client.CreateEmbedding(ctx, llm.EmbeddingRequest{
    Input: []string{"text1"},
    Model: stringPtr("text-embedding-3-small"), // 1536 dims
})
resp2, _ := client.CreateEmbedding(ctx, llm.EmbeddingRequest{
    Input: []string{"text2"},
    Model: stringPtr("text-embedding-3-large"), // 3072 dims
})
```

### Issue: Rate limiting

Implement exponential backoff:

```go
for retries := 0; retries < 3; retries++ {
    resp, err := client.CreateEmbedding(ctx, request)
    if err == nil {
        return resp, nil
    }
    if strings.Contains(err.Error(), "rate limit") {
        time.Sleep(time.Duration(1<<retries) * time.Second)
        continue
    }
    return nil, err
}
```

## Testing

See [embedding_test.go](embedding_test.go) for complete test examples.

Run integration tests:

```bash
# OpenAI tests
OPENAI_API_KEY=your-key go test -v -run TestOpenAIEmbedding

# Cohere tests
COHERE_API_KEY=your-key go test -v -run TestCohereEmbedding

# All tests
go test -v
```

## Examples

See [examples/embedding/main.go](examples/embedding/main.go) for runnable examples:

```bash
cd examples/embedding
export OPENAI_API_KEY=your-key
export COHERE_API_KEY=your-key
go run main.go
```

## References

- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- [Cohere Embed API](https://docs.cohere.com/reference/embed)
- [Vector Similarity Search](https://www.pinecone.io/learn/vector-similarity/)
- [Cosine Similarity Explained](https://en.wikipedia.org/wiki/Cosine_similarity)

## Support

For issues or questions:
1. Check [CHANGELOG.md](CHANGELOG.md) for recent changes
2. Review [README.md](README.md) for general usage
3. See [examples/](examples/) for code samples
4. Open an issue on GitHub
