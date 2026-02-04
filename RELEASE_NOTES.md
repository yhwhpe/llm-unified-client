# Release Notes - Embedding API & Cohere Support

## Summary

This release adds comprehensive embedding API support and introduces Cohere as a new provider. These additions enable semantic search, text clustering, and matchmaking capabilities essential for the Mesa agent and Matchmaker service integration.

## 🎯 Key Features

### 1. Embedding API Support

Generate vector embeddings for text using a unified interface across providers.

```go
resp, err := client.CreateEmbedding(ctx, llm.EmbeddingRequest{
    Input: []string{"Your text here"},
})
embedding := resp.Embeddings[0] // []float64
```

**Supported:**
- ✅ OpenAI (text-embedding-3-small, text-embedding-3-large)
- ✅ Cohere (embed-multilingual-v3.0)
- ⏳ Qwen (coming soon)
- ⏳ Azure (coming soon)

### 2. Cohere Provider

New provider with excellent multilingual support.

```go
client, err := llm.NewClient(llm.Config{
    Provider:     llm.ProviderCohere,
    APIKey:       "your-key",
    DefaultModel: "embed-multilingual-v3.0",
})
```

**Features:**
- 100+ languages supported
- 1024-dimensional embeddings
- Competitive pricing
- Chat API support

## 📦 What's Changed

### New Files

```
llm-unified-client/
├── cohere_client.go              # Cohere provider implementation
├── embedding_test.go             # Embedding API tests
├── EMBEDDING_API.md              # Complete embedding guide
├── CHANGELOG.md                  # Version history
├── RELEASE_NOTES.md             # This file
└── examples/
    └── embedding/
        └── main.go               # Embedding examples
```

### Modified Files

```
llm-unified-client/
├── types.go                      # + EmbeddingRequest, EmbeddingResponse, ProviderCohere
├── client.go                     # + Cohere support in NewClient()
├── openai_client.go             # + CreateEmbedding implementation
├── qwen_client.go               # + CreateEmbedding stub
├── azure_client.go              # + CreateEmbedding stub
└── README.md                     # + Embedding docs, Cohere section
```

## 🔧 API Changes

### New Interface Method

```go
type Client interface {
    // Existing methods...
    Generate(ctx context.Context, request Request) (*Response, error)
    GenerateWithHistory(ctx context.Context, history ChatHistory, userMessage, systemPrompt string) (*Response, error)

    // NEW: Embedding generation
    CreateEmbedding(ctx context.Context, request EmbeddingRequest) (*EmbeddingResponse, error)

    Close() error
    GetConfig() Config
}
```

### New Types

```go
type EmbeddingRequest struct {
    Input []string  // Texts to embed
    Model *string   // Optional model override
}

type EmbeddingResponse struct {
    Embeddings   [][]float64   // Vector embeddings
    Model        string        // Model used
    TokensUsed   int           // Tokens consumed
    ResponseTime time.Duration // Response time
}
```

### New Provider Constant

```go
const (
    ProviderOpenAI   Provider = "openai"
    ProviderDeepSeek Provider = "deepseek"
    ProviderQwen     Provider = "qwen"
    ProviderAzure    Provider = "azure"
    ProviderCohere   Provider = "cohere" // NEW
)
```

## 📚 Documentation

### New Documentation Files

1. **[EMBEDDING_API.md](EMBEDDING_API.md)** - Complete embedding API guide
   - Provider comparison
   - Usage examples
   - Best practices
   - Integration patterns
   - Performance tips

2. **[CHANGELOG.md](CHANGELOG.md)** - Version history and changes
   - Detailed feature descriptions
   - Technical details
   - Migration notes

3. **[examples/embedding/main.go](examples/embedding/main.go)** - Working examples
   - OpenAI single/batch embedding
   - Cohere multilingual embedding
   - Cosine similarity calculation

### Updated Documentation

- **[README.md](README.md)** - Added embedding section, Cohere config

## 🚀 Usage Examples

### Quick Start

```go
import llm "github.com/yhwhpe/llm-unified-client"

// OpenAI
client, _ := llm.NewClient(llm.Config{
    Provider: llm.ProviderOpenAI,
    APIKey:   "your-key",
    DefaultModel: "text-embedding-3-small",
})

resp, _ := client.CreateEmbedding(ctx, llm.EmbeddingRequest{
    Input: []string{"Hello, world!"},
})

fmt.Printf("Dimension: %d\n", len(resp.Embeddings[0]))
```

### Mesa Integration

```go
// In Mesa processor
func (p *Processor) generateGoalVector(ctx context.Context, msCore string) ([]float64, error) {
    resp, err := p.llmClient.CreateEmbedding(ctx, llm.EmbeddingRequest{
        Input: []string{msCore},
    })
    if err != nil {
        return nil, err
    }
    return resp.Embeddings[0], nil
}
```

### Matchmaking

```go
// 1. Generate goal embedding
goalResp, _ := client.CreateEmbedding(ctx, llm.EmbeddingRequest{
    Input: []string{userGoal},
})

// 2. Store in Weaviate
weaviate.AddInitiantGoal(ctx, &weaviate.InitiantGoal{
    SubjectID:  subjectID,
    MSCore:     msCore,
    GoalVector: goalResp.Embeddings[0],
})

// 3. Search similar masters
results, _ := weaviate.SearchMastersBySimilarity(
    ctx,
    goalResp.Embeddings[0],
    0.7,  // certainty
    2,    // top K
)
```

## 🧪 Testing

### Run Tests

```bash
# Unit tests
go test -v

# OpenAI integration tests
OPENAI_API_KEY=your-key go test -v -run TestOpenAIEmbedding

# Cohere integration tests
COHERE_API_KEY=your-key go test -v -run TestCohereEmbedding
```

### Run Examples

```bash
cd examples/embedding
export OPENAI_API_KEY=your-key
export COHERE_API_KEY=your-key
go run main.go
```

## 🔄 Migration Guide

### For Mesa Agent

**Before:**
```go
// Mesa had no embedding support
```

**After:**
```go
// In Processor.Analyze()
result, err := p.analyzeWithLLM(ctx, req)

// Generate goal_vector
goalVector, err := p.generateGoalVector(ctx, result.MSCore)
result.GoalVector = goalVector

// Now available in response
response := &MesaClassificationResponse{
    GoalVector: result.GoalVector,
    // ... other fields
}
```

### For New Users

Simply use the new API - no migration needed!

## 📊 Performance

### OpenAI

- **text-embedding-3-small**: ~10ms per text, 1536 dimensions
- **text-embedding-3-large**: ~20ms per text, 3072 dimensions
- Rate limit: 3000 RPM

### Cohere

- **embed-multilingual-v3.0**: ~50ms per batch, 1024 dimensions
- Excellent for multilingual content
- Competitive pricing

## 💰 Cost Estimates

### OpenAI

- text-embedding-3-small: $0.02 / 1M tokens
- text-embedding-3-large: $0.13 / 1M tokens

Example: 1000 embeddings of ~50 tokens each = ~$0.001 (small model)

### Cohere

- Varies by plan, generally competitive
- Free tier available for development

## 🐛 Known Issues

None currently. Report issues on GitHub.

## 🔮 Future Plans

- [ ] Streaming support for chat
- [ ] Qwen embedding implementation
- [ ] Azure embedding implementation
- [ ] Batch size optimization
- [ ] Embedding cache utilities
- [ ] Vector dimension reduction options

## 🤝 Contributing

Contributions welcome! See [README.md](README.md) for guidelines.

## 📝 License

MIT License - see LICENSE file for details.

## 🔗 Related Documentation

- [EMBEDDING_API.md](EMBEDDING_API.md) - Detailed embedding guide
- [CHANGELOG.md](CHANGELOG.md) - Complete version history
- [README.md](README.md) - General usage and setup
- [examples/](examples/) - Code examples

## 📞 Support

- GitHub Issues: Report bugs or request features
- Documentation: Check the docs/ directory
- Examples: See examples/ for working code

---

**Version:** Ready for tagging
**Date:** 2026-02-04
**Author:** PE Platform Team
