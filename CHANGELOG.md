# Changelog

All notable changes to the LLM Unified Client library will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

#### Embedding API Support
- Added `CreateEmbedding` method to `Client` interface for generating text embeddings
- New types: `EmbeddingRequest` and `EmbeddingResponse` for embedding operations
- Support for single and batch text embedding generation

#### Cohere Provider
- Added full support for Cohere AI provider (`ProviderCohere`)
- Implemented `CreateEmbedding` for Cohere with multilingual support (100+ languages)
- Default model: `embed-multilingual-v3.0` for embeddings
- Chat API support via Cohere's `/chat` endpoint
- Proper handling of Cohere-specific message format (USER/CHATBOT roles)

#### OpenAI Embedding Support
- Implemented `CreateEmbedding` for OpenAI provider
- Default model: `text-embedding-3-small`
- Support for custom embedding models via `EmbeddingRequest.Model`
- Proper handling of OpenAI embedding API response format with index-based ordering

#### Provider Updates
- Qwen: Added stub for embedding support (returns "not supported yet" error)
- Azure: Added stub for embedding support (returns "not supported yet" error)

#### Documentation
- Added comprehensive embedding examples in `examples/embedding/main.go`:
  - Single text embedding with OpenAI
  - Multilingual embedding with Cohere
  - Batch embedding with similarity calculation
- Updated README.md with:
  - Embedding API usage examples
  - Cohere configuration section
  - Embedding use cases (semantic search, clustering, classification, recommendation, matchmaking)
- Created CHANGELOG.md for version tracking

#### Examples
- New example: `examples/embedding/main.go` with three complete embedding scenarios
- Includes cosine similarity calculation helper function
- Demonstrates both OpenAI and Cohere embedding APIs

### Changed
- Extended `Client` interface with `CreateEmbedding(ctx, EmbeddingRequest) (*EmbeddingResponse, error)`
- Updated provider list in types.go to include `ProviderCohere`
- README.md now reflects 5 supported providers (was 4)
- Features list updated to highlight embedding generation capability

### Technical Details

#### Embedding Response Format
```go
type EmbeddingResponse struct {
    Embeddings   [][]float64   // Array of embedding vectors
    Model        string        // Model used for generation
    TokensUsed   int           // Total tokens consumed
    ResponseTime time.Duration // API response time
}
```

#### Supported Embedding Models
- **OpenAI**: `text-embedding-3-small`, `text-embedding-3-large`, `text-embedding-ada-002`
- **Cohere**: `embed-multilingual-v3.0`, `embed-english-v3.0`, `embed-multilingual-light-v3.0`

#### Use Cases Enabled
- Mesa agent goal vector generation for matchmaking
- Semantic search in master profiles
- Client-specialist matching based on cosine similarity
- Multilingual content analysis and clustering

### Migration Notes

#### For Existing Users
No breaking changes to existing chat/generation APIs. The embedding functionality is purely additive.

#### For New Embedding Users
```go
// Example: Generate embedding for matchmaking
client, _ := llm.NewClient(llm.Config{
    Provider: llm.ProviderOpenAI,
    APIKey:   os.Getenv("OPENAI_API_KEY"),
    DefaultModel: "text-embedding-3-small",
})

resp, _ := client.CreateEmbedding(ctx, llm.EmbeddingRequest{
    Input: []string{"User's goal and issue description"},
})

goalVector := resp.Embeddings[0] // Use for Weaviate storage
```

### Dependencies
No new external dependencies added. Uses only standard library (`net/http`, `encoding/json`, etc.).

### Testing
- Manual testing completed for OpenAI embeddings
- Manual testing completed for Cohere embeddings
- Batch embedding tested with multiple texts
- Build validation: `go build ./...` passes

---

## [0.1.0] - Initial Release

### Added
- Initial implementation with OpenAI, DeepSeek, Qwen, Azure support
- Unified client interface
- Chat history management
- Request/response types
- Error handling
- Configuration system
- Migration guide from old LLM package
