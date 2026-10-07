# ElBruno.LocalEmbeddings.EmbeddingGemma

Local text embeddings with Google's EmbeddingGemma 2 model, ONNX Runtime, and `Microsoft.Extensions.AI`.

## Install

```bash
dotnet add package ElBruno.LocalEmbeddings.EmbeddingGemma
```

## Quick start

```csharp
using ElBruno.LocalEmbeddings.EmbeddingGemma;
using ElBruno.LocalEmbeddings.EmbeddingGemma.Options;

await using var generator = await EmbeddingGemmaEmbeddingGenerator.CreateAsync(
    new EmbeddingGemmaOptions { OutputDimensions = 768 });

var query = EmbeddingGemmaPrompts.CreateSearchQuery("Which planet is known as the Red Planet?");
var document = EmbeddingGemmaPrompts.CreateDocument(
    "Mars is known for its reddish appearance.",
    "Mars");
var embeddings = await generator.GenerateAsync([query, document]);
Console.WriteLine($"Vector size: {embeddings[0].Vector.Length}");
```

The first creation downloads the pinned Q4 ONNX graph and external weights (~166 MiB) plus Google's SentencePiece tokenizer (~4.7 MB). Model files are cached under the LocalEmbeddings application-data directory. Downloads are pinned to immutable Hugging Face revisions and verified against SHA-256 hashes. Once cached, inference runs locally without sending text to a service.

This package currently supports **text only**. The underlying model has image, audio, and video encoders, but their preprocessing and APIs are not included here. The ONNX Q4 variant is used by default; output dimensions may be set to 128, 256, 512, or 768. Truncated vectors are re-normalized.

For setup, cache/offline use, limits, and DI examples, see the [EmbeddingGemma integration guide](../../docs/embeddinggemma-integration.md).
