# EmbeddingGemmaConsoleApp

A minimal semantic-search sample using Google's local, text-only EmbeddingGemma 2 integration.

The app embeds one search query and three titled documents using the recommended query/document prompt formats, then ranks the documents by cosine similarity. It requests 256-dimensional vectors; change `OutputDimensions` to `128`, `512`, or `768` to try the other supported sizes.

## Run

Requires the .NET 10 SDK. On first run, the package downloads the pinned Q4 ONNX model and SentencePiece tokenizer (about 171 MiB total). The files are cached locally for later runs.

```bash
dotnet run --project src/Samples/EmbeddingGemmaConsoleApp
```

For model licensing, offline use, configuration, and troubleshooting, see the [EmbeddingGemma integration guide](../../../docs/embeddinggemma-integration.md).
