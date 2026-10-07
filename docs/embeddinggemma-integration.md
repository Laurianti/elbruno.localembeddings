# EmbeddingGemma 2 Integration

`ElBruno.LocalEmbeddings.EmbeddingGemma` provides local **text-only** embeddings from Google's EmbeddingGemma 2 model through `IEmbeddingGenerator<string, Embedding<float>>`.

The base model is multimodal, but this integration only handles strings. Image, audio, and video inputs need separate preprocessors and are not supported by this package.

## Install and use

```bash
dotnet add package ElBruno.LocalEmbeddings.EmbeddingGemma
```

```csharp
using ElBruno.LocalEmbeddings.EmbeddingGemma;
using ElBruno.LocalEmbeddings.EmbeddingGemma.Options;

await using var generator = await EmbeddingGemmaEmbeddingGenerator.CreateAsync(
    new EmbeddingGemmaOptions
    {
        OutputDimensions = 768,
        MaxSequenceLength = 8192
    });

var query = EmbeddingGemmaPrompts.CreateSearchQuery("Which planet is known as the Red Planet?");
var passage = EmbeddingGemmaPrompts.CreateDocument(
    "Mars is known for its reddish appearance.",
    title: "Mars");
var vectors = await generator.GenerateAsync([query, passage]);
```

For a runnable semantic-search example that compares a query against several documents, run:

```bash
dotnet run --project src/Samples/EmbeddingGemmaConsoleApp
```

The sample requests 256-dimensional vectors and ranks its documents by cosine similarity. See [EmbeddingGemmaConsoleApp](../src/Samples/EmbeddingGemmaConsoleApp/README.md) for prerequisites and customization.

Task prefixes are not added automatically. The helper methods use Google's recommended retrieval formats:

- Query: `task: search result | query: {query}`
- Document: `title: {title or none} | text: {content}`

For symmetric tasks, use the same task prefix for every input (for example, `task: sentence similarity | query: {text}`). Avoid comparing vectors created for incompatible tasks or different output dimensions.

## Configuration

| Option | Default | Description |
|---|---:|---|
| `ModelPath` | `null` | Local model directory; it must contain `tokenizer.model` and `onnx/model_q4.onnx` with `onnx/model_q4.onnx_data`. |
| `CacheDirectory` | Platform application-data cache | Parent directory for downloaded model files. |
| `MaxSequenceLength` | `8192` | Maximum tokens including BOS/EOS; valid range is 3–8192. |
| `OutputDimensions` | `768` | Matryoshka vector size: 128, 256, 512, or 768. Truncated outputs are L2-normalized. |
| `EnsureModelDownloaded` | `true` | Download files if missing. Set false to require a valid existing cache or `ModelPath`. |
| `UseParallelExecution` | `true` | ONNX Runtime execution mode. |
| `InterOpNumThreads` | `null` | Optional ONNX Runtime inter-op thread count. |
| `IntraOpNumThreads` | `null` | Optional ONNX Runtime intra-op thread count. |

Dependency injection is available from `ElBruno.LocalEmbeddings.EmbeddingGemma.Extensions`:

```csharp
services.AddEmbeddingGemmaEmbeddings(options =>
{
    options.OutputDimensions = 256;
    options.CacheDirectory = @"D:\model-cache";
});
```

The DI factory initializes the model synchronously on first resolution. In async-first applications, create the generator with `CreateAsync` and register the instance as a singleton instead.

## Model files, revision, and licensing

The package downloads three files:

- `onnx-community/embeddinggemma-2-ONNX`, revision `daa72c51243991dfcaf9f9137d2c573d8f7790c0`: Q4 text graph and its external weight data.
- `google/embeddinggemma-2`, revision `914f7f89142e33e77833254d9c9b90c3cef7303b`: SentencePiece `tokenizer.model`.

Each file is checked against its pinned SHA-256 digest before it is used. The Q4 ONNX external weights are approximately 166 MiB (174 MB), the graph is below 1 MiB, and the tokenizer model is approximately 4.7 MB. The export includes Microsoft ONNX Runtime custom quantization operators and is tested with ONNX Runtime 1.24.4.

EmbeddingGemma 2 is published under Google's Apache 2.0 Gemma 4 license terms. The exporter repository indicates Apache 2.0 as well. Review the source model's [license and terms](https://ai.google.dev/gemma/docs/gemma_4_license) for your distribution and use case.

## Offline and local model use

After a successful download, set `EnsureModelDownloaded = false` to avoid network access:

```csharp
await using var generator = await EmbeddingGemmaEmbeddingGenerator.CreateAsync(
    new EmbeddingGemmaOptions { EnsureModelDownloaded = false });
```

Alternatively, point `ModelPath` at a directory containing the pinned model layout. Local paths are loaded directly; they are not downloaded or checksum-verified automatically.

## Output dimensions and context

Google supports 768-dimensional output and Matryoshka truncation to 512, 256, or 128 dimensions. This integration takes the leading dimensions and normalizes the truncated vector. Use the same dimension for every vector in an index.

Input length is capped at 8,192 tokens, including BOS and EOS. Longer inputs are truncated by the tokenizer. Chunk longer documents before embedding when complete-document retrieval matters.

The model-backed integration test is opt-in because it requires the large ONNX artifacts. With .NET 8 selected, set `EMBEDDINGGEMMA_DOWNLOAD_INTEGRATION=1` before running the EmbeddingGemma test project to download and verify the pinned model, then execute an end-to-end embedding check. Without this variable, tests use a model already present in `EMBEDDINGGEMMA_MODEL_PATH` or the default cache and skip the integration check when none is available.

## Troubleshooting

- **Model download/hash error:** Confirm access to Hugging Face and retry; downloaded files are pinned and verified, so a mismatched artifact is rejected.
- **Missing external data:** Keep `model_q4.onnx_data` beside the ONNX graph under the `onnx` directory.
- **Local path error:** Verify `tokenizer.model`, `onnx/model_q4.onnx`, and `onnx/model_q4.onnx_data` exist.
- **Memory pressure or slow startup:** The Q4 graph and external weights total about 166 MiB; including the tokenizer, the first download is about 171 MiB. Expect higher temporary memory requirements while ONNX Runtime loads the graph. Reduce batch size and sequence length for constrained machines.
- **Different search quality:** Use matching query/document task formats and keep index/query dimensions the same.

## References

- [Google EmbeddingGemma 2 documentation](https://ai.google.dev/gemma/docs/embeddinggemma)
- [Google model repository](https://huggingface.co/google/embeddinggemma-2)
- [Pinned ONNX export](https://huggingface.co/onnx-community/embeddinggemma-2-ONNX/tree/daa72c51243991dfcaf9f9137d2c573d8f7790c0)
- [Hugging Face SentencePiece tokenizer API](https://huggingface.co/google/embeddinggemma-2/blob/914f7f89142e33e77833254d9c9b90c3cef7303b/tokenizer.model)
