namespace ElBruno.LocalEmbeddings.EmbeddingGemma.Options;

/// <summary>
/// Configuration options for local text embedding generation with EmbeddingGemma 2.
/// </summary>
public sealed class EmbeddingGemmaOptions
{
    /// <summary>
    /// Gets or sets the path to a local model directory containing <c>tokenizer.model</c>
    /// and the pinned Q4 ONNX model files.
    /// </summary>
    public string? ModelPath { get; set; }

    /// <summary>
    /// Gets or sets the directory where model files are cached.
    /// If null, the default local application data directory is used.
    /// </summary>
    public string? CacheDirectory { get; set; }

    /// <summary>
    /// Gets or sets the maximum number of tokens, including BOS and EOS, per input.
    /// The default is 8192, which is the model's maximum text context length.
    /// </summary>
    public int MaxSequenceLength { get; set; } = 8192;

    /// <summary>
    /// Gets or sets the output vector size. Supported Matryoshka dimensions are 128, 256,
    /// 512, and 768.
    /// </summary>
    public int OutputDimensions { get; set; } = 768;

    /// <summary>
    /// Gets or sets whether missing model files should be downloaded during creation.
    /// Default is true. Set to false when <see cref="ModelPath"/> points to a local model.
    /// </summary>
    public bool EnsureModelDownloaded { get; set; } = true;

    /// <summary>
    /// Gets or sets whether ONNX Runtime should use parallel execution mode.
    /// </summary>
    public bool UseParallelExecution { get; set; } = true;

    /// <summary>
    /// Gets or sets the number of inter-op threads used by ONNX Runtime.
    /// If null, ONNX Runtime's default is used.
    /// </summary>
    public int? InterOpNumThreads { get; set; }

    /// <summary>
    /// Gets or sets the number of intra-op threads used by ONNX Runtime.
    /// If null, ONNX Runtime's default is used.
    /// </summary>
    public int? IntraOpNumThreads { get; set; }

    internal void Validate()
    {
        if (MaxSequenceLength is < 3 or > 8192)
        {
            throw new ArgumentOutOfRangeException(
                nameof(MaxSequenceLength),
                MaxSequenceLength,
                "Maximum sequence length must be between 3 and 8192 tokens.");
        }

        if (OutputDimensions is not (128 or 256 or 512 or 768))
        {
            throw new ArgumentOutOfRangeException(
                nameof(OutputDimensions),
                OutputDimensions,
                "Output dimensions must be one of 128, 256, 512, or 768.");
        }

        if (InterOpNumThreads is <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(InterOpNumThreads), "Thread count must be greater than zero when specified.");
        }

        if (IntraOpNumThreads is <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(IntraOpNumThreads), "Thread count must be greater than zero when specified.");
        }
    }
}
