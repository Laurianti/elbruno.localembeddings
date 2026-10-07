using ElBruno.LocalEmbeddings.EmbeddingGemma.Options;
using Microsoft.Extensions.AI;

namespace ElBruno.LocalEmbeddings.EmbeddingGemma;

/// <summary>
/// Generates local text embeddings using Google's EmbeddingGemma 2 model.
/// </summary>
/// <remarks>
/// Text input only is supported by this package. Image, audio, and video inputs are not
/// currently processed. Inputs are not task-prefixed automatically; use
/// <see cref="EmbeddingGemmaPrompts"/> for the model's recommended retrieval formats.
/// </remarks>
public sealed class EmbeddingGemmaEmbeddingGenerator : IEmbeddingGenerator<string, Embedding<float>>, IAsyncDisposable
{
    private static readonly HttpClient SharedDownloadHttpClient = new(new SocketsHttpHandler
    {
        PooledConnectionLifetime = TimeSpan.FromMinutes(2)
    });

    private readonly EmbeddingGemmaOnnxEmbeddingModel _model;
    private readonly EmbeddingGemmaTokenizer _tokenizer;
    private readonly EmbeddingGeneratorMetadata _metadata;
    private readonly int _outputDimensions;
    private bool _disposed;

    private EmbeddingGemmaEmbeddingGenerator(string modelDirectory, EmbeddingGemmaOptions options)
    {
        var modelPath = EmbeddingGemmaModelDownloader.GetModelPath(modelDirectory);
        _model = new EmbeddingGemmaOnnxEmbeddingModel();
        try
        {
            _model.Load(
                modelPath,
                options.UseParallelExecution,
                options.InterOpNumThreads,
                options.IntraOpNumThreads);
            _tokenizer = new EmbeddingGemmaTokenizer(modelDirectory, options.MaxSequenceLength);
            _outputDimensions = options.OutputDimensions;
            _metadata = new EmbeddingGeneratorMetadata(
                providerName: "ElBruno.LocalEmbeddings.EmbeddingGemma",
                providerUri: new Uri("https://github.com/elbruno/elbruno.localembeddings"),
                defaultModelId: "google/embeddinggemma-2",
                defaultModelDimensions: options.OutputDimensions);
        }
        catch
        {
            _model.Dispose();
            throw;
        }
    }

    /// <inheritdoc />
    public EmbeddingGeneratorMetadata Metadata => _metadata;

    /// <summary>
    /// Creates a generator using default options and downloads the pinned model if needed.
    /// </summary>
    public static Task<EmbeddingGemmaEmbeddingGenerator> CreateAsync(CancellationToken cancellationToken = default)
        => CreateAsync(new EmbeddingGemmaOptions(), cancellationToken);

    /// <summary>
    /// Creates a generator asynchronously using the specified options.
    /// </summary>
    public static async Task<EmbeddingGemmaEmbeddingGenerator> CreateAsync(
        EmbeddingGemmaOptions options,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(options);
        options.Validate();

        var modelDirectory = await ResolveModelDirectoryAsync(options, cancellationToken).ConfigureAwait(false);
        return new EmbeddingGemmaEmbeddingGenerator(modelDirectory, options);
    }

    /// <inheritdoc />
    public Task<GeneratedEmbeddings<Embedding<float>>> GenerateAsync(
        IEnumerable<string> values,
        EmbeddingGenerationOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(values);
        cancellationToken.ThrowIfCancellationRequested();

        IList<string> valueList = values as IList<string> ?? values.ToList();
        if (valueList.Count == 0)
        {
            return Task.FromResult(new GeneratedEmbeddings<Embedding<float>>());
        }

        var (inputIds, attentionMasks) = _tokenizer.TokenizeBatch(valueList, cancellationToken: cancellationToken);
        var vectors = _model.GenerateEmbeddings(inputIds, attentionMasks, _outputDimensions, cancellationToken);
        return Task.FromResult(new GeneratedEmbeddings<Embedding<float>>(
            vectors.Select(static vector => new Embedding<float>(vector))));
    }

    /// <inheritdoc />
    public TService? GetService<TService>(object? key = null) where TService : class
    {
        if (typeof(TService) == typeof(EmbeddingGeneratorMetadata))
        {
            return Metadata as TService;
        }

        return typeof(TService) == typeof(IEmbeddingGenerator<string, Embedding<float>>)
            ? (TService)(object)this
            : null;
    }

    /// <inheritdoc />
    public object? GetService(Type serviceType, object? key = null)
    {
        ArgumentNullException.ThrowIfNull(serviceType);
        if (serviceType == typeof(EmbeddingGeneratorMetadata))
        {
            return Metadata;
        }

        return serviceType == typeof(IEmbeddingGenerator<string, Embedding<float>>) ? this : null;
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _model.Dispose();
        _disposed = true;
    }

    /// <inheritdoc />
    public ValueTask DisposeAsync()
    {
        Dispose();
        return ValueTask.CompletedTask;
    }

    private static async Task<string> ResolveModelDirectoryAsync(
        EmbeddingGemmaOptions options,
        CancellationToken cancellationToken)
    {
        if (!string.IsNullOrWhiteSpace(options.ModelPath))
        {
            var modelPath = Path.GetFullPath(options.ModelPath);
            if (!File.Exists(EmbeddingGemmaModelDownloader.GetModelPath(modelPath)) ||
                !File.Exists(Path.Combine(
                    Path.GetDirectoryName(EmbeddingGemmaModelDownloader.GetModelPath(modelPath))!,
                    Path.GetFileName(EmbeddingGemmaModelDownloader.ModelDataRelativePath))) ||
                !File.Exists(Path.Combine(modelPath, EmbeddingGemmaModelDownloader.TokenizerFileName)))
            {
                throw new DirectoryNotFoundException(
                    $"The configured model directory '{modelPath}' must contain the pinned ONNX graph, its external data, and tokenizer.model.");
            }

            return modelPath;
        }

        var downloader = new EmbeddingGemmaModelDownloader(SharedDownloadHttpClient, options.CacheDirectory);
        if (options.EnsureModelDownloaded)
        {
            return await downloader.EnsureModelAsync(cancellationToken: cancellationToken).ConfigureAwait(false);
        }

        var cacheDirectory = options.CacheDirectory ?? Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "ElBruno", "LocalEmbeddings", "models");
        var expectedDirectory = Path.GetFullPath(Path.Combine(cacheDirectory, "embeddinggemma-2"));
        if (!EmbeddingGemmaModelDownloader.FilesAreValid(expectedDirectory))
        {
            throw new DirectoryNotFoundException(
                $"EmbeddingGemma model files were not found or failed integrity checks in '{expectedDirectory}'. " +
                "Set EnsureModelDownloaded to true or provide ModelPath.");
        }

        return expectedDirectory;
    }
}
