using ElBruno.LocalEmbeddings.EmbeddingGemma.Options;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;
using Microsoft.Extensions.Options;

namespace ElBruno.LocalEmbeddings.EmbeddingGemma.Extensions;

/// <summary>
/// Dependency injection extensions for EmbeddingGemma 2 text embeddings.
/// </summary>
public static class ServiceCollectionExtensions
{
    /// <summary>
    /// Default configuration section name for EmbeddingGemma options.
    /// </summary>
    public const string DefaultConfigurationSectionName = "EmbeddingGemma";

    /// <summary>
    /// Registers <see cref="EmbeddingGemmaEmbeddingGenerator"/> with an optional options callback.
    /// </summary>
    public static IServiceCollection AddEmbeddingGemmaEmbeddings(
        this IServiceCollection services,
        Action<EmbeddingGemmaOptions>? configure = null)
    {
        ArgumentNullException.ThrowIfNull(services);
        var optionsBuilder = services.AddOptions<EmbeddingGemmaOptions>();
        if (configure is not null)
        {
            optionsBuilder.Configure(configure);
        }

        return AddGenerator(services);
    }

    /// <summary>
    /// Registers <see cref="EmbeddingGemmaEmbeddingGenerator"/> with a preconfigured options instance.
    /// </summary>
    public static IServiceCollection AddEmbeddingGemmaEmbeddings(
        this IServiceCollection services,
        EmbeddingGemmaOptions options)
    {
        ArgumentNullException.ThrowIfNull(services);
        ArgumentNullException.ThrowIfNull(options);
        services.AddOptions<EmbeddingGemmaOptions>().Configure(configured =>
        {
            configured.ModelPath = options.ModelPath;
            configured.CacheDirectory = options.CacheDirectory;
            configured.MaxSequenceLength = options.MaxSequenceLength;
            configured.OutputDimensions = options.OutputDimensions;
            configured.EnsureModelDownloaded = options.EnsureModelDownloaded;
            configured.UseParallelExecution = options.UseParallelExecution;
            configured.InterOpNumThreads = options.InterOpNumThreads;
            configured.IntraOpNumThreads = options.IntraOpNumThreads;
        });
        return AddGenerator(services);
    }

    /// <summary>
    /// Registers <see cref="EmbeddingGemmaEmbeddingGenerator"/> using configuration values
    /// from the <c>EmbeddingGemma</c> section.
    /// </summary>
    public static IServiceCollection AddEmbeddingGemmaEmbeddings(
        this IServiceCollection services,
        IConfiguration configuration)
    {
        ArgumentNullException.ThrowIfNull(services);
        ArgumentNullException.ThrowIfNull(configuration);
        services.AddOptions<EmbeddingGemmaOptions>().Bind(configuration.GetSection(DefaultConfigurationSectionName));
        return AddGenerator(services);
    }

    private static IServiceCollection AddGenerator(IServiceCollection services)
    {
        services.TryAddSingleton<IEmbeddingGenerator<string, Embedding<float>>>(serviceProvider =>
        {
            var options = serviceProvider.GetRequiredService<IOptions<EmbeddingGemmaOptions>>().Value;
            return EmbeddingGemmaEmbeddingGenerator.CreateAsync(options).GetAwaiter().GetResult();
        });
        return services;
    }
}
