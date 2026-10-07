using ElBruno.LocalEmbeddings.EmbeddingGemma.Options;

namespace ElBruno.LocalEmbeddings.EmbeddingGemma.Tests;

[Trait("Category", "Integration")]
public sealed class EmbeddingGemmaIntegrationTests
{
    [SkippableFact]
    public async Task GenerateAsync_ProducesNormalizedTextEmbedding()
    {
        var modelDirectory = FindModelDirectory();
        var downloadModel = string.Equals(
            Environment.GetEnvironmentVariable("EMBEDDINGGEMMA_DOWNLOAD_INTEGRATION"),
            "1",
            StringComparison.Ordinal);
        Skip.If(
            modelDirectory is null && !downloadModel,
            "EmbeddingGemma Q4 model files are not available locally. Set EMBEDDINGGEMMA_DOWNLOAD_INTEGRATION=1 to download them.");

        await using var generator = await EmbeddingGemmaEmbeddingGenerator.CreateAsync(new EmbeddingGemmaOptions
        {
            ModelPath = modelDirectory,
            EnsureModelDownloaded = downloadModel,
            OutputDimensions = 256
        });

        var embeddings = await generator.GenerateAsync(
        [
            EmbeddingGemmaPrompts.CreateSearchQuery("Which planet is known as the Red Planet?"),
            EmbeddingGemmaPrompts.CreateDocument("Mars is known for its reddish appearance.", "Mars")
        ]);

        Assert.Equal(2, embeddings.Count);
        foreach (var embedding in embeddings)
        {
            var vector = embedding.Vector.ToArray();
            Assert.Equal(256, vector.Length);
            Assert.All(vector, value => Assert.True(float.IsFinite(value)));
            Assert.Equal(1f, MathF.Sqrt(vector.Sum(value => value * value)), 4);
        }
    }

    private static string? FindModelDirectory()
    {
        var configuredPath = Environment.GetEnvironmentVariable("EMBEDDINGGEMMA_MODEL_PATH");
        if (!string.IsNullOrWhiteSpace(configuredPath) &&
            File.Exists(EmbeddingGemmaModelDownloader.GetModelPath(configuredPath)) &&
            File.Exists(Path.Combine(configuredPath, EmbeddingGemmaModelDownloader.TokenizerFileName)))
        {
            return configuredPath;
        }

        var appData = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
        var cacheDirectory = Path.Combine(appData, "ElBruno", "LocalEmbeddings", "models", "embeddinggemma-2");
        return File.Exists(EmbeddingGemmaModelDownloader.GetModelPath(cacheDirectory)) &&
               File.Exists(Path.Combine(cacheDirectory, EmbeddingGemmaModelDownloader.TokenizerFileName))
            ? cacheDirectory
            : null;
    }
}
