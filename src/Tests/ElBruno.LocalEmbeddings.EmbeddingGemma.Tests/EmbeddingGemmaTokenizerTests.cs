namespace ElBruno.LocalEmbeddings.EmbeddingGemma.Tests;

public sealed class EmbeddingGemmaTokenizerTests
{
    [Fact]
    public void Constructor_RejectsMaxLengthOutsideModelLimits()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new EmbeddingGemmaTokenizer("missing", maxLength: 2));
        Assert.Throws<ArgumentOutOfRangeException>(() => new EmbeddingGemmaTokenizer("missing", maxLength: 8193));
    }

    [Fact]
    public void Constructor_RejectsMissingModel()
    {
        Assert.Throws<FileNotFoundException>(() => new EmbeddingGemmaTokenizer("missing-tokenizer.model"));
    }

    [SkippableFact]
    [Trait("Category", "Integration")]
    public void Tokenize_UsesReferenceGemmaTokenIds()
    {
        var modelDirectory = FindModelDirectory();
        Skip.If(modelDirectory is null, "EmbeddingGemma tokenizer.model is not available locally.");

        var tokenizer = new EmbeddingGemmaTokenizer(modelDirectory!, maxLength: 128);
        var (inputIds, attentionMask) = tokenizer.Tokenize("hello world");

        Assert.Equal(new long[] { 2, 23391, 1902, 1 }, inputIds);
        Assert.Equal(new long[] { 1, 1, 1, 1 }, attentionMask);
    }

    private static string? FindModelDirectory()
    {
        var configuredPath = Environment.GetEnvironmentVariable("EMBEDDINGGEMMA_MODEL_PATH");
        if (!string.IsNullOrWhiteSpace(configuredPath) &&
            File.Exists(Path.Combine(configuredPath, "tokenizer.model")))
        {
            return configuredPath;
        }

        var appData = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
        var cacheDirectory = Path.Combine(appData, "ElBruno", "LocalEmbeddings", "models", "embeddinggemma-2");
        return File.Exists(Path.Combine(cacheDirectory, "tokenizer.model")) ? cacheDirectory : null;
    }
}
