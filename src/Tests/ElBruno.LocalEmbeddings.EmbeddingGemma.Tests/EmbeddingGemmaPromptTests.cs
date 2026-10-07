using ElBruno.LocalEmbeddings.EmbeddingGemma.Extensions;
using ElBruno.LocalEmbeddings.EmbeddingGemma.Options;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Options;

namespace ElBruno.LocalEmbeddings.EmbeddingGemma.Tests;

public sealed class EmbeddingGemmaPromptTests
{
    [Fact]
    public void CreateSearchQuery_UsesRecommendedQueryPrefix()
    {
        Assert.Equal(
            "task: search result | query: what is an embedding?",
            EmbeddingGemmaPrompts.CreateSearchQuery("what is an embedding?"));
    }

    [Fact]
    public void CreateDocument_UsesNoneWhenTitleIsMissing()
    {
        Assert.Equal(
            "title: none | text: document text",
            EmbeddingGemmaPrompts.CreateDocument("document text"));
    }

    [Fact]
    public void CreateDocument_UsesProvidedTitle()
    {
        Assert.Equal(
            "title: Guide | text: document text",
            EmbeddingGemmaPrompts.CreateDocument("document text", "Guide"));
    }

    [Fact]
    public void AddEmbeddingGemmaEmbeddings_ConfiguresOptions()
    {
        var services = new ServiceCollection();
        services.AddEmbeddingGemmaEmbeddings(options =>
        {
            options.OutputDimensions = 256;
            options.MaxSequenceLength = 1024;
            options.EnsureModelDownloaded = false;
        });

        using var provider = services.BuildServiceProvider();
        var options = provider.GetRequiredService<IOptions<EmbeddingGemmaOptions>>().Value;

        Assert.Equal(256, options.OutputDimensions);
        Assert.Equal(1024, options.MaxSequenceLength);
        Assert.False(options.EnsureModelDownloaded);
    }
}
