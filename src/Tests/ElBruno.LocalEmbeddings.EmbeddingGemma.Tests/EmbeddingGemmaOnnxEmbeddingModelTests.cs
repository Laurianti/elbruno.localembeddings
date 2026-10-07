using Microsoft.ML.OnnxRuntime.Tensors;

namespace ElBruno.LocalEmbeddings.EmbeddingGemma.Tests;

public sealed class EmbeddingGemmaOnnxEmbeddingModelTests
{
    [Fact]
    public void TruncateAndNormalize_TruncatesAndProducesUnitVector()
    {
        var values = new float[768];
        values[0] = 3;
        values[1] = 4;
        var tensor = new DenseTensor<float>(values, [1, 768]);

        var embeddings = EmbeddingGemmaOnnxEmbeddingModel.TruncateAndNormalize(tensor, 1, 128);

        Assert.Single(embeddings);
        Assert.Equal(128, embeddings[0].Length);
        Assert.Equal(0.6f, embeddings[0][0], 5);
        Assert.Equal(0.8f, embeddings[0][1], 5);
        Assert.Equal(1f, MathF.Sqrt(embeddings[0].Sum(value => value * value)), 5);
    }

    [Fact]
    public void TruncateAndNormalize_RejectsUnexpectedGraphOutputShape()
    {
        var tensor = new DenseTensor<float>(new float[512], [1, 512]);

        Assert.Throws<InvalidOperationException>(
            () => EmbeddingGemmaOnnxEmbeddingModel.TruncateAndNormalize(tensor, 1, 256));
    }
}
