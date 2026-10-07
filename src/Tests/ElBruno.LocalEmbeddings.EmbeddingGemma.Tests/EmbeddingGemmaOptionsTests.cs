using ElBruno.LocalEmbeddings.EmbeddingGemma.Options;

namespace ElBruno.LocalEmbeddings.EmbeddingGemma.Tests;

public sealed class EmbeddingGemmaOptionsTests
{
    [Theory]
    [InlineData(128)]
    [InlineData(256)]
    [InlineData(512)]
    [InlineData(768)]
    public void Validate_AcceptsSupportedOutputDimensions(int dimensions)
    {
        var options = new EmbeddingGemmaOptions { OutputDimensions = dimensions };

        options.Validate();
    }

    [Theory]
    [InlineData(0)]
    [InlineData(127)]
    [InlineData(384)]
    [InlineData(1024)]
    public void Validate_RejectsUnsupportedOutputDimensions(int dimensions)
    {
        var options = new EmbeddingGemmaOptions { OutputDimensions = dimensions };

        Assert.Throws<ArgumentOutOfRangeException>(options.Validate);
    }

    [Theory]
    [InlineData(2)]
    [InlineData(8193)]
    public void Validate_RejectsSequenceLengthOutsideModelLimits(int maxLength)
    {
        var options = new EmbeddingGemmaOptions { MaxSequenceLength = maxLength };

        Assert.Throws<ArgumentOutOfRangeException>(options.Validate);
    }

    [Fact]
    public void Validate_RejectsNonPositiveThreadCounts()
    {
        var options = new EmbeddingGemmaOptions { InterOpNumThreads = 0 };

        Assert.Throws<ArgumentOutOfRangeException>(options.Validate);
    }
}
