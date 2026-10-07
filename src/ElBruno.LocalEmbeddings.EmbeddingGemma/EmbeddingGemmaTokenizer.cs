using Microsoft.ML.Tokenizers;

namespace ElBruno.LocalEmbeddings.EmbeddingGemma;

/// <summary>
/// Encodes text with the SentencePiece tokenizer distributed with EmbeddingGemma 2.
/// </summary>
public sealed class EmbeddingGemmaTokenizer
{
    private const int BosTokenId = 2;
    private const int EosTokenId = 1;
    private const int PadTokenId = 0;

    private readonly SentencePieceTokenizer _tokenizer;
    private readonly int _maxLength;

    /// <summary>
    /// Creates a tokenizer from Google's SentencePiece model file.
    /// </summary>
    /// <param name="tokenizerPath">Path to <c>tokenizer.model</c> or its containing directory.</param>
    /// <param name="maxLength">Maximum encoded length including BOS and EOS.</param>
    public EmbeddingGemmaTokenizer(string tokenizerPath, int maxLength = 8192)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(tokenizerPath);
        if (maxLength is < 3 or > 8192)
        {
            throw new ArgumentOutOfRangeException(nameof(maxLength), maxLength, "Maximum sequence length must be between 3 and 8192 tokens.");
        }

        var actualPath = Directory.Exists(tokenizerPath)
            ? Path.Combine(tokenizerPath, "tokenizer.model")
            : tokenizerPath;

        if (!File.Exists(actualPath))
        {
            throw new FileNotFoundException("EmbeddingGemma SentencePiece tokenizer.model file was not found.", actualPath);
        }

        using var stream = File.OpenRead(actualPath);
        _tokenizer = SentencePieceTokenizer.Create(stream, addBeginningOfSentence: false, addEndOfSentence: false);
        ValidateSpecialTokenId("<pad>", PadTokenId);
        ValidateSpecialTokenId("<eos>", EosTokenId);
        ValidateSpecialTokenId("<bos>", BosTokenId);
        _maxLength = maxLength;
    }

    /// <summary>
    /// Gets the maximum encoded length, including special tokens.
    /// </summary>
    public int MaxLength => _maxLength;

    /// <summary>
    /// Encodes text, prepending the model's BOS token and appending EOS.
    /// </summary>
    public (long[] InputIds, long[] AttentionMask) Tokenize(string text, int? maxLength = null)
    {
        ArgumentNullException.ThrowIfNull(text);

        var effectiveMaxLength = maxLength ?? _maxLength;
        ValidateMaxLength(effectiveMaxLength);

        var contentTokenLimit = effectiveMaxLength - 2;
        var contentIds = _tokenizer.EncodeToIds(
            text,
            addBeginningOfSentence: false,
            addEndOfSentence: false,
            maxTokenCount: contentTokenLimit,
            out _,
            out _);

        var inputIds = new long[contentIds.Count + 2];
        var attentionMask = new long[inputIds.Length];
        inputIds[0] = BosTokenId;
        attentionMask[0] = 1;

        for (int i = 0; i < contentIds.Count; i++)
        {
            inputIds[i + 1] = contentIds[i];
            attentionMask[i + 1] = 1;
        }

        inputIds[^1] = EosTokenId;
        attentionMask[^1] = 1;
        return (inputIds, attentionMask);
    }

    /// <summary>
    /// Encodes a batch and dynamically pads sequences to the longest item in the batch.
    /// </summary>
    public (long[][] InputIds, long[][] AttentionMasks) TokenizeBatch(
        IEnumerable<string> texts,
        int? maxLength = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(texts);
        var effectiveMaxLength = maxLength ?? _maxLength;
        ValidateMaxLength(effectiveMaxLength);

        IList<string> textList = texts as IList<string> ?? texts.ToList();
        if (textList.Count == 0)
        {
            return ([], []);
        }

        var encoded = new (long[] InputIds, long[] AttentionMask)[textList.Count];
        var batchLength = 0;
        for (int i = 0; i < textList.Count; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            encoded[i] = Tokenize(textList[i], effectiveMaxLength);
            batchLength = Math.Max(batchLength, encoded[i].InputIds.Length);
        }

        var inputIds = new long[textList.Count][];
        var attentionMasks = new long[textList.Count][];
        for (int i = 0; i < textList.Count; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            inputIds[i] = new long[batchLength];
            attentionMasks[i] = new long[batchLength];
            Array.Copy(encoded[i].InputIds, inputIds[i], encoded[i].InputIds.Length);
            Array.Copy(encoded[i].AttentionMask, attentionMasks[i], encoded[i].AttentionMask.Length);
        }

        return (inputIds, attentionMasks);
    }

    /// <summary>
    /// Counts the tokens including BOS and EOS, capped by the configured maximum length.
    /// </summary>
    public int CountTokens(string text, int? maxLength = null)
        => Tokenize(text, maxLength).InputIds.Length;

    private void ValidateSpecialTokenId(string token, int expectedId)
    {
        if (!_tokenizer.Vocabulary.TryGetValue(token, out var actualId) || actualId != expectedId)
        {
            throw new InvalidDataException(
                $"The tokenizer model does not match EmbeddingGemma 2: expected {token} to have ID {expectedId}.");
        }
    }

    private static void ValidateMaxLength(int maxLength)
    {
        if (maxLength is < 3 or > 8192)
        {
            throw new ArgumentOutOfRangeException(nameof(maxLength), maxLength, "Maximum sequence length must be between 3 and 8192 tokens.");
        }
    }
}
