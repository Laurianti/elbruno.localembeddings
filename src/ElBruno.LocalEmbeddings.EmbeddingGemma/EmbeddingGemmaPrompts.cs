namespace ElBruno.LocalEmbeddings.EmbeddingGemma;

/// <summary>
/// Helpers for formatting text with EmbeddingGemma 2 task instructions.
/// </summary>
public static class EmbeddingGemmaPrompts
{
    /// <summary>
    /// Formats a search query using the recommended asymmetric retrieval instruction.
    /// </summary>
    public static string CreateSearchQuery(string query)
    {
        ArgumentNullException.ThrowIfNull(query);
        return $"task: search result | query: {query}";
    }

    /// <summary>
    /// Formats a document using the recommended asymmetric retrieval structure.
    /// </summary>
    /// <param name="text">Document text.</param>
    /// <param name="title">Optional document title; omitted titles are represented as "none".</param>
    public static string CreateDocument(string text, string? title = null)
    {
        ArgumentNullException.ThrowIfNull(text);
        var resolvedTitle = string.IsNullOrWhiteSpace(title) ? "none" : title;
        return $"title: {resolvedTitle} | text: {text}";
    }
}
