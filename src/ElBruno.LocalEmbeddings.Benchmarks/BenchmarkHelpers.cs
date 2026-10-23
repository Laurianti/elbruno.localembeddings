namespace ElBruno.LocalEmbeddings.Benchmarks;

internal static class BenchmarkHelpers
{
    private const string ModelCacheSubDir = "sentence-transformers_all-MiniLM-L6-v2";
    private const string HarrierModelCacheSubDir = "onnx-community_harrier-oss-v1-270m-ONNX";

    /// <summary>
    /// Returns the local model directory if it exists on disk, or null when no model is cached.
    /// </summary>
    internal static string? TryResolveModelDirectory()
    {
        // The cache directory of the library first, then the one without ElBruno that the benchmarks used before.
        foreach (var cacheDir in new[] { ResolveBaseCacheDir(), ResolveLegacyCacheDir() })
        {
            var modelDir = Path.Combine(cacheDir, ModelCacheSubDir);
            if (Directory.Exists(modelDir))
            {
                return modelDir;
            }
        }

        return null;
    }

    /// <summary>
    /// Returns the local Harrier model directory if it exists on disk, or null when no model is cached.
    /// </summary>
    internal static string? TryResolveHarrierModelDirectory()
    {
        var cacheDir = ResolveHarrierCacheDir();
        var modelDir = Path.Combine(cacheDir, HarrierModelCacheSubDir);

        if (!Directory.Exists(modelDir))
            return null;

        // Verify tokenizer.json exists (required for Harrier)
        return File.Exists(Path.Combine(modelDir, "tokenizer.json")) ? modelDir : null;
    }

    private static string ResolveBaseCacheDir() =>
        Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "ElBruno", "LocalEmbeddings", "models");

    private static string ResolveLegacyCacheDir() =>
        OperatingSystem.IsWindows()
            ? Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                "LocalEmbeddings", "models")
            : Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
                ".local", "share", "LocalEmbeddings", "models");

    private static string ResolveHarrierCacheDir() =>
        Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "ElBruno", "LocalEmbeddings", "models");
}
