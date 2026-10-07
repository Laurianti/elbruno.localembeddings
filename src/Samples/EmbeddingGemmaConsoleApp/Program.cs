using ElBruno.LocalEmbeddings.EmbeddingGemma;
using ElBruno.LocalEmbeddings.EmbeddingGemma.Options;
using ElBruno.LocalEmbeddings.Extensions;

Console.WriteLine("EmbeddingGemma 2 local text-embedding sample");
Console.WriteLine("The first run downloads approximately 171 MiB of model and tokenizer files.");
Console.WriteLine();

var options = new EmbeddingGemmaOptions
{
    OutputDimensions = 256
};

await using var generator = await EmbeddingGemmaEmbeddingGenerator.CreateAsync(options);

var documents = new (string Title, string Text)[]
{
    ("Mars", "Mars is the red planet and the fourth planet from the Sun."),
    ("Jupiter", "Jupiter is the largest planet in our solar system."),
    ("Venus", "Venus has a thick atmosphere and is the hottest planet.")
};
var query = EmbeddingGemmaPrompts.CreateSearchQuery("Which planet is known as the Red Planet?");
var inputs = new[] { query }
    .Concat(documents.Select(document =>
        EmbeddingGemmaPrompts.CreateDocument(document.Text, document.Title)));
var embeddings = await generator.GenerateAsync(inputs);

Console.WriteLine($"Model: {generator.Metadata.DefaultModelId}");
Console.WriteLine($"Dimensions: {embeddings[0].Vector.Length}");
Console.WriteLine("Search results:");

var results = documents
    .Select((document, index) => new
    {
        document.Title,
        document.Text,
        Score = embeddings[0].CosineSimilarity(embeddings[index + 1])
    })
    .OrderByDescending(result => result.Score);

foreach (var result in results)
{
    Console.WriteLine($"  {result.Score:F3}  {result.Title}: {result.Text}");
}
