using System.Buffers;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Numerics.Tensors;
using OnnxTensor = Microsoft.ML.OnnxRuntime.Tensors.Tensor<float>;

namespace ElBruno.LocalEmbeddings.EmbeddingGemma;

/// <summary>
/// Runs the pinned Q4 EmbeddingGemma 2 ONNX graph for text-only inputs.
/// </summary>
public sealed class EmbeddingGemmaOnnxEmbeddingModel : IDisposable
{
    private const int ModelDimension = 768;
    private const int ModalityFeatureDimension = 512;
    private InferenceSession? _session;
    private bool _disposed;

    /// <summary>
    /// Gets the model's native 768-dimensional output size.
    /// </summary>
    public int EmbeddingDimension => ModelDimension;

    /// <summary>
    /// Loads the ONNX graph and its adjacent external tensor data.
    /// </summary>
    public void Load(
        string modelPath,
        bool useParallelExecution = true,
        int? interOpNumThreads = null,
        int? intraOpNumThreads = null)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentException.ThrowIfNullOrWhiteSpace(modelPath);
        if (!File.Exists(modelPath))
        {
            throw new FileNotFoundException("EmbeddingGemma ONNX model file was not found.", modelPath);
        }

        if (interOpNumThreads is <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(interOpNumThreads), "Thread count must be greater than zero when specified.");
        }

        if (intraOpNumThreads is <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(intraOpNumThreads), "Thread count must be greater than zero when specified.");
        }

        if (_session is not null)
        {
            throw new InvalidOperationException("A model is already loaded. Dispose this instance and create a new one to load a different model.");
        }

        using var sessionOptions = new SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
            ExecutionMode = useParallelExecution ? ExecutionMode.ORT_PARALLEL : ExecutionMode.ORT_SEQUENTIAL
        };
        if (interOpNumThreads.HasValue)
        {
            sessionOptions.InterOpNumThreads = interOpNumThreads.Value;
        }

        if (intraOpNumThreads.HasValue)
        {
            sessionOptions.IntraOpNumThreads = intraOpNumThreads.Value;
        }

        _session = new InferenceSession(modelPath, sessionOptions);
    }

    /// <summary>
    /// Generates embeddings for a padded batch of token IDs. The ONNX graph returns the
    /// sentence embedding directly and expects zero-length modality feature tensors for text-only input.
    /// </summary>
    public float[][] GenerateEmbeddings(long[][] inputIds, long[][] attentionMasks, int outputDimensions, CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(inputIds);
        ArgumentNullException.ThrowIfNull(attentionMasks);
        if (outputDimensions is not (128 or 256 or 512 or 768))
        {
            throw new ArgumentOutOfRangeException(nameof(outputDimensions));
        }

        if (inputIds.Length != attentionMasks.Length)
        {
            throw new ArgumentException("Input IDs and attention masks must contain the same batch size.");
        }

        if (inputIds.Length == 0)
        {
            return [];
        }

        if (_session is null)
        {
            throw new InvalidOperationException("No model is loaded. Call Load() first.");
        }

        var batchSize = inputIds.Length;
        var sequenceLength = inputIds[0].Length;
        if (sequenceLength == 0)
        {
            throw new ArgumentException("Input sequences cannot be empty.", nameof(inputIds));
        }

        for (int i = 0; i < batchSize; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (inputIds[i].Length != sequenceLength || attentionMasks[i].Length != sequenceLength)
            {
                throw new ArgumentException("All input IDs and attention masks must have the same sequence length.");
            }
        }

        var totalSize = checked(batchSize * sequenceLength);
        var flatInputIds = ArrayPool<long>.Shared.Rent(totalSize);
        var flatAttentionMasks = ArrayPool<long>.Shared.Rent(totalSize);
        try
        {
            for (int i = 0; i < batchSize; i++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                Array.Copy(inputIds[i], 0, flatInputIds, i * sequenceLength, sequenceLength);
                Array.Copy(attentionMasks[i], 0, flatAttentionMasks, i * sequenceLength, sequenceLength);
            }

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(
                    "input_ids",
                    new DenseTensor<long>(flatInputIds.AsMemory(0, totalSize), [batchSize, sequenceLength])),
                NamedOnnxValue.CreateFromTensor(
                    "attention_mask",
                    new DenseTensor<long>(flatAttentionMasks.AsMemory(0, totalSize), [batchSize, sequenceLength])),
                NamedOnnxValue.CreateFromTensor(
                    "image_features",
                    new DenseTensor<float>(Memory<float>.Empty, [0, ModalityFeatureDimension])),
                NamedOnnxValue.CreateFromTensor(
                    "video_features",
                    new DenseTensor<float>(Memory<float>.Empty, [0, ModalityFeatureDimension])),
                NamedOnnxValue.CreateFromTensor(
                    "audio_features",
                    new DenseTensor<float>(Memory<float>.Empty, [0, ModalityFeatureDimension]))
            };

            using var results = _session.Run(inputs, ["sentence_embedding"]);
            cancellationToken.ThrowIfCancellationRequested();
            var output = results.First().AsTensor<float>();
            return TruncateAndNormalize(output, batchSize, outputDimensions);
        }
        finally
        {
            ArrayPool<long>.Shared.Return(flatInputIds);
            ArrayPool<long>.Shared.Return(flatAttentionMasks);
        }
    }

    internal static float[][] TruncateAndNormalize(OnnxTensor output, int batchSize, int outputDimensions)
    {
        ArgumentNullException.ThrowIfNull(output);
        if (output.Dimensions.Length != 2 || output.Dimensions[0] != batchSize || output.Dimensions[1] != ModelDimension)
        {
            throw new InvalidOperationException("EmbeddingGemma ONNX output must have shape [batch, 768].");
        }

        var denseTensor = (DenseTensor<float>)output;
        var span = denseTensor.Buffer.Span;
        var embeddings = new float[batchSize][];
        for (int batch = 0; batch < batchSize; batch++)
        {
            var vector = span.Slice(batch * ModelDimension, outputDimensions).ToArray();
            var norm = TensorPrimitives.Norm(vector);
            if (norm > 0)
            {
                TensorPrimitives.Divide(vector, norm, vector);
            }

            embeddings[batch] = vector;
        }

        return embeddings;
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _session?.Dispose();
        _session = null;
        _disposed = true;
    }
}
