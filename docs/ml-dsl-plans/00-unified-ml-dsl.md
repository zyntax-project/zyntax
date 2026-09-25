# ZynML: The ML-First Programming Language

**Product Vision**: A unified domain-specific language for ML inference, data processing, and visualization - powered by Zyntax infrastructure with an interactive notebook interface.

## Branding

```
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║     ███████╗██╗   ██╗███╗   ██╗███╗   ███╗██╗                ║
║     ╚══███╔╝╚██╗ ██╔╝████╗  ██║████╗ ████║██║                ║
║       ███╔╝  ╚████╔╝ ██╔██╗ ██║██╔████╔██║██║                ║
║      ███╔╝    ╚██╔╝  ██║╚██╗██║██║╚██╔╝██║██║                ║
║     ███████╗   ██║   ██║ ╚████║██║ ╚═╝ ██║███████╗           ║
║     ╚══════╝   ╚═╝   ╚═╝  ╚═══╝╚═╝     ╚═╝╚══════╝           ║
║                                                               ║
║              The ML-First Programming Language                ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
```

### Product Suite

| Product | Description |
|---------|-------------|
| **ZynML** | The unified ML domain-specific language |
| **ZynBook** | Interactive notebook IDE for ZynML |
| **ZynML CLI** | Command-line tools for ZynML |
| **ZynML Plugins** | Extensible plugin ecosystem |

### File Extensions

| Extension | Purpose |
|-----------|---------|
| `.zynml` | ZynML source files |
| `.zynbook` | ZynBook notebook files |
| `.zynpkg` | ZynML package files |

## Product Overview

### ZynML Language
A single, cohesive DSL that covers all ML inference use cases with consistent syntax, type system, and runtime semantics.

### ZynBook Notebook
A Jupyter-inspired interactive environment for:
- Writing and executing ZynML code
- Rendering outputs (images, charts, tables, audio)
- Documentation with markdown
- Reproducible ML experiments

## Unified Language Design

### Core Principles

1. **Consistent Syntax** - Same patterns across all domains
2. **Type-Safe** - Strong typing with inference
3. **Composable** - Pipelines chain naturally
4. **Renderable** - First-class output visualization
5. **Extensible** - Plugin system for new capabilities

### Unified Syntax

```zynml
// ============================================================================
// ZynML - Unified ML DSL
// ============================================================================

// --- Module and Import System ---
module recommendation_pipeline

import zynml.tensor as T
import zynml.vector as V
import zynml.image as I
import zynml.audio as A
import zynml.text as X
import zynml.stats as S
import zynml.ml as M
import zynml.viz as Z       // Visualization

// --- Type System ---
// Primitives: int, float, bool, string
// Tensors: tensor[shape, dtype]
// Collections: list[T], dict[K, V], set[T]
// Domain types: image, audio, text, embedding, timeseries

// Type aliases
type Embedding = tensor[384, float32]
type Image = tensor[height, width, channels, uint8]
type Spectrogram = tensor[time, freq, float32]

// --- Data Loading ---
// Unified `load` syntax with format inference

// Tensors/Embeddings
let product_embeddings = load("products.npy") as tensor[100000, 384]
let model_weights = load("model.safetensors") as weights

// Images
let photo = load("image.jpg") as image
let photos = load("images/*.jpg") as list[image]

// Audio
let recording = load("speech.wav") as audio
let music = load("song.mp3") as audio

// Text/Documents
let corpus = load("documents.jsonl") as list[text]
let vocab = load("vocab.txt") as vocabulary

// Structured data
let transactions = load("data.parquet") as dataframe
let config = load("config.yaml") as dict

// Streaming sources
let sensor_stream = stream("mqtt://sensors/+") as timeseries
let audio_input = stream("microphone") as audio

// --- Model Loading ---
// Unified model loading with architecture inference

let encoder = model("sentence-bert.onnx") {
    input: text -> tensor[512],
    output: Embedding
}

let classifier = model("mobilenet_v3.safetensors") {
    architecture: "mobilenet_v3",
    input: image[224, 224, 3],
    output: tensor[1000],
    quantization: int8
}

let transcriber = model("whisper-tiny.safetensors") {
    architecture: "whisper",
    input: audio,
    output: text
}

let detector = model("yolov8n.safetensors") {
    architecture: "yolo",
    input: image[640, 640, 3],
    output: list[detection]
}

// --- Pipeline Definition ---
// Unified pipeline syntax with typed inputs/outputs

pipeline image_search(query: text, top_k: int = 10) -> list[{image, score: float}]:
    """Search images by text description."""

    // Encode query
    let query_embedding = encoder.forward(query)
    let query_norm = V.normalize(query_embedding)

    // Search
    let scores = V.cosine_similarity(query_norm, product_embeddings)
    let results = V.topk(scores, k=top_k)

    // Return with images
    return [
        {image: load(image_paths[idx]), score: score}
        for idx, score in results
    ]

pipeline classify_and_explain(img: image) -> {label: string, confidence: float, heatmap: image}:
    """Classify image with visual explanation."""

    // Preprocess
    let tensor = I.resize(img, 224, 224)
                |> I.normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                |> I.to_tensor()

    // Forward with gradient for GradCAM
    let logits, activations = classifier.forward(tensor, return_activations=true)
    let probs = M.softmax(logits)
    let top_class = M.argmax(probs)

    // Generate heatmap
    let heatmap = M.gradcam(activations, top_class)
                 |> I.resize(img.width, img.height)
                 |> I.apply_colormap("jet")

    return {
        label: labels[top_class],
        confidence: probs[top_class],
        heatmap: I.overlay(img, heatmap, alpha=0.5)
    }

pipeline transcribe_audio(audio_input: audio) -> {text: string, segments: list[segment]}:
    """Transcribe audio with timestamps."""

    // Preprocess
    let resampled = A.resample(audio_input, 16000)
    let mel = A.mel_spectrogram(resampled, n_mels=80, n_fft=400)
    let features = A.log(mel + 1e-6) |> A.normalize()

    // Encode
    let encoder_out = transcriber.encoder.forward(features)

    // Decode with timestamps
    let tokens, timestamps = transcriber.decoder.forward(
        encoder_out,
        return_timestamps=true
    )

    // Build segments
    let segments = [
        {text: decode(t), start: ts.start, end: ts.end}
        for t, ts in zip(tokens, timestamps)
    ]

    return {
        text: join(segments.text, " "),
        segments: segments
    }

pipeline detect_anomalies(data: timeseries, window: duration = 1h) -> list[anomaly]:
    """Detect anomalies in time series data."""

    // Compute features over sliding window
    let features = S.rolling(data, window) {
        mean: S.mean(value),
        std: S.std(value),
        trend: S.slope(value),
        range: S.max(value) - S.min(value)
    }

    // Score anomalies
    let scores = M.isolation_forest(features)

    // Find anomalies above threshold
    return [
        {timestamp: t, score: s, features: f}
        for t, s, f in zip(data.timestamps, scores, features)
        if s > 0.7
    ]

// --- Operators and Chaining ---
// Pipe operator for natural data flow

let result = input
    |> preprocess()
    |> model.forward()
    |> postprocess()
    |> format_output()

// Parallel execution with `&`
let (embeddings, metadata) = (
    texts |> encode_batch() &
    texts |> extract_metadata()
)

// Conditional with `?:`
let output = confidence > 0.9 ? high_conf_path() : low_conf_path()

// --- GPU Compute Dispatch (target syntax) ---
// Custom math kernels, run as CPU SIMD or on a GPU chosen by @device.
// Status 2026-09-25: only the in-place elementwise form
// (`for i in r { arr[i] = arr[i] OP scalar }`) and a directly yielded value
// lower today, on CPU SIMD; reduce(+) has no operator slot yet, and @device
// and @async are parsed and ignored. Kernels are a typed subset: a body the
// compiler cannot lower is a compile error. See 09-gpu-compute-system.md.

// Simple element-wise compute
let result = compute(tensor) {
    // SIMD/GPU kernel syntax
    @kernel elementwise
    for i in 0..len:
        out[i] = sin(x[i]) * exp(-x[i] * x[i])
}

// Matrix computation with explicit parallelism
let C = compute(A, B) {
    @kernel matmul
    @workgroup(16, 16)
    for i in 0..M, j in 0..N:
        var sum = 0.0
        for k in 0..K:
            sum += A[i, k] * B[k, j]
        out[i, j] = sum
}

// Reduction kernel
let total = compute(data) {
    @kernel reduce(+)
    for i in 0..len:
        yield data[i] * data[i]  // sum of squares
}

// Fused operations (single kernel launch)
let normalized = compute(x) {
    @kernel fused
    let mean = reduce(+, x) / len(x)
    let variance = reduce(+, (x - mean)^2) / len(x)
    out = (x - mean) / sqrt(variance + 1e-6)
}

// Convolution kernel
let conv_out = compute(input, kernel) {
    @kernel conv2d
    @workgroup(8, 8, 4)  // tile size
    for b in 0..batch, oc in 0..out_channels, oh in 0..out_h, ow in 0..out_w:
        var sum = 0.0
        for ic in 0..in_channels, kh in 0..kernel_h, kw in 0..kernel_w:
            sum += input[b, ic, oh + kh, ow + kw] * kernel[oc, ic, kh, kw]
        out[b, oc, oh, ow] = sum
}

// Attention kernel (fused softmax + matmul)
let attn_out = compute(Q, K, V) {
    @kernel attention
    @workgroup(32, 1)  // per-head parallelism
    for b in 0..batch, h in 0..heads, i in 0..seq_len:
        // Compute attention scores
        var max_score = -inf
        for j in 0..seq_len:
            let score = dot(Q[b, h, i, :], K[b, h, j, :]) / sqrt(head_dim)
            max_score = max(max_score, score)

        // Softmax with numerical stability
        var sum_exp = 0.0
        for j in 0..seq_len:
            let score = dot(Q[b, h, i, :], K[b, h, j, :]) / sqrt(head_dim)
            sum_exp += exp(score - max_score)

        // Weighted sum of values
        for d in 0..head_dim:
            var acc = 0.0
            for j in 0..seq_len:
                let score = dot(Q[b, h, i, :], K[b, h, j, :]) / sqrt(head_dim)
                let weight = exp(score - max_score) / sum_exp
                acc += weight * V[b, h, j, d]
            out[b, h, i, d] = acc
}

// Device selection
let result = compute(data) @device("cuda:0") {
    @kernel custom
    ...
}

// Automatic device selection by size (small kernels on CPU SIMD)
let result = compute(data) @device("auto") {
    @kernel custom
    ...
}

// Async compute (on Zyntax fibers and async)
let future_result = compute(data) @async {
    @kernel expensive_op
    ...
}
let other_work = do_something_else()
let result = await future_result  // Wait for GPU result

// --- Visualization (for ZynBook) ---
// First-class rendering support

// Display image
render photo

// Display with options
render photo {
    title: "Original Image",
    width: 400
}

// Display multiple
render grid([photo, processed, heatmap], cols=3) {
    titles: ["Original", "Processed", "Attention"]
}

// Charts
render chart(data) {
    type: "line",
    x: "timestamp",
    y: "value",
    color: "category"
}

render histogram(scores, bins=50) {
    title: "Score Distribution"
}

render scatter(embeddings_2d) {
    color: labels,
    hover: metadata
}

// Audio player
render audio_player(recording) {
    waveform: true,
    spectrogram: true
}

// Table
render table(results) {
    columns: ["image", "label", "confidence"],
    sortable: true,
    pagesize: 10
}

// Interactive
render interactive(slider(0, 100, default=50)) as threshold {
    let filtered = results |> filter(r => r.score > threshold)
    render table(filtered)
}

// --- Control Flow ---

// For loops
for img in images:
    let result = classify(img)
    render result

// Parallel map
let results = images |> parallel_map(classify, workers=4)

// Conditional
if confidence > 0.9:
    render "High confidence: {label}"
else:
    render "Low confidence, please review"

// Match/Switch
match detection.class:
    case "person": process_person(detection)
    case "vehicle": process_vehicle(detection)
    case _: ignore()

// --- Error Handling ---

try:
    let result = risky_operation()
catch ModelError as e:
    render error("Model failed: {e.message}")
catch IOError as e:
    render warning("IO error, using cached: {e.message}")
    let result = load_cached()

// --- Streaming ---

// Process stream with windowing
stream sensor_data
    |> window(size=100, stride=10)
    |> map(detect_anomalies)
    |> filter(a => a.severity > "warning")
    |> sink(alert_system)

// Real-time rendering
stream microphone
    |> chunk(duration=5s, overlap=1s)
    |> map(transcribe_audio)
    |> render_live(transcript_view)

// --- Caching and Memoization ---

@cache(ttl=1h)
pipeline expensive_embedding(text: text) -> Embedding:
    return encoder.forward(text)

@memoize
fn compute_statistics(data: tensor) -> stats:
    return S.describe(data)

// --- Configuration ---

config {
    device: "cpu",              // target: or "cuda", "metal" (no device config is implemented)
    precision: "float32",       // or "float16", "int8"
    batch_size: 32,
    cache_dir: "~/.zynml/cache",
    log_level: "info"
}

// Per-pipeline config
pipeline fast_inference(x: tensor) -> tensor:
    @config { precision: "int8", batch_size: 64 }
    return model.forward(x)
```

### Type System

```zynml
// === ZynML Type System ===

// Primitive types
int         // 64-bit integer
float       // 64-bit float
bool        // boolean
string      // UTF-8 string

// Numeric variants
int8, int16, int32, int64
uint8, uint16, uint32, uint64
float16, float32, float64

// Tensor types
tensor[shape, dtype]
tensor[384]                    // 1D, inferred dtype
tensor[224, 224, 3, uint8]     // Explicit shape and dtype
tensor[?, 384]                 // Dynamic first dimension

// Domain types (sugar over tensors)
image                          // tensor[H, W, C, uint8]
image[224, 224]                // Fixed size image
audio                          // tensor[samples] + sample_rate
audio[16000]                   // Specific sample rate
text                           // String with tokenization support
embedding                      // tensor[dim, float32]
embedding[384]                 // Specific dimension

// Collection types
list[T]                        // Dynamic array
dict[K, V]                     // Hash map
set[T]                         // Unique set
tuple[T1, T2, ...]             // Fixed tuple

// Structured types
struct Point:
    x: float
    y: float

struct Detection:
    bbox: tuple[float, float, float, float]
    class: string
    confidence: float

struct Segment:
    text: string
    start: float
    end: float

// Optional and Result types
option[T]                      // T or none
result[T, E]                   // Ok(T) or Err(E)

// Function types
fn(T1, T2) -> R                // Function signature
pipeline(T) -> R               // Pipeline signature

// Type aliases
type Embedding384 = tensor[384, float32]
type BBox = tuple[float, float, float, float]
type ClassScores = dict[string, float]

// Generics
fn map[T, R](items: list[T], f: fn(T) -> R) -> list[R]

// Constraints
fn normalize[T: Numeric](data: tensor[?, T]) -> tensor[?, float32]
```

### Standard Library Modules

```zynml
// === ZynML Standard Library ===

// zynml.tensor (T) - Core tensor operations
T.zeros(shape)                 // Create zero tensor
T.ones(shape)                  // Create ones tensor
T.rand(shape)                  // Random tensor
T.load(path)                   // Load from file
T.save(tensor, path)           // Save to file
T.reshape(tensor, shape)       // Reshape
T.transpose(tensor, axes)      // Transpose
T.concat(tensors, axis)        // Concatenate
T.split(tensor, chunks, axis)  // Split
T.slice(tensor, ranges)        // Slice

// zynml.vector (V) - Vector/embedding operations
V.normalize(vec)               // L2 normalize
V.dot(a, b)                    // Dot product
V.cosine_similarity(a, b)      // Cosine similarity
V.euclidean_distance(a, b)     // L2 distance
V.topk(scores, k)              // Top-k indices and values
V.search(query, index, k)      // Index search
V.build_index(vectors, type)   // Build search index

// zynml.image (I) - Image operations
I.load(path)                   // Load image
I.save(img, path)              // Save image
I.resize(img, w, h)            // Resize
I.crop(img, x, y, w, h)        // Crop
I.rotate(img, angle)           // Rotate
I.flip(img, axis)              // Flip
I.normalize(img, mean, std)    // Normalize
I.to_tensor(img)               // Convert to tensor
I.from_tensor(tensor)          // Convert from tensor
I.blend(a, b, alpha)           // Alpha blend
I.overlay(base, overlay, pos)  // Overlay images
I.apply_colormap(gray, map)    // Apply colormap
I.draw_boxes(img, boxes)       // Draw bounding boxes
I.draw_text(img, text, pos)    // Draw text

// zynml.audio (A) - Audio operations
A.load(path)                   // Load audio
A.save(audio, path)            // Save audio
A.resample(audio, rate)        // Resample
A.trim(audio, start, end)      // Trim
A.pad(audio, length)           // Pad to length
A.concat(audios)               // Concatenate
A.mix(audios, weights)         // Mix audio
A.stft(audio, n_fft, hop)      // Short-time Fourier transform
A.istft(spec, hop)             // Inverse STFT
A.mel_spectrogram(audio, ...)  // Mel spectrogram
A.mfcc(audio, n_mfcc)          // MFCC features
A.spectrogram(audio)           // Power spectrogram

// zynml.text (X) - Text operations
X.tokenize(text, vocab)        // Tokenize
X.detokenize(tokens, vocab)    // Detokenize
X.embed(text, model)           // Get embedding
X.encode(text, model)          // Encode to features
X.chunk(text, size, overlap)   // Chunk text
X.split_sentences(text)        // Split into sentences

// zynml.stats (S) - Statistics
S.mean(data)                   // Mean
S.std(data)                    // Standard deviation
S.var(data)                    // Variance
S.min(data), S.max(data)       // Min/Max
S.median(data)                 // Median
S.percentile(data, p)          // Percentile
S.describe(data)               // Summary statistics
S.correlation(a, b)            // Correlation
S.covariance(data)             // Covariance matrix
S.rolling(data, window)        // Rolling window
S.ewma(data, alpha)            // Exponential moving average
S.zscore(data)                 // Z-score normalization

// zynml.ml (M) - ML operations
M.softmax(logits)              // Softmax
M.sigmoid(x)                   // Sigmoid
M.relu(x)                      // ReLU
M.argmax(x)                    // Argmax
M.topk(x, k)                   // Top-k
M.nms(boxes, scores, thresh)   // Non-max suppression
M.quantize(model, method)      // Quantize model
M.gradcam(activations, class)  // GradCAM visualization
M.isolation_forest(data)       // Anomaly scoring
M.kmeans(data, k)              // K-means clustering
M.pca(data, n_components)      // PCA reduction

// zynml.viz (Z) - Visualization
Z.plot(data, type, options)    // Generic plot
Z.line(x, y, ...)              // Line chart
Z.scatter(x, y, ...)           // Scatter plot
Z.bar(categories, values)      // Bar chart
Z.histogram(data, bins)        // Histogram
Z.heatmap(matrix)              // Heatmap
Z.confusion_matrix(pred, true) // Confusion matrix
Z.image_grid(images, cols)     // Image grid
Z.audio_waveform(audio)        // Waveform plot
Z.spectrogram_plot(spec)       // Spectrogram visualization

// zynml.io (IO) - Input/Output
IO.read(path)                  // Read file
IO.write(path, data)           // Write file
IO.glob(pattern)               // Glob files
IO.http_get(url)               // HTTP GET
IO.http_post(url, data)        // HTTP POST
IO.mqtt_subscribe(topic)       // MQTT subscribe
IO.redis_get(key)              // Redis get
IO.redis_set(key, value)       // Redis set
```

## ZynBook: Interactive Notebook

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        ZynBook Frontend                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   Editor    │  │   Output    │  │      Sidebar            │  │
│  │  (Monaco)   │  │  Renderer   │  │  - File Browser         │  │
│  │             │  │  - Images   │  │  - Variables Explorer   │  │
│  │  Cell 1     │  │  - Charts   │  │  - Model Inspector      │  │
│  │  ─────────  │  │  - Audio    │  │  - Pipeline Visualizer  │  │
│  │  Cell 2     │  │  - Tables   │  │  - Performance Profiler │  │
│  │  ─────────  │  │  - Video    │  │                         │  │
│  │  Cell 3     │  │  - 3D       │  │                         │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ WebSocket
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        ZynBook Backend                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   Session   │  │   ZynML     │  │      Plugin Manager     │  │
│  │   Manager   │  │   Runtime   │  │                         │  │
│  │             │  │             │  │  - zrtl_simd            │  │
│  │  - State    │  │  - Parser   │  │  - zrtl_image           │  │
│  │  - History  │  │  - Compiler │  │  - zrtl_audio           │  │
│  │  - Cache    │  │  - JIT      │  │  - zrtl_ml              │  │
│  │             │  │  - Execute  │  │  - zrtl_viz             │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Zyntax Infrastructure                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   ZynPEG    │  │  Cranelift  │  │         ZRTL            │  │
│  │   Parser    │  │   Backend   │  │    Plugin System        │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Notebook File Format (.zynbook)

```json
{
  "version": "1.0",
  "metadata": {
    "title": "Image Classification Demo",
    "author": "User",
    "created": "2024-01-15T10:30:00Z",
    "modified": "2024-01-15T14:22:00Z",
    "tags": ["ml", "classification", "demo"],
    "kernel": "zynml-1.0"
  },
  "config": {
    "device": "cpu",
    "cache_enabled": true
  },
  "cells": [
    {
      "id": "cell-001",
      "type": "markdown",
      "content": "# Image Classification\n\nThis notebook demonstrates image classification using MobileNetV3."
    },
    {
      "id": "cell-002",
      "type": "code",
      "content": "let model = model(\"mobilenet_v3.safetensors\") {\n    architecture: \"mobilenet_v3\",\n    input: image[224, 224, 3],\n    output: tensor[1000]\n}",
      "outputs": [
        {
          "type": "text",
          "content": "Model loaded: mobilenet_v3 (5.4MB, 1000 classes)"
        }
      ],
      "execution_count": 1,
      "execution_time_ms": 234
    },
    {
      "id": "cell-003",
      "type": "code",
      "content": "let img = load(\"cat.jpg\") as image\nrender img { title: \"Input Image\", width: 400 }",
      "outputs": [
        {
          "type": "image",
          "format": "png",
          "data": "base64...",
          "width": 400,
          "height": 300,
          "title": "Input Image"
        }
      ],
      "execution_count": 2,
      "execution_time_ms": 45
    },
    {
      "id": "cell-004",
      "type": "code",
      "content": "let result = classify_and_explain(img)\nrender grid([img, result.heatmap], cols=2) {\n    titles: [\"Original\", \"Attention Map\"]\n}\nrender table([{label: result.label, confidence: result.confidence}])",
      "outputs": [
        {
          "type": "image_grid",
          "images": ["base64...", "base64..."],
          "cols": 2,
          "titles": ["Original", "Attention Map"]
        },
        {
          "type": "table",
          "columns": ["label", "confidence"],
          "data": [["tabby cat", 0.847]]
        }
      ],
      "execution_count": 3,
      "execution_time_ms": 156
    }
  ]
}
```

### Output Types and Renderers

```rust
// Output types supported by ZynBook

enum RenderOutput {
    // Text
    Text { content: String, style: TextStyle },
    Code { content: String, language: String },
    Markdown { content: String },

    // Rich
    Html { content: String },
    Latex { content: String },

    // Media
    Image {
        data: Vec<u8>,
        format: ImageFormat,  // PNG, JPEG, SVG, WebP
        width: Option<u32>,
        height: Option<u32>,
        title: Option<String>,
    },
    ImageGrid {
        images: Vec<Image>,
        cols: u32,
        titles: Vec<String>,
    },
    Audio {
        data: Vec<u8>,
        format: AudioFormat,  // WAV, MP3
        sample_rate: u32,
        show_waveform: bool,
        show_spectrogram: bool,
    },
    Video {
        data: Vec<u8>,
        format: VideoFormat,
        width: u32,
        height: u32,
    },

    // Data
    Table {
        columns: Vec<String>,
        data: Vec<Vec<Value>>,
        sortable: bool,
        filterable: bool,
        page_size: u32,
    },
    DataFrame {
        schema: Schema,
        data: Vec<Row>,
        summary: Option<Summary>,
    },
    Json { content: Value, collapsible: bool },

    // Charts (using Vega-Lite spec)
    Chart {
        spec: VegaLiteSpec,
        width: u32,
        height: u32,
        interactive: bool,
    },

    // 3D (using Three.js)
    Scene3D {
        objects: Vec<Object3D>,
        camera: Camera,
        controls: bool,
    },
    PointCloud {
        points: Vec<[f32; 3]>,
        colors: Option<Vec<[u8; 3]>>,
        size: f32,
    },

    // ML-specific
    ConfusionMatrix {
        labels: Vec<String>,
        matrix: Vec<Vec<u32>>,
    },
    ModelSummary {
        layers: Vec<LayerInfo>,
        total_params: u64,
        trainable_params: u64,
    },
    AttentionMap {
        tokens: Vec<String>,
        weights: Vec<Vec<f32>>,
    },

    // Interactive
    Widget {
        widget_type: WidgetType,
        state: Value,
        on_change: String,  // Code to execute on change
    },

    // Progress
    ProgressBar {
        current: u64,
        total: u64,
        message: String,
    },

    // Errors
    Error { message: String, traceback: Vec<String> },
    Warning { message: String },
}
```

### Frontend Components

```typescript
// ZynBook Frontend Architecture (React/TypeScript)

// Main components
interface ZynBookApp {
  notebook: Notebook;
  kernel: KernelConnection;
  state: AppState;
}

// Cell component
interface CellProps {
  cell: Cell;
  isActive: boolean;
  onExecute: () => void;
  onUpdate: (content: string) => void;
}

// Output renderers
const OutputRenderers = {
  text: TextRenderer,
  image: ImageRenderer,
  image_grid: ImageGridRenderer,
  audio: AudioRenderer,
  table: TableRenderer,
  chart: ChartRenderer,      // Vega-Lite
  scene3d: Scene3DRenderer,  // Three.js
  confusion_matrix: ConfusionMatrixRenderer,
  attention_map: AttentionMapRenderer,
  widget: WidgetRenderer,
  progress: ProgressRenderer,
  error: ErrorRenderer,
};

// Code editor with ZynML support
const ZynMLEditor = {
  language: 'zynml',
  theme: 'zynbook-dark',
  features: [
    'syntax-highlighting',
    'autocomplete',
    'hover-documentation',
    'error-markers',
    'code-folding',
    'bracket-matching',
  ],
};

// Sidebar panels
const SidebarPanels = {
  files: FileBrowserPanel,
  variables: VariablesPanel,    // Inspect current variables
  models: ModelInspectorPanel,  // View loaded models
  pipelines: PipelinePanel,     // Visualize pipeline graph
  profiler: ProfilerPanel,      // Performance metrics
  history: HistoryPanel,        // Execution history
};
```

### Backend Implementation

```rust
// ZynBook Backend (Rust)

pub struct ZynBookBackend {
    sessions: HashMap<SessionId, Session>,
    runtime: ZynMLRuntime,
    plugin_manager: PluginManager,
}

pub struct Session {
    id: SessionId,
    notebook: Notebook,
    state: ExecutionState,
    variables: HashMap<String, Value>,
    models: HashMap<String, ModelHandle>,
    history: Vec<ExecutionRecord>,
}

pub struct ExecutionState {
    current_cell: Option<CellId>,
    is_running: bool,
    interrupt_flag: Arc<AtomicBool>,
}

impl ZynBookBackend {
    /// Execute a cell and return outputs
    pub async fn execute_cell(
        &mut self,
        session_id: SessionId,
        cell_id: CellId,
    ) -> Result<Vec<RenderOutput>> {
        let session = self.sessions.get_mut(&session_id)?;
        let cell = session.notebook.get_cell(cell_id)?;

        // Parse the cell content
        let ast = self.runtime.parse(&cell.content)?;

        // Type check
        let typed_ast = self.runtime.type_check(ast, &session.variables)?;

        // Compile to IR
        let ir = self.runtime.lower(typed_ast)?;

        // JIT compile
        let code = self.runtime.compile(ir)?;

        // Execute with output capture
        let outputs = self.runtime.execute(code, &mut session.state)?;

        // Update variables
        session.variables.extend(outputs.bindings);

        // Record history
        session.history.push(ExecutionRecord {
            cell_id,
            timestamp: Utc::now(),
            duration: outputs.duration,
            success: true,
        });

        Ok(outputs.renders)
    }

    /// Interrupt execution
    pub fn interrupt(&mut self, session_id: SessionId) {
        if let Some(session) = self.sessions.get_mut(&session_id) {
            session.state.interrupt_flag.store(true, Ordering::SeqCst);
        }
    }

    /// Get variable value for inspection
    pub fn inspect_variable(
        &self,
        session_id: SessionId,
        name: &str,
    ) -> Result<VariableInfo> {
        let session = self.sessions.get(&session_id)?;
        let value = session.variables.get(name)?;

        Ok(VariableInfo {
            name: name.to_string(),
            type_: value.type_name(),
            shape: value.shape(),
            preview: value.preview(100),  // First 100 elements
            memory_bytes: value.memory_size(),
        })
    }
}
```

### WebSocket Protocol

```typescript
// Client -> Server messages
type ClientMessage =
  | { type: 'execute_cell', session_id: string, cell_id: string }
  | { type: 'interrupt', session_id: string }
  | { type: 'complete', session_id: string, code: string, cursor: number }
  | { type: 'inspect', session_id: string, name: string }
  | { type: 'kernel_info' }
  | { type: 'widget_update', widget_id: string, value: any };

// Server -> Client messages
type ServerMessage =
  | { type: 'execute_result', cell_id: string, outputs: RenderOutput[] }
  | { type: 'execute_error', cell_id: string, error: Error }
  | { type: 'stream', cell_id: string, name: 'stdout' | 'stderr', text: string }
  | { type: 'display_data', cell_id: string, output: RenderOutput }
  | { type: 'execute_status', status: 'busy' | 'idle' }
  | { type: 'complete_reply', matches: Completion[] }
  | { type: 'inspect_reply', info: VariableInfo }
  | { type: 'kernel_info_reply', info: KernelInfo };
```

## Implementation Roadmap

### Phase 1: Core Language (Weeks 1-4)

1. **Week 1-2: Grammar and Parser**
   - [ ] Define unified ZynML grammar
   - [ ] Implement in ZynPEG
   - [ ] Basic type system

2. **Week 3-4: Type Checker and Compiler**
   - [ ] Type inference
   - [ ] Type checking
   - [ ] IR lowering
   - [ ] Cranelift backend updates

### Phase 2: Standard Library (Weeks 5-8)

1. **Week 5-6: Core Modules**
   - [ ] zynml.tensor
   - [ ] zynml.vector
   - [ ] zynml.stats

2. **Week 7-8: Domain Modules**
   - [ ] zynml.image
   - [ ] zynml.audio
   - [ ] zynml.text
   - [ ] zynml.ml

### Phase 3: ZynBook Backend (Weeks 9-12)

1. **Week 9-10: Runtime**
   - [ ] Session management
   - [ ] Cell execution
   - [ ] Variable tracking

2. **Week 11-12: Output System**
   - [ ] Render output types
   - [ ] Visualization library
   - [ ] WebSocket protocol

### Phase 4: ZynBook Frontend (Weeks 13-16)

1. **Week 13-14: Core UI**
   - [ ] Notebook editor
   - [ ] Cell management
   - [ ] Output renderers (basic)

2. **Week 15-16: Rich Features**
   - [ ] Chart rendering
   - [ ] Audio/Video players
   - [ ] Interactive widgets
   - [ ] Sidebar panels

### Phase 5: Polish and Launch (Weeks 17-20)

1. **Week 17-18: Documentation**
   - [ ] Language reference
   - [ ] Tutorial notebooks
   - [ ] API documentation

2. **Week 19-20: Testing and Release**
   - [ ] End-to-end testing
   - [ ] Performance optimization
   - [ ] Packaging and distribution

## Competitive Positioning

| Feature | Jupyter | Google Colab | ZynBook |
|---------|---------|--------------|---------|
| Language | Python | Python | ZynML (native) |
| ML-first syntax | No | No | **Yes** |
| Native types for ML | No | No | **Yes** |
| Built-in viz | Via libs | Via libs | **Native** |
| JIT Compilation | No | No | **Yes** |
| Offline-first | Yes | No | **Yes** |
| Edge deployment | No | No | **Yes** |
| Pipeline viz | No | No | **Yes** |
| Type safety | No | No | **Yes** |

## Example Notebooks

### 1. Image Search Demo

```zynml
// Cell 1 - Setup
module image_search_demo

let embeddings = load("product_embeddings.npy") as tensor[50000, 384]
let image_paths = load("product_paths.json") as list[string]
let encoder = model("clip-vit-b32.safetensors")

render text("Loaded {len(image_paths)} product images")

// Cell 2 - Search Function
pipeline search(query: text, k: int = 5) -> list[{image, score: float}]:
    let q = encoder.encode_text(query) |> V.normalize()
    let scores = V.cosine_similarity(q, embeddings)
    let top = V.topk(scores, k)

    return [
        {image: load(image_paths[i]) as image, score: s}
        for i, s in top
    ]

// Cell 3 - Interactive Search
render interactive(text_input(placeholder="Search products...")) as query {
    if len(query) > 0:
        let results = search(query, k=8)
        render image_grid([r.image for r in results], cols=4) {
            captions: ["{r.score:.2f}" for r in results]
        }
}
```

### 2. Audio Transcription Demo

```zynml
// Cell 1 - Load Model
let whisper = model("whisper-small.safetensors") {
    architecture: "whisper"
}

// Cell 2 - Load and Visualize Audio
let audio = load("interview.wav") as audio
render audio_player(audio) {
    waveform: true,
    spectrogram: true
}

// Cell 3 - Transcribe
let result = transcribe(audio)
render markdown("## Transcription\n\n" + result.text)

// Cell 4 - Show Segments with Timeline
render table(result.segments) {
    columns: ["start", "end", "text"],
    formatters: {
        start: time_format,
        end: time_format
    }
}

// Cell 5 - Word Cloud
let words = X.tokenize(result.text) |> S.word_frequency()
render chart(words) {
    type: "wordcloud",
    size_field: "count"
}
```

### 3. Anomaly Detection Demo

```zynml
// Cell 1 - Load Data
let data = load("sensor_data.parquet") as dataframe
render chart(data) {
    type: "line",
    x: "timestamp",
    y: "value",
    color: "sensor_id"
}

// Cell 2 - Detect Anomalies
let anomalies = detect_anomalies(data, window=1h)
render text("Found {len(anomalies)} anomalies")

// Cell 3 - Visualize
render chart(data) {
    type: "line",
    x: "timestamp",
    y: "value",
    layers: [
        {type: "point", data: anomalies, color: "red", size: 10}
    ]
}

// Cell 4 - Anomaly Details
render table(anomalies) {
    columns: ["timestamp", "score", "sensor_id"],
    sortable: true
}
```
