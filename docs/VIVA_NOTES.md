# Viva Notes — AI-Powered Semantic Search in Video

## 30-Second Project Summary

"We built a system that lets you search *inside* videos using plain
English, instead of scrubbing through them manually. It supports two
modes: text search, which finds moments based on what's *said* (using
Whisper for transcription and a Sentence Transformer for meaning-based
matching), and visual search, which finds moments based on what's *shown*
(using CLIP to match your text query against sampled video frames). Both
use FAISS for fast similarity search, and every result links to an exact
timestamp the video player jumps straight to. All models are pretrained —
we didn't train anything; our videos are the searchable corpus."

## 1. Why is this project relevant?

Video is one of the least searchable media formats — text is trivially
searchable (Ctrl+F), but video content is locked inside audio/visual
streams. As more learning and work moves to video (lectures, recorded
meetings, tutorials), the inability to search *inside* them wastes time.

## 2. Why is conventional video search insufficient?

Platforms like YouTube search video *titles, descriptions, tags* — metadata
about the whole video. They don't search *inside* the video's actual
spoken or visual content at a specific timestamp.

## 3. What does "semantic search" mean?

Search based on *meaning*, not exact keyword matching. A semantic search
for "gradient descent" can also match a sentence like "the optimizer
updates weights using the slope of the loss function" — no shared words,
but the same meaning.

## 4. What is an embedding?

A fixed-length list of numbers (a vector) that represents the *meaning* of
a piece of text or an image. Similar meanings produce vectors that are
close together in this numeric space; unrelated meanings produce vectors
far apart. This is what lets a computer "compare meaning" mathematically.

## 5. Why do we use Sentence Transformers?

To convert transcript text (and the user's query) into embeddings that
capture meaning rather than exact wording — this is what enables semantic
(not just keyword) matching.

## 6. What does Whisper do?

Converts spoken audio into written text, while preserving *when* each
piece of speech occurred (timestamps) — this is essential, since we need
to point the user to an exact moment, not just say "this video mentions X."

## 7. Why are timestamps important?

Without them, we could only say "this video is relevant" — not "this video
is relevant AT THIS MOMENT." Timestamps are what let the video player seek
directly to the answer.

## 8. What does CLIP do?

CLIP has two encoders — one for images, one for text — trained together so
that matching image/text pairs produce similar vectors. This lets us embed
a video frame and a text query into the *same* space and directly compare
them, enabling "search a diagram by describing it."

## 9. How does visual search work?

1. Sample frames from the video periodically (not every frame).
2. Embed every sampled frame with CLIP's image encoder.
3. Embed the user's text query with CLIP's text encoder.
4. Find the frames whose embeddings are closest to the query's embedding.
5. Map the matching frame back to its timestamp in the video.

## 10. What does FAISS do?

FAISS is a *library* for fast vector similarity search — given a query
vector, it efficiently finds which stored vectors are closest to it. It has
no understanding of language or images; it only performs numeric similarity
comparisons.

## 11. Model vs Embedding vs Vector Index vs Search Algorithm

- **Model** — the pretrained neural network that converts raw input
  (text or image) into a vector (e.g. Whisper, Sentence Transformer, CLIP).
- **Embedding** — the actual output vector produced by a model for one
  specific input.
- **Vector Index** — the data structure (built by FAISS) that stores many
  embeddings for efficient searching.
- **Search Algorithm** — the method used to find close vectors inside the
  index (we use exact inner-product search via `IndexFlatIP`, which is
  mathematically equivalent to cosine similarity since our vectors are
  normalized).

## 12. Are the models trained on our 6 videos?

**No.** Whisper, the Sentence Transformer, and CLIP are all used exactly as
pretrained — no training or fine-tuning happens in this project. Our 6
videos are the *searchable corpus*: content we search inside, not data used
to adjust any model's internal weights. This is an important distinction —
a training dataset would need thousands/millions of examples; our videos
are just the content being indexed and retrieved.

## 13. Why do we sample frames instead of using every frame?

A 10-minute video at 30fps has ~18,000 frames. Embedding all of them wastes
storage and processing time for near-zero benefit, since visual content
(like a slide) typically stays on screen for several seconds. Sampling one
frame every 5 seconds (configurable) still reliably catches slides/diagrams
while using a fraction of the compute.

## 14. Why 16kHz mono audio?

Whisper's underlying model was trained on 16kHz mono audio. Feeding it that
exact format avoids implicit resampling and keeps transcription accuracy
predictable.

## 15. Text search vs Visual search

| | Text Search | Visual Search |
|---|---|---|
| Based on | what is *said* | what is *shown* |
| Query embedded with | Sentence Transformer | CLIP text encoder |
| Content embedded with | Sentence Transformer (on transcript) | CLIP image encoder (on frames) |
| Finds | spoken explanations | diagrams, slides, visual demonstrations |

## 16. Limitations

- Frame sampling can miss briefly-shown visual content, and (an FFmpeg
  filter quirk) sometimes doesn't capture a frame very close to the video's
  end.
- Text and visual scores aren't combined (no multimodal fusion yet).
- Small, CPU-friendly models trade some accuracy for speed.
- Single-user, local system — no auth, no cloud deployment.

## 17. Future Improvements

- Multimodal fusion (combine text + visual relevance into one ranked list).
- RAG: pass retrieved segments to an LLM to generate a natural-language
  answer instead of just returning raw segments.
- Reranking, query expansion, temporal context windows.
- Formal evaluation: Precision@K, Recall@K, timestamp accuracy, latency.
