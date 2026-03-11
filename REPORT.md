# Code-Mixed Pedagogical Flow Extractor — Report

## Task

Build a pipeline that takes educational videos in code-mixed Indian languages and outputs a structured prerequisite map of the concepts taught — showing which concepts depend on which others, in what order.

---

## Video Sources

| ID | URL | Topic | Language Mix |
|----|-----|-------|-------------|
| video_1 | https://www.youtube.com/watch?v=DM0RUrheUQ0 | Rotational Motion (JEE) | Hindi-English |
| video_2 | https://www.youtube.com/watch?v=nzaHG0dme4g | Linked Lists (DSA) | Hindi-English |
| video_3 | https://www.youtube.com/watch?v=tWPa-rZiGM8 | System Calls (OS) | Hindi-English |
| video_4 | https://www.youtube.com/watch?v=VJs99aeqUe4 | Python Strings | Telugu-English |
| video_5 | https://www.youtube.com/watch?v=_99lZBc-lFY | Functions in C | Tamil-English |

Videos were selected to cover multiple domains (physics, CS, programming) and three different language combinations (Hindi-English, Telugu-English, Tamil-English), to test the pipeline's generalization.

---

## Pipeline Architecture

```
YouTube URL
    -> yt-dlp + FFmpeg           -> data/audio/video_N.mp3
    -> Whisper (local, medium)   -> data/transcripts/video_N_transcript.json
    -> Groq LLM (translation)    -> data/transcripts/video_N_translated.json
    -> spaCy + KeyBERT           -> data/outputs/json/video_N_concepts.json
    -> NLI + positional          -> data/outputs/json/video_N_prerequisites.json
    -> networkx + pyvis           -> data/outputs/graphs/video_N_graph.html
```

### Step 1: Audio Download (`pipeline/ingestion/downloader.py`)

Uses `yt-dlp` to download the best available audio stream from YouTube, then converts it to MP3 at 192kbps using FFmpeg. We download audio-only to avoid wasting disk space on video frames that Whisper doesn't use.

### Step 2: Transcription (`pipeline/ingestion/transcriber.py`)

Uses OpenAI Whisper (medium model, 1.4GB, runs locally on CPU) with three important settings:
- `language=None` — auto-detect per segment, needed because code-mixed audio switches languages mid-sentence
- `condition_on_previous_text=False` — prevents hallucination feedback loops on CPU, where Whisper would otherwise repeat the same sentence for the entire video
- `fp16=False` — CPU does not support 16-bit float operations

Output is a JSON with the full transcript text and a list of segments, each with start/end timestamps and the transcribed text.

### Step 3: Translation (`pipeline/preprocessing/translator.py`)

All downstream NLP tools (spaCy, KeyBERT, NLI) are English-only models. They cannot parse Hindi or Telugu words, so technical terms embedded in those languages would be missed entirely. We use the Groq API (llama-3.3-70b-versatile) to translate each segment individually to English, with the instruction to keep technical terms in their original English form.

Translating segment-by-segment rather than the full transcript at once keeps requests within Groq's token limits and preserves the per-segment timestamps, which are needed for the positional scoring step later.

Both the original mixed-language text and the English translation are saved alongside each other, so the original can be referenced for evaluation.

### Step 4: Concept Extraction (`pipeline/extraction/concept_extractor.py`)

Uses two models in combination:

**spaCy (`en_core_web_sm`)** parses the translated text grammatically and extracts all noun phrases as candidates — things like "moment of inertia", "angular velocity", "kinetic energy". This gives us linguistically valid phrases rather than arbitrary n-grams.

**KeyBERT (`all-MiniLM-L6-v2`)** takes those candidates and scores each one by cosine similarity against the embedding of the full document. Phrases that are representative of the overall topic score high; generic phrases like "the teacher" or "the example" score low. Maximal Marginal Relevance (MMR) with diversity=0.5 is used to prevent the top results from being near-duplicate variations of the same concept.

The rationale for this combination: spaCy ensures candidates are grammatically valid noun phrases, while KeyBERT provides a relevance score grounded in semantic similarity — neither step involves hardcoding any domain vocabulary.

### Step 5: Prerequisite Mapping (`pipeline/extraction/prerequisite_mapper.py`)

This is the core NLP step. For each pair of extracted concepts, we determine if one is a prerequisite of the other.

**Co-occurrence filter:** We first check which concept pairs appear together within a 3-segment window (~30 seconds of speech). Only pairs that co-occur get passed to the NLI model. This reduces the number of NLI calls from O(n^2) over all pairs to a small subset.

**NLI scoring (`cross-encoder/nli-MiniLM2-L6-H768`):** For each co-occurring pair (A, B), we construct a hypothesis — "To understand B, one must first understand A" — and use the NLI model to measure how much the surrounding context (the actual transcript segment text) entails that hypothesis. We check both directions (A→B and B→A) and pick the one with higher entailment. This approach reads actual meaning from text rather than pattern-matching on signal phrases like "first understand X", which would not generalize across teaching styles or languages.

**Positional scoring:** If concept A appears more than 120 seconds before concept B in the video, this gives a weak supporting signal that A might be a prerequisite — teachers generally teach prerequisites first. This is weighted at 30% of the final score because it is a rough heuristic, not a semantic signal.

**Final confidence = 0.7 × NLI score + 0.3 × positional score.** Edges below 0.5 are discarded.

### Step 6: Knowledge Graph (`pipeline/graph/knowledge_graph.py`)

Uses `networkx` to build a directed graph where nodes are concepts and edges point from prerequisite to dependent concept. `pyvis` renders this as an interactive HTML file — nodes are color-coded by role (red = foundational, purple = intermediate, blue = advanced/leaf), edge thickness reflects confidence score, and hovering over an edge shows the NLI confidence and the evidence text from the transcript.

A topological sort of the graph gives the recommended learning order — the sequence in which concepts should be studied such that no concept is encountered before its prerequisites.

---

## Architectural Choices

**Why use an LLM only for translation and not for concept extraction or prerequisite mapping?**

Using an LLM to answer "what are the concepts in this transcript?" or "which concepts are prerequisites of which?" produces results that are not grounded in any real measurement — the model answers based on its training data about physics or CS, not based on what the teacher actually said. KeyBERT produces a score that reflects genuine semantic relevance to the document. The NLI model's entailment score reflects whether the transcript text actually supports the prerequisite hypothesis. These are measurable signals.

**Why translate before running NLP tools rather than using multilingual models?**

Multilingual models (mBERT, XLM-R) are trained on full monolingual texts, not code-mixed speech where Hindi and English switch at the word level mid-sentence. Their embeddings for code-mixed input are less reliable than a good English-only model receiving clean translated input. The translation step also standardizes inconsistent romanization and colloquial abbreviations automatically.

**Why segment-by-segment translation rather than full-text?**

Preserves timestamps. Each translated segment is still associated with its start and end time, which the positional scoring step depends on.

---

## Output Structure

For each video, the pipeline produces:

- `video_N_transcript.json` — raw Whisper output: detected language, full text, segments with timestamps
- `video_N_translated.json` — same segments with original and translated text side by side
- `video_N_concepts.json` — list of extracted concepts with KeyBERT relevance scores
- `video_N_prerequisites.json` — prerequisite edges with confidence, NLI score, positional score, and evidence text
- `video_N_graph.html` — interactive browser-based knowledge graph
- `video_N_graph_metadata.json` — graph summary including recommended learning order

The intermediate files are kept separately so any step can be re-run independently without re-running the full pipeline. This is important because Whisper takes ~10 minutes per video on CPU, and translation makes multiple API calls.

---

## Results: video_1 (Rotational Motion, Hindi-English)

The pipeline ran successfully end-to-end on video_1 but produced poor concept extraction results — only 2 concepts were extracted ("force", "cheese") and 0 prerequisite edges were found.

**Root cause:** Whisper medium on CPU produced highly inaccurate transcription for this audio. The teacher speaks fast with a strong regional accent, and the audio contains long stretches of silence (whiteboard writing). Whisper produced phonetically plausible but semantically meaningless Hindi text — words like "प्रेंग", "विनेषे", "अस्कॉहार" which are not real Hindi words. The Groq LLM faithfully translated this garbage into equally meaningless English ("emulsion scar tissue", "the preng", "Cheese is perpendicular to the plane").

As a result, spaCy found noun phrases but none of them were real technical terms, and KeyBERT had no meaningful document to score them against. The concept extraction and prerequisite mapping stages themselves worked correctly — there was simply no valid input reaching them.

The audio also has three large silence gaps (2:08-5:08 and 5:16-6:46), where Whisper produced no segments at all. This is expected behavior — Whisper does not hallucinate speech over silence.

**The bottleneck is transcription quality.** The downstream NLP steps (translation, KeyBERT, NLI) are functioning correctly and would produce meaningful output given an accurate transcript.

---

## Setup

**Requirements:** Python 3.10+, FFmpeg installed on system

```bash
git clone https://github.com/yoimdepressed/iREL.git
cd iREL
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Create a `.env` file:
```
GROQ_API_KEY=your_groq_api_key_here
```

Run the full pipeline:
```bash
python main.py
```

Run a single step:
```bash
python main.py --step transcribe
python main.py --step translate
python main.py --step extract
python main.py --step map
python main.py --step graph
```

To process additional videos, uncomment their entries in `config.yaml`.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| openai-whisper | Local speech transcription |
| yt-dlp | YouTube audio download |
| groq | LLM API for translation |
| spacy + en_core_web_sm | Noun phrase extraction |
| keybert + sentence-transformers | Semantic concept scoring |
| transformers | NLI model for prerequisite detection |
| networkx | Graph construction and topological sort |
| pyvis | Interactive HTML graph rendering |
| pyyaml, python-dotenv | Config and environment loading |
