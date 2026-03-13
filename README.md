# Code-Mixed Pedagogical Flow Extractor

An end-to-end NLP pipeline that ingests educational YouTube videos in code-mixed Indian languages (Hindi-English, Telugu-English) and automatically extracts a structured **prerequisite knowledge graph** -- showing which concepts must be understood before others, with every edge grounded in an evidence quote from the transcript.

---

## The Problem

Indian educational content on YouTube is overwhelmingly code-mixed: instructors switch fluidly between Hindi (or Telugu) and English mid-sentence, often within the same phrase. Standard NLP tools fail on this input because they expect monolingual text. The goal of this project is to handle that messy, real-world input and produce something pedagogically useful: a directed graph of *what to learn first* from a given video.

---

## Pipeline Architecture

```
YouTube URL
    |
    v
[1] DOWNLOAD         yt-dlp + FFmpeg -> .mp3
    |
    v
[2] TRANSCRIBE       OpenAI Whisper large (GPU / CPU)
                     language=None -> auto-detect
                     Outputs Devanagari/Latin code-mixed segments with timestamps
    |
    v
[3] TRANSLATE        Groq API (llama-3.3-70b-versatile, temp=0.2)
                     Per-segment translation to fluent English
                     Skips already-English segments (<10% non-ASCII chars)
                     Detects and drops garbled segments (3+ Unicode scripts mixed)
    |
    v
[4] EXTRACT          spaCy (en_core_web_sm) + KeyBERT (all-MiniLM-L6-v2)
                     Noun phrase candidates from POS/NER/compound sequences
                     Multi-word concepts: scored by term frequency
                     Single-word concepts: scored by KeyBERT semantic similarity
                     Filters: OOV ratio, stop-word phrases, equation variables (a2, p1...)
                     Lemma deduplication removes plural/variant forms
    |
    v
[5] MAP              NLI CrossEncoder (cross-encoder/nli-MiniLM2-L6-H768)
                     Co-occurrence window: 10 segments (~800 chars of context)
                     Tests both directions: "To understand B, one must first understand A"
                     Blends NLI entailment score (60%) + positional signal (40%)
                     Positional signal: does A appear significantly earlier in the video?
    |
    v
[6] GRAPH            NetworkX DiGraph + custom D3.js HTML
                     Topological sort -> recommended learning order
                     Cycle detection: iteratively removes lowest-confidence cycle edges
                     Interactive HTML: zoom, drag, hover to highlight neighbours + see evidence
```

---

## Architectural Choices and Rationale

### Why Whisper `large` instead of a cloud ASR API?
Whisper `large` with `language=None` auto-detects the dominant language per segment and handles spontaneous code-switching better than cloud APIs (Google STT, AssemblyAI), which expect a single declared language. The `large` model is used with fp16 on GPU (auto-detected via `torch.cuda.is_available()`) and falls back to CPU fp32 locally. The tradeoff is speed vs. accuracy -- Whisper large is significantly better than `medium` on fast colloquial Hindi/Telugu speech.

### Why Groq for translation instead of a local model?
Reliably translating code-mixed Indic text requires a large multilingual LLM (70B+ parameters). Running this locally on CPU is impractical. Groq's free tier with `llama-3.3-70b-versatile` gives fast, high-quality per-segment translation at zero cost, with temperature `0.2` to minimise hallucination.

### Why spaCy + KeyBERT instead of asking an LLM to extract concepts?
LLM-based extraction is non-deterministic and has no grounding in term frequency. spaCy's POS tagger and noun chunker produce syntactically valid noun-phrase candidates (things that *can* be concepts). KeyBERT scores single-word candidates by semantic similarity to the full document embedding. Multi-word compound terms (e.g., `kernel mode`, `linked list`) are scored by raw term frequency instead -- KeyBERT suppresses multi-word phrases in favour of their individual tokens, so TF is the more reliable signal there. The two pools are interleaved (70% multi-word, 30% single-word) to prefer precise compound terminology over generic single words.

### Why NLI for prerequisite mapping instead of co-occurrence or keyword rules?
Rule-based methods (e.g., detecting the word "before" between two concepts) are fragile and miss implicit ordering. The NLI CrossEncoder evaluates whether a given context *entails* the hypothesis `"To understand B, one must first understand A"` -- a semantic judgement grounded in the actual transcript text. Both directions are tested per pair and the stronger entailment direction wins. A positional signal (A appearing significantly earlier in the video) is blended in at 40% weight, reflecting the teaching habit of introducing prerequisites before dependent concepts.

### Why D3.js for the graph instead of Pyvis?
Pyvis generates thousands of lines of bloated HTML by embedding its entire JS runtime inline. Our D3.js implementation produces a clean ~230-line self-contained HTML file (D3 loaded from CDN), with full zoom/pan/drag, hover-to-highlight neighbourhood, confidence-scaled edge thickness, and proper directed arrow rendering -- all in readable, maintainable code.

### Why a Directed Acyclic Graph (DAG) in JSON format?

Pedagogical flows are inherently directional and non-circular (you cannot have Concept A require Concept B, while Concept B requires Concept A). Representing the output as a DAG ensures mathematical soundness. The computational format chosen is a JSON edge list ([{"source": "A", "target": "B", "confidence": 0.8}]). This structure is highly scalable, language-agnostic, easily ingestible by graph databases (like Neo4j), and perfectly structured for the D3.js frontend to render topological sorts.

### Why this file structure for outputs?

| File | Purpose |
|------|---------|
| `{id}_transcript.json` | Raw Whisper output with timestamps -- baseline for debugging ASR quality |
| `{id}_translated.json` | Segment-by-segment English translation -- evaluable independently |
| `{id}_concepts.json` | Extracted concepts with scores -- inspectable intermediate result |
| `{id}_prerequisites.json` | Prerequisite edges with `confidence`, `nli_score`, `positional_score`, `evidence` quote |
| `{id}_graph_metadata.json` | Graph summary + recommended learning order -- machine-readable final output |
| `{id}_graph.html` | Interactive D3.js visualization -- human-readable final output |

Each intermediate file is written to disk so any single step can be re-run independently. This matters because Whisper transcription takes 10-20 minutes per video and Groq has a 100k tokens/day free limit.

---

## Video Sources

| # | Source | Topic | Language Mix | URL |
|---|--------|-------|-------------|-----|
| 1 | Fundamenthol | Fluid Mechanics | Hindi-English | [link](https://www.youtube.com/watch?v=6xeENYd-aLw) |
| 2 | Next Toppers | Chemical Equilibrium | Hindi-English | [link](https://www.youtube.com/watch?v=z3JYqn2cNzg) |
| 3 | Gate Smashers | System Calls in OS | Hindi-English | [link](https://www.youtube.com/watch?v=tWPa-rZiGM8) |
| 4 | Gate Smashers | C Programming Basics | Hindi-English | [link](https://www.youtube.com/watch?v=ec-Cd4jKFWc) |
| 5 | Atish Jain | C Programming | Telugu-English | [link](https://www.youtube.com/watch?v=8qi4wcYhcl8) |

Videos were chosen to span different subjects (fluid mechanics, chemistry, OS, programming), different language mixes (Hindi-dominant, Telugu-dominant), and different teaching styles (conceptual explanation vs. rapid revision vs. beginner course intro).

---

## Results Summary

| Video | Concepts | Edges | Top Prerequisite Edges |
|-------|----------|-------|------------------------|
| video_1 -- Fluid Mechanics | 4 | 3 | velocity -> viscosity (0.609), speed -> viscosity (0.494) |
| video_2 -- Chemical Equilibrium | 20 | 39 | equilibrium -> concentration (0.611), stoichiometric -> solubility (0.539) |
| video_3 -- System Calls | 5 | 5 | windows -> access (0.494), security -> access (0.465) |
| video_4 -- C Basics | 7 | 7 | exams -> structure (0.711), programming -> structure (0.699) |
| video_5 -- C Programming (Telugu) | 3 | 2 | format -> output (0.497), strings -> output (0.464) |

**Recommended learning orders (topological sort):**
- video_1: `velocity -> speed -> flow -> viscosity`
- video_2: `equilibrium -> acid -> ion -> base -> chemistry -> reactions -> concentration -> theory -> ...`
- video_3: `windows -> security -> protection -> access -> processes`
- video_4: `statements -> syllabus -> programming -> exams -> concepts -> structure`
- video_5: `format -> strings -> output`

---

## Known Limitations

- **Whisper on fast colloquial speech**: Whisper sometimes mishears fast Hindi or Telugu as garbled text. Videos with slower, clearer speech produce higher-quality transcripts and more prerequisite edges. This is a fundamental ASR limitation.
- **Rapid revision / intro videos produce fewer edges**: Videos that only list topics without explaining *why* one concept precedes another give the NLI model little to work with. video_1 (fluid mechanics quick overview) and video_5 (short Telugu intro) reflect this in their low edge counts -- the content itself lacks explicit prerequisite language.
- **Romanized Hinglish passes the English filter**: Segments where Hindi is written in Latin script (e.g., `"pointer ko samjho pehle"`) have fewer than 10% non-ASCII characters and are not sent for translation. The OOV filter in the concept extractor rejects most leaked romanized tokens, but occasional false survivors are possible.
- **Long-range concept dependencies**: The NLI model operates over a 10-segment sliding window (~800 chars). Concepts introduced very far apart in a long video may never co-occur in any window, so no edge is created between them even if one is a clear prerequisite of the other.

---

## Project Structure

```
iREL/
+-- pipeline/
|   +-- ingestion/
|   |   +-- __init__.py
|   |   +-- downloader.py           # yt-dlp audio download
|   |   +-- transcriber.py          # Whisper large, auto fp16 on GPU
|   +-- preprocessing/
|   |   +-- __init__.py
|   |   +-- translator.py           # Groq per-segment translation, garble detection
|   +-- extraction/
|   |   +-- __init__.py
|   |   +-- concept_extractor.py    # spaCy + KeyBERT, TF scoring, lemma dedup
|   |   +-- prerequisite_mapper.py  # NLI CrossEncoder, positional signal
|   +-- graph/
|       +-- __init__.py
|       +-- knowledge_graph.py      # NetworkX DiGraph + D3.js HTML graph
+-- data/
|   +-- audio/                      # downloaded .mp3 files (gitignored)
|   +-- transcripts/                # Whisper + translated JSON (gitignored)
|   +-- outputs/
|       +-- json/                   # concepts, prerequisites, graph metadata JSON
|       +-- graphs/                 # interactive D3.js HTML graphs
+-- config.yaml                     # all settings: video URLs, model params, thresholds
+-- main.py                         # single entry point, --step flag for partial runs
+-- requirements.txt
+-- .env                            # GROQ_API_KEY (not committed)
```

---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/yoimdepressed/iREL.git
cd iREL
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
```

### 3. Install Python dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 4. Install FFmpeg (required for audio extraction)
```bash
sudo apt install ffmpeg          # Ubuntu/Debian
brew install ffmpeg              # Mac
```

### 5. Set up your Groq API key
Get a free key at [console.groq.com/keys](https://console.groq.com/keys). The free tier provides 100,000 tokens/day -- enough for 2-3 videos per day.

Create a `.env` file in the project root:
```
GROQ_API_KEY=your_key_here
```

### 6. (Optional) GPU acceleration
For faster transcription, a CUDA-capable GPU is recommended. The pipeline auto-detects GPU availability and uses fp16 if available, otherwise falls back to CPU fp32. Google Colab with a T4 GPU works well for running the full pipeline.

---

## Running the Pipeline

Run all 6 steps end-to-end:
```bash
python main.py
```

Run a single step (for all videos listed in `config.yaml`):
```bash
python main.py --step download      # download audio from YouTube
python main.py --step transcribe    # transcribe with Whisper large (~15 min/video on CPU, ~3 min on GPU)
python main.py --step translate     # translate to English with Groq
python main.py --step extract       # extract concepts (spaCy + KeyBERT)
python main.py --step map           # map prerequisites (NLI CrossEncoder)
python main.py --step graph         # build interactive D3.js knowledge graph
```

Each step reads its inputs from and writes its outputs to `data/` -- so if translation is already done, you can re-run only `extract` onward without re-translating.

**Note:** To process a single video without touching the others, temporarily comment out the other entries in the `videos:` list in `config.yaml`.

---

## Output

For each video:

| File | Description |
|------|-------------|
| `data/outputs/json/{id}_concepts.json` | Extracted concepts with confidence scores |
| `data/outputs/json/{id}_prerequisites.json` | Prerequisite edges: `from`, `to`, `confidence`, `nli_score`, `positional_score`, `evidence` |
| `data/outputs/json/{id}_graph_metadata.json` | Total concepts, edges, recommended learning order |
| `data/outputs/graphs/{id}_graph.html` | Interactive graph -- open in any browser, no server needed |

**Graph colour coding:** Red = foundational (no prerequisites), Purple = intermediate, Teal = advanced (leaf node). Edge thickness scales with confidence. Hover over any node to highlight its direct connections. Hover over edges to see the NLI evidence quote from the transcript.
