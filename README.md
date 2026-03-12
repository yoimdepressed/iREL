# Code-Mixed Pedagogical Flow Extractor

An end-to-end NLP pipeline that takes educational YouTube videos in code-mixed Indian languages (Hindi-English, Telugu-English, Tamil-English) and automatically extracts a structured **prerequisite knowledge graph** — showing which concepts must be understood before others, grounded in actual evidence from the transcript.

---

## The Problem

Indian educational content on YouTube is overwhelmingly code-mixed: instructors switch fluidly between Hindi (or Telugu/Tamil) and English mid-sentence. Standard NLP tools fail on this because they expect either pure English or pure Hindi. The goal of this project is to handle that messy, real-world input and still produce something useful: a map of *what to learn first* from a given video.

---

## Pipeline Architecture

```
YouTube URL
    │
    ▼
[1] DOWNLOAD         yt-dlp + FFmpeg → .mp3
    │
    ▼
[2] TRANSCRIBE       OpenAI Whisper (medium, local CPU)
                     Auto language detection — outputs Devanagari/Latin mix
    │
    ▼
[3] TRANSLATE        Groq API (llama-3.3-70b-versatile)
                     Per-segment translation to fluent English
                     Skips already-English segments (<10% non-ASCII)
                     Detects and drops garbled segments (3+ Unicode scripts)
    │
    ▼
[4] EXTRACT          spaCy (en_core_web_sm) + KeyBERT (all-MiniLM-L6-v2)
                     Noun phrase candidates → TF-scored multi-word concepts
                     KeyBERT semantic scoring for single-word concepts
                     OOV filter removes leaked romanized Hindi/telugu..
    │
    ▼
[5] MAP              NLI CrossEncoder (cross-encoder/nli-MiniLM2-L6-H768)
                     Tests hypothesis: "To understand B, one must first understand A"
                     Combined with positional signal (does A appear before B in video?)
    │
    ▼
[6] GRAPH            NetworkX DiGraph + Pyvis
                     Interactive HTML visualization
                     Topological sort → recommended learning order
```

---

## Architectural Choices and Rationale

### Why local Whisper instead of a cloud ASR API?
Whisper `medium` handles code-mixed audio reasonably well with `language=None` (auto-detect). Cloud APIs like Google STT or AssemblyAI typically expect one declared language and perform worse on spontaneous language-switching. The tradeoff is speed (CPU transcription takes ~10–20 minutes per video) vs. accuracy on this specific task.

### Why Groq for translation instead of a local model?
Translation of code-mixed Indian text requires a large, multilingual LLM. Running a 7B+ model locally on CPU is impractical. Groq's free tier with `llama-3.3-70b-versatile` gives fast, high-quality translation at zero cost. Temperature is set to `0.2` to reduce hallucination.

### Why spaCy + KeyBERT instead of asking an LLM to extract concepts?
LLM-based concept extraction is brittle — it requires careful prompting, is non-deterministic, and has no grounding in actual term frequency. spaCy's POS tagger and noun chunker extract candidates that are syntactically valid noun phrases (things that *can* be concepts). KeyBERT then scores them by semantic similarity to the full document using sentence embeddings, ensuring extracted concepts are actually central to the content. This two-stage approach is both reproducible and interpretable.

### Why separate TF scoring for multi-word vs. KeyBERT for single-word?
KeyBERT works well for single words but suppresses multi-word phrases in favour of their individual tokens. A phrase like `linked list` gets a lower KeyBERT score than `list` alone, because `list` has broader semantic coverage. So multi-word candidates are scored by **term frequency** (how often the exact compound phrase appears) — if a teacher says `slow pointer` 8 times, it is definitely a key concept. Single words are scored by KeyBERT. The final output interleaves both pools (70% multi-word, 30% single-word) to prefer precise compound terms.

### Why NLI for prerequisite detection instead of co-occurrence or keyword rules?
Rule-based approaches (e.g., "if the word 'before' appears between two concepts, they're ordered") are fragile and language-specific. A Natural Language Inference model evaluates whether the context *entails* the hypothesis "To understand B, one must first understand A" — this is a semantic judgement grounded in the actual transcript evidence. Both directions (A→B and B→A) are tested and the stronger entailment wins. A positional signal (does A appear significantly earlier in the video?) is blended in at 30% weight to break ties.

### Why this output structure?

| File | Purpose |
|---|---|
| `{id}_transcript.json` | Raw Whisper output with timestamps — for debugging |
| `{id}_translated.json` | Segment-by-segment English translation — allows evaluating translation quality independently |
| `{id}_concepts.json` | Extracted concepts with scores — intermediate result, inspectable |
| `{id}_prerequisites.json` | Prerequisite edges with confidence scores **and evidence quotes** — the core result |
| `{id}_graph_metadata.json` | Graph summary + recommended learning order — machine-readable final output |
| `{id}_graph.html` | Interactive visualization — human-readable final output |

Each intermediate file is saved to disk so any single step can be re-run without repeating the entire pipeline. This is important because Whisper takes 10–20 minutes and Groq has daily token limits.

---

## Video Sources

| # | Channel | Topic | Language Mix | URL |
|---|---------|-------|-------------|-----|
| 1 | Fundamenthol | Fluid Mechanics | Hindi-English | [link](https://www.youtube.com/watch?v=6xeENYd-aLw) |
| 2 | Shradha Khapra | Middle of Linked List | Hindi-English | [link](https://www.youtube.com/watch?v=nzaHG0dme4g) |
| 3 | Gate Smashers | System Calls in OS | Hindi-English | [link](https://www.youtube.com/watch?v=tWPa-rZiGM8) |
| 4 | (Hindi CS) | C Programming Basics | Hindi-English | [link](https://www.youtube.com/watch?v=ec-Cd4jKFWc) |
| 5 | (Telugu CS) | C Programming | Telugu-English | [link](https://www.youtube.com/watch?v=8qi4wcYhcl8) |

Videos were chosen to cover: different subjects (data structures, OS, programming basics), different language mixes (Hindi-dominant, English-dominant, Telugu), and different instructor styles (slow/clear vs. fast/colloquial).

---

## Sample Results

**Video 2 — Linked Lists (Hindi-English, Shradha Khapra)**

Concepts extracted: `linked list`, `fast pointer`, `slow pointer`, `middle node`, `pseudo code`, `null value`, `data structures`, `single loop`, `temporary pointer`, `single pass`, `brute force`, ...

Key prerequisite edges found:
```
linked list → null value        (confidence: 0.865)
linked list → pseudo code       (confidence: 0.825)
null value  → fast pointer      (confidence: 0.501)
pseudo code → null value        (confidence: 0.503)
```

Recommended learning order: `linked list → slow pointer → pseudo code → null value → fast pointer → execution`

**Video 3 — System Calls in OS (Hindi-English, Gate Smashers)**

Concepts extracted: `system call`, `operating system`, `user mode`, `kernel mode`, `shared memory`, `api`, `linux`, `processes`, `printf` ...

Key prerequisite edges found:
```
system call → c program         (confidence: 0.801)
system call → linux             (confidence: 0.599)
operating system → c program    (confidence: 0.429)
```

---

## Known Limitations

- **Whisper hallucination on fast Hindi speech**: Whisper `medium` sometimes transcribes fast, colloquial Hindi as garbled Devanagari or mishears English words (e.g., "syllabus" → "slavus"). This is a fundamental ASR limitation, not a pipeline bug. Videos with slow, clear speech (like video_2) produce much better results.
- **Romanized Hinglish is passed through untranslated**: Segments where the speaker writes Hindi in Latin script (e.g., "pointer ko samjho pehle") pass the non-ASCII check and are not sent to Groq for translation. The concept extractor's OOV filter rejects most leaked romanized words, but some may survive.
- **NLI confidence on short evidence windows**: The NLI model scores entailment over a ~500 character context window. For concepts that are introduced far apart in the video, the evidence window may not contain enough context for a reliable score.

---

## Project Structure

```
iREL/
├── pipeline/
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── downloader.py           # yt-dlp audio download
│   │   └── transcriber.py          # local Whisper transcription
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── translator.py           # Groq translation, garble detection
│   ├── extraction/
│   │   ├── __init__.py
│   │   ├── concept_extractor.py    # spaCy + KeyBERT concept extraction
│   │   └── prerequisite_mapper.py  # NLI prerequisite mapping
│   └── graph/
│       ├── __init__.py
│       └── knowledge_graph.py      # NetworkX + Pyvis graph builder
├── data/
│   ├── audio/                      # downloaded .mp3 files (gitignored)
│   ├── transcripts/                # Whisper + translated JSON (gitignored)
│   └── outputs/
│       ├── json/                   # concepts, prerequisites, graph metadata
│       └── graphs/                 # interactive HTML graphs
├── config.yaml                     # all settings: video URLs, model params, paths
├── main.py                         # single entry point
├── requirements.txt
└── .env                            # API keys (not committed — see setup below)
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
```

### 5. Set up your Groq API key
Get a free key at [console.groq.com/keys](https://console.groq.com/keys). The free tier includes 100,000 tokens/day which is enough for ~2–3 videos per day.

Create a `.env` file in the project root:
```
GROQ_API_KEY=your_key_here
```

### 6. Configure which videos to run
Edit `config.yaml` — uncomment the videos you want to process. Only one video at a time is recommended to stay within Groq's daily token limit.

---

## Running the Pipeline

Run all 6 steps end-to-end:
```bash
python main.py
```

Run a single step:
```bash
python main.py --step download      # download audio from YouTube
python main.py --step transcribe    # transcribe with Whisper (slow — ~15 min/video on CPU)
python main.py --step translate     # translate to English with Groq
python main.py --step extract       # extract concepts (spaCy + KeyBERT)
python main.py --step map           # map prerequisites (NLI)
python main.py --step graph         # build interactive knowledge graph
```

Each step reads from and writes to `data/` — so if translation is already done, you can re-run only `extract` onward without re-translating.

---

## Output

For each video you get:

| File | Description |
|---|---|
| `data/outputs/json/{id}_concepts.json` | Extracted concepts with confidence scores |
| `data/outputs/json/{id}_prerequisites.json` | Prerequisite edges: `from`, `to`, `confidence`, `evidence` quote |
| `data/outputs/json/{id}_graph_metadata.json` | Total concepts, edges, recommended learning order |
| `data/outputs/graphs/{id}_graph.html` | Interactive graph — open in any browser |

The HTML graph uses color coding: 🔴 red = foundational (no prerequisites), 🟣 purple = intermediate, 🔵 blue = advanced (leaf node). Edge thickness represents confidence.

