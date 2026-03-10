# Code-Mixed Pedagogical Flow Extractor

An NLP pipeline that takes educational videos in code-mixed Indian languages (Hinglish, Telugu-English, Tamil-English) and automatically extracts a structured prerequisite map of the concepts taught.

## What it does

1. **Downloads** audio from YouTube videos
2. **Transcribes** the audio using OpenAI Whisper (auto language detection for code-mixed audio)
3. **Extracts** technical concepts from the transcript using Gemini
4. **Standardizes** colloquial Indic terms to proper English academic terminology
5. **Maps prerequisites** — determines which concepts must be understood before others, grounded in actual quotes from the transcript
6. **Visualizes** everything as an interactive HTML knowledge graph with confidence scores

## Video Sources

| # | Source | Topic | Language Mix |
|---|--------|-------|-------------|
| 1 | [Video 1](https://www.youtube.com/watch?v=DM0RUrheUQ0) | Rotational Motion | Hindi-English |
| 2 | [Video 2](https://www.youtube.com/watch?v=nzaHG0dme4g) | Linked lists | Hindi-English |
| 3 | [Video 3](https://www.youtube.com/watch?v=tWPa-rZiGM8) | Sys calls in OS | Hindi-English |
| 4 | [Video 4](https://www.youtube.com/watch?v=VJs99aeqUe4) | Python Strings | Telugu-English |
| 5 | [Video 5](https://www.youtube.com/watch?v=_99lZBc-lFY) | Functions in C | Tamil-English |

## Project Structure

```
iREL/
├── pipeline/
│   ├── ingestion/
│   │   ├── downloader.py       # downloads audio from YouTube
│   │   └── transcriber.py      # transcribes with Whisper
│   ├── preprocessing/
│   │   └── normalizer.py       # cleans text, detects prerequisite signal phrases
│   ├── extraction/
│   │   ├── concept_extractor.py    # pulls concepts + standardizes terminology
│   │   └── prerequisite_mapper.py  # maps concept dependencies with evidence
│   └── graph/
│       └── knowledge_graph.py  # builds interactive HTML graph
├── data/
│   ├── audio/                  # downloaded mp3 files (gitignored)
│   ├── transcripts/            # whisper JSON output (gitignored)
│   └── outputs/
│       ├── json/               # concepts + prerequisites JSON
│       └── graphs/             # interactive HTML graphs
├── config.yaml                 # all settings: video URLs, model config, paths
├── main.py                     # single entry point to run everything
├── requirements.txt
└── .env                        # API keys 
```

## Setup

### 1. Clone and enter the repo
```bash
git clone https://github.com/yoimdepressed/iREL.git
cd iREL
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your Gemini API key
Get a free key at https://aistudio.google.com/app/apikey

Create a `.env` file:
```
GEMINI_API_KEY=your_key_here
```

### 5. Install FFmpeg (required for audio processing)
```bash
sudo apt install ffmpeg   # Linux
```

## Running the Pipeline

Run everything end-to-end:
```bash
python main.py
```

Run individual steps:
```bash
python main.py --step download      # just download audio
python main.py --step transcribe    # just transcribe
python main.py --step extract       # just extract concepts
python main.py --step map           # just map prerequisites
python main.py --step graph         # just build the graph
```

## Output

For each video you get:
- `data/outputs/json/{video_id}_concepts.json` — raw and standardized concepts
- `data/outputs/json/{video_id}_prerequisites.json` — prerequisite edges with confidence + evidence quotes
- `data/outputs/json/{video_id}_graph_metadata.json` — graph summary + recommended learning order
- `data/outputs/graphs/{video_id}_graph.html` — interactive visualization (open in browser)

## Design Decisions

**Why Whisper with `language=None`?**
Auto-detection handles code-mixed audio much better than forcing a single language. Whisper's multilingual model was trained on diverse data including code-switched speech.

**Why evidence-backed prerequisite edges?**
Every dependency claim is grounded in an exact quote from the transcript. This prevents LLM hallucination and makes the output verifiable.

**Why confidence scores on edges?**
Not all prerequisite signals are equally strong. A teacher explicitly saying "pehle yeh samjho" is stronger evidence than a vague ordering. Weak edges (below threshold) are stored but filtered from the main output.

**Why interactive HTML (Pyvis)?**
Static graph images lose information. Interactive graphs let you hover over edges to see the evidence quote, drag nodes around, and explore the dependency chain naturally.

**Why a unified config.yaml?**
All settings — video URLs, model sizes, thresholds, paths — live in one place. Nothing is hardcoded. Adding a new video is just adding 4 lines to config.yaml.
