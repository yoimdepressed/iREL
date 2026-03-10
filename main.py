"""
main.py — Code-Mixed Pedagogical Flow Extractor

Pipeline: Download → Transcribe (Whisper) → Translate (Groq) → Extract (KeyBERT+spaCy) → Map (NLI) → Graph

Usage:
    python main.py                      # run all steps
    python main.py --step download
    python main.py --step transcribe
    python main.py --step translate
    python main.py --step extract
    python main.py --step map
    python main.py --step graph
"""

import argparse
import yaml
import os
import sys
import json

sys.path.insert(0, os.path.dirname(__file__))

from pipeline.ingestion.downloader import download_audio
from pipeline.ingestion.transcriber import transcribe_audio, save_transcript
from pipeline.preprocessing.translator import translate_transcript, save_translated
from pipeline.extraction.concept_extractor import extract_concepts
from pipeline.extraction.prerequisite_mapper import map_prerequisites
from pipeline.graph.knowledge_graph import build_graph, compute_learning_order, save_interactive_graph, save_graph_metadata


def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


def run_pipeline(config, step=None):
    videos = config["videos"]
    paths = config["paths"]

    for video in videos:
        vid_id = video["id"]
        print(f"\n{'='*60}")
        print(f"Processing: {video['source']} ({video['language_mix']})")
        print(f"{'='*60}")

        audio_path = os.path.join(paths["audio_dir"], f"{vid_id}.mp3")
        transcript_path = os.path.join(paths["transcripts_dir"], f"{vid_id}_transcript.json")
        translated_path = os.path.join(paths["transcripts_dir"], f"{vid_id}_translated.json")
        concepts_path = os.path.join(paths["json_dir"], f"{vid_id}_concepts.json")
        prereq_path = os.path.join(paths["json_dir"], f"{vid_id}_prerequisites.json")

        # Step 1 — Download audio
        if step in (None, "download"):
            print("\n--- Step 1: Downloading audio ---")
            download_audio(video["url"], vid_id, paths["audio_dir"])

        # Step 2 — Transcribe with local Whisper
        if step in (None, "transcribe"):
            print("\n--- Step 2: Transcribing with Whisper ---")
            if not os.path.exists(audio_path):
                print(f"Audio not found: {audio_path}")
                continue
            transcript = transcribe_audio(audio_path, config["whisper"]["model_size"])
            save_transcript(transcript, vid_id, paths["transcripts_dir"])

        # Step 3 — Translate to English with Groq
        if step in (None, "translate"):
            print("\n--- Step 3: Translating to English ---")
            if not os.path.exists(transcript_path):
                print(f"Transcript not found: {transcript_path}")
                continue
            with open(transcript_path, "r", encoding="utf-8") as f:
                transcript_data = json.load(f)
            translated = translate_transcript(transcript_data, config)
            save_translated(translated, vid_id, paths["transcripts_dir"])

        # Step 4 — Extract concepts with KeyBERT + spaCy
        if step in (None, "extract"):
            print("\n--- Step 4: Extracting concepts ---")
            if not os.path.exists(translated_path):
                print(f"Translated transcript not found: {translated_path}")
                continue
            with open(translated_path, "r", encoding="utf-8") as f:
                trans_data = json.load(f)

            top_n = config["extraction"]["top_n_concepts"]
            min_score = config["extraction"]["min_keyword_score"]
            concepts = extract_concepts(trans_data["translated_full_text"], top_n, min_score)

            result = {
                "video_id": vid_id,
                "language_mix": video["language_mix"],
                "concepts": concepts,
            }
            os.makedirs(paths["json_dir"], exist_ok=True)
            with open(concepts_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"Concepts saved: {concepts_path}")

        # Step 5 — Map prerequisites with NLI
        if step in (None, "map"):
            print("\n--- Step 5: Mapping prerequisites ---")
            if not os.path.exists(translated_path) or not os.path.exists(concepts_path):
                print(f"Missing files for {vid_id}")
                continue
            with open(translated_path, "r", encoding="utf-8") as f:
                trans_data = json.load(f)
            with open(concepts_path, "r", encoding="utf-8") as f:
                concepts_data = json.load(f)

            edges = map_prerequisites(trans_data, concepts_data["concepts"], config)

            prereq_result = {
                "video_id": vid_id,
                "language_mix": video["language_mix"],
                "prerequisite_edges": edges,
            }
            os.makedirs(paths["json_dir"], exist_ok=True)
            with open(prereq_path, "w", encoding="utf-8") as f:
                json.dump(prereq_result, f, ensure_ascii=False, indent=2)
            print(f"Prerequisites saved: {prereq_path}")

        # Step 6 — Build knowledge graph
        if step in (None, "graph"):
            print("\n--- Step 6: Building knowledge graph ---")
            if not os.path.exists(prereq_path):
                print(f"Prerequisites not found: {prereq_path}")
                continue
            with open(prereq_path, "r", encoding="utf-8") as f:
                prereq_data = json.load(f)

            G = build_graph(prereq_data["prerequisite_edges"])
            order = compute_learning_order(G)

            if order:
                print(f"\nRecommended learning order:")
                print(f"  {' → '.join(order)}")

            save_interactive_graph(G, vid_id, paths["graphs_dir"])
            save_graph_metadata(G, order, vid_id, paths["json_dir"])

    print(f"\n{'='*60}")
    print("Pipeline complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Code-Mixed Pedagogical Flow Extractor")
    parser.add_argument(
        "--step",
        choices=["download", "transcribe", "translate", "extract", "map", "graph"],
        default=None,
        help="Run a specific step. Default runs all.",
    )
    args = parser.parse_args()

    config = load_config()
    run_pipeline(config, step=args.step)
