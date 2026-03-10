import os
import json
import yaml
import spacy
from keybert import KeyBERT


def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


# load models once
nlp = spacy.load("en_core_web_sm")
kw_model = KeyBERT(model="all-MiniLM-L6-v2")


def get_noun_phrase_candidates(text: str) -> list:
    """Use spaCy to extract noun phrase candidates from the text."""
    doc = nlp(text)
    candidates = set()

    for chunk in doc.noun_chunks:
        # clean up: remove determiners/pronouns at the start
        phrase = chunk.text.strip().lower()
        # skip very short or very long phrases
        if len(phrase.split()) < 1 or len(phrase.split()) > 5:
            continue
        # skip common non-technical phrases
        if phrase in ("we", "you", "i", "it", "they", "this", "that", "he", "she"):
            continue
        candidates.add(phrase)

    return list(candidates)


def extract_concepts(text: str, top_n: int = 20, min_score: float = 0.15) -> list:
    """
    Extract key concepts from translated transcript text.
    1. spaCy finds noun phrase candidates
    2. KeyBERT scores them by relevance to the document
    3. Return top concepts above the score threshold
    """
    # get candidate phrases from spaCy
    candidates = get_noun_phrase_candidates(text)

    if not candidates:
        print("Warning: no noun phrases found")
        return []

    # let KeyBERT score each candidate against the full document
    keywords = kw_model.extract_keywords(
        text,
        candidates=candidates,
        top_n=top_n,
        use_mmr=True,       # Maximal Marginal Relevance — reduces redundancy
        diversity=0.5,
    )

    # filter by minimum score
    concepts = []
    for keyword, score in keywords:
        if score >= min_score:
            concepts.append({"concept": keyword, "score": round(score, 4)})

    print(f"Extracted {len(concepts)} concepts from {len(candidates)} candidates")
    return concepts


if __name__ == "__main__":
    config = load_config()
    transcript_dir = config["paths"]["transcripts_dir"]
    json_dir = config["paths"]["json_dir"]
    os.makedirs(json_dir, exist_ok=True)

    top_n = config["extraction"]["top_n_concepts"]
    min_score = config["extraction"]["min_keyword_score"]

    for video in config["videos"]:
        translated_path = os.path.join(transcript_dir, f"{video['id']}_translated.json")
        if not os.path.exists(translated_path):
            print(f"Translated transcript not found for {video['id']}")
            continue

        with open(translated_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        text = data["translated_full_text"]
        print(f"\nExtracting concepts from: {video['id']}")
        concepts = extract_concepts(text, top_n, min_score)

        for c in concepts:
            print(f"  {c['concept']}: {c['score']}")

        result = {
            "video_id": video["id"],
            "language_mix": video["language_mix"],
            "concepts": concepts,
        }

        out_path = os.path.join(json_dir, f"{video['id']}_concepts.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Saved: {out_path}")
