import os
import json
import yaml
import re
from sentence_transformers import CrossEncoder


def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


# NLI model — classifies if one sentence entails/contradicts/is neutral to another
nli_model = None


def get_nli_model():
    global nli_model
    if nli_model is None:
        print("Loading NLI model...")
        nli_model = CrossEncoder("cross-encoder/nli-MiniLM2-L6-H768")
    return nli_model


def split_into_sentences(text: str) -> list:
    """Split text into sentences. Simple but works for transcripts."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def find_co_occurring_pairs(segments: list, concepts: list) -> list:
    """
    Find concept pairs that appear in the same segment or nearby segments.
    This is our pre-filter — only these pairs get checked by NLI.
    Returns list of (concept_a, concept_b, segment_text, time_a, time_b).
    """
    concept_names = [c["concept"] for c in concepts]
    pairs = []
    seen_pairs = set()

    # for each segment, find which concepts appear in it
    concept_locations = {}  # concept -> list of (segment_index, start_time)
    for i, seg in enumerate(segments):
        text_lower = seg["translated"].lower() if "translated" in seg else seg.get("text", "").lower()
        for concept in concept_names:
            if concept.lower() in text_lower:
                if concept not in concept_locations:
                    concept_locations[concept] = []
                concept_locations[concept].append((i, seg.get("start", 0)))

    # find pairs that co-occur within a window of 3 segments
    for i, seg in enumerate(segments):
        text_lower = seg["translated"].lower() if "translated" in seg else seg.get("text", "").lower()
        window_text = text_lower

        # also include next 4 segments for context (wider window for educational content)
        for j in range(1, 5):
            if i + j < len(segments):
                next_seg = segments[i + j]
                next_text = next_seg["translated"].lower() if "translated" in next_seg else next_seg.get("text", "").lower()
                window_text += " " + next_text

        concepts_in_window = [c for c in concept_names if c.lower() in window_text]

        # create pairs from concepts found in this window
        for a_idx, concept_a in enumerate(concepts_in_window):
            for concept_b in concepts_in_window[a_idx + 1:]:
                pair_key = tuple(sorted([concept_a, concept_b]))
                if pair_key not in seen_pairs:
                    seen_pairs.add(pair_key)
                    # get first appearance times
                    time_a = concept_locations.get(concept_a, [(0, 0)])[0][1]
                    time_b = concept_locations.get(concept_b, [(0, 0)])[0][1]
                    pairs.append((concept_a, concept_b, window_text[:500], time_a, time_b))

    return pairs


def compute_nli_score(concept_a: str, concept_b: str, context: str, model) -> float:
    """
    Use NLI to check: does the context suggest concept_a is a prerequisite for concept_b?
    We frame it as: "To understand {B}, one must first understand {A}" and check
    if the context entails this.
    """
    hypothesis = f"To understand {concept_b}, one must first understand {concept_a}."
    # NLI model returns [contradiction, entailment, neutral] logits
    scores = model.predict([(context, hypothesis)])[0]
    # Convert raw logits to probabilities via softmax
    import numpy as np
    exp_scores = np.exp(scores - np.max(scores))  # subtract max for numerical stability
    probs = exp_scores / exp_scores.sum()
    entailment_prob = float(probs[1])  # index 1 = entailment
    return entailment_prob


def compute_positional_score(time_a: float, time_b: float, gap_threshold: float = 60) -> float:
    """
    If concept_a appears significantly before concept_b,
    gives a signal that A might be prerequisite to B.
    Teachers typically introduce prerequisites before dependent concepts.
    Returns 0-1 score.
    """
    gap = time_b - time_a
    if gap <= 0:
        return 0.0
    if gap < gap_threshold:
        return 0.3  # small gap — moderate signal (still meaningful in educational context)
    # scale: 60s gap = 0.5, 300s+ gap = 1.0
    score = min(1.0, 0.5 + (gap - gap_threshold) / 480)
    return round(score, 3)


def map_prerequisites(translated_data: dict, concepts: list, config: dict) -> list:
    """
    Main prerequisite mapping function.
    1. Find co-occurring concept pairs (pre-filter)
    2. Run NLI on filtered pairs
    3. Add positional signal
    4. Combine scores and return edges
    """
    model = get_nli_model()
    segments = translated_data["segments"]

    nli_weight = config["prerequisites"]["nli_weight"]
    pos_weight = config["prerequisites"]["positional_weight"]
    min_conf = config["prerequisites"]["min_confidence"]
    gap_threshold = config["prerequisites"]["positional_gap"]

    # step 1: co-occurrence filter
    pairs = find_co_occurring_pairs(segments, concepts)
    print(f"Found {len(pairs)} co-occurring concept pairs to check")

    edges = []
    for concept_a, concept_b, context, time_a, time_b in pairs:
        # step 2: NLI — check both directions
        # score_a_to_b tests hypothesis: "To understand B, one must first understand A" (A is prereq of B)
        # score_b_to_a tests hypothesis: "To understand A, one must first understand B" (B is prereq of A)
        score_a_to_b = compute_nli_score(concept_a, concept_b, context, model)
        score_b_to_a = compute_nli_score(concept_b, concept_a, context, model)

        # pick the direction with higher entailment — the winner becomes the edge "prereq → dependent"
        if score_a_to_b > score_b_to_a:
            prereq, dependent = concept_a, concept_b
            nli_score = score_a_to_b
            t_prereq, t_dep = time_a, time_b
        else:
            prereq, dependent = concept_b, concept_a
            nli_score = score_b_to_a
            t_prereq, t_dep = time_b, time_a

        # step 3: positional score
        pos_score = compute_positional_score(t_prereq, t_dep, gap_threshold)

        # step 4: combine
        final_score = round(nli_weight * nli_score + pos_weight * pos_score, 4)

        if final_score >= min_conf:
            edges.append({
                "from": prereq,
                "to": dependent,
                "confidence": final_score,
                "nli_score": round(nli_score, 4),
                "positional_score": pos_score,
                "evidence": context[:300],
            })

    # sort by confidence
    edges.sort(key=lambda e: e["confidence"], reverse=True)
    print(f"Found {len(edges)} prerequisite edges above threshold {min_conf}")
    return edges


if __name__ == "__main__":
    config = load_config()
    transcript_dir = config["paths"]["transcripts_dir"]
    json_dir = config["paths"]["json_dir"]
    os.makedirs(json_dir, exist_ok=True)

    for video in config["videos"]:
        translated_path = os.path.join(transcript_dir, f"{video['id']}_translated.json")
        concepts_path = os.path.join(json_dir, f"{video['id']}_concepts.json")

        if not os.path.exists(translated_path) or not os.path.exists(concepts_path):
            print(f"Missing files for {video['id']} — run earlier steps first")
            continue

        with open(translated_path, "r", encoding="utf-8") as f:
            translated_data = json.load(f)
        with open(concepts_path, "r", encoding="utf-8") as f:
            concepts_data = json.load(f)

        print(f"\nMapping prerequisites for: {video['id']}")
        edges = map_prerequisites(translated_data, concepts_data["concepts"], config)

        result = {
            "video_id": video["id"],
            "language_mix": video["language_mix"],
            "prerequisite_edges": edges,
        }

        out_path = os.path.join(json_dir, f"{video['id']}_prerequisites.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Saved: {out_path}")
