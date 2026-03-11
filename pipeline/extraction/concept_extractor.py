import os
import json
import re
import yaml
import math
import spacy
from keybert import KeyBERT
from collections import Counter


def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


# load models once — spaCy for POS/NER/lemma, KeyBERT for semantic scoring
nlp = spacy.load("en_core_web_sm")
kw_model = KeyBERT(model="all-MiniLM-L6-v2")


def is_valid_candidate(phrase: str) -> bool:
    """
    Generic validity check — works for ANY domain, no hardcoding.
    Uses spaCy's built-in stop word vocabulary to reject generic words.
    """
    # Reject anything containing non-ASCII (hallucinated script leaked from LLM)
    if re.search(r'[^\x00-\x7F]', phrase):
        return False
    # Reject purely numeric tokens
    if re.fullmatch(r'[\d\s\-\.]+', phrase):
        return False
    # Reject single characters
    if len(phrase.strip()) <= 1:
        return False
    # Reject phrases where EVERY token is a spaCy stop word
    tokens = phrase.lower().split()
    if all(nlp.vocab[t].is_stop for t in tokens):
        return False
    return True


def get_noun_phrase_candidates(text: str) -> list:
    """
    Extract candidate phrases generically using spaCy POS tags and NER.
    - Noun chunks that contain at least one non-stop-word noun/proper noun
    - Named entities (PERSON, ORG, etc. are filtered; keep only useful ones)
    - Multi-word expressions: consecutive ADJ/NOUN sequences (catches "linked list",
      "slow pointer", "binary search" etc. that spaCy may split into separate chunks)
    - VERB+NOUN compounds: catches past-participle modifiers like "linked list",
      "sorted array" where the modifier is tagged as VERB by spaCy
    No hardcoded word lists — works for any subject domain.
    """
    doc = nlp(text)
    candidates = set()

    # 1. Standard noun chunks — strip leading determiners/pronouns
    for chunk in doc.noun_chunks:
        tokens = [t for t in chunk if not t.is_stop and not t.is_punct and t.pos_ not in ("DET", "PRON")]
        if not tokens:
            continue
        phrase = " ".join(t.text for t in tokens).strip().lower()
        if 1 <= len(phrase.split()) <= 5 and is_valid_candidate(phrase):
            candidates.add(phrase)

    # 2. Named entities — skip purely social/location entities that aren't technical
    skip_ent_labels = {"GPE", "LOC", "PERSON", "NORP", "DATE", "TIME", "MONEY",
                       "PERCENT", "CARDINAL", "ORDINAL", "QUANTITY"}
    for ent in doc.ents:
        if ent.label_ in skip_ent_labels:
            continue
        phrase = ent.text.strip().lower()
        if is_valid_candidate(phrase) and 1 <= len(phrase.split()) <= 4:
            candidates.add(phrase)

    # 3. Consecutive ADJ/NOUN/PROPN/VERB(participle) sequences — catches compound terms
    #    Including VERB covers past participles like "linked" in "linked list",
    #    "sorted" in "sorted array" — spaCy tags these as VBN (VERB)
    i = 0
    tokens_list = [t for t in doc if not t.is_punct]
    compound_pos = {"ADJ", "NOUN", "PROPN"}
    while i < len(tokens_list):
        t = tokens_list[i]
        # start a sequence from ADJ/NOUN/PROPN or a participle VERB (VBN/VBG)
        if (t.pos_ in compound_pos or (t.pos_ == "VERB" and t.tag_ in ("VBN", "VBG"))) and not t.is_stop:
            seq = [t]
            j = i + 1
            while j < len(tokens_list) and (
                tokens_list[j].pos_ in compound_pos or
                (tokens_list[j].pos_ == "VERB" and tokens_list[j].tag_ in ("VBN", "VBG"))
            ):
                seq.append(tokens_list[j])
                j += 1
            # emit all sub-sequences of length 2-4 that end with NOUN/PROPN
            for start in range(len(seq)):
                for end in range(start + 2, min(start + 5, len(seq) + 1)):
                    sub = seq[start:end]
                    if sub[-1].pos_ in ("NOUN", "PROPN"):
                        phrase = " ".join(s.text for s in sub).strip().lower()
                        if is_valid_candidate(phrase):
                            candidates.add(phrase)
            i = j
        else:
            i += 1

    return list(candidates)


def compute_tf_score(candidate: str, text: str) -> float:
    """
    Compute normalized term frequency for a candidate phrase in the text.
    Uses word-boundary regex so 'link' doesn't match inside 'linked'.
    Applies a multi-word bonus: compound terms that appear together are
    more significant than isolated single words.
    """
    text_lower = text.lower()
    # exact phrase match with word boundaries
    pattern = r'\b' + re.escape(candidate.lower()) + r'\b'
    count = len(re.findall(pattern, text_lower))
    total_words = len(text_lower.split())
    word_count = len(candidate.split())
    # base TF
    tf = count / max(total_words, 1)
    # multi-word bonus: scale up by (1 + log(word_count))
    # a 2-word phrase appearing 10x is worth more than a 1-word appearing 10x
    tf *= (1 + math.log(word_count + 1))
    return tf


def lemmatize_phrase(phrase: str) -> str:
    """Lemmatize each word in a phrase using spaCy."""
    doc = nlp(phrase)
    return " ".join(t.lemma_.lower() for t in doc)


def subsumption_filter(scored_concepts: list) -> list:
    """
    Remove single-word concepts that are subsumed by multi-word concepts.
    E.g., if "linked list" is present, drop "list". If "fast pointer" is present, drop "pointer".
    Uses both raw tokens and lemmatized forms for matching.
    Also removes shorter multi-word phrases fully contained in longer ones
    (e.g., drop "link list" if "linked list" exists).
    """
    # collect all multi-word concepts and their tokens + lemmas
    multi_word = {c["concept"] for c in scored_concepts if len(c["concept"].split()) >= 2}
    subsumed_tokens = set()
    subsumed_lemmas = set()
    for mw in multi_word:
        for token in mw.split():
            subsumed_tokens.add(token)
            subsumed_lemmas.add(lemmatize_phrase(token))

    filtered = []
    for c in scored_concepts:
        concept = c["concept"]
        words = concept.split()
        if len(words) == 1:
            word = words[0]
            lemma = lemmatize_phrase(word)
            if word in subsumed_tokens or lemma in subsumed_lemmas:
                # single word is covered by a multi-word concept
                continue
        filtered.append(c)

    return filtered


def extract_concepts(text: str, top_n: int = 20, min_score: float = 0.15) -> list:
    """
    Extract key concepts from translated transcript text.
    1. Strip non-ASCII garbage
    2. spaCy extracts noun phrase candidates (POS tags + NER, no hardcoding)
    3. Split into multi-word and single-word pools
    4. Multi-word: scored by TF (frequency) — KeyBERT doesn't work well for phrases
    5. Single-word: scored by KeyBERT semantic relevance — captures document theme
    6. Subsumption: remove single words already in multi-word concepts
    7. Lemma dedup, interleave with multi-word priority
    """
    # Strip non-ASCII characters
    clean_text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()

    candidates = get_noun_phrase_candidates(clean_text)

    if not candidates:
        print("Warning: no noun phrase candidates found")
        return []

    print(f"  Candidates from spaCy: {len(candidates)}")

    # --- STEP 1: Split into multi-word and single-word pools ---
    multi_word_cands = [c for c in candidates if len(c.split()) >= 2]
    single_word_cands = [c for c in candidates if len(c.split()) == 1]

    # --- STEP 2: Score multi-word candidates by TF ---
    # Count exact phrase occurrences (word-boundary match)
    multi_freq = {}
    text_lower = clean_text.lower()
    for c in multi_word_cands:
        pattern = r'\b' + re.escape(c) + r'\b'
        multi_freq[c] = len(re.findall(pattern, text_lower))

    # Filter: must appear at least 2 times (avoid one-off phrases)
    multi_word_cands = [c for c in multi_word_cands if multi_freq[c] >= 2]

    # Normalize TF scores to 0-1
    if multi_word_cands:
        max_freq = max(multi_freq[c] for c in multi_word_cands)
        min_freq = min(multi_freq[c] for c in multi_word_cands)
        freq_range = max_freq - min_freq if max_freq > min_freq else 1
        multi_scored = [(c, (multi_freq[c] - min_freq) / freq_range) for c in multi_word_cands]
        multi_scored.sort(key=lambda x: x[1], reverse=True)
    else:
        multi_scored = []

    # --- STEP 3: Score single-word candidates by KeyBERT ---
    if single_word_cands:
        kws = kw_model.extract_keywords(
            clean_text,
            candidates=single_word_cands,
            top_n=len(single_word_cands),
            use_mmr=True,
            diversity=0.3,
        )
        single_scored = [(kw.lower(), score) for kw, score in kws]
    else:
        single_scored = []

    # --- STEP 4: Deduplicate multi-word results ---
    # Enhanced dedup: if two phrases share the same core content words
    # (e.g., "slow pointing" vs "slow pointer"), keep only the higher-scored one
    seen_lemmas = set()
    seen_root_keys = set()  # root keys = sorted set of word roots (first 4 chars)
    multi_concepts = []
    for kw, score in multi_scored:
        if not is_valid_candidate(kw):
            continue
        lemma = lemmatize_phrase(kw)
        if lemma in seen_lemmas:
            continue
        # Generate root key: sorted roots of each word (first 4 chars captures stem)
        words = [w for w in kw.split() if not nlp.vocab[w].is_stop]
        root_key = tuple(sorted(w[:4] for w in words)) if words else (kw,)
        if root_key in seen_root_keys:
            continue
        seen_lemmas.add(lemma)
        seen_root_keys.add(root_key)
        multi_concepts.append({"concept": kw, "score": round(score, 4)})

    # --- STEP 4b: Redundancy filter for multi-word concepts ---
    # Remove longer phrases that are just modifiers of a shorter concept already captured.
    # E.g., "entire linked list" is redundant if "linked list" exists,
    # "odd sized linked list" is redundant if "linked list" or "sized linked list" exists.
    # Generic: if concept A is a contiguous substring of concept B, drop B (keep shorter).
    concept_strings = [c["concept"] for c in multi_concepts]
    redundant = set()
    for i, longer in enumerate(concept_strings):
        for j, shorter in enumerate(concept_strings):
            if i == j:
                continue
            # shorter must be strictly shorter and a contiguous substring
            if len(shorter.split()) < len(longer.split()) and shorter in longer:
                redundant.add(i)
                break
    multi_concepts = [c for i, c in enumerate(multi_concepts) if i not in redundant]

    # --- STEP 5: Subsumption — remove single words present in multi-word concepts ---
    subsumed_tokens = set()
    subsumed_lemmas = set()
    for mc in multi_concepts:
        for token in mc["concept"].split():
            subsumed_tokens.add(token)
            subsumed_lemmas.add(lemmatize_phrase(token))

    # Also filter generic noise: words appearing in >25% of sentences are too common
    sentences = [s.strip() for s in re.split(r'[.!?]+', clean_text) if len(s.strip()) > 5]
    total_sents = max(len(sentences), 1)

    single_concepts = []
    for kw, score in single_scored:
        if score < min_score:
            continue
        if not is_valid_candidate(kw):
            continue
        # Skip if subsumed by a multi-word concept
        if kw in subsumed_tokens or lemmatize_phrase(kw) in subsumed_lemmas:
            continue
        # Skip overly generic words (appear in too many sentences)
        sent_count = sum(1 for s in sentences if re.search(r'\b' + re.escape(kw) + r'\b', s.lower()))
        if sent_count / total_sents > 0.25:
            continue
        lemma = lemmatize_phrase(kw)
        if lemma in seen_lemmas:
            continue
        seen_lemmas.add(lemma)
        single_concepts.append({"concept": kw, "score": round(score, 4)})

    # --- STEP 6: Interleave — prioritize multi-word concepts ---
    # Take up to 70% from multi-word pool, rest from single-word pool
    multi_limit = max(1, int(top_n * 0.7))
    single_limit = top_n - min(len(multi_concepts), multi_limit)

    concepts = multi_concepts[:multi_limit]
    concepts.extend(single_concepts[:single_limit])
    concepts = concepts[:top_n]

    print(f"Extracted {len(concepts)} concepts ({len(multi_concepts)} multi-word, {len(single_concepts)} single-word) from {len(candidates)} candidates")
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
