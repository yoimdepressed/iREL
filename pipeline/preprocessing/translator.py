import os
import json
import re
import unicodedata
import yaml
import time
from dotenv import load_dotenv
from groq import Groq

load_dotenv()


def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in .env")
    return Groq(api_key=api_key)


# Unicode block ranges for script detection
_SCRIPT_RANGES = [
    ("Devanagari", 0x0900, 0x097F),
    ("Latin", 0x0041, 0x024F),
    ("Cyrillic", 0x0400, 0x04FF),
    ("CJK", 0x4E00, 0x9FFF),
    ("Arabic", 0x0600, 0x06FF),
    ("Tamil", 0x0B80, 0x0BFF),
    ("Telugu", 0x0C00, 0x0C7F),
    ("Bengali", 0x0980, 0x09FF),
    ("Gujarati", 0x0A80, 0x0AFF),
    ("Greek", 0x0370, 0x03FF),
    ("Hebrew", 0x0590, 0x05FF),
]


def detect_scripts(text: str) -> set:
    """Return the set of Unicode script names present in the text."""
    found = set()
    for ch in text:
        cp = ord(ch)
        for name, start, end in _SCRIPT_RANGES:
            if start <= cp <= end:
                found.add(name)
                break
    return found


def is_garbled(text: str) -> bool:
    scripts = detect_scripts(text)
    return len(scripts) >= 3


SYSTEM_PROMPT = (
    "You are a translation assistant for code-mixed Indian educational videos. "
    "The input is a single sentence that may contain Hindi, Telugu, Tamil, or other Indian language words "
    "(either in their native script or romanized in Latin letters) mixed with English.\n\n"
    "Your job: produce a single, natural, fluent English sentence that conveys the full meaning of the input.\n\n"
    "Rules:\n"
    "1. Output ONLY the translated sentence — no explanations, no notes, no labels.\n"
    "2. Keep all technical/programming terms exactly as they are.\n"
    "3. Do NOT add new information. Do NOT repeat phrases.\n"
    "4. If the input is already fully in English, return it unchanged.\n"
    "5. Prioritize meaning and natural English flow over word-for-word literalness."
)


def translate_segment(text: str, client, model: str, temperature: float) -> str:
    """Translate a single transcript segment to English using Groq."""
    non_ascii_chars = len(re.findall(r'[^\x00-\x7F]', text))
    if non_ascii_chars / max(len(text), 1) < 0.1:
        return text

    prompt = (
        f"Translate the following sentence to natural, fluent English. "
        f"Preserve all technical/programming terms. Output only the translated sentence.\n\n"
        f"Sentence: {text}"
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=500,
    )
    result = response.choices[0].message.content.strip()

    # Safety check: if LLM returned something way longer than input, it hallucinated — use original
    if len(result) > len(text) * 2.5:
        return text

    # Safety check: if LLM output still has more non-ASCII than the input, it hallucinated
    result_non_ascii = len(re.findall(r'[^\x00-\x7F]', result))
    if result_non_ascii > non_ascii_chars:
        return text

    return result


def translate_transcript(transcript_data: dict, config: dict) -> dict:
    """
    Translate all segments from the transcript to English, one by one.
    Preserves original text alongside translation for evaluation.
    """
    client = get_groq_client()
    model = config["groq"]["model"]
    temp = config["groq"]["temperature"]
    segments = transcript_data["segments"]

    translated_segments = []
    print(f"Translating {len(segments)} segments...")

    for i, seg in enumerate(segments):
        original = seg["text"]

        # skip empty segments
        if not original.strip():
            translated_segments.append({
                "start": seg["start"],
                "end": seg["end"],
                "original": original,
                "translated": "",
            })
            continue

        # skip garbled segments (Whisper mixing 3+ Unicode scripts = corrupted output)
        # Sending these to Groq produces hallucinated nonsense
        if is_garbled(original):
            print(f"  [WARN] Segment {i} is garbled (mixed scripts), keeping original")
            translated_segments.append({
                "start": seg["start"],
                "end": seg["end"],
                "original": original,
                "translated": original,  # keep as-is; concept extractor will handle non-ASCII
            })
            continue

        try:
            translated = translate_segment(original, client, model, temp)
        except Exception as e:
            print(f"  Segment {i} failed: {e}, using original text")
            translated = original
            time.sleep(2)  # back off on error

        translated_segments.append({
            "start": seg["start"],
            "end": seg["end"],
            "original": original,
            "translated": translated,
        })

        # small delay to respect rate limits
        if i % 10 == 0 and i > 0:
            print(f"  Translated {i}/{len(segments)} segments...")
            time.sleep(0.5)

    # build full translated text
    full_translated = " ".join(s["translated"] for s in translated_segments if s["translated"])

    print(f"Translation complete. {len(translated_segments)} segments processed.")

    return {
        "detected_language": transcript_data.get("detected_language", "unknown"),
        "original_full_text": transcript_data["full_text"],
        "translated_full_text": full_translated,
        "segments": translated_segments,
    }


def save_translated(data: dict, video_id: str, output_dir: str) -> str:
    """Save translated transcript as JSON."""
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{video_id}_translated.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Translated transcript saved: {out_path}")
    return out_path


if __name__ == "__main__":
    config = load_config()
    transcript_dir = config["paths"]["transcripts_dir"]

    for video in config["videos"]:
        transcript_path = os.path.join(transcript_dir, f"{video['id']}_transcript.json")
        if not os.path.exists(transcript_path):
            print(f"Transcript not found for {video['id']} — run transcriber.py first")
            continue

        with open(transcript_path, "r", encoding="utf-8") as f:
            transcript_data = json.load(f)

        print(f"\nTranslating: {video['source']} ({video['language_mix']})")
        translated = translate_transcript(transcript_data, config)
        save_translated(translated, video["id"], transcript_dir)
