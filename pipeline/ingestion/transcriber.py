import os
import json
import yaml
import torch
import whisper


def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


def transcribe_audio(audio_path: str, model_size: str = "large") -> dict:
    """Transcribe audio using local Whisper. Returns segments with timestamps."""
    use_fp16 = torch.cuda.is_available()  # fp16 only on GPU (Colab T4); CPU needs fp16=False
    print(f"Loading Whisper {model_size} model... (GPU: {use_fp16})")
    model = whisper.load_model(model_size)

    print(f"Transcribing: {audio_path}")
    result = model.transcribe(
        audio_path,
        task="transcribe",                    # preserve original language — don't auto-translate
        language=None,                        # auto-detect — needed for code-mixed
        condition_on_previous_text=False,     # prevents hallucination loops
        fp16=use_fp16,                        # True on GPU (faster), False on CPU (not supported)
        verbose=True,
    )

    segments = []
    for seg in result.get("segments", []):
        segments.append({
            "start": round(seg["start"], 2),
            "end": round(seg["end"], 2),
            "text": seg["text"].strip(),
        })

    detected_lang = result.get("language", "unknown")
    full_text = result.get("text", "")

    print(f"Detected language: {detected_lang}")
    print(f"Total segments: {len(segments)}")
    print(f"Preview: {full_text[:200]}...")

    return {
        "detected_language": detected_lang,
        "full_text": full_text,
        "segments": segments,
    }


def save_transcript(data: dict, video_id: str, output_dir: str) -> str:
    """Save transcript dict as JSON."""
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{video_id}_transcript.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Transcript saved: {out_path}")
    return out_path


if __name__ == "__main__":
    config = load_config()
    audio_dir = config["paths"]["audio_dir"]
    transcript_dir = config["paths"]["transcripts_dir"]
    model_size = config["whisper"]["model_size"]

    for video in config["videos"]:
        audio_path = os.path.join(audio_dir, f"{video['id']}.mp3")
        if not os.path.exists(audio_path):
            print(f"Audio not found: {audio_path} — run downloader.py first")
            continue

        print(f"\nProcessing: {video['source']}")
        data = transcribe_audio(audio_path, model_size)
        save_transcript(data, video["id"], transcript_dir)
