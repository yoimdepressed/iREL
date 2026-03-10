import os
import yt_dlp
import yaml


def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


def download_audio(video_url: str, video_id: str, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True) # creating audio folder
    output_path = os.path.join(output_dir, video_id)

    options = {
        "format": "bestaudio/best",
        "outtmpl": output_path + ".%(ext)s",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        "quiet": False,
    }

    with yt_dlp.YoutubeDL(options) as downloader:
        downloader.download([video_url])

    return output_path + ".mp3"


if __name__ == "__main__":
    config = load_config()
    audio_dir = config["paths"]["audio_dir"]

    for video in config["videos"]:
        print(f"\nDownloading: {video['source']}")
        saved_path = download_audio(video["url"], video["id"], audio_dir)
        print(f"Saved to: {saved_path}")
