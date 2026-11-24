import yt_dlp
import os
import subprocess
import wave
import imageio_ffmpeg

DOWNLOAD_DIR = 'downloades'
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# Set global ffmpeg path for subprocess
ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

def download_youtube_audio(url: str) -> str:
    """Download audio from YouTube and convert to WAV with deterministic filename."""
    import glob

    ydl_info_opts = {
        "quiet": True,
        "extractor_args": {"youtube": {"player_client": ["android", "web"]}},
    }
    with yt_dlp.YoutubeDL(ydl_info_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        video_id = info.get("id", "audio")

    target_wav = os.path.join(DOWNLOAD_DIR, f"{video_id}.wav")
    if os.path.exists(target_wav) and os.path.getsize(target_wav) > 1000:
        print(f"Using already cached audio: {target_wav}")
        return target_wav

    raw_template = os.path.join(DOWNLOAD_DIR, f"temp_{video_id}.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": raw_template,
        "quiet": True,
        "extractor_args": {"youtube": {"player_client": ["android", "web"]}},
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    raw_candidates = glob.glob(os.path.join(DOWNLOAD_DIR, f"temp_{video_id}.*"))
    if not raw_candidates:
        raise FileNotFoundError(f"Failed to download audio for video {video_id}")

    raw_file = raw_candidates[0]
    try:
        subprocess.run(
            [ffmpeg_path, "-y", "-i", raw_file, "-vn", "-ac", "1", "-ar", "16000", target_wav],
            check=True,
            capture_output=True
        )
    finally:
        for f in raw_candidates:
            try:
                os.remove(f)
            except OSError:
                pass

    return target_wav


def convert_to_wav(input_path: str) -> str:
    """
    Convert any audio/video file to 16kHz mono WAV using ffmpeg directly.
    Does NOT require ffprobe, which avoids Windows FileNotFoundError.
    """
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    clean_base = "".join(c for c in base_name if c.isalnum() or c in ("-", "_")).strip() or "uploaded_media"
    output_path = os.path.join(DOWNLOAD_DIR, f"{clean_base}_converted.wav")

    cmd = [
        ffmpeg_path,
        "-y",
        "-i", input_path,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg conversion failed: {result.stderr}")

    return output_path


def slice_wav(wav_path: str, segment_seconds: int, suffix: str) -> list[str]:
    """Split a WAV file into fixed-duration WAV segments using Python's stdlib wave module."""
    pieces: list[str] = []
    with wave.open(wav_path, "rb") as src:
        params = src.getparams()
        frames_per_piece = max(1, params.framerate * segment_seconds)
        total_frames = src.getnframes()
        for i, _ in enumerate(range(0, total_frames, frames_per_piece)):
            frames = src.readframes(frames_per_piece)
            if not frames:
                break
            piece_path = f"{wav_path}_{suffix}_{i}.wav"
            with wave.open(piece_path, "wb") as dst:
                dst.setparams(params)
                dst.writeframes(frames)
            pieces.append(piece_path)

    return pieces or [wav_path]


def chunk_audio(wav_path: str, chunk_minutes: int = 10) -> list:
    return slice_wav(wav_path, chunk_minutes * 60, "chunk")


def process_input(source: str) -> list:
    if source.startswith("http://") or source.startswith("https://"):
        print("Detected YouTube URL. Downloading audio...")
        wav_path = download_youtube_audio(source)
    else:
        print("Detected local file. Converting to WAV...")
        wav_path = convert_to_wav(source)

    print(f"Chunking audio from: {wav_path} ...")
    chunks = chunk_audio(wav_path)
    print(f"Audio ready — {len(chunks)} chunk(s) created.")
    return chunks


