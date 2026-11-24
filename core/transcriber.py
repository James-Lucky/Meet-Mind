import whisper
import os
import subprocess
import threading
import requests
import numpy as np
import torch
import imageio_ffmpeg
from dotenv import load_dotenv
from utils.audio_processor import slice_wav

load_dotenv()

# Sarvam's sync STT-translate API rejects audio longer than 30s.
# We slice each chunk into 25s pieces (with a 5s safety margin) before sending.
SARVAM_PIECE_SECONDS = 25


WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")


SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
SARVAM_STT_TRANSLATE_URL = "https://api.sarvam.ai/speech-to-text-translate"
SARVAM_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v2.5")

_model = None
_whisper_lock = threading.Lock()


def load_model():

    global _model  

    if _model is None: 
        print(f"Loading Whisper model: {WHISPER_MODEL} ...")
        _model = whisper.load_model(WHISPER_MODEL) 
        print("Whisper model loaded.")
    return _model 


def _clear_decoder_hooks(model) -> None:
    """Remove any leftover forward hooks from a previously interrupted decode."""
    decoder = getattr(model, "decoder", None)
    if decoder is None:
        return
    for module in decoder.modules():
        if getattr(module, "_forward_hooks", None):
            module._forward_hooks.clear()


def _load_audio(file_path: str, sr: int = 16000) -> np.ndarray:
    """Decode audio into a 16kHz mono float32 waveform using imageio-ffmpeg's bundled binary."""
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe,
        "-nostdin",
        "-threads", "0",
        "-i", file_path,
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sr),
        "-",
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, check=True).stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio with ffmpeg: {e.stderr.decode(errors='replace')}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def transcribe_chunk_whisper(chunk_path: str) -> str:
    audio = _load_audio(chunk_path)
    if audio.size < 400:
        return ""

    use_fp16 = torch.cuda.is_available()
    with _whisper_lock:
        model = load_model()
        _clear_decoder_hooks(model)
        try:
            result = model.transcribe(audio, task="transcribe", fp16=use_fp16)
        except RuntimeError:
            _clear_decoder_hooks(model)
            result = model.transcribe(
                audio,
                task="transcribe",
                fp16=use_fp16,
                condition_on_previous_text=False,
            )
    return result["text"]


def _send_to_sarvam(piece_path: str) -> str:
    """Send one ≤30s WAV file to Sarvam and return the English transcript."""
    headers = {"api-subscription-key": SARVAM_API_KEY}

    with open(piece_path, "rb") as f:
        files = {"file": (os.path.basename(piece_path), f, "audio/wav")}
        data = {"model": SARVAM_MODEL, "with_diarization": "false"}
        response = requests.post(
            SARVAM_STT_TRANSLATE_URL,
            headers=headers,
            files=files,
            data=data,
            timeout=120,
        )

    if not response.ok:
        print(f"\n❌ Sarvam returned {response.status_code}")
        print(f"Response body: {response.text}\n")
        response.raise_for_status()

    return response.json().get("transcript", "")


def transcribe_chunk_sarvam(chunk_path: str) -> str:
    """
    Sarvam sync API only accepts ≤30s audio. We split this chunk into
    25-second pieces, send each separately, and join the transcripts.
    """
    if not SARVAM_API_KEY:
        raise RuntimeError("SARVAM_API_KEY is not set in environment / .env")

    piece_paths = slice_wav(chunk_path, SARVAM_PIECE_SECONDS, "sv")
    full_text = ""
    total_pieces = len(piece_paths)

    for i, piece_path in enumerate(piece_paths):
        try:
            print(f"  → Sarvam piece {i + 1}/{total_pieces} ...")
            full_text += _send_to_sarvam(piece_path) + " "
        finally:
            if piece_path != chunk_path and os.path.exists(piece_path):
                os.remove(piece_path)

    return full_text.strip()


def transcribe_chunk(chunk_path: str, language: str = "english") -> str:
    """
    Route one chunk to Whisper or Sarvam depending on language choice.
    - english  → Whisper (local model)
    - hinglish → Sarvam if API key available; falls back to Whisper
    """
    if language.lower() == "hinglish":
        if SARVAM_API_KEY:
            try:
                return transcribe_chunk_sarvam(chunk_path)
            except Exception as e:
                print(f"⚠️ Sarvam transcription failed: {e}. Falling back to Whisper...")
                return transcribe_chunk_whisper(chunk_path)
        else:
            print("SARVAM_API_KEY not found in .env. Using Whisper for Hinglish/Hindi transcription.")
            return transcribe_chunk_whisper(chunk_path)

    return transcribe_chunk_whisper(chunk_path)


def transcribe_all(chunks: list, language: str = "english") -> str:

    full_transcript = "" 

    use_sarvam = (language.lower() == "hinglish") and bool(SARVAM_API_KEY)
    engine = "Sarvam AI" if use_sarvam else "Whisper"
    print(f"Using {engine} for transcription.")

    for i, chunk in enumerate(chunks):  

        print(f"Transcribing chunk {i + 1}/{len(chunks)}...")

        text = transcribe_chunk(chunk, language=language)  

        full_transcript += text + " "  

    print("Transcription complete.")

    return full_transcript.strip()  
