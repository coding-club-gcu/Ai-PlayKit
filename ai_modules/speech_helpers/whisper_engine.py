import os
import time
import numpy as np

class WhisperTranscriberEngine:
    """Fast Speech Recognition Engine powered by OpenAI Whisper."""

    def __init__(self):
        self.models = {}
        self.current_model_size = None
        self.current_model = None

    def load_model(self, model_size="tiny", progress_callback=None):
        """Lazy-loads requested Whisper model (tiny, base, small, etc.)."""
        if model_size in self.models:
            self.current_model = self.models[model_size]
            self.current_model_size = model_size
            return True

        try:
            import whisper
            if progress_callback:
                progress_callback(f"⏳ Loading Whisper '{model_size}' AI Model...")
            print(f"[WhisperEngine] Loading Whisper '{model_size}' model...")
            model = whisper.load_model(model_size)
            self.models[model_size] = model
            self.current_model = model
            self.current_model_size = model_size
            print(f"[WhisperEngine] Loaded Whisper '{model_size}' successfully.")
            return True
        except Exception as e:
            print(f"[WhisperEngine] Error loading Whisper model: {e}")
            return False

    def transcribe_audio_buffer(self, audio_data_float32, sample_rate=16000, model_size="tiny", language=None):
        """
        Transcribes 1D float32 audio array (normalized to [-1.0, 1.0]).
        Returns dict containing 'text', 'language', and 'segments'.
        """
        if not self.load_model(model_size):
            return {"text": "", "language": "en", "segments": []}

        try:
            # Whisper expects 16kHz audio array
            if len(audio_data_float32) == 0:
                return {"text": "", "language": "en", "segments": []}

            # Pre-pad short audio buffers if necessary
            min_samples = sample_rate * 1 # 1 second minimum
            if len(audio_data_float32) < min_samples:
                audio_data_float32 = np.pad(audio_data_float32, (0, min_samples - len(audio_data_float32)))

            # Options
            kwargs = {"fp16": False}
            if language and language.lower() != "auto-detect":
                kwargs["language"] = language.lower()

            result = self.current_model.transcribe(audio_data_float32, **kwargs)
            return {
                "text": result.get("text", "").strip(),
                "language": result.get("language", "en"),
                "segments": result.get("segments", [])
            }
        except Exception as e:
            print(f"[WhisperEngine] Buffer Transcription Error: {e}")
            return {"text": "", "language": "en", "segments": []}

    def transcribe_file(self, filepath, model_size="tiny", language=None, progress_callback=None):
        """Transcribes audio file (.wav, .mp3, .m4a, .flac)."""
        if not self.load_model(model_size, progress_callback):
            return {"text": "", "language": "en", "segments": []}

        try:
            if progress_callback:
                progress_callback("🎙️ Transcribing audio file with Whisper AI...")

            kwargs = {"fp16": False}
            if language and language.lower() != "auto-detect":
                kwargs["language"] = language.lower()

            result = self.current_model.transcribe(filepath, **kwargs)
            return {
                "text": result.get("text", "").strip(),
                "language": result.get("language", "en"),
                "segments": result.get("segments", [])
            }
        except Exception as e:
            print(f"[WhisperEngine] File Transcription Error: {e}")
            return {"text": f"Error: {e}", "language": "en", "segments": []}
