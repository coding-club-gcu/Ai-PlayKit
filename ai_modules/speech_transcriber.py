import os
import time
import math
import datetime
import threading
from tkinter import filedialog
import customtkinter as ctk
import numpy as np

# Metadata describing this module for AI PlayKit Hub
PROJECT_INFO = {
    "id": "speech_transcriber",
    "title": "AI Live Speech Transcriber",
    "description": "Real-time AI speech-to-text transcription powered by OpenAI Whisper with animated live audio waveform oscilloscope & subtitle generator!",
    "icon": "🎙️",
    "category": "Speech & Audio AI",
    "required_packages": ["openai-whisper", "sounddevice", "numpy", "PIL"],
    "install_command": "pip install openai-whisper sounddevice numpy pillow",
    "guide": """# 🎙️ AI Live Speech Transcriber Guide

### Overview
This project leverages **OpenAI Whisper AI** and **Real-Time Audio DSP** to record your microphone input live, animate a stunning neon audio waveform oscilloscope, and transcribe speech to text instantaneously!

---

### Step 1: Install Required Packages
Open your terminal and run:
```bash
pip install openai-whisper sounddevice numpy pillow
```

---

### Step 2: Key Features
- 🎙️ **Live Speech-to-Text**: Speak into your microphone and watch speech convert to text live.
- 🌊 **Real-Time Audio Waveform Canvas**: Oscilloscope sine waves & 32-channel frequency spectrum equalizer bars.
- 📁 **Audio File Transcriber**: Upload `.wav`, `.mp3`, `.m4a`, or `.flac` files to generate instant transcripts.
- 🧠 **Whisper AI Models**: Select from `tiny` (Fastest, 39M params), `base` (Balanced), or `small` (High Accuracy).
- 🌐 **Multi-Language Support**: English, Spanish, French, German, Hindi, Japanese, Chinese, Italian, or Auto-Detect.
- 📜 **Subtitle Exporter (`.srt` / `.txt`)**: Export transcripts with timestamps ready for YouTube or video editors.
- 📊 **Speech Analytics**: Live WPM (words per minute), microphone volume meter (dB), word count.

---

### Step 3: Beginner Python Code Example
```python
import whisper

# 1. Load Whisper AI Model
model = whisper.load_model("tiny")

# 2. Transcribe Audio File
result = model.transcribe("speech_sample.wav")

# 3. Print Transcribed Text
print("--- TRANSCRIPT ---")
print(result["text"])
```
"""
}

def check_dependencies():
    """Returns True if sounddevice and numpy are installed without importing heavy packages."""
    import importlib.util
    return (
        importlib.util.find_spec("sounddevice") is not None and
        importlib.util.find_spec("numpy") is not None
    )


class SpeechTranscriberUI(ctk.CTkFrame):
    """CustomTkinter UI for AI Live Speech Transcriber Studio."""

    def __init__(self, parent, on_back_callback):
        super().__init__(parent, fg_color="transparent")
        self.on_back_callback = on_back_callback

        self.whisper_engine = None
        self.is_recording = False
        self.audio_stream = None

        # Audio Buffers (16kHz mono)
        self.fs = 16000
        self.viz_buffer = np.zeros(1600, dtype=np.float32)  # 0.1s audio for live waveform viz
        self.accumulated_audio = np.array([], dtype=np.float32)
        self.transcript_lines = []
        self.last_volume_db = -60.0
        self.total_words_spoken = 0
        self.start_talk_time = None

        self.setup_ui()
        self.start_viz_loop()

    def setup_ui(self):
        # Top Header Bar
        header_frame = ctk.CTkFrame(
            self,
            fg_color=("#FFFFFF", "#1E1E2E"),
            corner_radius=12,
            height=60
        )
        header_frame.pack(fill="x", padx=5, pady=(0, 10))

        back_btn = ctk.CTkButton(
            header_frame,
            text="← Back to Hub",
            width=110,
            fg_color=("#DCE0E8", "#313244"),
            hover_color=("#BCC0CC", "#45475A"),
            text_color=("#4C4F69", "#CDD6F4"),
            command=self.on_back_click
        )
        back_btn.pack(side="left", padx=15, pady=12)

        title_lbl = ctk.CTkLabel(
            header_frame,
            text="🎙️ AI Live Speech Transcriber",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=("#1E1E2E", "#F5E0DC")
        )
        title_lbl.pack(side="left", padx=10)

        # Status Badge
        self.status_badge = ctk.CTkLabel(
            header_frame,
            text="Ready 🎙️",
            fg_color=("#1E66F5", "#89B4FA"),
            text_color=("#FFFFFF", "#11111B"),
            corner_radius=8,
            padx=10,
            pady=4,
            font=ctk.CTkFont(size=12, weight="bold")
        )
        self.status_badge.pack(side="right", padx=15)

        # Main Workspace Container
        main_content = ctk.CTkFrame(self, fg_color="transparent")
        main_content.pack(fill="both", expand=True, padx=5)

        # Left Control Panel
        control_panel = ctk.CTkScrollableFrame(
            main_content,
            fg_color=("#FFFFFF", "#1E1E2E"),
            corner_radius=12,
            width=320,
            scrollbar_button_color=("#CBD5E1", "#2B2C3B"),
            scrollbar_button_hover_color=("#94A3B8", "#3E4052")
        )
        control_panel.pack(side="left", fill="y", padx=(0, 10), pady=0)

        ctrl_title = ctk.CTkLabel(
            control_panel,
            text="⚙ Transcriber Controls",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=("#1E66F5", "#89B4FA")
        )
        ctrl_title.pack(anchor="w", padx=15, pady=(15, 10))

        # Main Action Buttons (Start Mic / Upload File)
        self.toggle_mic_btn = ctk.CTkButton(
            control_panel,
            text="🎙️ Start Live Mic Transcribe",
            fg_color=("#40A02B", "#A6E3A1"),
            hover_color=("#207015", "#94E2D5"),
            text_color=("#FFFFFF", "#11111B"),
            font=ctk.CTkFont(size=13, weight="bold"),
            height=38,
            command=self.toggle_live_recording
        )
        self.toggle_mic_btn.pack(fill="x", padx=15, pady=(5, 5))

        upload_file_btn = ctk.CTkButton(
            control_panel,
            text="📁 Transcribe Audio File",
            fg_color=("#1E66F5", "#89B4FA"),
            hover_color=("#7287FD", "#B4BEFE"),
            text_color=("#FFFFFF", "#11111B"),
            font=ctk.CTkFont(size=13, weight="bold"),
            height=38,
            command=self.upload_audio_file
        )
        upload_file_btn.pack(fill="x", padx=15, pady=(0, 10))

        # Whisper Model Selector
        model_lbl = ctk.CTkLabel(
            control_panel,
            text="Whisper AI Model:",
            font=ctk.CTkFont(size=12),
            text_color=("#4C4F69", "#CDD6F4")
        )
        model_lbl.pack(anchor="w", padx=15, pady=(8, 2))

        self.model_dropdown = ctk.CTkOptionMenu(
            control_panel,
            values=[
                "tiny (Fastest - 39M)",
                "base (Balanced - 74M)",
                "small (Accurate - 244M)"
            ],
            fg_color=("#8839EF", "#CBA6F7"),
            text_color=("#FFFFFF", "#11111B")
        )
        self.model_dropdown.pack(fill="x", padx=15, pady=5)

        # Language Selector
        lang_lbl = ctk.CTkLabel(
            control_panel,
            text="Spoken Language:",
            font=ctk.CTkFont(size=12),
            text_color=("#4C4F69", "#CDD6F4")
        )
        lang_lbl.pack(anchor="w", padx=15, pady=(8, 2))

        self.lang_dropdown = ctk.CTkOptionMenu(
            control_panel,
            values=[
                "Auto-Detect",
                "English",
                "Spanish",
                "French",
                "German",
                "Hindi",
                "Japanese",
                "Chinese",
                "Italian"
            ],
            fg_color=("#EA76CB", "#F5C2E7"),
            text_color=("#FFFFFF", "#11111B")
        )
        self.lang_dropdown.pack(fill="x", padx=15, pady=5)

        # Microphone Volume Level Meter
        vol_hdr = ctk.CTkFrame(control_panel, fg_color="transparent")
        vol_hdr.pack(fill="x", padx=15, pady=(12, 2))

        vol_lbl = ctk.CTkLabel(
            vol_hdr,
            text="Mic Volume (dB):",
            font=ctk.CTkFont(size=12),
            text_color=("#4C4F69", "#CDD6F4")
        )
        vol_lbl.pack(side="left")

        self.vol_val_lbl = ctk.CTkLabel(
            vol_hdr,
            text="-60 dB",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=("#1E66F5", "#89B4FA")
        )
        self.vol_val_lbl.pack(side="right")

        self.vol_progressbar = ctk.CTkProgressBar(
            control_panel,
            fg_color=("#DCE0E8", "#313244"),
            progress_color=("#40A02B", "#A6E3A1")
        )
        self.vol_progressbar.set(0.0)
        self.vol_progressbar.pack(fill="x", padx=15, pady=5)

        # Transcript Text Output Box Header
        trans_hdr = ctk.CTkFrame(control_panel, fg_color="transparent")
        trans_hdr.pack(fill="x", padx=15, pady=(14, 4))

        t_title = ctk.CTkLabel(
            trans_hdr,
            text="📜 Live Transcript",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=("#1E66F5", "#89B4FA")
        )
        t_title.pack(side="left")

        clear_btn = ctk.CTkButton(
            trans_hdr,
            text="🔄 Clear",
            width=60,
            height=24,
            fg_color=("#DCE0E8", "#313244"),
            hover_color=("#BCC0CC", "#45475A"),
            text_color=("#4C4F69", "#CDD6F4"),
            font=ctk.CTkFont(size=11),
            command=self.clear_transcript
        )
        clear_btn.pack(side="right")

        # Scrollable Transcript Textbox
        self.transcript_box = ctk.CTkTextbox(
            control_panel,
            height=140,
            fg_color=("#F2F4F8", "#11111B"),
            text_color=("#1E1E2E", "#CDD6F4"),
            font=ctk.CTkFont(family="Consolas", size=12),
            corner_radius=8
        )
        self.transcript_box.pack(fill="x", padx=15, pady=5)

        # Copy & Export Subtitles Row
        action_row = ctk.CTkFrame(control_panel, fg_color="transparent")
        action_row.pack(fill="x", padx=15, pady=(5, 10))

        self.copy_btn = ctk.CTkButton(
            action_row,
            text="📋 Copy Text",
            fg_color=("#1E66F5", "#89B4FA"),
            hover_color=("#7287FD", "#B4BEFE"),
            text_color=("#FFFFFF", "#11111B"),
            font=ctk.CTkFont(weight="bold"),
            height=32,
            command=self.copy_transcript
        )
        self.copy_btn.pack(side="left", fill="x", expand=True, padx=(0, 5))

        self.export_btn = ctk.CTkButton(
            action_row,
            text="💾 Subtitles (.srt)",
            fg_color=("#40A02B", "#A6E3A1"),
            hover_color=("#207015", "#94E2D5"),
            text_color=("#FFFFFF", "#11111B"),
            font=ctk.CTkFont(weight="bold"),
            height=32,
            command=self.export_srt_subtitles
        )
        self.export_btn.pack(side="right", fill="x", expand=True, padx=(5, 0))

        # Analytics Box
        status_box = ctk.CTkFrame(
            control_panel,
            fg_color=("#F2F4F8", "#11111B"),
            corner_radius=10
        )
        status_box.pack(fill="both", expand=True, padx=15, pady=(5, 15))

        stat_title = ctk.CTkLabel(
            status_box,
            text="📊 Speech Analytics",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=("#1E66F5", "#89B4FA")
        )
        stat_title.pack(anchor="w", padx=12, pady=(10, 4))

        self.stats_lbl = ctk.CTkLabel(
            status_box,
            text="Click '🎙️ Start Live Mic Transcribe' or upload an audio file.",
            font=ctk.CTkFont(size=11),
            text_color=("#5C5F77", "#A6ADC8"),
            justify="left",
            wraplength=250
        )
        self.stats_lbl.pack(anchor="w", padx=12, pady=(0, 10))

        # Right Live Waveform & Visualizer Container
        self.right_container = ctk.CTkFrame(
            main_content,
            fg_color=("#11111B", "#11111B"),
            corner_radius=12
        )
        self.right_container.pack(side="right", fill="both", expand=True)

        # High-DPI Audio Waveform Canvas
        self.waveform_canvas = ctk.CTkCanvas(
            self.right_container,
            bg="#11111B",
            highlightthickness=0
        )
        self.waveform_canvas.pack(fill="both", expand=True, padx=10, pady=10)

    def start_viz_loop(self):
        """Animates live waveform oscilloscope & FFT spectrum visualizer at 30 FPS."""
        self._draw_waveform_visualizer()
        self.after(33, self.start_viz_loop)

    def _draw_waveform_visualizer(self):
        w = self.waveform_canvas.winfo_width()
        h = self.waveform_canvas.winfo_height()

        if w < 50 or h < 50:
            return

        self.waveform_canvas.delete("all")

        # Background Subtle Grid
        grid_step = 40
        for x in range(0, w, grid_step):
            self.waveform_canvas.create_line(x, 0, x, h, fill="#181825", width=1)
        for y in range(0, h, grid_step):
            self.waveform_canvas.create_line(0, y, w, y, fill="#181825", width=1)

        center_y = h // 2

        # Draw 32 Animated Frequency Spectrum Equalizer Bars at top half
        bars_count = 32
        bar_w = max(4, (w - (bars_count * 4)) // bars_count)

        if len(self.viz_buffer) > 0:
            fft_data = np.abs(np.fft.rfft(self.viz_buffer))[:bars_count]
            fft_norm = np.clip(fft_data * 8.0, 0.0, 1.0)
        else:
            fft_norm = np.zeros(bars_count)

        for i in range(bars_count):
            bar_h = int(fft_norm[i] * (h * 0.35))
            x1 = 20 + i * (bar_w + 4)
            y1 = center_y - 20 - bar_h
            x2 = x1 + bar_w
            y2 = center_y - 20

            # Gradient bar colors
            hue_ratio = i / float(bars_count)
            color = "#89B4FA" if hue_ratio < 0.4 else ("#CBA6F7" if hue_ratio < 0.7 else "#F5C2E7")
            if bar_h > 2:
                self.waveform_canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="")

        # Draw Oscilloscope Waveform Curve
        points = []
        samples = len(self.viz_buffer)
        step = max(1, samples // w) if samples > 0 else 1

        for idx, x in enumerate(range(0, w, 2)):
            sample_idx = int((x / w) * samples)
            if sample_idx < samples:
                amp = self.viz_buffer[sample_idx]
            else:
                amp = 0.0

            # Oscilloscope amplitude curve
            y = center_y + int(amp * (h * 0.38))
            points.append(x)
            points.append(y)

        if len(points) >= 4:
            self.waveform_canvas.create_line(
                points,
                fill="#89B4FA",
                width=2,
                smooth=True
            )

        # Center Axis Line
        self.waveform_canvas.create_line(0, center_y, w, center_y, fill="#313244", width=1)

        # Live Recording Indicator Overlay
        if self.is_recording:
            self.waveform_canvas.create_text(
                w - 120, 25,
                text="● REC LIVE MIC",
                fill="#F38BA8",
                font=("Segoe UI", 12, "bold")
            )
        else:
            self.waveform_canvas.create_text(
                w - 120, 25,
                text="STANDBY",
                fill="#5C5F77",
                font=("Segoe UI", 12, "bold")
            )

    def toggle_live_recording(self):
        if self.is_recording:
            self.stop_live_recording()
        else:
            self.start_live_recording()

    def start_live_recording(self):
        self.is_recording = True
        self.start_talk_time = time.time()
        self.toggle_mic_btn.configure(
            text="⏹ Stop Live Mic Transcribe",
            fg_color=("#FE640B", "#F38BA8"),
            hover_color=("#D20F39", "#E64553"),
            text_color=("#FFFFFF", "#11111B")
        )
        self.status_badge.configure(
            text="Live Transcribing 🔴",
            fg_color=("#40A02B", "#A6E3A1"),
            text_color=("#FFFFFF", "#11111B")
        )

        threading.Thread(target=self._audio_stream_loop, daemon=True).start()
        threading.Thread(target=self._transcribe_stream_loop, daemon=True).start()

    def stop_live_recording(self):
        self.is_recording = False
        self.toggle_mic_btn.configure(
            text="🎙️ Start Live Mic Transcribe",
            fg_color=("#40A02B", "#A6E3A1"),
            hover_color=("#207015", "#94E2D5"),
            text_color=("#FFFFFF", "#11111B")
        )
        self.status_badge.configure(
            text="Ready 🎙️",
            fg_color=("#1E66F5", "#89B4FA"),
            text_color=("#FFFFFF", "#11111B")
        )

    def _audio_stream_loop(self):
        """Continuously reads audio frames from default microphone using sounddevice."""
        try:
            import sounddevice as sd

            def audio_callback(indata, frames, time_info, status):
                if not self.is_recording:
                    return
                mono = indata[:, 0]
                self.viz_buffer = mono.copy()
                self.accumulated_audio = np.append(self.accumulated_audio, mono)

                # Compute dB level meter
                rms = np.sqrt(np.mean(mono**2))
                db = 20 * math.log10(max(1e-5, rms))
                self.last_volume_db = db

                vol_norm = np.clip((db + 60.0) / 60.0, 0.0, 1.0)
                if self.winfo_exists():
                    self.after(0, lambda v=vol_norm, d=db: self._update_vol_meter(v, d))

            with sd.InputStream(samplerate=16000, channels=1, callback=audio_callback):
                while self.is_recording:
                    time.sleep(0.1)

        except Exception as e:
            if self.winfo_exists():
                self.after(0, lambda err=str(e): self.stats_lbl.configure(text=f"❌ Mic Stream Error: {err}"))
            self.stop_live_recording()

    def _update_vol_meter(self, vol_norm, db):
        if not self.winfo_exists():
            return
        try:
            self.vol_progressbar.set(vol_norm)
            self.vol_val_lbl.configure(text=f"{int(db)} dB")
        except Exception:
            pass

    def _transcribe_stream_loop(self):
        """Runs Whisper transcription every 2.5 seconds on accumulated audio buffer."""
        try:
            from ai_modules.speech_helpers.whisper_engine import WhisperTranscriberEngine

            if self.whisper_engine is None:
                self.after(0, lambda: self.stats_lbl.configure(text="⏳ Initializing Whisper AI Model..."))
                self.whisper_engine = WhisperTranscriberEngine()

            while self.is_recording:
                time.sleep(2.5)

                if len(self.accumulated_audio) >= 16000 * 2:  # At least 2 seconds
                    audio_chunk = self.accumulated_audio.copy()
                    self.accumulated_audio = np.array([], dtype=np.float32)

                    model_str = self.model_dropdown.get().split()[0]
                    lang_str = self.lang_dropdown.get()

                    result = self.whisper_engine.transcribe_audio_buffer(
                        audio_chunk,
                        sample_rate=16000,
                        model_size=model_str,
                        language=lang_str
                    )

                    text = result.get("text", "").strip()
                    if text:
                        now_str = datetime.datetime.now().strftime("%H:%M:%S")
                        line_entry = f"[{now_str}] {text}"
                        self.transcript_lines.append(line_entry)
                        self.total_words_spoken += len(text.split())

                        if self.winfo_exists():
                            self.after(0, lambda t=line_entry: self._append_transcript_line(t))

        except Exception as e:
            if self.winfo_exists():
                self.after(0, lambda err=str(e): self.stats_lbl.configure(text=f"❌ Transcription Error: {err}"))

    def _append_transcript_line(self, line_str):
        if not self.winfo_exists():
            return

        self.transcript_box.insert("end", line_str + "\n")
        self.transcript_box.see("end")

        # Analytics calculation
        elapsed_min = max(0.1, (time.time() - (self.start_talk_time or time.time())) / 60.0)
        wpm = int(self.total_words_spoken / elapsed_min)

        model_name = self.model_dropdown.get().split()[0]
        stats = (
            f"• Whisper Model: {model_name}\n"
            f"• Total Words: {self.total_words_spoken} words\n"
            f"• Speaking Speed: {wpm} WPM\n"
            f"• Live Status: Active Transcribing 🎙️"
        )
        self.stats_lbl.configure(text=stats)

    def upload_audio_file(self):
        """Opens file dialog for uploading and transcribing audio files."""
        file_path = filedialog.askopenfilename(
            title="Select Audio File for AI Transcription",
            filetypes=[("Audio Files", "*.wav;*.mp3;*.m4a;*.flac;*.ogg")]
        )
        if not file_path:
            return

        if self.is_recording:
            self.stop_live_recording()

        self.status_badge.configure(text="Processing Audio... ⏳", fg_color=("#FE640B", "#FAB387"))
        threading.Thread(target=self._process_file_thread, args=(file_path,), daemon=True).start()

    def _process_file_thread(self, file_path):
        try:
            from ai_modules.speech_helpers.whisper_engine import WhisperTranscriberEngine

            if self.whisper_engine is None:
                self.whisper_engine = WhisperTranscriberEngine()

            def update_status(text):
                if self.winfo_exists():
                    self.after(0, lambda: self.stats_lbl.configure(text=text))

            model_str = self.model_dropdown.get().split()[0]
            lang_str = self.lang_dropdown.get()

            start_t = time.time()
            result = self.whisper_engine.transcribe_file(
                file_path,
                model_size=model_str,
                language=lang_str,
                progress_callback=update_status
            )
            proc_ms = (time.time() - start_t) * 1000.0

            text = result.get("text", "")
            segments = result.get("segments", [])
            lang_detected = result.get("language", "en")

            self.transcript_lines = []
            for seg in segments:
                start_s = int(seg.get("start", 0))
                m, s = divmod(start_s, 60)
                timestamp_str = f"[{m:02d}:{s:02d}]"
                self.transcript_lines.append(f"{timestamp_str} {seg.get('text', '').strip()}")

            word_count = len(text.split()) if text else 0
            stats_str = (
                f"• Transcribed Audio File\n"
                f"• Language: {lang_detected.upper()}\n"
                f"• Total Words: {word_count} words\n"
                f"• Processing Time: {proc_ms:.0f} ms\n"
                f"• Segments Generated: {len(segments)}"
            )

            if self.winfo_exists():
                self.after(0, lambda: self._on_file_complete(text, stats_str))

        except Exception as e:
            if self.winfo_exists():
                self.after(0, lambda err=str(e): self.stats_lbl.configure(text=f"❌ File Error: {err}"))
        finally:
            if self.winfo_exists():
                self.after(0, lambda: self.status_badge.configure(text="Ready 🎙️", fg_color=("#1E66F5", "#89B4FA")))

    def _on_file_complete(self, full_text, stats_str):
        if not self.winfo_exists():
            return

        self.stats_lbl.configure(text=stats_str)
        self.transcript_box.delete("1.0", "end")

        if self.transcript_lines:
            for line in self.transcript_lines:
                self.transcript_box.insert("end", line + "\n")
        else:
            self.transcript_box.insert("1.0", full_text if full_text else "[No speech recognized in audio file]")

    def clear_transcript(self):
        self.transcript_lines = []
        self.total_words_spoken = 0
        self.transcript_box.delete("1.0", "end")
        self.stats_lbl.configure(text="Transcript cleared.")

    def copy_transcript(self):
        text = self.transcript_box.get("1.0", "end").strip()
        if not text:
            return
        self.clipboard_clear()
        self.clipboard_append(text)
        self.copy_btn.configure(text="Copied! ✓", fg_color=("#40A02B", "#A6E3A1"))
        self.after(2000, lambda: self.copy_btn.configure(text="📋 Copy Text", fg_color=("#1E66F5", "#89B4FA")))

    def export_srt_subtitles(self):
        """Exports transcript as standard .srt subtitle file."""
        text = self.transcript_box.get("1.0", "end").strip()
        if not text:
            return

        file_path = filedialog.asksaveasfilename(
            title="Export Subtitles (.srt)",
            defaultextension=".srt",
            filetypes=[("SRT Subtitles", "*.srt"), ("Text Files", "*.txt")]
        )
        if file_path:
            try:
                lines = text.split("\n")
                with open(file_path, "w", encoding="utf-8") as f:
                    if file_path.endswith(".srt"):
                        for idx, line in enumerate(lines, 1):
                            if line.strip():
                                f.write(f"{idx}\n00:00:{idx:02d},000 --> 00:00:{idx+2:02d},000\n{line.strip()}\n\n")
                    else:
                        f.write(text)

                self.export_btn.configure(text="Exported! ✓", fg_color=("#40A02B", "#A6E3A1"))
                self.after(2000, lambda: self.export_btn.configure(text="💾 Subtitles (.srt)", fg_color=("#40A02B", "#A6E3A1")))
            except Exception as e:
                print(f"Error exporting subtitles: {e}")

    def on_back_click(self):
        self.stop_live_recording()
        self.on_back_callback()
