import os
import time
import datetime
import threading
from tkinter import filedialog
import customtkinter as ctk

# Metadata describing this module for AI PlayKit Hub
PROJECT_INFO = {
    "id": "text_ocr_scanner",
    "title": "AI Text OCR Scanner",
    "description": "Extract text live from any document, photo, or webcam stream with deep learning text recognition & text-to-speech read aloud!",
    "icon": "🔍",
    "category": "Vision & Perception",
    "required_packages": ["easyocr", "pyttsx3", "cv2", "PIL"],
    "install_command": "pip install easyocr pyttsx3 opencv-python pillow",
    "guide": """# 🔍 AI Text OCR Scanner Guide

### Overview
This project uses **Deep Learning Text Recognition (EasyOCR)** and **OpenCV** to scan documents, signs, receipts, or live webcam video streams, extracting text lines and drawing glowing bounding box overlays in real-time!

---

### Step 1: Install Required Packages
Open your terminal and run:
```bash
pip install easyocr pyttsx3 opencv-python pillow
```

---

### Step 2: Key Features
- 📁 **Document & Photo Upload**: Scan any `.jpg`, `.png`, `.webp`, or `.bmp` file from your computer.
- 🎥 **Live Webcam Scanner**: Scan physical documents or signs live in front of your camera.
- 🔲 **Bounding Box Overlays**: Real-time bounding boxes drawn over detected text with confidence scores.
- 🔊 **Read Aloud (TTS)**: Listen to your scanned document read out loud offline with built-in voice synthesis.
- 📋 **One-Click Clipboard Copy**: Instantly copy extracted text to your computer's clipboard.
- 💾 **Export Text File (`.txt`)**: Save scanned text directly to a text file.
- 🌐 **Multi-Language Support**: English, Spanish, French, German, and Hindi text recognition.

---

### Step 3: Beginner Python Code Example
```python
import cv2
import easyocr

# 1. Initialize EasyOCR Reader for English
reader = easyocr.Reader(['en'], gpu=False)

# 2. Read input photo or document
image_path = "document.jpg"
results = reader.readtext(image_path)

# 3. Print extracted text lines
print("--- EXTRACTED TEXT ---")
for bbox, text, confidence in results:
    print(f"[{confidence*100:.1f}%] {text}")
```
"""
}

def check_dependencies():
    """Returns True if cv2 and PIL are installed without importing heavy packages."""
    import importlib.util
    return (
        importlib.util.find_spec("cv2") is not None and
        importlib.util.find_spec("PIL") is not None
    )


class TextOCRScannerUI(ctk.CTkFrame):
    """CustomTkinter UI for AI Text OCR Scanner Studio."""

    def __init__(self, parent, on_back_callback):
        super().__init__(parent, fg_color="transparent")
        self.on_back_callback = on_back_callback

        self.cap = None
        self.is_running_webcam = False
        self.ocr_engine = None

        self.uploaded_image_bgr = None
        self.latest_annotated_bgr = None
        self.extracted_text_content = ""
        self.current_ctk_img = None
        self.is_scanning = False
        self.is_speaking = False

        self.setup_ui()

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
            text="🔍 AI Text OCR Scanner",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=("#1E1E2E", "#F5E0DC")
        )
        title_lbl.pack(side="left", padx=10)

        # Status Badge
        self.status_badge = ctk.CTkLabel(
            header_frame,
            text="Ready 🔍",
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
            text="⚙ Scanner Controls",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=("#1E66F5", "#89B4FA")
        )
        ctrl_title.pack(anchor="w", padx=15, pady=(15, 10))

        # Action Buttons Row (Upload Document / Start Webcam)
        upload_btn = ctk.CTkButton(
            control_panel,
            text="📁 Upload Document/Photo",
            fg_color=("#1E66F5", "#89B4FA"),
            hover_color=("#7287FD", "#B4BEFE"),
            text_color=("#FFFFFF", "#11111B"),
            font=ctk.CTkFont(size=13, weight="bold"),
            height=38,
            command=self.upload_document
        )
        upload_btn.pack(fill="x", padx=15, pady=(5, 5))

        self.toggle_cam_btn = ctk.CTkButton(
            control_panel,
            text="▶ Start Live Webcam Scan",
            fg_color=("#40A02B", "#A6E3A1"),
            hover_color=("#207015", "#94E2D5"),
            text_color=("#FFFFFF", "#11111B"),
            font=ctk.CTkFont(size=13, weight="bold"),
            height=38,
            command=self.toggle_webcam
        )
        self.toggle_cam_btn.pack(fill="x", padx=15, pady=(0, 10))

        # Language Selector
        lang_lbl = ctk.CTkLabel(
            control_panel,
            text="OCR Language:",
            font=ctk.CTkFont(size=12),
            text_color=("#4C4F69", "#CDD6F4")
        )
        lang_lbl.pack(anchor="w", padx=15, pady=(8, 2))

        self.lang_dropdown = ctk.CTkOptionMenu(
            control_panel,
            values=[
                "English (en)",
                "Spanish (es)",
                "French (fr)",
                "German (de)",
                "Hindi (hi)"
            ],
            fg_color=("#EA76CB", "#F5C2E7"),
            text_color=("#FFFFFF", "#11111B"),
            command=self.on_lang_selected
        )
        self.lang_dropdown.pack(fill="x", padx=15, pady=5)

        # Contrast Enhancement Checkbox
        self.contrast_var = ctk.BooleanVar(value=True)
        contrast_chk = ctk.CTkCheckBox(
            control_panel,
            text="Enhance Image Contrast (CLAHE)",
            variable=self.contrast_var,
            text_color=("#4C4F69", "#CDD6F4"),
            command=self.reprocess_uploaded_document
        )
        contrast_chk.pack(anchor="w", padx=15, pady=8)

        # Minimum Confidence Slider
        conf_lbl_frame = ctk.CTkFrame(control_panel, fg_color="transparent")
        conf_lbl_frame.pack(fill="x", padx=15, pady=(10, 2))

        c_lbl = ctk.CTkLabel(
            conf_lbl_frame,
            text="Min Confidence:",
            font=ctk.CTkFont(size=12),
            text_color=("#4C4F69", "#CDD6F4")
        )
        c_lbl.pack(side="left")

        self.conf_val_lbl = ctk.CTkLabel(
            conf_lbl_frame,
            text="20%",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=("#1E66F5", "#89B4FA")
        )
        self.conf_val_lbl.pack(side="right")

        self.conf_slider = ctk.CTkSlider(
            control_panel,
            from_=0.05,
            to=0.85,
            number_of_steps=16,
            command=self.on_conf_change
        )
        self.conf_slider.set(0.20)
        self.conf_slider.pack(fill="x", padx=15, pady=5)

        # Extracted Text Box Header & Controls
        text_panel_hdr = ctk.CTkFrame(control_panel, fg_color="transparent")
        text_panel_hdr.pack(fill="x", padx=15, pady=(14, 4))

        t_title = ctk.CTkLabel(
            text_panel_hdr,
            text="📋 Extracted Text",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=("#1E66F5", "#89B4FA")
        )
        t_title.pack(side="left")

        # Speech TTS Read Aloud Button
        self.tts_btn = ctk.CTkButton(
            text_panel_hdr,
            text="🔊 Read Aloud",
            width=90,
            height=26,
            fg_color=("#DCE0E8", "#313244"),
            hover_color=("#BCC0CC", "#45475A"),
            text_color=("#4C4F69", "#CDD6F4"),
            font=ctk.CTkFont(size=11, weight="bold"),
            command=self.read_aloud
        )
        self.tts_btn.pack(side="right")

        # Scrollable Text Box for Recognized Text Output
        self.text_box = ctk.CTkTextbox(
            control_panel,
            height=140,
            fg_color=("#F2F4F8", "#11111B"),
            text_color=("#1E1E2E", "#CDD6F4"),
            font=ctk.CTkFont(family="Consolas", size=12),
            corner_radius=8
        )
        self.text_box.pack(fill="x", padx=15, pady=5)

        # Copy & Export Action Row
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
            command=self.copy_to_clipboard
        )
        self.copy_btn.pack(side="left", fill="x", expand=True, padx=(0, 5))

        self.export_btn = ctk.CTkButton(
            action_row,
            text="💾 Export .txt",
            fg_color=("#40A02B", "#A6E3A1"),
            hover_color=("#207015", "#94E2D5"),
            text_color=("#FFFFFF", "#11111B"),
            font=ctk.CTkFont(weight="bold"),
            height=32,
            command=self.export_text_file
        )
        self.export_btn.pack(side="right", fill="x", expand=True, padx=(5, 0))

        # Scan Stats Box
        status_box = ctk.CTkFrame(
            control_panel,
            fg_color=("#F2F4F8", "#11111B"),
            corner_radius=10
        )
        status_box.pack(fill="both", expand=True, padx=15, pady=(5, 15))

        stat_title = ctk.CTkLabel(
            status_box,
            text="📊 OCR Analytics",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=("#1E66F5", "#89B4FA")
        )
        stat_title.pack(anchor="w", padx=12, pady=(10, 4))

        self.stats_lbl = ctk.CTkLabel(
            status_box,
            text="Upload a document/photo or start live camera scan.",
            font=ctk.CTkFont(size=11),
            text_color=("#5C5F77", "#A6ADC8"),
            justify="left",
            wraplength=250
        )
        self.stats_lbl.pack(anchor="w", padx=12, pady=(0, 10))

        # Right Live View Container
        self.image_container = ctk.CTkFrame(
            main_content,
            fg_color=("#F2F4F8", "#181825"),
            corner_radius=12
        )
        self.image_container.pack(side="right", fill="both", expand=True)

        self.image_lbl = ctk.CTkLabel(
            self.image_container,
            text="🔍 AI Text OCR Scanner\nClick '📁 Upload Document/Photo' or '▶ Start Live Webcam Scan' to scan text live!",
            font=ctk.CTkFont(size=15),
            text_color=("#5C5F77", "#A6ADC8"),
            justify="center",
            wraplength=540
        )
        self.image_lbl.pack(expand=True, padx=20, pady=20)

    def on_conf_change(self, val):
        self.conf_val_lbl.configure(text=f"{int(val * 100)}%")
        if self.uploaded_image_bgr is not None and not self.is_running_webcam:
            self.reprocess_uploaded_document()

    def on_lang_selected(self, selected_lang):
        if self.uploaded_image_bgr is not None and not self.is_running_webcam:
            self.reprocess_uploaded_document()

    def upload_document(self):
        """Opens file dialog for uploading any photo/document."""
        file_path = filedialog.askopenfilename(
            title="Select Document/Photo for Text OCR Scanning",
            filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.webp;*.bmp")]
        )
        if not file_path:
            return

        try:
            import cv2
            if self.is_running_webcam:
                self.stop_webcam()

            img = cv2.imread(file_path)
            if img is None:
                self.stats_lbl.configure(text="❌ Error loading selected document file.")
                return

            self.uploaded_image_bgr = img
            self.reprocess_uploaded_document()
        except Exception as e:
            self.stats_lbl.configure(text=f"❌ Document Upload Error: {e}")

    def reprocess_uploaded_document(self):
        """Applies OCR text recognition to currently uploaded document in a background thread."""
        if self.uploaded_image_bgr is None or self.is_scanning:
            return

        self.is_scanning = True
        self.status_badge.configure(text="Scanning Text... ⏳", fg_color=("#FE640B", "#FAB387"))

        threading.Thread(target=self._process_uploaded_thread, daemon=True).start()

    def _process_uploaded_thread(self):
        try:
            import cv2
            from ai_modules.ocr_helpers.ocr_engine import OCREngine

            if self.ocr_engine is None:
                self.ocr_engine = OCREngine()

            lang_str = self.lang_dropdown.get()
            lang_code = lang_str.split("(")[-1].replace(")", "").strip()
            min_conf = self.conf_slider.get()
            enhance = self.contrast_var.get()

            def update_status(text):
                if self.winfo_exists():
                    self.after(0, lambda: self.stats_lbl.configure(text=text))

            start_t = time.time()
            annotated_bgr, text_content, detections = self.ocr_engine.process_image(
                self.uploaded_image_bgr,
                lang_code=lang_code,
                contrast_enhance=enhance,
                min_confidence=min_conf,
                progress_callback=update_status
            )
            proc_ms = (time.time() - start_t) * 1000.0

            self.latest_annotated_bgr = annotated_bgr
            self.extracted_text_content = text_content

            word_count = len(text_content.split()) if text_content.strip() else 0
            char_count = len(text_content)
            box_count = len(detections)
            avg_conf = (sum(d['confidence'] for d in detections) / max(1, len(detections))) * 100.0

            h, w = self.uploaded_image_bgr.shape[:2]
            stats_str = (
                f"• Text Blocks Detected: {box_count}\n"
                f"• Word Count: {word_count} words\n"
                f"• Character Count: {char_count} chars\n"
                f"• Average Confidence: {avg_conf:.1f}%\n"
                f"• Image Dimensions: {w}x{h} px\n"
                f"• Scan Time: {proc_ms:.0f} ms"
            )

            if self.winfo_exists():
                self.after(0, lambda: self._on_scan_complete(text_content, stats_str))

        except Exception as e:
            if self.winfo_exists():
                self.after(0, lambda err=str(e): self._on_cam_error(f"OCR Error: {err}"))
        finally:
            self.is_scanning = False

    def _on_scan_complete(self, text_content, stats_str):
        if not self.winfo_exists():
            return

        self.status_badge.configure(text="Scanned ✓", fg_color=("#40A02B", "#A6E3A1"))
        self.stats_lbl.configure(text=stats_str)

        self.text_box.delete("1.0", "end")
        if text_content.strip():
            self.text_box.insert("1.0", text_content)
        else:
            self.text_box.insert("1.0", "[No text detected with current threshold settings]")

        self._render_output_view()

    def _render_output_view(self):
        if self.latest_annotated_bgr is None or not self.winfo_exists():
            return

        import cv2
        from PIL import Image

        rgb = cv2.cvtColor(self.latest_annotated_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        c_w = self.image_container.winfo_width()
        c_h = self.image_container.winfo_height()

        if c_w < 100 or c_h < 100:
            target_w, target_h = 640, 480
        else:
            img_w, img_h = pil_img.size
            aspect = img_w / img_h
            target_w = max(150, c_w - 30)
            target_h = int(target_w / aspect)

            if target_h > c_h - 30:
                target_h = max(150, c_h - 30)
                target_w = int(target_h * aspect)

        ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(target_w, target_h))
        self.current_ctk_img = ctk_img
        try:
            self.image_lbl.configure(text="", image=self.current_ctk_img)
        except Exception:
            pass

    def copy_to_clipboard(self):
        text = self.text_box.get("1.0", "end").strip()
        if not text:
            return
        self.clipboard_clear()
        self.clipboard_append(text)
        self.copy_btn.configure(text="Copied! ✓", fg_color=("#40A02B", "#A6E3A1"))
        self.after(2000, lambda: self.copy_btn.configure(text="📋 Copy Text", fg_color=("#1E66F5", "#89B4FA")))

    def export_text_file(self):
        text = self.text_box.get("1.0", "end").strip()
        if not text:
            return

        file_path = filedialog.asksaveasfilename(
            title="Save Extracted Text File",
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        if file_path:
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(text)
                self.export_btn.configure(text="Exported! ✓", fg_color=("#40A02B", "#A6E3A1"))
                self.after(2000, lambda: self.export_btn.configure(text="💾 Export .txt", fg_color=("#40A02B", "#A6E3A1")))
            except Exception as e:
                print(f"Error saving text file: {e}")

    def read_aloud(self):
        """Toggles offline speech synthesis read aloud."""
        if self.is_speaking:
            self.stop_speech()
            return

        text = self.text_box.get("1.0", "end").strip()
        if not text or text.startswith("[No text"):
            return

        if self.ocr_engine is None:
            from ai_modules.ocr_helpers.ocr_engine import OCREngine
            self.ocr_engine = OCREngine()

        self.is_speaking = True
        self.tts_btn.configure(
            text="⏹ Stop Speech",
            fg_color=("#FE640B", "#F38BA8"),
            hover_color=("#D20F39", "#E64553"),
            text_color=("#FFFFFF", "#11111B")
        )

        def _on_speech_finished():
            if self.winfo_exists():
                self.after(0, self._reset_tts_button)

        self.ocr_engine.speak_text(text, on_finished_callback=_on_speech_finished)

    def stop_speech(self):
        """Interrupts and stops active speech synthesis immediately."""
        if self.ocr_engine:
            self.ocr_engine.stop_speaking()
        self._reset_tts_button()

    def _reset_tts_button(self):
        self.is_speaking = False
        if self.winfo_exists():
            try:
                self.tts_btn.configure(
                    text="🔊 Read Aloud",
                    fg_color=("#DCE0E8", "#313244"),
                    hover_color=("#BCC0CC", "#45475A"),
                    text_color=("#4C4F69", "#CDD6F4")
                )
            except Exception:
                pass

    def toggle_webcam(self):
        if self.is_running_webcam:
            self.stop_webcam()
        else:
            self.start_webcam()

    def start_webcam(self):
        self.uploaded_image_bgr = None
        self.is_running_webcam = True
        self.toggle_cam_btn.configure(
            text="⏹ Stop Live Webcam Scan",
            fg_color=("#FE640B", "#F38BA8"),
            hover_color=("#D20F39", "#E64553"),
            text_color=("#FFFFFF", "#11111B")
        )
        self.status_badge.configure(
            text="Webcam Scanning 🔴",
            fg_color=("#40A02B", "#A6E3A1"),
            text_color=("#FFFFFF", "#11111B")
        )

        threading.Thread(target=self._webcam_loop, daemon=True).start()

    def stop_webcam(self):
        self.stop_speech()
        self.is_running_webcam = False
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        self.toggle_cam_btn.configure(
            text="▶ Start Live Webcam Scan",
            fg_color=("#40A02B", "#A6E3A1"),
            hover_color=("#207015", "#94E2D5"),
            text_color=("#FFFFFF", "#11111B")
        )
        self.status_badge.configure(
            text="Ready 🔍",
            fg_color=("#1E66F5", "#89B4FA"),
            text_color=("#FFFFFF", "#11111B")
        )
        if self.uploaded_image_bgr is None:
            try:
                self.image_lbl.configure(
                    text="🔍 AI Text OCR Scanner\nClick '📁 Upload Document/Photo' or '▶ Start Live Webcam Scan' to scan text live!",
                    image=None
                )
            except Exception:
                pass
            self.stats_lbl.configure(text="Webcam stopped.")

    def on_back_click(self):
        self.stop_speech()
        self.stop_webcam()
        self.on_back_callback()

    def _webcam_loop(self):
        try:
            import cv2
            from ai_modules.ocr_helpers.ocr_engine import OCREngine

            if self.ocr_engine is None:
                self.after(0, lambda: self.stats_lbl.configure(text="⏳ Initializing AI OCR Engine..."))
                self.ocr_engine = OCREngine()

            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.after(0, lambda: self._on_cam_error("Could not open webcam (Camera index 0). Please check camera permissions."))
                return

            last_time = time.time()
            fps = 30.0

            while self.is_running_webcam and self.cap.isOpened():
                start_proc_time = time.time()
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    break

                # Flip horizontally for natural mirror view
                frame = cv2.flip(frame, 1)

                lang_str = self.lang_dropdown.get()
                lang_code = lang_str.split("(")[-1].replace(")", "").strip()
                min_conf = self.conf_slider.get()
                enhance = self.contrast_var.get()

                # Process webcam frame at 640 size for smooth real-time scanning
                annotated_bgr, text_content, detections = self.ocr_engine.process_image(
                    frame,
                    lang_code=lang_code,
                    contrast_enhance=enhance,
                    min_confidence=min_conf,
                    target_size=640
                )

                self.latest_annotated_bgr = annotated_bgr
                self.uploaded_image_bgr = frame
                self.extracted_text_content = text_content

                # Performance Stats
                proc_ms = (time.time() - start_proc_time) * 1000.0
                curr_time = time.time()
                fps = 0.9 * fps + 0.1 * (1.0 / max(0.001, curr_time - last_time))
                last_time = curr_time

                word_count = len(text_content.split()) if text_content.strip() else 0
                stats_str = (
                    f"• Live Camera Stream\n"
                    f"• Words Detected: {word_count}\n"
                    f"• Processing Speed: {proc_ms:.0f} ms\n"
                    f"• Performance: {int(fps)} FPS"
                )

                if self.is_running_webcam and self.winfo_exists():
                    self.after(0, lambda text=stats_str, txt=text_content: self._update_webcam_ui(text, txt))

                time.sleep(0.05)

        except Exception as e:
            if self.is_running_webcam and self.winfo_exists():
                self.after(0, lambda err=str(e): self._on_cam_error(f"Webcam OCR Error: {err}"))
        finally:
            if self.cap:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None

    def _update_webcam_ui(self, stats_text, text_content):
        if not self.is_running_webcam or not self.winfo_exists():
            return
        try:
            self.stats_lbl.configure(text=stats_text)
            self.text_box.delete("1.0", "end")
            if text_content.strip():
                self.text_box.insert("1.0", text_content)
            else:
                self.text_box.insert("1.0", "[Scanning for text... Hold document steady in camera field]")
        except Exception:
            pass
        self._render_output_view()

    def _on_cam_error(self, err_msg):
        if not self.winfo_exists():
            return
        self.stop_webcam()
        clean_msg = str(err_msg).replace("\t", " ").strip()
        if len(clean_msg) > 280:
            clean_msg = clean_msg[:280] + "..."
        try:
            self.image_lbl.configure(text=f"❌ OCR Scanner Error:\n\n{clean_msg}", image=None)
        except Exception:
            pass
