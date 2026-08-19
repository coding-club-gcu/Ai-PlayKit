import os
import time
import threading
from tkinter import filedialog
import customtkinter as ctk

# Metadata describing this module for AI PlayKit Hub
PROJECT_INFO = {
    "id": "document_summarizer",
    "title": "AI Document Summarizer",
    "description": "100% offline NLP engine to summarize long articles, PDFs, Word docs, code, and text into key bullet points & sentiment analytics!",
    "icon": "📄",
    "category": "NLP & Language AI",
    "required_packages": ["pypdf", "docx", "nltk", "textblob", "PIL"],
    "install_command": "pip install pypdf python-docx nltk textblob pillow",
    "guide": """# 📄 AI Document Summarizer Guide

### Overview
This project uses an **Offline Natural Language Processing (NLP) Engine** based on **TextRank** and **TF-IDF Term Scoring** to read documents, PDFs, or articles and automatically generate concise bulleted summaries, sentiment analytics, and key term extractions!

---

### Step 1: Install Required Packages
Open your terminal and run:
```bash
pip install pypdf python-docx nltk textblob pillow
```

---

### Step 2: Key Features
- 📁 **Multi-Format Support**: Upload `.pdf`, `.docx`, `.txt`, `.md`, `.py`, `.json`, `.csv` files.
- ⚡ **100% Offline & Private**: Zero API keys or cloud servers required. Works entirely locally on your machine.
- 🎚️ **Compression Ratio Slider**: Choose summary detail from 10% (Ultra Concise) to 50% (Detailed Summary).
- 📌 **Multiple Summary Modes**: Bullet Points, Key Takeaways list, or Executive Paragraphs.
- 😊 **NLP Sentiment Gauge**: Detects document tone (Positive 😊 / Neutral 😐 / Negative 😟) and subjectivity score.
- 🏷️ **Top Key Terms Extractor**: Extracts top 10 most important terms with TF-IDF ranks.
- 🔊 **Read Summary Aloud**: Listen to generated summaries using built-in offline Text-to-Speech synthesis.

---

### Step 3: Beginner Python Code Example
```python
from ai_modules.nlp_helpers.summarizer_engine import NLPSummarizerEngine

# 1. Initialize NLP Engine
nlp = NLPSummarizerEngine()

# 2. Extract text from PDF document
document_text = nlp.extract_text_from_file("report.pdf")

# 3. Generate 25% summary & sentiment analysis
summary, keywords = nlp.summarize(document_text, ratio=0.25, mode="bullet")
sentiment_data = nlp.analyze_sentiment(document_text)

print("--- AI SUMMARY ---")
print(summary)
print(f"Sentiment: {sentiment_data['sentiment']} {sentiment_data['icon']}")
```
"""
}

def check_dependencies():
    """Returns True if pypdf or PyPDF2 or docx are installed without importing heavy packages."""
    import importlib.util
    return (
        importlib.util.find_spec("pypdf") is not None or
        importlib.util.find_spec("PyPDF2") is not None
    )


class DocumentSummarizerUI(ctk.CTkFrame):
    """CustomTkinter UI for AI Document Summarizer Studio."""

    def __init__(self, parent, on_back_callback):
        super().__init__(parent, fg_color="transparent")
        self.on_back_callback = on_back_callback

        self.nlp_engine = None
        self.is_processing = False

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
            text="📄 AI Document Summarizer",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=("#1E1E2E", "#F5E0DC")
        )
        title_lbl.pack(side="left", padx=10)

        # Status Badge
        self.status_badge = ctk.CTkLabel(
            header_frame,
            text="Ready 📄",
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
            text="⚙ NLP Controls",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=("#1E66F5", "#89B4FA")
        )
        ctrl_title.pack(anchor="w", padx=15, pady=(15, 10))

        # Main Action Buttons (Upload File / Generate Summary)
        upload_btn = ctk.CTkButton(
            control_panel,
            text="📁 Upload Document (PDF/DOCX/TXT)",
            fg_color=("#1E66F5", "#89B4FA"),
            hover_color=("#7287FD", "#B4BEFE"),
            text_color=("#FFFFFF", "#11111B"),
            font=ctk.CTkFont(size=13, weight="bold"),
            height=38,
            command=self.upload_document
        )
        upload_btn.pack(fill="x", padx=15, pady=(5, 5))

        self.summarize_btn = ctk.CTkButton(
            control_panel,
            text="⚡ Generate AI Summary",
            fg_color=("#40A02B", "#A6E3A1"),
            hover_color=("#207015", "#94E2D5"),
            text_color=("#FFFFFF", "#11111B"),
            font=ctk.CTkFont(size=13, weight="bold"),
            height=38,
            command=self.generate_summary
        )
        self.summarize_btn.pack(fill="x", padx=15, pady=(0, 10))



        # Compression Ratio Slider
        ratio_hdr = ctk.CTkFrame(control_panel, fg_color="transparent")
        ratio_hdr.pack(fill="x", padx=15, pady=(10, 2))

        r_lbl = ctk.CTkLabel(
            ratio_hdr,
            text="Compression Detail:",
            font=ctk.CTkFont(size=12),
            text_color=("#4C4F69", "#CDD6F4")
        )
        r_lbl.pack(side="left")

        self.ratio_val_lbl = ctk.CTkLabel(
            ratio_hdr,
            text="25%",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=("#1E66F5", "#89B4FA")
        )
        self.ratio_val_lbl.pack(side="right")

        self.ratio_slider = ctk.CTkSlider(
            control_panel,
            from_=0.10,
            to=0.50,
            number_of_steps=8,
            command=self.on_ratio_change
        )
        self.ratio_slider.set(0.25)
        self.ratio_slider.pack(fill="x", padx=15, pady=5)

        # Sentiment Analysis Card
        sent_card = ctk.CTkFrame(
            control_panel,
            fg_color=("#F2F4F8", "#11111B"),
            corner_radius=10
        )
        sent_card.pack(fill="x", padx=15, pady=(12, 10))

        s_title = ctk.CTkLabel(
            sent_card,
            text="😊 Sentiment Analytics",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=("#1E66F5", "#89B4FA")
        )
        s_title.pack(anchor="w", padx=12, pady=(10, 4))

        self.sent_val_lbl = ctk.CTkLabel(
            sent_card,
            text="Mood: Neutral 😐\nPolarity: 0.00\nSubjectivity: 0.50",
            font=ctk.CTkFont(size=11),
            text_color=("#5C5F77", "#A6ADC8"),
            justify="left"
        )
        self.sent_val_lbl.pack(anchor="w", padx=12, pady=(0, 10))

        # AI Auto-Tags Card
        tag_card = ctk.CTkFrame(
            control_panel,
            fg_color=("#F2F4F8", "#11111B"),
            corner_radius=10
        )
        tag_card.pack(fill="x", padx=15, pady=(0, 10))

        t_title = ctk.CTkLabel(
            tag_card,
            text="🏷️ AI Auto-Tags & Category",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=("#8839EF", "#CBA6F7")
        )
        t_title.pack(anchor="w", padx=12, pady=(10, 4))

        self.tag_val_lbl = ctk.CTkLabel(
            tag_card,
            text="Category: 📄 General Document\nTags: #Document #Text",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=("#1E66F5", "#89B4FA"),
            justify="left",
            wraplength=250
        )
        self.tag_val_lbl.pack(anchor="w", padx=12, pady=(0, 10))

        # Top Key Terms Box
        kw_card = ctk.CTkFrame(
            control_panel,
            fg_color=("#F2F4F8", "#11111B"),
            corner_radius=10
        )
        kw_card.pack(fill="x", padx=15, pady=(0, 10))

        kw_title = ctk.CTkLabel(
            kw_card,
            text="🔑 Top Extracted Keywords",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=("#1E66F5", "#89B4FA")
        )
        kw_title.pack(anchor="w", padx=12, pady=(10, 4))

        self.kw_val_lbl = ctk.CTkLabel(
            kw_card,
            text="Upload or paste text to extract keywords.",
            font=ctk.CTkFont(size=11),
            text_color=("#5C5F77", "#A6ADC8"),
            justify="left",
            wraplength=250
        )
        self.kw_val_lbl.pack(anchor="w", padx=12, pady=(0, 10))

        # Document Stats Card
        doc_stats_box = ctk.CTkFrame(
            control_panel,
            fg_color=("#F2F4F8", "#11111B"),
            corner_radius=10
        )
        doc_stats_box.pack(fill="both", expand=True, padx=15, pady=(0, 15))

        d_title = ctk.CTkLabel(
            doc_stats_box,
            text="📊 Document Stats",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=("#1E66F5", "#89B4FA")
        )
        d_title.pack(anchor="w", padx=12, pady=(10, 4))

        self.stats_lbl = ctk.CTkLabel(
            doc_stats_box,
            text="Upload a document or paste text to summarize.",
            font=ctk.CTkFont(size=11),
            text_color=("#5C5F77", "#A6ADC8"),
            justify="left",
            wraplength=250
        )
        self.stats_lbl.pack(anchor="w", padx=12, pady=(0, 10))

        # Right Text Editors Container
        right_container = ctk.CTkFrame(main_content, fg_color="transparent")
        right_container.pack(side="right", fill="both", expand=True)

        # Top Container: Source Document Text
        src_frame = ctk.CTkFrame(
            right_container,
            fg_color=("#FFFFFF", "#1E1E2E"),
            corner_radius=12
        )
        src_frame.pack(fill="both", expand=True, pady=(0, 5))

        src_hdr = ctk.CTkFrame(src_frame, fg_color="transparent")
        src_hdr.pack(fill="x", padx=15, pady=(10, 5))

        src_lbl = ctk.CTkLabel(
            src_hdr,
            text="📥 Source Document Text (Paste or Upload)",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=("#1E66F5", "#89B4FA")
        )
        src_lbl.pack(side="left")

        clear_src_btn = ctk.CTkButton(
            src_hdr,
            text="🔄 Clear Text",
            width=80,
            height=26,
            fg_color=("#DCE0E8", "#313244"),
            hover_color=("#BCC0CC", "#45475A"),
            text_color=("#4C4F69", "#CDD6F4"),
            font=ctk.CTkFont(size=11),
            command=self.clear_source_text
        )
        clear_src_btn.pack(side="right")

        self.src_textbox = ctk.CTkTextbox(
            src_frame,
            fg_color=("#F2F4F8", "#11111B"),
            text_color=("#1E1E2E", "#CDD6F4"),
            font=ctk.CTkFont(family="Consolas", size=12),
            corner_radius=8
        )
        self.src_textbox.pack(fill="both", expand=True, padx=15, pady=(0, 12))
        self.src_textbox.insert("1.0", "Paste any document, article, PDF text, or code here to generate an instant AI summary!")

        # Bottom Container: AI Generated Summary
        sum_frame = ctk.CTkFrame(
            right_container,
            fg_color=("#FFFFFF", "#1E1E2E"),
            corner_radius=12
        )
        sum_frame.pack(fill="both", expand=True, pady=(5, 0))

        sum_hdr = ctk.CTkFrame(sum_frame, fg_color="transparent")
        sum_hdr.pack(fill="x", padx=15, pady=(10, 5))

        sum_lbl = ctk.CTkLabel(
            sum_hdr,
            text="📌 Generated AI Summary",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=("#40A02B", "#A6E3A1")
        )
        sum_lbl.pack(side="left")

        # Speech Read Aloud Button
        self.tts_btn = ctk.CTkButton(
            sum_hdr,
            text="🔊 Read Aloud",
            width=90,
            height=26,
            fg_color=("#DCE0E8", "#313244"),
            hover_color=("#BCC0CC", "#45475A"),
            text_color=("#4C4F69", "#CDD6F4"),
            font=ctk.CTkFont(size=11, weight="bold"),
            command=self.read_summary_aloud
        )
        self.tts_btn.pack(side="right", padx=(5, 0))

        copy_sum_btn = ctk.CTkButton(
            sum_hdr,
            text="📋 Copy Summary",
            width=110,
            height=26,
            fg_color=("#1E66F5", "#89B4FA"),
            hover_color=("#7287FD", "#B4BEFE"),
            text_color=("#FFFFFF", "#11111B"),
            font=ctk.CTkFont(size=11, weight="bold"),
            command=self.copy_summary
        )
        copy_sum_btn.pack(side="right", padx=5)

        export_sum_btn = ctk.CTkButton(
            sum_hdr,
            text="💾 Save (.txt)",
            width=90,
            height=26,
            fg_color=("#40A02B", "#A6E3A1"),
            hover_color=("#207015", "#94E2D5"),
            text_color=("#FFFFFF", "#11111B"),
            font=ctk.CTkFont(size=11, weight="bold"),
            command=self.export_summary
        )
        export_sum_btn.pack(side="right")

        self.sum_textbox = ctk.CTkTextbox(
            sum_frame,
            fg_color=("#F2F4F8", "#11111B"),
            text_color=("#1E1E2E", "#CDD6F4"),
            font=ctk.CTkFont(family="Consolas", size=12),
            corner_radius=8
        )
        self.sum_textbox.pack(fill="both", expand=True, padx=15, pady=(0, 12))
        self.sum_textbox.insert("1.0", "[Click '⚡ Generate AI Summary' to view results]")

    def on_ratio_change(self, val):
        self.ratio_val_lbl.configure(text=f"{int(val * 100)}%")
        self.generate_summary()

    def upload_document(self):
        file_path = filedialog.askopenfilename(
            title="Select Document File to Summarize",
            filetypes=[
                ("Supported Documents", "*.pdf;*.docx;*.txt;*.md;*.py;*.json;*.csv"),
                ("PDF Documents", "*.pdf"),
                ("Word Documents", "*.docx"),
                ("Text Files", "*.txt;*.md"),
                ("All Files", "*.*")
            ]
        )
        if not file_path:
            return

        self.status_badge.configure(text="Reading File... ⏳", fg_color=("#FE640B", "#FAB387"))
        threading.Thread(target=self._read_file_thread, args=(file_path,), daemon=True).start()

    def _read_file_thread(self, file_path):
        try:
            from ai_modules.nlp_helpers.summarizer_engine import NLPSummarizerEngine
            if self.nlp_engine is None:
                self.nlp_engine = NLPSummarizerEngine()

            text = self.nlp_engine.extract_text_from_file(file_path)
            if self.winfo_exists():
                self.after(0, lambda: self._on_file_read_success(text, file_path))
        except Exception as e:
            if self.winfo_exists():
                self.after(0, lambda err=str(e): self.stats_lbl.configure(text=f"❌ Error reading file: {err}"))

    def _on_file_read_success(self, text, file_path):
        if not self.winfo_exists():
            return
        self.src_textbox.delete("1.0", "end")
        if text.strip():
            self.src_textbox.insert("1.0", text)
            self.generate_summary()
        else:
            self.src_textbox.insert("1.0", f"[Could not extract text from {os.path.basename(file_path)}]")
            self.status_badge.configure(text="Ready 📄", fg_color=("#1E66F5", "#89B4FA"))

    def generate_summary(self):
        src_text = self.src_textbox.get("1.0", "end").strip()
        if not src_text or src_text.startswith("Paste any document"):
            return

        if self.is_processing:
            return

        self.is_processing = True
        self.status_badge.configure(text="Summarizing... ⏳", fg_color=("#FE640B", "#FAB387"))
        threading.Thread(target=self._summary_thread, args=(src_text,), daemon=True).start()

    def _summary_thread(self, text):
        try:
            from ai_modules.nlp_helpers.summarizer_engine import NLPSummarizerEngine
            if self.nlp_engine is None:
                self.nlp_engine = NLPSummarizerEngine()

            mode_key = "bullet"

            ratio = self.ratio_slider.get()

            start_t = time.time()
            summary_out, keywords = self.nlp_engine.summarize(text, ratio=ratio, mode=mode_key)
            sentiment_data = self.nlp_engine.analyze_sentiment(text)
            tag_data = self.nlp_engine.generate_auto_tags(text)
            proc_ms = (time.time() - start_t) * 1000.0

            words = text.split()
            word_cnt = len(words)
            char_cnt = len(text)
            read_time_min = max(1, int(word_cnt / 200))  # avg reading speed 200 WPM

            stats_str = (
                f"• Original Words: {word_cnt} words\n"
                f"• Character Count: {char_cnt} chars\n"
                f"• Est. Reading Time: ~{read_time_min} min\n"
                f"• Summarize Time: {proc_ms:.0f} ms"
            )

            if self.winfo_exists():
                self.after(0, lambda: self._on_summary_complete(summary_out, keywords, sentiment_data, tag_data, stats_str))

        except Exception as e:
            if self.winfo_exists():
                self.after(0, lambda err=str(e): self.stats_lbl.configure(text=f"❌ Summarize Error: {err}"))
        finally:
            self.is_processing = False

    def _on_summary_complete(self, summary_out, keywords, sentiment_data, tag_data, stats_str):
        if not self.winfo_exists():
            return

        self.status_badge.configure(text="Summarized ✓", fg_color=("#40A02B", "#A6E3A1"))
        self.stats_lbl.configure(text=stats_str)

        # Update Sentiment Card
        s_text = (
            f"Mood: {sentiment_data['sentiment']} {sentiment_data['icon']}\n"
            f"Polarity Score: {sentiment_data['score']:+.2f}\n"
            f"Subjectivity: {sentiment_data['subjectivity']:.2f}"
        )
        self.sent_val_lbl.configure(text=s_text)

        # Update Auto-Tags Card
        tags_str = " ".join(tag_data.get('tags', []))
        t_text = (
            f"Category: {tag_data.get('category', '📄 General')}\n"
            f"Tags: {tags_str if tags_str else '#Document'}"
        )
        self.tag_val_lbl.configure(text=t_text)

        # Update Keywords Card
        kw_str = ", ".join(f"{w} ({cnt})" for w, cnt in keywords)
        self.kw_val_lbl.configure(text=kw_str if kw_str else "No keywords found.")

        # Update Summary Textbox
        self.sum_textbox.delete("1.0", "end")
        self.sum_textbox.insert("1.0", summary_out)

    def clear_source_text(self):
        self.src_textbox.delete("1.0", "end")
        self.sum_textbox.delete("1.0", "end")
        self.kw_val_lbl.configure(text="Upload or paste text to extract keywords.")
        self.tag_val_lbl.configure(text="Category: 📄 General Document\nTags: #Document #Text")
        self.sent_val_lbl.configure(text="Mood: Neutral 😐\nPolarity: 0.00\nSubjectivity: 0.50")
        self.stats_lbl.configure(text="Cleared source text.")

    def copy_summary(self):
        text = self.sum_textbox.get("1.0", "end").strip()
        if not text:
            return
        self.clipboard_clear()
        self.clipboard_append(text)

    def export_summary(self):
        text = self.sum_textbox.get("1.0", "end").strip()
        if not text:
            return

        file_path = filedialog.asksaveasfilename(
            title="Save AI Summary File",
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt"), ("Markdown", "*.md")]
        )
        if file_path:
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(text)
            except Exception as e:
                print(f"Error saving summary file: {e}")

    def read_summary_aloud(self):
        text = self.sum_textbox.get("1.0", "end").strip()
        if not text or text.startswith("[Click"):
            return

        try:
            import pyttsx3
            def _tts_thread():
                try:
                    engine = pyttsx3.init()
                    engine.setProperty('rate', 160)
                    engine.say(text[:800])
                    engine.runAndWait()
                except Exception as ex:
                    print(f"TTS Thread Error: {ex}")

            threading.Thread(target=_tts_thread, daemon=True).start()
        except Exception as e:
            print(f"TTS Exception: {e}")

    def on_back_click(self):
        self.on_back_callback()
