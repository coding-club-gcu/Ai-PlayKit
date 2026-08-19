import os
import sys
import time
import gc
import re
import threading
import customtkinter as ctk

# Metadata describing this module for AI PlayKit Hub
PROJECT_INFO = {
    "id": "huggingface_llm",
    "title": "Local LLM Studio",
    "description": "Download, load, and chat with local Hugging Face LLMs (e.g. Qwen, TinyLlama, SmolLM) with 100% offline privacy and zero API costs!",
    "icon": "🤗",
    "category": "Local AI & LLMs",
    "required_packages": ["transformers", "torch", "huggingface_hub"],
    "install_command": "pip install transformers torch huggingface_hub",
    "guide": """# 🤗 Hugging Face Local LLM Studio Guide

### Overview
Run state-of-the-art Open Source Language Models (LLMs) locally on your device! You can paste any Hugging Face model link or Repo ID, download model weights directly, load them into GPU or CPU memory, and chat with total offline privacy.

---

### Step 1: Install Required Libraries
Open your terminal and run:
```bash
pip install transformers torch huggingface_hub
```

---

### Step 2: Key Features
- 🔗 **Paste Any Hugging Face Link or Model ID**: Simply paste a URL like `https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct` or a Repo ID like `Qwen/Qwen2.5-0.5B-Instruct`.
- ⚡ **Preset Quick Pick**: 1-click select lightweight models like Qwen 0.5B, SmolLM2 360M, or TinyLlama 1.1B.
- 🚀 **Hardware Acceleration**: Automatic GPU (NVIDIA CUDA) or CPU execution with custom precision (`float16`, `float32`, `bfloat16`).
- 💬 **Real-Time Token Streaming**: Watch responses stream word-by-word into the chat window.
- ⚙️ **Hyperparameter Control**: Adjust Max New Tokens, Temperature, Top-P, and System Prompt.
- 🔒 **100% Private & Offline**: No cloud API keys or external servers required once model weights are downloaded.

---

### Step 3: Beginner Python Code Example
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 1. Specify model repository ID
model_id = "Qwen/Qwen2.5-0.5B-Instruct"

# 2. Download and load tokenizer & model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

# 3. Format prompt with Chat Template
messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "Explain gravity in 2 simple sentences."}
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# 4. Generate response
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.7)
response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

print(response)
```
"""
}

def check_dependencies():
    """Returns True if required python packages are installed."""
    import importlib.util
    return (
        importlib.util.find_spec("transformers") is not None and
        importlib.util.find_spec("torch") is not None
    )


def extract_hf_repo_id(url_or_id: str) -> str:
    """Extracts clean HuggingFace model repo ID from full URL or raw string."""
    s = url_or_id.strip()
    if not s:
        return ""
    # Handle full HuggingFace URLs (e.g. https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/tree/main)
    if "huggingface.co/" in s:
        s = s.split("huggingface.co/")[-1]
    s = s.split("/tree/")[0].split("/blob/")[0].split("?")[0].strip("/")
    parts = s.split("/")
    if len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}"
    return s


# Popular, lightweight preset open-source LLMs suitable for quick local testing
PRESET_MODELS = {
    "Qwen 2.5 0.5B Instruct (Ultra Fast ~1GB)": "Qwen/Qwen2.5-0.5B-Instruct",
    "SmolLM2 360M Instruct (Tiny ~700MB)": "HuggingFaceTB/SmolLM2-360M-Instruct",
    "TinyLlama 1.1B Chat (Fast ~2.2GB)": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "Phi-3 Mini 4k Instruct (~7GB)": "microsoft/Phi-3-mini-404k-instruct",
    "Llama 3.2 1B Instruct (~2.4GB)": "meta-llama/Llama-3.2-1B-Instruct",
}


# Dedicated local models cache directory inside AI PlayKit folder
MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "huggingface"))
os.makedirs(MODELS_DIR, exist_ok=True)
os.environ["HF_HOME"] = MODELS_DIR
os.environ["TRANSFORMERS_CACHE"] = MODELS_DIR


class HuggingFaceLLMEngine:
    """Backend engine for downloading, loading, and generating responses from HF Local LLMs."""

    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.current_model_id = ""
        self.device = "cpu"
        self.torch_dtype = None
        self.is_loading = False
        self.is_generating = False
        self._stop_requested = False

    def load_model(self, model_id, device_choice="Auto", precision_choice="float16", status_callback=None):
        """Downloads & loads model and tokenizer in a background thread into local project directory."""
        self.is_loading = True
        self._stop_requested = False

        def _log(msg):
            if status_callback:
                status_callback(msg)

        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM

            repo_id = extract_hf_repo_id(model_id)
            if not repo_id:
                raise ValueError("Invalid Hugging Face model ID or link provided.")

            _log(f"⏳ Connecting to Hugging Face Hub for '{repo_id}'...")

            # Determine Torch Device & Precision
            if device_choice == "CUDA (GPU)" and torch.cuda.is_available():
                self.device = "cuda"
            elif device_choice == "CPU":
                self.device = "cpu"
            else:  # Auto
                self.device = "cuda" if torch.cuda.is_available() else "cpu"

            if precision_choice == "float16" and self.device == "cuda":
                self.torch_dtype = torch.float16
            elif precision_choice == "bfloat16" and self.device == "cuda":
                self.torch_dtype = torch.bfloat16
            else:
                self.torch_dtype = torch.float32

            # Free previous model memory if loaded
            self.unload_model()

            _log(f"📥 Downloading tokenizer for '{repo_id}' into local folder 'models/huggingface'...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                repo_id,
                cache_dir=MODELS_DIR,
                trust_remote_code=True
            )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            _log(f"📥 Downloading model weights for '{repo_id}' into '{MODELS_DIR}' on {self.device.upper()} ({precision_choice})...\nThis may take a moment on first run...")

            if self.device == "cuda":
                self.model = AutoModelForCausalLM.from_pretrained(
                    repo_id,
                    cache_dir=MODELS_DIR,
                    torch_dtype=self.torch_dtype,
                    device_map="auto",
                    trust_remote_code=True
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    repo_id,
                    cache_dir=MODELS_DIR,
                    torch_dtype=self.torch_dtype,
                    trust_remote_code=True
                ).to(self.device)

            self.model.eval()
            self.current_model_id = repo_id
            self.is_loading = False

            device_name = f"GPU ({torch.cuda.get_device_name(0)})" if self.device == "cuda" else "CPU"
            _log(f"SUCCESS: Model '{repo_id}' loaded on {device_name}! Stored in 'models/huggingface'")
            return True, f"Loaded: {repo_id} ({self.device.upper()})"

        except Exception as e:
            self.is_loading = False
            self.unload_model()
            err_msg = str(e)
            _log(f"❌ Error loading model: {err_msg}")
            return False, f"Load Error: {err_msg}"

    def generate_stream(self, messages, system_prompt="", max_tokens=256, temperature=0.7, top_p=0.9, token_callback=None):
        """Generates response tokens and yields them via callback using TextIteratorStreamer."""
        if not self.model or not self.tokenizer:
            if token_callback:
                token_callback("⚠️ No model loaded. Please download/load a model first.")
            return

        self.is_generating = True
        self._stop_requested = False

        try:
            import torch
            from transformers import TextIteratorStreamer

            formatted_messages = []
            if system_prompt.strip():
                formatted_messages.append({"role": "system", "content": system_prompt.strip()})

            for m in messages:
                formatted_messages.append(m)

            # Apply chat template if supported by model's tokenizer
            try:
                prompt_text = self.tokenizer.apply_chat_template(
                    formatted_messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception:
                # Fallback simple conversation formatting
                prompt_text = ""
                if system_prompt.strip():
                    prompt_text += f"System: {system_prompt.strip()}\n\n"
                for msg in messages:
                    role = "User" if msg["role"] == "user" else "Assistant"
                    prompt_text += f"{role}: {msg['content']}\n"
                prompt_text += "Assistant: "

            inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)

            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True
            )

            generation_kwargs = dict(
                **inputs,
                streamer=streamer,
                max_new_tokens=int(max_tokens),
                temperature=float(max(0.01, temperature)),
                top_p=float(top_p),
                do_sample=temperature > 0.05,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

            # Run generate in a background worker thread
            gen_thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs, daemon=True)
            gen_thread.start()

            # Stream tokens to callback
            for new_text in streamer:
                if self._stop_requested:
                    break
                if token_callback:
                    token_callback(new_text)

        except Exception as e:
            if token_callback:
                token_callback(f"\n❌ Generation Error: {str(e)}")
        finally:
            self.is_generating = False

    def stop_generation(self):
        self._stop_requested = True

    def unload_model(self):
        """Unloads model & tokenizer to release RAM / VRAM."""
        self.model = None
        self.tokenizer = None
        self.current_model_id = ""
        self.is_loading = False
        self.is_generating = False

        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


class HuggingFaceLLMUI(ctk.CTkFrame):
    """CustomTkinter UI for Hugging Face Local LLM Studio."""

    def __init__(self, parent, on_back_callback):
        super().__init__(parent, fg_color="transparent")
        self.on_back_callback = on_back_callback

        self.engine = HuggingFaceLLMEngine()
        self.messages_history = []
        self.current_assistant_bubble = None

        self.setup_ui()

    def setup_ui(self):
        # Top Navigation Header Bar
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
            text="🤗 Hugging Face Local LLM Studio",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=("#1E1E2E", "#F5E0DC")
        )
        title_lbl.pack(side="left", padx=10)

        # Status Badge
        self.status_badge = ctk.CTkLabel(
            header_frame,
            text="No Model Loaded 🔌",
            fg_color=("#FE640B", "#F38BA8"),
            text_color=("#FFFFFF", "#11111B"),
            corner_radius=8,
            padx=10,
            pady=4,
            font=ctk.CTkFont(size=12, weight="bold")
        )
        self.status_badge.pack(side="right", padx=15)

        # Main Layout Container
        main_content = ctk.CTkFrame(self, fg_color="transparent")
        main_content.pack(fill="both", expand=True, padx=5)

        # Top Model Setup Card
        setup_box = ctk.CTkFrame(
            main_content,
            fg_color=("#FFFFFF", "#1E1E2E"),
            corner_radius=12
        )
        setup_box.pack(fill="x", pady=(0, 10), ipady=4)

        box_title = ctk.CTkLabel(
            setup_box,
            text="⚙️ Hugging Face Model Loader & Settings",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=("#1E66F5", "#89B4FA")
        )
        box_title.pack(anchor="w", padx=15, pady=(10, 5))

        # Row 1: Quick Preset Dropdown + Model Link/ID Entry
        row1 = ctk.CTkFrame(setup_box, fg_color="transparent")
        row1.pack(fill="x", padx=15, pady=4)

        preset_lbl = ctk.CTkLabel(
            row1,
            text="Quick Presets:",
            font=ctk.CTkFont(size=12),
            text_color=("#4C4F69", "#CDD6F4")
        )
        preset_lbl.pack(side="left", padx=(0, 8))

        self.preset_dropdown = ctk.CTkOptionMenu(
            row1,
            values=list(PRESET_MODELS.keys()),
            width=280,
            fg_color=("#1E66F5", "#89B4FA"),
            text_color=("#FFFFFF", "#11111B"),
            command=self.on_preset_selected
        )
        self.preset_dropdown.pack(side="left", padx=(0, 15))

        url_lbl = ctk.CTkLabel(
            row1,
            text="Model Link / ID:",
            font=ctk.CTkFont(size=12),
            text_color=("#4C4F69", "#CDD6F4")
        )
        url_lbl.pack(side="left", padx=(0, 8))

        self.model_entry = ctk.CTkEntry(
            row1,
            placeholder_text="Paste HF link or ID (e.g. Qwen/Qwen2.5-0.5B-Instruct)",
            fg_color=("#F2F4F8", "#11111B"),
            text_color=("#1E1E2E", "#CDD6F4"),
            font=ctk.CTkFont(size=12)
        )
        self.model_entry.pack(side="left", fill="x", expand=True)
        self.model_entry.insert(0, list(PRESET_MODELS.values())[0])

        # Row 2: Device, Precision, and Action Buttons
        row2 = ctk.CTkFrame(setup_box, fg_color="transparent")
        row2.pack(fill="x", padx=15, pady=(4, 10))

        dev_lbl = ctk.CTkLabel(
            row2,
            text="Hardware:",
            font=ctk.CTkFont(size=12),
            text_color=("#4C4F69", "#CDD6F4")
        )
        dev_lbl.pack(side="left", padx=(0, 8))

        self.device_dropdown = ctk.CTkOptionMenu(
            row2,
            values=["Auto (GPU/CPU)", "CUDA (GPU)", "CPU"],
            width=140,
            fg_color=("#DCE0E8", "#313244"),
            button_color=("#BCC0CC", "#45475A"),
            text_color=("#4C4F69", "#CDD6F4")
        )
        self.device_dropdown.pack(side="left", padx=(0, 15))

        prec_lbl = ctk.CTkLabel(
            row2,
            text="Precision:",
            font=ctk.CTkFont(size=12),
            text_color=("#4C4F69", "#CDD6F4")
        )
        prec_lbl.pack(side="left", padx=(0, 8))

        self.prec_dropdown = ctk.CTkOptionMenu(
            row2,
            values=["float16", "float32", "bfloat16"],
            width=110,
            fg_color=("#DCE0E8", "#313244"),
            button_color=("#BCC0CC", "#45475A"),
            text_color=("#4C4F69", "#CDD6F4")
        )
        self.prec_dropdown.pack(side="left", padx=(0, 15))

        self.load_btn = ctk.CTkButton(
            row2,
            text="📥 Download & Load Model",
            fg_color=("#40A02B", "#A6E3A1"),
            hover_color=("#207015", "#94E2D5"),
            text_color=("#FFFFFF", "#11111B"),
            font=ctk.CTkFont(size=13, weight="bold"),
            command=self.start_load_model
        )
        self.load_btn.pack(side="left", padx=(0, 10))

        self.unload_btn = ctk.CTkButton(
            row2,
            text="🔌 Unload",
            width=90,
            fg_color=("#DCE0E8", "#313244"),
            hover_color=("#BCC0CC", "#45475A"),
            text_color=("#4C4F69", "#CDD6F4"),
            command=self.unload_model
        )
        self.unload_btn.pack(side="left")

        # Download / Status Log Banner
        self.log_lbl = ctk.CTkLabel(
            setup_box,
            text="Select or paste a Hugging Face Model ID. Models are saved directly to 'AI PlayKit/models/huggingface/'.",
            font=ctk.CTkFont(size=11, slant="italic"),
            text_color=("#5C5F77", "#A6ADC8"),
            anchor="w"
        )
        self.log_lbl.pack(fill="x", padx=15, pady=(0, 8))

        # Bottom Workspace: Left Sidebar Controls + Right Chat Interface
        chat_workspace = ctk.CTkFrame(main_content, fg_color="transparent")
        chat_workspace.pack(fill="both", expand=True)

        # Left Parameters Sidebar (Sliders & System Prompt)
        sidebar = ctk.CTkFrame(
            chat_workspace,
            fg_color=("#FFFFFF", "#1E1E2E"),
            corner_radius=12,
            width=260
        )
        sidebar.pack(side="left", fill="y", padx=(0, 10))

        sb_title = ctk.CTkLabel(
            sidebar,
            text="🎛️ Parameters",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=("#1E66F5", "#89B4FA")
        )
        sb_title.pack(anchor="w", padx=15, pady=(12, 8))

        # System Prompt
        sys_lbl = ctk.CTkLabel(
            sidebar,
            text="System Prompt:",
            font=ctk.CTkFont(size=12),
            text_color=("#4C4F69", "#CDD6F4")
        )
        sys_lbl.pack(anchor="w", padx=15, pady=(4, 2))

        self.sys_prompt_entry = ctk.CTkTextbox(
            sidebar,
            height=60,
            fg_color=("#F2F4F8", "#11111B"),
            text_color=("#1E1E2E", "#CDD6F4"),
            font=ctk.CTkFont(size=11),
            corner_radius=6
        )
        self.sys_prompt_entry.pack(fill="x", padx=15, pady=(0, 10))
        self.sys_prompt_entry.insert("1.0", "You are a helpful, concise AI assistant.")

        # Max Tokens Slider
        tokens_hdr = ctk.CTkFrame(sidebar, fg_color="transparent")
        tokens_hdr.pack(fill="x", padx=15, pady=(6, 2))
        ctk.CTkLabel(tokens_hdr, text="Max New Tokens:", font=ctk.CTkFont(size=11), text_color=("#4C4F69", "#CDD6F4")).pack(side="left")
        self.tokens_val_lbl = ctk.CTkLabel(tokens_hdr, text="256", font=ctk.CTkFont(size=11, weight="bold"), text_color=("#1E66F5", "#89B4FA"))
        self.tokens_val_lbl.pack(side="right")

        self.tokens_slider = ctk.CTkSlider(
            sidebar, from_=32, to=1024, number_of_steps=31, command=lambda v: self.tokens_val_lbl.configure(text=str(int(v)))
        )
        self.tokens_slider.set(256)
        self.tokens_slider.pack(fill="x", padx=15, pady=(0, 10))

        # Temperature Slider
        temp_hdr = ctk.CTkFrame(sidebar, fg_color="transparent")
        temp_hdr.pack(fill="x", padx=15, pady=(6, 2))
        ctk.CTkLabel(temp_hdr, text="Temperature:", font=ctk.CTkFont(size=11), text_color=("#4C4F69", "#CDD6F4")).pack(side="left")
        self.temp_val_lbl = ctk.CTkLabel(temp_hdr, text="0.7", font=ctk.CTkFont(size=11, weight="bold"), text_color=("#1E66F5", "#89B4FA"))
        self.temp_val_lbl.pack(side="right")

        self.temp_slider = ctk.CTkSlider(
            sidebar, from_=0.1, to=1.5, number_of_steps=14, command=lambda v: self.temp_val_lbl.configure(text=f"{v:.1f}")
        )
        self.temp_slider.set(0.7)
        self.temp_slider.pack(fill="x", padx=15, pady=(0, 10))

        # Top-P Slider
        topp_hdr = ctk.CTkFrame(sidebar, fg_color="transparent")
        topp_hdr.pack(fill="x", padx=15, pady=(6, 2))
        ctk.CTkLabel(topp_hdr, text="Top-P Sampling:", font=ctk.CTkFont(size=11), text_color=("#4C4F69", "#CDD6F4")).pack(side="left")
        self.topp_val_lbl = ctk.CTkLabel(topp_hdr, text="0.9", font=ctk.CTkFont(size=11, weight="bold"), text_color=("#1E66F5", "#89B4FA"))
        self.topp_val_lbl.pack(side="right")

        self.topp_slider = ctk.CTkSlider(
            sidebar, from_=0.1, to=1.0, number_of_steps=9, command=lambda v: self.topp_val_lbl.configure(text=f"{v:.1f}")
        )
        self.topp_slider.set(0.9)
        self.topp_slider.pack(fill="x", padx=15, pady=(0, 15))

        # Right Chat Area Container
        chat_container = ctk.CTkFrame(
            chat_workspace,
            fg_color=("#F2F4F8", "#181825"),
            corner_radius=12
        )
        chat_container.pack(side="right", fill="both", expand=True)

        # Scrollable Chat Messages List
        self.chat_scroll = ctk.CTkScrollableFrame(
            chat_container,
            fg_color="transparent",
            scrollbar_button_color=("#CBD5E1", "#2B2C3B"),
            scrollbar_button_hover_color=("#94A3B8", "#3E4052")
        )
        self.chat_scroll.pack(fill="both", expand=True, padx=15, pady=12)

        # Welcome message bubble
        self.add_message_bubble(
            "Hugging Face AI",
            "👋 Welcome to Hugging Face Local LLM Studio!\nSelect a quick model preset or paste any HF repository link above, click 'Download & Load Model', and start chatting offline!",
            is_user=False
        )

        # Bottom Input Row
        input_row = ctk.CTkFrame(
            chat_container,
            fg_color=("#FFFFFF", "#1E1E2E"),
            corner_radius=10,
            height=50
        )
        input_row.pack(fill="x", padx=15, pady=(5, 12))

        self.msg_entry = ctk.CTkEntry(
            input_row,
            placeholder_text="Type your question or prompt here...",
            border_width=0,
            fg_color="transparent",
            text_color=("#1E1E2E", "#CDD6F4"),
            font=ctk.CTkFont(size=13)
        )
        self.msg_entry.pack(side="left", fill="x", expand=True, padx=15, pady=8)
        self.msg_entry.bind("<Return>", lambda event: self.send_message())

        self.stop_btn = ctk.CTkButton(
            input_row,
            text="🛑 Stop",
            width=65,
            fg_color=("#FE640B", "#F38BA8"),
            hover_color=("#E64553", "#E78284"),
            text_color=("#FFFFFF", "#11111B"),
            command=self.stop_generation,
            state="disabled"
        )
        self.stop_btn.pack(side="right", padx=(0, 5), pady=8)

        clear_btn = ctk.CTkButton(
            input_row,
            text="Clear",
            width=60,
            fg_color=("#DCE0E8", "#45475A"),
            hover_color=("#BCC0CC", "#585B70"),
            text_color=("#4C4F69", "#CDD6F4"),
            command=self.clear_chat
        )
        clear_btn.pack(side="right", padx=(0, 5), pady=8)

        self.send_btn = ctk.CTkButton(
            input_row,
            text="Send 📤",
            width=90,
            fg_color=("#1E66F5", "#89B4FA"),
            hover_color=("#7287FD", "#B4BEFE"),
            text_color=("#FFFFFF", "#11111B"),
            font=ctk.CTkFont(weight="bold"),
            command=self.send_message
        )
        self.send_btn.pack(side="right", padx=(0, 5), pady=8)

    def on_preset_selected(self, selected_label):
        repo_id = PRESET_MODELS.get(selected_label, "")
        if repo_id:
            self.model_entry.delete(0, "end")
            self.model_entry.insert(0, repo_id)

    def start_load_model(self):
        raw_input = self.model_entry.get().strip()
        repo_id = extract_hf_repo_id(raw_input)
        if not repo_id:
            self.log_lbl.configure(text="❌ Please enter a valid Hugging Face Model ID or URL.", text_color=("#E64553", "#F38BA8"))
            return

        device_val = self.device_dropdown.get()
        prec_val = self.prec_dropdown.get()

        self.load_btn.configure(state="disabled", text="⏳ Loading...")
        self.status_badge.configure(text="Loading Model... ⏳", fg_color=("#FE640B", "#FAB387"))
        self.log_lbl.configure(text=f"⏳ Initiating download/load for '{repo_id}'...", text_color=("#1E66F5", "#89B4FA"))

        def _update_log(msg):
            if self.winfo_exists():
                self.after(0, lambda m=msg: self.log_lbl.configure(
                    text=m,
                    text_color=("#40A02B", "#A6E3A1") if "SUCCESS" in m else (("#E64553", "#F38BA8") if "❌" in m else ("#1E66F5", "#89B4FA"))
                ))

        threading.Thread(
            target=self._load_worker,
            args=(repo_id, device_val, prec_val, _update_log),
            daemon=True
        ).start()

    def _load_worker(self, repo_id, device_val, prec_val, update_log_fn):
        success, status_str = self.engine.load_model(
            model_id=repo_id,
            device_choice=device_val,
            precision_choice=prec_val,
            status_callback=update_log_fn
        )
        if self.winfo_exists():
            self.after(0, lambda: self._on_load_finished(success, status_str, repo_id))

    def _on_load_finished(self, success, status_str, repo_id):
        if not self.winfo_exists():
            return
        self.load_btn.configure(state="normal", text="📥 Download & Load Model")
        if success:
            short_id = repo_id.split("/")[-1]
            self.status_badge.configure(
                text=f"Ready: {short_id} ✓",
                fg_color=("#40A02B", "#A6E3A1")
            )
        else:
            self.status_badge.configure(
                text="Load Failed ❌",
                fg_color=("#E64553", "#F38BA8")
            )

    def unload_model(self):
        self.engine.unload_model()
        self.status_badge.configure(text="No Model Loaded 🔌", fg_color=("#FE640B", "#F38BA8"))
        self.log_lbl.configure(text="Model unloaded. RAM/VRAM released.", text_color=("#5C5F77", "#A6ADC8"))

    def add_message_bubble(self, sender, text, is_user=False):
        bubble_bg = ("#1E66F5", "#89B4FA") if is_user else ("#E6E9EF", "#313244")
        text_color = ("#FFFFFF", "#11111B") if is_user else ("#1E1E2E", "#CDD6F4")
        sender_color = ("#FFFFFF", "#181825") if is_user else ("#1E66F5", "#89B4FA")
        align = "e" if is_user else "w"

        outer_frame = ctk.CTkFrame(self.chat_scroll, fg_color="transparent")
        outer_frame.pack(fill="x", pady=6, anchor=align)

        bubble = ctk.CTkFrame(
            outer_frame,
            fg_color=bubble_bg,
            corner_radius=12
        )
        bubble.pack(side="right" if is_user else "left", padx=(15, 15), ipadx=4, ipady=4)

        sender_lbl = ctk.CTkLabel(
            bubble,
            text=sender,
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=sender_color
        )
        sender_lbl.pack(anchor="w", padx=(16, 16), pady=(8, 0))

        msg_lbl = ctk.CTkLabel(
            bubble,
            text=text,
            font=ctk.CTkFont(size=13),
            text_color=text_color,
            justify="left",
            wraplength=480
        )
        msg_lbl.pack(anchor="w", padx=(16, 16), pady=(2, 8))

        self.chat_scroll._parent_canvas.yview_moveto(1.0)
        return msg_lbl

    def send_message(self):
        user_text = self.msg_entry.get().strip()
        if not user_text:
            return

        if not self.engine.model or not self.engine.tokenizer:
            self.add_message_bubble(
                "System",
                "⚠️ Please load a model first using '📥 Download & Load Model' above!",
                is_user=False
            )
            return

        if self.engine.is_generating:
            return

        self.add_message_bubble("You", user_text, is_user=True)
        self.messages_history.append({"role": "user", "content": user_text})
        self.msg_entry.delete(0, "end")

        self.send_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.status_badge.configure(text="Generating... 💬", fg_color=("#1E66F5", "#89B4FA"))

        # Create assistant placeholder bubble
        self.current_assistant_bubble = self.add_message_bubble("Local LLM", "...", is_user=False)
        self._accumulated_text = ""

        sys_prompt = self.sys_prompt_entry.get("1.0", "end").strip()
        max_t = self.tokens_slider.get()
        temp_v = self.temp_slider.get()
        top_p_v = self.topp_slider.get()

        threading.Thread(
            target=self._generation_worker,
            args=(list(self.messages_history), sys_prompt, max_t, temp_v, top_p_v),
            daemon=True
        ).start()

    def _generation_worker(self, messages, sys_prompt, max_t, temp_v, top_p_v):
        def _on_token(token_text):
            if self.winfo_exists():
                self.after(0, lambda t=token_text: self._append_token(t))

        self.engine.generate_stream(
            messages=messages,
            system_prompt=sys_prompt,
            max_tokens=max_t,
            temperature=temp_v,
            top_p=top_p_v,
            token_callback=_on_token
        )

        if self.winfo_exists():
            self.after(0, self._on_generation_finished)

    def _append_token(self, token_text):
        if not self.winfo_exists() or self.current_assistant_bubble is None:
            return
        self._accumulated_text += token_text
        self.current_assistant_bubble.configure(text=self._accumulated_text)
        self.chat_scroll._parent_canvas.yview_moveto(1.0)

    def _on_generation_finished(self):
        if not self.winfo_exists():
            return
        if self._accumulated_text:
            self.messages_history.append({"role": "assistant", "content": self._accumulated_text})
        self.send_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")

        short_id = self.engine.current_model_id.split("/")[-1] if self.engine.current_model_id else "Model"
        self.status_badge.configure(text=f"Ready: {short_id} ✓", fg_color=("#40A02B", "#A6E3A1"))

    def stop_generation(self):
        self.engine.stop_generation()

    def clear_chat(self):
        for child in self.chat_scroll.winfo_children():
            child.destroy()
        self.messages_history = []
        self.add_message_bubble(
            "Hugging Face AI",
            "Chat history cleared! Ask me anything.",
            is_user=False
        )

    def on_back_click(self):
        self.engine.unload_model()
        self.on_back_callback()
