import os
import threading
import customtkinter as ctk

# Metadata describing this project for AI PlayKit Hub
PROJECT_INFO = {
    "id": "gemini_chat",
    "title": "Gemini AI Chatbot",
    "description": "Interactive AI chat powered by Google Gemini API. Learn multi-turn chat, prompt handling, and API integration.",
    "icon": "🤖",
    "category": "Text & Chat",
    "required_packages": ["google.genai"],
    "install_command": "pip install google-genai",
    "guide": """# 🤖 Gemini AI Chatbot Guide

### Overview
This project connects your python application to Google's powerful Gemini AI model. You can ask questions, brainstorm ideas, debug code, or generate text!

---

### Step 1: Install Python SDK
Open your terminal or command prompt and run:
```bash
pip install google-genai
```

---

### Step 2: Get a Free Gemini API Key
1. Go to Google AI Studio: **https://aistudio.google.com/**
2. Sign in with your Google Account.
3. Click **"Create API key"**.
4. Copy the API key and paste it in the API Key box inside the app.

---

### Step 3: Minimal Python Code for Beginners
```python
from google import genai

# 1. Initialize the client with your key
client = genai.Client(api_key="YOUR_GEMINI_API_KEY")

# 2. Ask Gemini a question
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Explain Quantum Computing in 2 simple sentences."
)

# 3. Print the response
print(response.text)
```
"""
}

def check_dependencies():
    """Returns True if required python packages are installed without importing heavy packages."""
    import importlib.util
    return importlib.util.find_spec("google.genai") is not None


class GeminiChatUI(ctk.CTkFrame):
    """Clean CustomTkinter UI for Gemini Chatbot with full Light/Dark theme support."""

    def __init__(self, parent, on_back_callback):
        super().__init__(parent, fg_color="transparent")
        self.on_back_callback = on_back_callback
        self.api_key = ""
        self.client = None

        self.key_file = os.path.join(os.path.dirname(__file__), ".gemini_key")
        self.load_saved_key()

        self.setup_ui()

    def load_saved_key(self):
        """Loads saved API key if exists so student doesn't re-type it."""
        if os.path.exists(self.key_file):
            try:
                with open(self.key_file, "r") as f:
                    self.api_key = f.read().strip()
            except Exception:
                pass

    def save_key(self, key):
        """Saves API key locally."""
        try:
            with open(self.key_file, "w") as f:
                f.write(key)
        except Exception:
            pass

    def setup_ui(self):
        # Top Navigation Header bar
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
            command=self.on_back_callback
        )
        back_btn.pack(side="left", padx=15, pady=12)

        title_lbl = ctk.CTkLabel(
            header_frame,
            text="🤖 Gemini AI Chatbot",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=("#1E1E2E", "#F5E0DC")
        )
        title_lbl.pack(side="left", padx=10)

        # Status badge
        self.status_badge = ctk.CTkLabel(
            header_frame,
            text="Disconnected" if not self.api_key else "Key Configured",
            fg_color=("#FE640B", "#F38BA8") if not self.api_key else ("#40A02B", "#A6E3A1"),
            text_color=("#FFFFFF", "#11111B"),
            corner_radius=8,
            padx=10,
            pady=4,
            font=ctk.CTkFont(size=12, weight="bold")
        )
        self.status_badge.pack(side="right", padx=15)

        # Main Content Layout
        content_frame = ctk.CTkFrame(self, fg_color="transparent")
        content_frame.pack(fill="both", expand=True, padx=5)

        # API Setup Card (Top Section)
        setup_box = ctk.CTkFrame(
            content_frame,
            fg_color=("#FFFFFF", "#1E1E2E"),
            corner_radius=12
        )
        setup_box.pack(fill="x", pady=(0, 10), ipady=4)

        box_title = ctk.CTkLabel(
            setup_box,
            text="🔑 API Connection Settings",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=("#1E66F5", "#89B4FA")
        )
        box_title.pack(anchor="w", padx=15, pady=(10, 5))

        key_row = ctk.CTkFrame(setup_box, fg_color="transparent")
        key_row.pack(fill="x", padx=15, pady=5)

        key_lbl = ctk.CTkLabel(
            key_row,
            text="Gemini API Key:",
            font=ctk.CTkFont(size=12),
            text_color=("#4C4F69", "#CDD6F4")
        )
        key_lbl.pack(side="left", padx=(0, 10))

        self.key_entry = ctk.CTkEntry(
            key_row,
            placeholder_text="Paste your Gemini API key (e.g. AIzaSy...)",
            show="*",
            fg_color=("#F2F4F8", "#11111B"),
            text_color=("#1E1E2E", "#CDD6F4"),
            width=320
        )
        self.key_entry.pack(side="left", padx=(0, 10), fill="x", expand=True)
        if self.api_key:
            self.key_entry.insert(0, self.api_key)

        self.show_key_var = ctk.BooleanVar(value=False)
        show_chk = ctk.CTkCheckBox(
            key_row,
            text="Show Key",
            variable=self.show_key_var,
            text_color=("#4C4F69", "#CDD6F4"),
            width=80,
            command=self.toggle_key_visibility
        )
        show_chk.pack(side="left", padx=(0, 10))

        model_lbl = ctk.CTkLabel(
            key_row,
            text="Model:",
            font=ctk.CTkFont(size=12),
            text_color=("#4C4F69", "#CDD6F4")
        )
        model_lbl.pack(side="left", padx=(10, 5))

        self.model_dropdown = ctk.CTkComboBox(
            key_row,
            values=["gemini-3.6-flash", "gemini-3.6-pro"],
            fg_color=("#1E66F5", "#89B4FA"),
            text_color=("#FFFFFF", "#11111B"),
            width=160
        )
        self.model_dropdown.pack(side="left", padx=(0, 10))

        connect_btn = ctk.CTkButton(
            key_row,
            text="Connect API",
            width=100,
            fg_color=("#1E66F5", "#89B4FA"),
            hover_color=("#7287FD", "#B4BEFE"),
            text_color=("#FFFFFF", "#11111B"),
            font=ctk.CTkFont(weight="bold"),
            command=self.connect_api
        )
        connect_btn.pack(side="left")

        # Chat container frame
        chat_container = ctk.CTkFrame(
            content_frame,
            fg_color=("#F2F4F8", "#181825"),
            corner_radius=12
        )
        chat_container.pack(fill="both", expand=True)

        # Scrollable Chat History
        self.chat_scroll = ctk.CTkScrollableFrame(
            chat_container,
            fg_color="transparent",
            scrollbar_button_color=("#CBD5E1", "#2B2C3B"),
            scrollbar_button_hover_color=("#94A3B8", "#3E4052")
        )
        self.chat_scroll.pack(fill="both", expand=True, padx=15, pady=12)

        # Welcome message inside chat
        self.add_message_bubble(
            "AI Assistant",
            "👋 Hello! Welcome to Gemini AI Chatbot.\nEnter your Gemini API key above and type any message below to start chatting!",
            is_user=False
        )

        # Input Area (Entry + Buttons)
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

        clear_btn = ctk.CTkButton(
            input_row,
            text="Clear",
            width=60,
            fg_color=("#DCE0E8", "#45475A"),
            hover_color=("#BCC0CC", "#585B70"),
            text_color=("#4C4F69", "#CDD6F4"),
            command=self.clear_chat
        )
        clear_btn.pack(side="right", padx=(0, 8), pady=8)

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

        if self.api_key:
            self.connect_api()

    def toggle_key_visibility(self):
        show = "" if self.show_key_var.get() else "*"
        self.key_entry.configure(show=show)

    def connect_api(self):
        key = self.key_entry.get().strip()
        if not key:
            self.status_badge.configure(
                text="No Key Entered",
                fg_color=("#FE640B", "#F38BA8"),
                text_color=("#FFFFFF", "#11111B")
            )
            return

        self.api_key = key
        self.save_key(key)
        try:
            from google import genai
            self.client = genai.Client(api_key=self.api_key)
            self.status_badge.configure(
                text="Connected ✓",
                fg_color=("#40A02B", "#A6E3A1"),
                text_color=("#FFFFFF", "#11111B")
            )
        except Exception as e:
            self.status_badge.configure(
                text="Connection Error",
                fg_color=("#FE640B", "#F38BA8"),
                text_color=("#FFFFFF", "#11111B")
            )

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
            wraplength=500
        )
        msg_lbl.pack(anchor="w", padx=(16, 16), pady=(2, 8))

        self.chat_scroll._parent_canvas.yview_moveto(1.0)

    def send_message(self):
        user_text = self.msg_entry.get().strip()
        if not user_text:
            return

        if not self.api_key or not self.client:
            self.add_message_bubble("System", "⚠️ Please enter a valid Gemini API key and click 'Connect API' first!", is_user=False)
            return

        self.add_message_bubble("You", user_text, is_user=True)
        self.msg_entry.delete(0, "end")

        self.send_btn.configure(state="disabled", text="Thinking...")

        threading.Thread(target=self._fetch_gemini_response, args=(user_text,), daemon=True).start()

    def _fetch_gemini_response(self, prompt):
        model_name = self.model_dropdown.get()
        try:
            response = self.client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            reply = response.text if response and response.text else "No response generated."
        except Exception as e:
            reply = f"❌ API Error: {str(e)}"

        self.after(0, lambda: self._on_response_received(reply))

    def _on_response_received(self, reply):
        self.add_message_bubble("Gemini AI", reply, is_user=False)
        self.send_btn.configure(state="normal", text="Send 📤")

    def clear_chat(self):
        for child in self.chat_scroll.winfo_children():
            child.destroy()
        self.add_message_bubble(
            "AI Assistant",
            "Chat cleared! Ask me anything.",
            is_user=False
        )
