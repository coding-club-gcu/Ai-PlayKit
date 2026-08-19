import os
import time
import datetime
import threading
from tkinter import filedialog
import customtkinter as ctk

# Metadata describing this module for AI PlayKit Hub
PROJECT_INFO = {
    "id": "style_transfer",
    "title": "AI Neural Style Transfer",
    "description": "Transform any uploaded photo or live webcam feed into masterpiece paintings using pre-trained Neural Style Transfer AI models!",
    "icon": "🎨",
    "category": "Generative & Creative",
    "required_packages": ["cv2", "PIL"],
    "install_command": "pip install opencv-python pillow",
    "guide": """# 🎨 AI Neural Style Transfer Studio Guide

### Overview
This project uses **Feedforward Convolutional Neural Networks (CNNs)** trained on artistic masterpieces (such as Van Gogh's *Starry Night*, Picasso's *La Muse*, and Munch's *The Scream*) to re-render any uploaded photo or webcam stream in the style of famous artwork in milliseconds!

---

### Step 1: Install Required Packages
Open your terminal and run:
```bash
pip install opencv-python pillow
```

---

### Step 2: Key Features
- 📁 **Upload Any Photo**: Upload any `.jpg`, `.png`, or `.webp` picture from your computer and transform it instantly!
- 🎥 **Live Webcam Stylization**: Experience real-time neural style transfer live from your webcam.
- 🎨 **7 Pre-Trained AI Models**:
  1. 🌌 **Starry Night** (Vincent van Gogh)
  2. 🍬 **Candy Art** (Vibrant Abstract)
  3. 🖼️ **Stained Glass Mosaic** (Tessellated Glass)
  4. 🎨 **Udnie** (Cubist Expressionism)
  5. 🪶 **Peacock Feathers** (Intricate Textures)
  6. 🎭 **La Muse** (Pablo Picasso)
  7. 😱 **The Scream** (Edvard Munch)
- 🎚️ **Style Intensity Slider**: Blend the neural artwork seamlessly with your original photo.
- 🎨 **Preserve Original Colors**: Transfer artistic textures while keeping your photo's original color palette.
- 🖼️ **Real Artwork Preview**: View original masterpiece painting reference images directly in the control panel!
- 💾 **Export Artwork**: Save high-resolution AI artwork to your disk.

---

### Step 3: Beginner Python Code Example
```python
import cv2

# 1. Load pre-trained Torch neural style model in OpenCV DNN
net = cv2.dnn.readNetFromTorch("starry_night.t7")

# 2. Load input photo
image = cv2.imread("my_photo.jpg")
(h, w) = image.shape[:2]

# 3. Create 4D input blob for neural network
blob = cv2.dnn.blobFromImage(
    image, 1.0, (w, h), 
    (103.939, 116.779, 123.68), 
    swapRB=False, crop=False
)

# 4. Perform forward pass inference
net.setInput(blob)
out = net.forward()

# 5. Post-process tensor output
out = out.reshape(3, out.shape[2], out.shape[3])
out[0] += 103.939; out[1] += 116.779; out[2] += 123.68
out = (out / 255.0).transpose(1, 2, 0)
stylized_bgr = (out * 255.0).clip(0, 255).astype("uint8")

# 6. Display stylized output
cv2.imshow("Neural Style Transfer", stylized_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()
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


class StyleTransferUI(ctk.CTkFrame):
    """CustomTkinter UI for AI Neural Style Transfer Studio."""

    def __init__(self, parent, on_back_callback):
        super().__init__(parent, fg_color="transparent")
        self.on_back_callback = on_back_callback

        self.cap = None
        self.is_running_webcam = False
        self.style_engine = None

        self.uploaded_image_bgr = None
        self.latest_stylized_bgr = None
        self.current_ctk_img = None
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
            text="🎨 AI Neural Style Transfer Studio",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=("#1E1E2E", "#F5E0DC")
        )
        title_lbl.pack(side="left", padx=10)

        # Status Badge
        self.status_badge = ctk.CTkLabel(
            header_frame,
            text="Ready 🎨",
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
            width=310,
            scrollbar_button_color=("#CBD5E1", "#2B2C3B"),
            scrollbar_button_hover_color=("#94A3B8", "#3E4052")
        )
        control_panel.pack(side="left", fill="y", padx=(0, 10), pady=0)

        ctrl_title = ctk.CTkLabel(
            control_panel,
            text="⚙ Style Settings",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=("#1E66F5", "#89B4FA")
        )
        ctrl_title.pack(anchor="w", padx=15, pady=(15, 10))

        # Action Buttons Row (Upload Photo / Start Webcam)
        upload_btn = ctk.CTkButton(
            control_panel,
            text="📁 Upload Photo",
            fg_color=("#1E66F5", "#89B4FA"),
            hover_color=("#7287FD", "#B4BEFE"),
            text_color=("#FFFFFF", "#11111B"),
            font=ctk.CTkFont(size=13, weight="bold"),
            height=38,
            command=self.upload_photo
        )
        upload_btn.pack(fill="x", padx=15, pady=(5, 5))

        self.toggle_cam_btn = ctk.CTkButton(
            control_panel,
            text="▶ Start Live Webcam",
            fg_color=("#40A02B", "#A6E3A1"),
            hover_color=("#207015", "#94E2D5"),
            text_color=("#FFFFFF", "#11111B"),
            font=ctk.CTkFont(size=13, weight="bold"),
            height=38,
            command=self.toggle_webcam
        )
        self.toggle_cam_btn.pack(fill="x", padx=15, pady=(0, 10))

        # AI Style Model Selector
        model_lbl = ctk.CTkLabel(
            control_panel,
            text="AI Masterpiece Style:",
            font=ctk.CTkFont(size=12),
            text_color=("#4C4F69", "#CDD6F4")
        )
        model_lbl.pack(anchor="w", padx=15, pady=(10, 2))

        self.model_dropdown = ctk.CTkOptionMenu(
            control_panel,
            values=[
                "🌌 Starry Night (Van Gogh)",
                "🍬 Candy Art (Abstract)",
                "🪞 Stained Glass Mosaic",
                "🎨 Udnie (Expressionism)",
                "🪶 Peacock Feathers",
                "🎭 La Muse (Picasso)",
                "😱 The Scream (Munch)"
            ],
            fg_color=("#EA76CB", "#F5C2E7"),
            text_color=("#FFFFFF", "#11111B"),
            command=self.on_style_selected
        )
        self.model_dropdown.pack(fill="x", padx=15, pady=5)

        # Original Painting Reference Preview (below OptionMenu)
        self.preview_card = ctk.CTkFrame(
            control_panel,
            fg_color=("#F2F4F8", "#11111B"),
            corner_radius=10
        )
        self.preview_card.pack(fill="x", padx=15, pady=(5, 10))

        preview_hdr = ctk.CTkLabel(
            self.preview_card,
            text="🖼️ Original Painting Reference",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=("#1E66F5", "#89B4FA")
        )
        preview_hdr.pack(anchor="w", padx=10, pady=(8, 4))

        self.style_preview_lbl = ctk.CTkLabel(
            self.preview_card,
            text="",
            corner_radius=8
        )
        self.style_preview_lbl.pack(padx=10, pady=(0, 8))

        # Style Intensity / Blend Slider
        blend_lbl_frame = ctk.CTkFrame(control_panel, fg_color="transparent")
        blend_lbl_frame.pack(fill="x", padx=15, pady=(12, 2))

        b_lbl = ctk.CTkLabel(
            blend_lbl_frame,
            text="Style Intensity:",
            font=ctk.CTkFont(size=12),
            text_color=("#4C4F69", "#CDD6F4")
        )
        b_lbl.pack(side="left")

        self.blend_val_lbl = ctk.CTkLabel(
            blend_lbl_frame,
            text="85%",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=("#1E66F5", "#89B4FA")
        )
        self.blend_val_lbl.pack(side="right")

        self.blend_slider = ctk.CTkSlider(
            control_panel,
            from_=0.10,
            to=1.0,
            number_of_steps=18,
            command=self.on_blend_change
        )
        self.blend_slider.set(0.85)
        self.blend_slider.pack(fill="x", padx=15, pady=5)

        # Preserve Colors Checkbox
        self.preserve_color_var = ctk.BooleanVar(value=False)
        preserve_chk = ctk.CTkCheckBox(
            control_panel,
            text="Preserve Original Colors",
            variable=self.preserve_color_var,
            text_color=("#4C4F69", "#CDD6F4"),
            command=self.reprocess_uploaded_image
        )
        preserve_chk.pack(anchor="w", padx=15, pady=10)

        # Save Artwork Button
        self.save_btn = ctk.CTkButton(
            control_panel,
            text="💾 Export Artwork",
            fg_color=("#40A02B", "#A6E3A1"),
            hover_color=("#207015", "#94E2D5"),
            text_color=("#FFFFFF", "#11111B"),
            font=ctk.CTkFont(weight="bold"),
            height=36,
            command=self.save_artwork
        )
        self.save_btn.pack(fill="x", padx=15, pady=12)

        # Reset Button
        reset_btn = ctk.CTkButton(
            control_panel,
            text="🔄 Clear Photo",
            fg_color=("#DCE0E8", "#313244"),
            hover_color=("#BCC0CC", "#45475A"),
            text_color=("#4C4F69", "#CDD6F4"),
            command=self.clear_photo
        )
        reset_btn.pack(fill="x", padx=15, pady=(0, 12))

        # Model Info & Performance Stats Box
        status_box = ctk.CTkFrame(
            control_panel,
            fg_color=("#F2F4F8", "#11111B"),
            corner_radius=10
        )
        status_box.pack(fill="both", expand=True, padx=15, pady=(5, 15))

        stat_title = ctk.CTkLabel(
            status_box,
            text="📊 AI Model Info",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=("#1E66F5", "#89B4FA")
        )
        stat_title.pack(anchor="w", padx=12, pady=(10, 4))

        self.stats_lbl = ctk.CTkLabel(
            status_box,
            text="Upload a photo or start webcam to apply neural art styling.",
            font=ctk.CTkFont(size=11),
            text_color=("#5C5F77", "#A6ADC8"),
            justify="left",
            wraplength=250
        )
        self.stats_lbl.pack(anchor="w", padx=12, pady=(0, 10))

        # Right Live Video / Image View Container
        self.image_container = ctk.CTkFrame(
            main_content,
            fg_color=("#F2F4F8", "#181825"),
            corner_radius=12
        )
        self.image_container.pack(side="right", fill="both", expand=True)

        self.image_lbl = ctk.CTkLabel(
            self.image_container,
            text="🎨 AI Neural Style Transfer Studio\nClick '📁 Upload Photo' or '▶ Start Live Webcam' to style your image!",
            font=ctk.CTkFont(size=15),
            text_color=("#5C5F77", "#A6ADC8"),
            justify="center",
            wraplength=540
        )
        self.image_lbl.pack(expand=True, padx=20, pady=20)

        # Initialize original painting preview for current selection
        self.update_style_preview(self.model_dropdown.get())

    def update_style_preview(self, selected_style):
        """Displays original artwork image reference below the style OptionMenu."""
        try:
            from PIL import Image
            from ai_modules.style_helpers.style_engine import NeuralStyleEngine
            if self.style_engine is None:
                self.style_engine = NeuralStyleEngine()

            preview_path = self.style_engine.get_style_preview_path(selected_style)
            if preview_path and os.path.exists(preview_path):
                pil_img = Image.open(preview_path)
                w, h = pil_img.size
                target_w = 250
                target_h = int(h * (target_w / w))
                if target_h > 180:
                    target_h = 180
                    target_w = int(w * (target_h / h))

                ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(target_w, target_h))
                self.style_preview_lbl.configure(image=ctk_img, text="")
            else:
                self.style_preview_lbl.configure(image=None, text="Preview unavailable")
        except Exception as e:
            print(f"Error updating style preview: {e}")

    def on_blend_change(self, val):
        self.blend_val_lbl.configure(text=f"{int(val * 100)}%")
        if self.uploaded_image_bgr is not None and not self.is_running_webcam:
            self.reprocess_uploaded_image()

    def on_style_selected(self, selected_style):
        self.update_style_preview(selected_style)
        if self.uploaded_image_bgr is not None and not self.is_running_webcam:
            self.reprocess_uploaded_image()

    def upload_photo(self):
        """Opens file dialog for uploading any photo."""
        file_path = filedialog.askopenfilename(
            title="Select Photo for AI Style Transfer",
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
                self.stats_lbl.configure(text="❌ Error loading selected image file.")
                return

            self.uploaded_image_bgr = img
            self.reprocess_uploaded_image()
        except Exception as e:
            self.stats_lbl.configure(text=f"❌ Image Upload Error: {e}")

    def reprocess_uploaded_image(self):
        """Applies neural style transfer to currently uploaded photo in a background thread."""
        if self.uploaded_image_bgr is None or self.is_processing:
            return

        self.is_processing = True
        self.status_badge.configure(text="Processing AI... ⏳", fg_color=("#FE640B", "#FAB387"))

        threading.Thread(target=self._process_uploaded_thread, daemon=True).start()

    def _process_uploaded_thread(self):
        try:
            import cv2
            from ai_modules.style_helpers.style_engine import NeuralStyleEngine, MODEL_METADATA

            if self.style_engine is None:
                self.style_engine = NeuralStyleEngine()

            selected_style = self.model_dropdown.get()
            blend_ratio = self.blend_slider.get()
            preserve_colors = self.preserve_color_var.get()

            def update_status(text):
                if self.winfo_exists():
                    self.after(0, lambda: self.stats_lbl.configure(text=text))

            start_t = time.time()
            stylized = self.style_engine.stylize_image(
                self.uploaded_image_bgr,
                style_name=selected_style,
                blend_ratio=blend_ratio,
                preserve_color=preserve_colors,
                progress_callback=update_status
            )
            proc_ms = (time.time() - start_t) * 1000.0

            self.latest_stylized_bgr = stylized

            meta = MODEL_METADATA.get(selected_style, {})
            desc = meta.get("description", "")
            h, w = self.uploaded_image_bgr.shape[:2]

            stats_str = (
                f"• Style: {selected_style.split('(')[0].strip()}\n"
                f"• Image Dimensions: {w}x{h} px\n"
                f"• Processing Time: {proc_ms:.0f} ms\n"
                f"• Colors Preserved: {'Yes ✓' if preserve_colors else 'No (Artistic)'}\n\n"
                f"💡 {desc}"
            )

            if self.winfo_exists():
                self.after(0, lambda: self._on_stylize_complete(stats_str))

        except Exception as e:
            if self.winfo_exists():
                self.after(0, lambda err=str(e): self._on_cam_error(f"Stylize Error: {err}"))
        finally:
            self.is_processing = False

    def _on_stylize_complete(self, stats_str):
        if not self.winfo_exists():
            return

        self.status_badge.configure(text="Stylized ✓", fg_color=("#40A02B", "#A6E3A1"))
        self.stats_lbl.configure(text=stats_str)
        self._render_output_view()

    def _render_output_view(self):
        if self.latest_stylized_bgr is None or not self.winfo_exists():
            return

        import cv2
        from PIL import Image

        rgb = cv2.cvtColor(self.latest_stylized_bgr, cv2.COLOR_BGR2RGB)
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

    def toggle_webcam(self):
        if self.is_running_webcam:
            self.stop_webcam()
        else:
            self.start_webcam()

    def start_webcam(self):
        self.uploaded_image_bgr = None
        self.is_running_webcam = True
        self.toggle_cam_btn.configure(
            text="⏹ Stop Live Webcam",
            fg_color=("#FE640B", "#F38BA8"),
            hover_color=("#D20F39", "#E64553"),
            text_color=("#FFFFFF", "#11111B")
        )
        self.status_badge.configure(
            text="Webcam Stylizing 🔴",
            fg_color=("#40A02B", "#A6E3A1"),
            text_color=("#FFFFFF", "#11111B")
        )

        threading.Thread(target=self._webcam_loop, daemon=True).start()

    def stop_webcam(self):
        self.is_running_webcam = False
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        self.toggle_cam_btn.configure(
            text="▶ Start Live Webcam",
            fg_color=("#40A02B", "#A6E3A1"),
            hover_color=("#207015", "#94E2D5"),
            text_color=("#FFFFFF", "#11111B")
        )
        self.status_badge.configure(
            text="Ready 🎨",
            fg_color=("#1E66F5", "#89B4FA"),
            text_color=("#FFFFFF", "#11111B")
        )
        if self.uploaded_image_bgr is None:
            try:
                self.image_lbl.configure(
                    text="🎨 AI Neural Style Transfer Studio\nClick '📁 Upload Photo' or '▶ Start Live Webcam' to style your image!",
                    image=None
                )
            except Exception:
                pass
            self.stats_lbl.configure(text="Webcam stopped.")

    def save_artwork(self):
        """Saves stylized image artwork to disk."""
        if self.latest_stylized_bgr is None:
            self.save_btn.configure(text="⚠️ No Artwork Active!", fg_color=("#FE640B", "#FAB387"))
            self.after(2000, lambda: self.save_btn.configure(text="💾 Export Artwork", fg_color=("#40A02B", "#A6E3A1")))
            return

        try:
            import cv2
            captures_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "captures")
            os.makedirs(captures_dir, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"neural_art_{timestamp}.png"
            filepath = os.path.join(captures_dir, filename)

            cv2.imwrite(filepath, self.latest_stylized_bgr)

            self.save_btn.configure(
                text=f"Saved! ✓ ({filename})",
                fg_color=("#40A02B", "#A6E3A1"),
                text_color=("#FFFFFF", "#11111B")
            )
            self.after(2500, lambda: self.save_btn.configure(
                text="💾 Export Artwork",
                fg_color=("#40A02B", "#A6E3A1"),
                text_color=("#FFFFFF", "#11111B")
            ))
        except Exception as e:
            print(f"Error saving artwork: {e}")

    def clear_photo(self):
        self.stop_webcam()
        self.uploaded_image_bgr = None
        self.latest_stylized_bgr = None
        self.current_ctk_img = None
        try:
            self.image_lbl.configure(
                text="🎨 AI Neural Style Transfer Studio\nClick '📁 Upload Photo' or '▶ Start Live Webcam' to style your image!",
                image=None
            )
        except Exception:
            pass
        self.stats_lbl.configure(text="Cleared photo.")

    def on_back_click(self):
        self.stop_webcam()
        self.on_back_callback()

    def _webcam_loop(self):
        try:
            import cv2
            from ai_modules.style_helpers.style_engine import NeuralStyleEngine

            if self.style_engine is None:
                self.after(0, lambda: self.stats_lbl.configure(text="⏳ Initializing AI Neural Style Engine..."))
                self.style_engine = NeuralStyleEngine()

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

                selected_style = self.model_dropdown.get()
                blend_ratio = self.blend_slider.get()
                preserve_colors = self.preserve_color_var.get()

                # Process webcam frame at 480 size for smooth real-time performance
                stylized_bgr = self.style_engine.stylize_image(
                    frame,
                    style_name=selected_style,
                    blend_ratio=blend_ratio,
                    preserve_color=preserve_colors,
                    target_size=480
                )

                self.latest_stylized_bgr = stylized_bgr
                self.uploaded_image_bgr = frame

                # Stats
                proc_ms = (time.time() - start_proc_time) * 1000.0
                curr_time = time.time()
                fps = 0.9 * fps + 0.1 * (1.0 / max(0.001, curr_time - last_time))
                last_time = curr_time

                stats_str = (
                    f"• Live Webcam Stream\n"
                    f"• Style: {selected_style.split('(')[0].strip()}\n"
                    f"• Processing Speed: {proc_ms:.0f} ms\n"
                    f"• Live Performance: {int(fps)} FPS"
                )

                if self.is_running_webcam and self.winfo_exists():
                    self.after(0, lambda text=stats_str: self._update_webcam_ui(text))

                time.sleep(0.02)

        except Exception as e:
            if self.is_running_webcam and self.winfo_exists():
                self.after(0, lambda err=str(e): self._on_cam_error(f"Webcam Style Error: {err}"))
        finally:
            if self.cap:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None

    def _update_webcam_ui(self, stats_text):
        if not self.is_running_webcam or not self.winfo_exists():
            return
        try:
            self.stats_lbl.configure(text=stats_text)
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
            self.image_lbl.configure(text=f"❌ Style Transfer Error:\n\n{clean_msg}", image=None)
        except Exception:
            pass