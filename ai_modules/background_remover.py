import os
import time
import datetime
import threading
import customtkinter as ctk

# Metadata describing this module for AI PlayKit Hub
PROJECT_INFO = {
    "id": "background_remover",
    "title": "AI Live Background Remover",
    "description": "Real-time human detection and background removal using MediaPipe. Swap virtual backdrops, apply bokeh blur, green screen, and cyber halo glow effects live!",
    "icon": "✂️",
    "category": "Vision & Perception",
    "required_packages": ["mediapipe", "cv2", "PIL"],
    "install_command": "pip install mediapipe opencv-python pillow",
    "guide": """# ✂️ AI Live Background Remover Guide

### Overview
This project uses **MediaPipe Selfie Segmenter** to detect human silhouettes in real-time from webcam video, automatically separating person pixels from background pixels. You can instantly replace your background with **Sci-Fi grids**, **Bokeh nature**, **Studio Green Screen**, **DSLR Blur**, or **Cyber Neon Halos**!

---

### Step 1: Install Required Packages
Open your terminal and run:
```bash
pip install mediapipe opencv-python pillow
```

---

### Step 2: Key Features & Modes
- 🎭 **Virtual Backdrops**: Replace your background with Sci-Fi Cyber Grid, Synthwave Sunset, Cosmic Space, Tropical Bokeh, or Modern Office.
- 🌫️ **Background Blur**: Simulate a DSLR depth-of-field bokeh effect while keeping your person sharp.
- 🟩 **Chroma Key Green**: Output solid green screen (#00FF00) for streaming in OBS or video recording.
- 🎨 **Color Pop**: Keep yourself in full color while turning the background black & white!
- ✨ **Cyber Halo Glow**: Add a neon glowing border around your body silhouette.
- 🔲 **Transparent Cutout**: Isolate person onto a checkerboard canvas.
- 📸 **Snapshot Saver**: Save high-resolution PNG cutouts directly to your computer!

---

### Step 3: Beginner Python Code Example
```python
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. Download & load MediaPipe Selfie Segmenter
base_options = python.BaseOptions(model_asset_path="selfie_segmenter.tflite")
options = vision.ImageSegmenterOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    output_confidence_masks=True
)
segmenter = vision.ImageSegmenter.create_from_options(options)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR webcam frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Segment human pixels
    result = segmenter.segment(mp_img)
    mask = result.confidence_masks[0].numpy_view()

    # Blend green screen background where human mask < 0.5
    green_bg = np.zeros_like(frame)
    green_bg[:, :] = (0, 255, 0)
    
    alpha = (mask > 0.5).astype(np.float32)
    alpha_3d = np.dstack([alpha, alpha, alpha])
    output = (frame * alpha_3d + green_bg * (1.0 - alpha_3d)).astype(np.uint8)

    cv2.imshow("AI Background Remover", output)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```
"""
}

def check_dependencies():
    """Returns True if mediapipe, cv2, and PIL are installed without importing heavy packages."""
    import importlib.util
    return (
        importlib.util.find_spec("mediapipe") is not None and
        importlib.util.find_spec("cv2") is not None and
        importlib.util.find_spec("PIL") is not None
    )


class BackgroundRemoverUI(ctk.CTkFrame):
    """CustomTkinter UI for Real-Time Human Background Removal & Replacement."""

    def __init__(self, parent, on_back_callback):
        super().__init__(parent, fg_color="transparent")
        self.on_back_callback = on_back_callback

        self.cap = None
        self.is_running = False
        self.segmenter = None
        self.effects_engine = None

        self.current_ctk_img = None
        self.latest_output_frame = None

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
            text="✂️ AI Live Background Remover",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=("#1E1E2E", "#F5E0DC")
        )
        title_lbl.pack(side="left", padx=10)

        # Status Badge
        self.status_badge = ctk.CTkLabel(
            header_frame,
            text="Camera Off ⏸",
            fg_color=("#FE640B", "#FAB387"),
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
            width=300,
            scrollbar_button_color=("#CBD5E1", "#2B2C3B"),
            scrollbar_button_hover_color=("#94A3B8", "#3E4052")
        )
        control_panel.pack(side="left", fill="y", padx=(0, 10), pady=0)

        ctrl_title = ctk.CTkLabel(
            control_panel,
            text="⚙ Remover Settings",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=("#1E66F5", "#89B4FA")
        )
        ctrl_title.pack(anchor="w", padx=15, pady=(15, 10))

        # Start / Stop Webcam Button
        self.toggle_cam_btn = ctk.CTkButton(
            control_panel,
            text="▶ Start Webcam",
            fg_color=("#40A02B", "#A6E3A1"),
            hover_color=("#207015", "#94E2D5"),
            text_color=("#FFFFFF", "#11111B"),
            font=ctk.CTkFont(size=14, weight="bold"),
            height=40,
            command=self.toggle_webcam
        )
        self.toggle_cam_btn.pack(fill="x", padx=15, pady=5)

        # Background Removal Mode Selector
        mode_lbl = ctk.CTkLabel(
            control_panel,
            text="Background Mode:",
            font=ctk.CTkFont(size=12),
            text_color=("#4C4F69", "#CDD6F4")
        )
        mode_lbl.pack(anchor="w", padx=15, pady=(12, 2))

        self.mode_dropdown = ctk.CTkOptionMenu(
            control_panel,
            values=[
                "🎭 Virtual Backdrop",
                "🌫️ Background Blur",
                "🟩 Chroma Key Green",
                "🎨 Color Pop (B&W Back)",
                "✨ Cyber Halo Glow",
                "🔲 Transparent Cutout"
            ],
            fg_color=("#1E66F5", "#89B4FA"),
            text_color=("#FFFFFF", "#11111B")
        )
        self.mode_dropdown.set("🎭 Virtual Backdrop")
        self.mode_dropdown.pack(fill="x", padx=15, pady=5)

        # Virtual Backdrop Selector
        backdrop_lbl = ctk.CTkLabel(
            control_panel,
            text="Backdrop Style:",
            font=ctk.CTkFont(size=12),
            text_color=("#4C4F69", "#CDD6F4")
        )
        backdrop_lbl.pack(anchor="w", padx=15, pady=(10, 2))

        self.backdrop_dropdown = ctk.CTkOptionMenu(
            control_panel,
            values=[
                "🚀 Sci-Fi Cyber Grid",
                "🌆 Neon Sunset",
                "🌌 Cosmic Galaxy",
                "🌿 Tropical Bokeh",
                "🏢 Modern Office",
                "🎨 Solid Custom Color"
            ],
            fg_color=("#EA76CB", "#F5C2E7"),
            text_color=("#FFFFFF", "#11111B")
        )
        self.backdrop_dropdown.pack(fill="x", padx=15, pady=5)

        # Background Blur Slider
        blur_lbl_frame = ctk.CTkFrame(control_panel, fg_color="transparent")
        blur_lbl_frame.pack(fill="x", padx=15, pady=(10, 2))

        b_lbl = ctk.CTkLabel(
            blur_lbl_frame,
            text="Blur Intensity:",
            font=ctk.CTkFont(size=12),
            text_color=("#4C4F69", "#CDD6F4")
        )
        b_lbl.pack(side="left")

        self.blur_val_lbl = ctk.CTkLabel(
            blur_lbl_frame,
            text="25px",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=("#1E66F5", "#89B4FA")
        )
        self.blur_val_lbl.pack(side="right")

        self.blur_slider = ctk.CTkSlider(
            control_panel,
            from_=5,
            to=65,
            number_of_steps=12,
            command=self.on_blur_change
        )
        self.blur_slider.set(25)
        self.blur_slider.pack(fill="x", padx=15, pady=5)

        # Human Threshold & Feathering Sliders
        thresh_lbl_frame = ctk.CTkFrame(control_panel, fg_color="transparent")
        thresh_lbl_frame.pack(fill="x", padx=15, pady=(10, 2))

        t_lbl = ctk.CTkLabel(
            thresh_lbl_frame,
            text="Detection Threshold:",
            font=ctk.CTkFont(size=12),
            text_color=("#4C4F69", "#CDD6F4")
        )
        t_lbl.pack(side="left")

        self.thresh_val_lbl = ctk.CTkLabel(
            thresh_lbl_frame,
            text="50%",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=("#1E66F5", "#89B4FA")
        )
        self.thresh_val_lbl.pack(side="right")

        self.thresh_slider = ctk.CTkSlider(
            control_panel,
            from_=0.15,
            to=0.85,
            number_of_steps=14,
            command=self.on_thresh_change
        )
        self.thresh_slider.set(0.50)
        self.thresh_slider.pack(fill="x", padx=15, pady=5)

        # Edge Feathering Slider
        smooth_lbl_frame = ctk.CTkFrame(control_panel, fg_color="transparent")
        smooth_lbl_frame.pack(fill="x", padx=15, pady=(10, 2))

        s_lbl = ctk.CTkLabel(
            smooth_lbl_frame,
            text="Edge Smoothing:",
            font=ctk.CTkFont(size=12),
            text_color=("#4C4F69", "#CDD6F4")
        )
        s_lbl.pack(side="left")

        self.smooth_val_lbl = ctk.CTkLabel(
            smooth_lbl_frame,
            text="3px",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=("#1E66F5", "#89B4FA")
        )
        self.smooth_val_lbl.pack(side="right")

        self.smooth_slider = ctk.CTkSlider(
            control_panel,
            from_=0,
            to=10,
            number_of_steps=10,
            command=self.on_smooth_change
        )
        self.smooth_slider.set(3)
        self.smooth_slider.pack(fill="x", padx=15, pady=5)


        # Snapshot Capture Button
        self.snapshot_btn = ctk.CTkButton(
            control_panel,
            text="📸 Save Snapshot",
            fg_color=("#DCE0E8", "#313244"),
            hover_color=("#BCC0CC", "#45475A"),
            text_color=("#4C4F69", "#CDD6F4"),
            font=ctk.CTkFont(weight="bold"),
            command=self.save_snapshot
        )
        self.snapshot_btn.pack(fill="x", padx=15, pady=12)

        # Stats & Human Presence Box
        status_box = ctk.CTkFrame(
            control_panel,
            fg_color=("#F2F4F8", "#11111B"),
            corner_radius=10
        )
        status_box.pack(fill="both", expand=True, padx=15, pady=(5, 15))

        stat_title = ctk.CTkLabel(
            status_box,
            text="📊 Detection Stats",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=("#1E66F5", "#89B4FA")
        )
        stat_title.pack(anchor="w", padx=12, pady=(10, 4))

        self.stats_lbl = ctk.CTkLabel(
            status_box,
            text="Webcam is off.\nClick 'Start Webcam' to remove background live.",
            font=ctk.CTkFont(size=11),
            text_color=("#5C5F77", "#A6ADC8"),
            justify="left",
            wraplength=240
        )
        self.stats_lbl.pack(anchor="w", padx=12, pady=(0, 10))

        # Right Live Video View Frame
        self.video_container = ctk.CTkFrame(
            main_content,
            fg_color=("#F2F4F8", "#181825"),
            corner_radius=12
        )
        self.video_container.pack(side="right", fill="both", expand=True)

        self.video_lbl = ctk.CTkLabel(
            self.video_container,
            text="🎥 Live Camera Feed\nClick '▶ Start Webcam' to start live background removal.",
            font=ctk.CTkFont(size=15),
            text_color=("#5C5F77", "#A6ADC8"),
            justify="center",
            wraplength=540
        )
        self.video_lbl.pack(expand=True, padx=20, pady=20)

    def on_blur_change(self, val):
        self.blur_val_lbl.configure(text=f"{int(val)}px")

    def on_thresh_change(self, val):
        self.thresh_val_lbl.configure(text=f"{int(val * 100)}%")

    def on_smooth_change(self, val):
        self.smooth_val_lbl.configure(text=f"{int(val)}px")

    def toggle_webcam(self):
        if self.is_running:
            self.stop_webcam()
        else:
            self.start_webcam()

    def start_webcam(self):
        self.is_running = True
        self.toggle_cam_btn.configure(
            text="⏹ Stop Webcam",
            fg_color=("#FE640B", "#F38BA8"),
            hover_color=("#D20F39", "#E64553"),
            text_color=("#FFFFFF", "#11111B")
        )
        self.status_badge.configure(
            text="Removing Background 🔴",
            fg_color=("#40A02B", "#A6E3A1"),
            text_color=("#FFFFFF", "#11111B")
        )

        threading.Thread(target=self._webcam_loop, daemon=True).start()

    def stop_webcam(self):
        self.is_running = False
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        if self.segmenter:
            self.segmenter.close()
            self.segmenter = None

        self.toggle_cam_btn.configure(
            text="▶ Start Webcam",
            fg_color=("#40A02B", "#A6E3A1"),
            hover_color=("#207015", "#94E2D5"),
            text_color=("#FFFFFF", "#11111B")
        )
        self.status_badge.configure(
            text="Camera Off ⏸",
            fg_color=("#FE640B", "#FAB387"),
            text_color=("#FFFFFF", "#11111B")
        )
        self.current_ctk_img = None
        self.latest_output_frame = None
        try:
            self.video_lbl.configure(
                text="🎥 Camera Stopped.\nClick '▶ Start Webcam' to resume.",
                image=None
            )
        except Exception:
            pass
        self.stats_lbl.configure(text="Webcam is off.")

    def save_snapshot(self):
        """Saves current processed output frame to captures directory."""
        if self.latest_output_frame is None:
            self.snapshot_btn.configure(text="⚠️ No Frame Active!", fg_color=("#FE640B", "#FAB387"))
            self.after(2000, lambda: self.snapshot_btn.configure(text="📸 Save Snapshot", fg_color=("#DCE0E8", "#313244")))
            return

        try:
            import cv2
            captures_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "captures")
            os.makedirs(captures_dir, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"cutout_{timestamp}.png"
            filepath = os.path.join(captures_dir, filename)

            cv2.imwrite(filepath, self.latest_output_frame)

            self.snapshot_btn.configure(
                text=f"Saved! ✓ ({filename})",
                fg_color=("#40A02B", "#A6E3A1"),
                text_color=("#FFFFFF", "#11111B")
            )
            self.after(2500, lambda: self.snapshot_btn.configure(
                text="📸 Save Snapshot",
                fg_color=("#DCE0E8", "#313244"),
                text_color=("#4C4F69", "#CDD6F4")
            ))
        except Exception as e:
            print(f"Error saving snapshot: {e}")

    def on_back_click(self):
        self.stop_webcam()
        self.on_back_callback()

    def _webcam_loop(self):
        try:
            import cv2
            from PIL import Image
            from ai_modules.background_helpers.segmenter_engine import SelfieSegmenterEngine, BackgroundEffectsEngine

            if self.segmenter is None:
                self.after(0, lambda: self.stats_lbl.configure(text="⏳ Loading Selfie Segmenter Model..."))
                self.segmenter = SelfieSegmenterEngine()

            if self.effects_engine is None:
                self.effects_engine = BackgroundEffectsEngine()

            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.after(0, lambda: self._on_cam_error("Could not open webcam (Camera index 0). Please check camera permissions."))
                return

            last_time = time.time()
            fps = 30.0

            while self.is_running and self.cap.isOpened():
                start_proc_time = time.time()
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    break

                # Flip horizontally for natural mirror view
                frame = cv2.flip(frame, 1)

                # Get sliders parameters
                threshold_val = self.thresh_slider.get()
                softness_val = self.smooth_slider.get() * 0.03
                edge_blur_val = int(self.smooth_slider.get())

                # Process segmenter mask
                alpha_mask, conf_mask = self.segmenter.process_frame(
                    frame,
                    threshold=threshold_val,
                    softness=softness_val,
                    edge_blur=edge_blur_val
                )

                # Active dropdown values
                active_mode = self.mode_dropdown.get()
                active_backdrop = self.backdrop_dropdown.get()
                blur_amount = int(self.blur_slider.get())
                halo_color = "Cyan"

                # Render background effect
                output_bgr = self.effects_engine.render_effect(
                    frame,
                    alpha_mask,
                    mode=active_mode,
                    backdrop_type=active_backdrop,
                    blur_amount=blur_amount,
                    halo_color=halo_color,
                    halo_width=5
                )

                self.latest_output_frame = output_bgr.copy()

                # Calculate performance stats
                proc_time_ms = (time.time() - start_proc_time) * 1000.0
                curr_time = time.time()
                fps = 0.9 * fps + 0.1 * (1.0 / max(0.001, curr_time - last_time))
                last_time = curr_time

                # Human coverage %
                coverage_pct = int((alpha_mask > 0.45).mean() * 100)
                human_detected = "Yes ✓" if coverage_pct > 3 else "No ❌"

                stats_str = (
                    f"• Human Detected: {human_detected}\n"
                    f"• Subject Coverage: {coverage_pct}%\n"
                    f"• Processing Speed: {proc_time_ms:.1f} ms\n"
                    f"• Live FPS: {int(fps)} FPS\n"
                    f"• Mode: {active_mode.split()[0]} {active_mode.split()[1] if len(active_mode.split())>1 else ''}"
                )

                # Convert BGR to RGB for CustomTkinter
                rgb_frame = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_frame)

                # Dynamic aspect ratio sizing based on container size
                c_w = self.video_container.winfo_width()
                c_h = self.video_container.winfo_height()

                if c_w < 100 or c_h < 100:
                    target_w, target_h = 600, 440
                else:
                    img_w, img_h = pil_img.size
                    aspect = img_w / img_h
                    target_w = max(150, c_w - 30)
                    target_h = int(target_w / aspect)

                    if target_h > c_h - 30:
                        target_h = max(150, c_h - 30)
                        target_w = int(target_h * aspect)

                ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(target_w, target_h))

                # Update UI on main thread if window exists
                if self.is_running and self.winfo_exists():
                    self.after(0, lambda img=ctk_img, text=stats_str: self._update_frame(img, text))

                time.sleep(0.02)  # ~30 FPS

        except Exception as e:
            if self.is_running and self.winfo_exists():
                self.after(0, lambda err=str(e): self._on_cam_error(err))
        finally:
            if self.cap:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None

    def _update_frame(self, ctk_img, stats_text):
        if not self.is_running or not self.winfo_exists():
            return
        self.current_ctk_img = ctk_img
        try:
            self.video_lbl.configure(text="", image=self.current_ctk_img)
        except Exception:
            pass
        try:
            self.stats_lbl.configure(text=stats_text)
        except Exception:
            pass

    def _on_cam_error(self, err_msg):
        if not self.winfo_exists():
            return
        self.stop_webcam()
        clean_msg = str(err_msg).replace("\t", " ").strip()
        if len(clean_msg) > 280:
            clean_msg = clean_msg[:280] + "..."
        self.current_ctk_img = None
        try:
            self.video_lbl.configure(text=f"❌ Background Remover Error:\n\n{clean_msg}", image=None)
        except Exception:
            pass
