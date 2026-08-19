import os
import time
import threading
import customtkinter as ctk

# Metadata describing this module for AI PlayKit Hub
PROJECT_INFO = {
    "id": "eye_gaze_tracker",
    "title": "AI Eye Gaze Tracker",
    "description": "Track 3D facial landmarks and eye gaze in real-time. Control a virtual laser pointer or play the Gaze Target Mini-Game!",
    "icon": "👁️",
    "category": "Vision & Perception",
    "required_packages": ["mediapipe", "cv2", "PIL", "numpy"],
    "install_command": "pip install mediapipe opencv-python pillow numpy",
    "guide": """# 👁️ AI Eye Gaze Tracker & Virtual Laser Pointer Guide

### Overview
This project uses **MediaPipe FaceLandmarker** with 478 3D facial landmarks to track eye irises, estimate gaze direction, detect blinks, and project a smooth virtual laser pointer on screen!

---

### Step 1: Install Required Packages
Run this in your terminal:
```bash
pip install mediapipe opencv-python pillow numpy
```

---

### Step 2: Key Modes & Controls
- 🔴 **Virtual Laser Pointer**: Point your eyes to control a glowing **Cyber Red** or **Neon Blue** laser pointer!
- 🎯 **Gaze Target Mini-Game**: Stare at targets to lock on, and blink or hold lock to hit targets and score points!
- 😉 **Blink & Wink Detection**: Real-time Eye Aspect Ratio (EAR) tracking to detect blinks and winks.

---

### Step 3: Beginner Python Code Example
```python
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path="face_landmarker.task")
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_faces=1
)
landmarker = vision.FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
print("Webcam starting... Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    results = landmarker.detect(mp_image)

    if results.face_landmarks:
        left_iris = results.face_landmarks[0][468]
        h, w, _ = frame.shape
        cv2.circle(frame, (int(left_iris.x * w), int(left_iris.y * h)), 5, (0, 255, 0), -1)

    cv2.imshow("Eye Gaze Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```
"""
}


def check_dependencies():
    """Returns True if mediapipe, cv2, PIL, and numpy are installed without importing heavy packages."""
    import importlib.util
    return (
        importlib.util.find_spec("mediapipe") is not None and
        importlib.util.find_spec("cv2") is not None and
        importlib.util.find_spec("PIL") is not None and
        importlib.util.find_spec("numpy") is not None
    )


class EyeGazeTrackerUI(ctk.CTkFrame):
    """CustomTkinter UI for Real-Time AI Eye Gaze Tracker & Virtual Laser Pointer."""

    def __init__(self, parent, on_back_callback):
        super().__init__(parent, fg_color="transparent")
        self.on_back_callback = on_back_callback

        self.cap = None
        self.is_running = False
        self.detector = None
        self.laser_engine = None

        self.current_ctk_img = None

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
            text="👁️ AI Eye Gaze Tracker & Virtual Laser Pointer",
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
        control_panel = ctk.CTkFrame(
            main_content,
            fg_color=("#FFFFFF", "#1E1E2E"),
            corner_radius=12,
            width=300
        )
        control_panel.pack(side="left", fill="y", padx=(0, 10), pady=0)
        control_panel.pack_propagate(False)

        ctrl_title = ctk.CTkLabel(
            control_panel,
            text="⚙ Gaze Settings",
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

        # Mode Selector Dropdown (Only 2 Modes)
        mode_lbl = ctk.CTkLabel(
            control_panel,
            text="Visualizer Mode:",
            font=ctk.CTkFont(size=12),
            text_color=("#4C4F69", "#CDD6F4")
        )
        mode_lbl.pack(anchor="w", padx=15, pady=(14, 2))

        self.mode_dropdown = ctk.CTkOptionMenu(
            control_panel,
            values=[
                "🔴 Virtual Laser Pointer",
                "🎯 Gaze Target Mini-Game"
            ],
            fg_color=("#1E66F5", "#89B4FA"),
            text_color=("#FFFFFF", "#11111B")
        )
        self.mode_dropdown.set("🔴 Virtual Laser Pointer")
        self.mode_dropdown.pack(fill="x", padx=15, pady=5)

        # Laser Pointer Style Dropdown (Only Cyber Red & Neon Blue)
        laser_lbl = ctk.CTkLabel(
            control_panel,
            text="Laser Pointer Style:",
            font=ctk.CTkFont(size=12),
            text_color=("#4C4F69", "#CDD6F4")
        )
        laser_lbl.pack(anchor="w", padx=15, pady=(12, 2))

        self.laser_dropdown = ctk.CTkOptionMenu(
            control_panel,
            values=[
                "🔴 Cyber Red Laser",
                "⚡ Neon Plasma Blue"
            ],
            fg_color=("#EA76CB", "#F5C2E7"),
            text_color=("#FFFFFF", "#11111B")
        )
        self.laser_dropdown.set("🔴 Cyber Red Laser")
        self.laser_dropdown.pack(fill="x", padx=15, pady=5)

        # Clear Canvas / Reset Game Button
        clear_btn = ctk.CTkButton(
            control_panel,
            text="🧹 Reset Game / Laser",
            fg_color=("#DCE0E8", "#313244"),
            hover_color=("#BCC0CC", "#45475A"),
            text_color=("#4C4F69", "#CDD6F4"),
            command=self.clear_canvas
        )
        clear_btn.pack(fill="x", padx=15, pady=12)

        # Checkboxes for Overlay Toggles
        self.show_mesh_var = ctk.BooleanVar(value=True)
        show_mesh_chk = ctk.CTkCheckBox(
            control_panel,
            text="Draw 3D Eye Mesh & Iris",
            variable=self.show_mesh_var,
            text_color=("#4C4F69", "#CDD6F4")
        )
        show_mesh_chk.pack(anchor="w", padx=15, pady=6)

        self.show_rays_var = ctk.BooleanVar(value=True)
        show_rays_chk = ctk.CTkCheckBox(
            control_panel,
            text="Draw 3D Gaze Vector Rays",
            variable=self.show_rays_var,
            text_color=("#4C4F69", "#CDD6F4")
        )
        show_rays_chk.pack(anchor="w", padx=15, pady=6)

        # Live Gaze Stats & Eye Status Box
        metrics_box = ctk.CTkFrame(
            control_panel,
            fg_color=("#F2F4F8", "#11111B"),
            corner_radius=10
        )
        metrics_box.pack(fill="both", expand=True, padx=15, pady=(15, 15))

        metrics_title = ctk.CTkLabel(
            metrics_box,
            text="📊 Live Gaze Info",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=("#1E66F5", "#89B4FA")
        )
        metrics_title.pack(anchor="w", padx=12, pady=(10, 4))

        self.metrics_lbl = ctk.CTkLabel(
            metrics_box,
            text="Webcam is off.\n\n👁️ Move eyes → Move Laser\n🎯 Stare at targets → Lock-on\n😉 Wink / Blink → Shoot Target",
            font=ctk.CTkFont(size=11),
            text_color=("#5C5F77", "#A6ADC8"),
            justify="left",
            wraplength=250
        )
        self.metrics_lbl.pack(anchor="w", padx=12, pady=(0, 10))

        # Right Live Video View Frame
        self.video_container = ctk.CTkFrame(
            main_content,
            fg_color=("#F2F4F8", "#181825"),
            corner_radius=12
        )
        self.video_container.pack(side="right", fill="both", expand=True)

        self.video_lbl = ctk.CTkLabel(
            self.video_container,
            text="🎥 Live Camera Feed\nClick '▶ Start Webcam' to begin tracking eye gaze & virtual laser pointer.",
            font=ctk.CTkFont(size=15),
            text_color=("#5C5F77", "#A6ADC8"),
            justify="center",
            wraplength=540
        )
        self.video_lbl.pack(expand=True, padx=20, pady=20)

    def clear_canvas(self):
        if self.laser_engine:
            self.laser_engine.clear()

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
            text="Tracking Active 🔴",
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

        if self.detector:
            self.detector.close()
            self.detector = None

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
        try:
            self.video_lbl.configure(
                text="🎥 Camera Stopped.\nClick '▶ Start Webcam' to resume.",
                image=None
            )
        except Exception:
            pass
        self.metrics_lbl.configure(text="Webcam is off.")

    def on_back_click(self):
        self.stop_webcam()
        self.on_back_callback()

    def _webcam_loop(self):
        try:
            import cv2
            from PIL import Image
            from ai_modules.eye_helpers.eye_tracker import EyeGazeDetector
            from ai_modules.eye_helpers.laser_canvas import LaserCanvasEngine

            if self.detector is None:
                self.after(0, lambda: self.metrics_lbl.configure(text="⏳ Loading MediaPipe 3D Face Landmarker..."))
                self.detector = EyeGazeDetector()

            if self.laser_engine is None:
                self.laser_engine = LaserCanvasEngine()

            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.after(0, lambda: self._on_cam_error("Could not open webcam (Index 0). Please check camera permissions."))
                return

            while self.is_running and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    break

                # Flip horizontally for mirror view
                frame = cv2.flip(frame, 1)

                # Process Frame with Eye Gaze Detector
                eye_data = self.detector.detect(frame)

                # Fetch UI Options
                selected_mode = self.mode_dropdown.get()
                selected_laser = self.laser_dropdown.get()
                show_mesh = self.show_mesh_var.get()
                show_rays = self.show_rays_var.get()

                # Sync Laser Pointer Style
                self.laser_engine.set_laser_style(selected_laser)

                if eye_data:
                    # Draw 3D Eye Mesh & Iris contour if enabled
                    if show_mesh:
                        self.detector.draw_eye_mesh(frame, eye_data)

                    # Draw 3D Gaze Vector Rays if enabled
                    if show_rays:
                        self.detector.draw_gaze_vectors(frame, eye_data)

                    # Render Virtual Laser Pointer or Target Game
                    self.laser_engine.update_and_render(frame, eye_data, selected_mode)

                    # Format Telemetry Text for Left Panel
                    metrics_text = (
                        f"👁️ Eye State: {eye_data['eye_state']}\n"
                        f"🎯 Direction: {eye_data['gaze_direction']}\n\n"
                        f"👀 EAR L: {eye_data['left_ear']:.2f} | R: {eye_data['right_ear']:.2f}\n"
                        f"😉 Blinks: {eye_data['blink_count']} (BPM: {eye_data['blinks_per_min']})\n"
                        f"😉 Left Winks: {eye_data['left_wink_count']} | Right: {eye_data['right_wink_count']}"
                    )
                    if "Game" in selected_mode:
                        g = self.laser_engine.game
                        metrics_text += f"\n\n🎮 Game Score: {g.score}\n🔥 Combo Streak: {g.streak}x"
                else:
                    metrics_text = "No face detected in webcam view.\nPosition your face in front of the camera."

                # Convert BGR frame to RGB for CustomTkinter
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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

                # Update UI safely on main thread if window exists
                if self.is_running and self.winfo_exists():
                    self.after(0, lambda img=ctk_img, text=metrics_text: self._update_frame(img, text))

                time.sleep(0.03)  # ~30 FPS

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

    def _update_frame(self, ctk_img, metrics_text):
        if not self.is_running or not self.winfo_exists():
            return
        self.current_ctk_img = ctk_img
        try:
            self.video_lbl.configure(text="", image=self.current_ctk_img)
        except Exception:
            pass
        try:
            self.metrics_lbl.configure(text=metrics_text)
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
            self.video_lbl.configure(text=f"❌ Eye Tracker Error:\n\n{clean_msg}", image=None)
        except Exception:
            pass
