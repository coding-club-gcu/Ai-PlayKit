import os
import time
import threading
import customtkinter as ctk

# Metadata describing this module for AI PlayKit Hub
PROJECT_INFO = {
    "id": "gesture_controller",
    "title": "AI Webcam Gesture Controller",
    "description": "Track hands live with MediaPipe, plant/erase blooming flowers in a garden, or draw & erase with a light blue pen!",
    "icon": "🖐️",
    "category": "Vision & Perception",
    "required_packages": ["mediapipe", "cv2", "PIL"],
    "install_command": "pip install mediapipe opencv-python pillow",
    "guide": """# 🖐️ AI Webcam Gesture Controller Guide

### Overview
This project uses **MediaPipe** to detect 21 3D hand landmarks in real-time. You can track hand skeletons, plant flowers in an **Interactive Flower Garden**, or draw with a **Light Blue Pen** using hand gestures!

---

### Step 1: Install Required Packages
Run this in your terminal:
```bash
pip install mediapipe opencv-python pillow
```

---

### Step 2: Hand Gestures Legend
- 🌸 **Pinch (Thumb + Index)**: Plant blooming flowers!
- ✌️ **Peace**: Spawn flying **Butterflies** 🦋!
- 👉 **Point Index**: Draw on screen with a **Light Blue Pen** ✏️!
- 🖐️ **Open Palm**: Eraser gesture! Move open palm to erase flowers or drawing strokes.


---

### Step 3: Beginner Python Code Example
```python
import cv2
import mediapipe as mp

# 1. Initialize MediaPipe Hands
mp_hands = mp.solutions.hands if hasattr(mp.solutions, 'hands') else None

cap = cv2.VideoCapture(0)

print("Webcam starting... Press 'q' to quit.")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip horizontally for natural mirror view
    frame = cv2.flip(frame, 1)

    cv2.imshow("Hand Gesture Controller", frame)
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


class GestureControllerUI(ctk.CTkFrame):
    """CustomTkinter UI for Real-Time MediaPipe Hand & Gesture Controller."""

    def __init__(self, parent, on_back_callback):
        super().__init__(parent, fg_color="transparent")
        self.on_back_callback = on_back_callback

        self.cap = None
        self.is_running = False
        self.detector = None
        self.garden = None
        self.draw_engine = None

        self.current_ctk_img = None

        self.setup_ui()

    def setup_ui(self):
        # Top Header bar
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
            text="🖐️ AI Webcam Gesture Controller",
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
            width=290
        )
        control_panel.pack(side="left", fill="y", padx=(0, 10), pady=0)
        control_panel.pack_propagate(False)

        ctrl_title = ctk.CTkLabel(
            control_panel,
            text="⚙ Controller Settings",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=("#1E66F5", "#89B4FA")
        )
        ctrl_title.pack(anchor="w", padx=15, pady=(15, 10))

        # Start / Stop Button
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

        # Mode Selector Dropdown
        mode_lbl = ctk.CTkLabel(
            control_panel,
            text="Visualizer Mode:",
            font=ctk.CTkFont(size=12),
            text_color=("#4C4F69", "#CDD6F4")
        )
        mode_lbl.pack(anchor="w", padx=15, pady=(15, 2))

        self.mode_dropdown = ctk.CTkOptionMenu(
            control_panel,
            values=["🌸 Interactive Flower Garden", "🎨 Draw & Erase (Light Blue Pen)", "✋ 3D Hand Skeleton & Gestures"],
            fg_color=("#1E66F5", "#89B4FA"),
            text_color=("#FFFFFF", "#11111B")
        )
        self.mode_dropdown.set("✋ 3D Hand Skeleton & Gestures")
        self.mode_dropdown.pack(fill="x", padx=15, pady=5)

        # Flower Style Dropdown
        flower_lbl = ctk.CTkLabel(
            control_panel,
            text="Flower Style:",
            font=ctk.CTkFont(size=12),
            text_color=("#4C4F69", "#CDD6F4")
        )
        flower_lbl.pack(anchor="w", padx=15, pady=(12, 2))

        self.flower_dropdown = ctk.CTkOptionMenu(
            control_panel,
            values=["Cherry Blossom 🌸", "Golden Lotus 🌻", "Cyber Violet 🪻", "Celestial Azure 🩵", "Royal Rose 🌹"],
            fg_color=("#EA76CB", "#F5C2E7"),
            text_color=("#FFFFFF", "#11111B")
        )
        self.flower_dropdown.pack(fill="x", padx=15, pady=5)

        # Clear Canvas Button
        clear_btn = ctk.CTkButton(
            control_panel,
            text="🧹 Clear Canvas",
            fg_color=("#DCE0E8", "#313244"),
            hover_color=("#BCC0CC", "#45475A"),
            text_color=("#4C4F69", "#CDD6F4"),
            command=self.clear_canvas
        )
        clear_btn.pack(fill="x", padx=15, pady=10)

        # Show Skeleton Overlay Checkbox
        self.show_skeleton_var = ctk.BooleanVar(value=True)
        show_skeleton_chk = ctk.CTkCheckBox(
            control_panel,
            text="Draw Hand Skeleton",
            variable=self.show_skeleton_var,
            text_color=("#4C4F69", "#CDD6F4")
        )
        show_skeleton_chk.pack(anchor="w", padx=15, pady=5)

        # Gesture Info / Status Box
        status_box = ctk.CTkFrame(
            control_panel,
            fg_color=("#F2F4F8", "#11111B"),
            corner_radius=10
        )
        status_box.pack(fill="both", expand=True, padx=15, pady=15)

        stat_title = ctk.CTkLabel(
            status_box,
            text="📊 Live Gesture Info",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=("#1E66F5", "#89B4FA")
        )
        stat_title.pack(anchor="w", padx=12, pady=(10, 4))

        self.gesture_summary_lbl = ctk.CTkLabel(
            status_box,
            text="Webcam is off.\n\n🌸 Garden: Pinch → Flower, Peace → Butterfly\n🎨 Draw: Point 👉 → Light Blue Pen\n🖐️ Erase: Open Palm",
            font=ctk.CTkFont(size=11),
            text_color=("#5C5F77", "#A6ADC8"),
            justify="left",
            wraplength=240
        )
        self.gesture_summary_lbl.pack(anchor="w", padx=12, pady=(0, 10))

        # Right Live Video View Frame
        self.video_container = ctk.CTkFrame(
            main_content,
            fg_color=("#F2F4F8", "#181825"),
            corner_radius=12
        )
        self.video_container.pack(side="right", fill="both", expand=True)

        self.video_lbl = ctk.CTkLabel(
            self.video_container,
            text="🎥 Live Camera Feed\nClick '▶ Start Webcam' to start hand tracking.",
            font=ctk.CTkFont(size=15),
            text_color=("#5C5F77", "#A6ADC8"),
            justify="center",
            wraplength=540
        )
        self.video_lbl.pack(expand=True, padx=20, pady=20)

    def clear_canvas(self):
        if self.garden:
            self.garden.clear()
        if self.draw_engine:
            self.draw_engine.clear()

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
        self.gesture_summary_lbl.configure(text="Webcam is off.")

    def on_back_click(self):
        self.stop_webcam()
        self.on_back_callback()

    def _webcam_loop(self):
        try:
            import cv2
            from PIL import Image
            from ai_modules.gesture_helpers.hand_detector import HandDetector
            from ai_modules.gesture_helpers.flower_garden import FlowerGardenEngine
            from ai_modules.gesture_helpers.draw_canvas import DrawCanvasEngine

            if self.detector is None:
                self.after(0, lambda: self.gesture_summary_lbl.configure(text="⏳ Loading MediaPipe Model..."))
                self.detector = HandDetector(num_hands=2)

            if self.garden is None:
                self.garden = FlowerGardenEngine()

            if self.draw_engine is None:
                self.draw_engine = DrawCanvasEngine()

            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.after(0, lambda: self._on_cam_error("Could not open webcam (Camera index 0). Please check camera permissions or close other apps."))
                return

            while self.is_running and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    break

                # Flip frame horizontally for mirror mode
                frame = cv2.flip(frame, 1)

                # Detect hands
                hands_data = self.detector.detect(frame)

                # Active UI settings
                selected_mode = self.mode_dropdown.get()
                selected_flower = self.flower_dropdown.get()
                show_skeleton = self.show_skeleton_var.get()

                is_garden_mode = "Garden" in selected_mode
                is_draw_mode = "Draw" in selected_mode

                # Update garden logic if in garden mode
                if is_garden_mode:
                    self.garden.process_hand_gestures(hands_data, selected_flower)
                    self.garden.draw(frame, hands_data)
                elif is_draw_mode:
                    self.draw_engine.process_hand_gestures(hands_data, frame.shape)
                    self.draw_engine.draw(frame, hands_data)

                # Draw Hand Skeleton overlay if enabled or in pure skeleton mode
                if show_skeleton or (not is_garden_mode and not is_draw_mode):
                    for hand in hands_data:
                        self.detector.draw_skeleton(frame, hand, draw_labels=True)

                # Build Live Gesture Summary text
                if hands_data:
                    lines = ["Active Hands Detected:"]
                    for idx, hand in enumerate(hands_data):
                        g = hand['gesture']
                        act = "Idle"
                        if g == 'PINCH':
                            act = "🌸 Planting Flower" if is_garden_mode else "🤏 Pinching"
                        elif g in ['OPEN_PALM', 'FIST']:
                            if is_garden_mode:
                                act = f"{'🖐️' if g=='OPEN_PALM' else '✊'} Erasing Flowers"
                            elif is_draw_mode:
                                act = "🖐️ Erasing Canvas" if g=='OPEN_PALM' else "✊ Fist Closed"
                            else:
                                act = f"{'🖐️' if g=='OPEN_PALM' else '✊'} Open/Closed"
                        elif g == 'PEACE':
                            act = "🦋 Spawning Butterflies" if is_garden_mode else "✌️ Peace Sign"
                        elif g == 'POINT':
                            act = "✏️ Drawing (Light Blue)" if is_draw_mode else "👉 Pointing"
                        lines.append(f"• {hand['handedness']}: {g} ({act})")
                    summary_str = "\n".join(lines)
                else:
                    summary_str = "No hands detected in webcam view.\nShow your hand to track!"

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

                # Update UI on main thread if window exists
                if self.is_running and self.winfo_exists():
                    self.after(0, lambda img=ctk_img, text=summary_str: self._update_frame(img, text))

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

    def _update_frame(self, ctk_img, summary_text):
        if not self.is_running or not self.winfo_exists():
            return
        self.current_ctk_img = ctk_img
        try:
            self.video_lbl.configure(text="", image=self.current_ctk_img)
        except Exception:
            pass
        try:
            self.gesture_summary_lbl.configure(text=summary_text)
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
            self.video_lbl.configure(text=f"❌ Gesture Detector Error:\n\n{clean_msg}", image=None)
        except Exception:
            pass
