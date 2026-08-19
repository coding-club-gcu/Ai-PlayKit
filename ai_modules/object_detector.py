import os
import time
import threading
import customtkinter as ctk

# Metadata describing this module for AI PlayKit Hub
PROJECT_INFO = {
    "id": "object_detector",
    "title": "YOLO Real-Time Object Detector",
    "description": "Real-time computer vision object detection using OpenCV & YOLO. Detects people, objects, and devices live from your webcam.",
    "icon": "🎯",
    "category": "Vision & Perception",
    "required_packages": ["cv2", "ultralytics", "PIL"],
    "install_command": "pip install opencv-python ultralytics pillow",
    "guide": """# 🎯 YOLO Real-Time Object Detector Guide

### Overview
This project uses **OpenCV** to capture live video from your webcam and **YOLO (You Only Look Once)** to detect objects in real-time with bounding boxes and labels!

---

### Step 1: Install Required Packages
Open your terminal or command prompt and run:
```bash
pip install opencv-python ultralytics pillow
```

---

### Step 2: Beginner Python Code Example
```python
import cv2
from ultralytics import YOLO

# 1. Load pre-trained lightweight YOLOv8 model
model = YOLO("yolov8n.pt")

# 2. Open default webcam (0)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 3. Perform object detection
    results = model(frame, conf=0.5)

    # 4. Draw bounding boxes on frame
    annotated_frame = results[0].plot()

    # 5. Display the result window
    cv2.imshow("YOLO Real-Time Detection", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```
"""
}

def check_dependencies():
    """Returns True if cv2, ultralytics, and PIL are installed without importing heavy packages."""
    import importlib.util
    return (
        importlib.util.find_spec("cv2") is not None and
        importlib.util.find_spec("ultralytics") is not None and
        importlib.util.find_spec("PIL") is not None
    )


class ObjectDetectorUI(ctk.CTkFrame):
    """CustomTkinter UI for Real-Time YOLO Object Detection."""

    def __init__(self, parent, on_back_callback):
        super().__init__(parent, fg_color="transparent")
        self.on_back_callback = on_back_callback

        self.cap = None
        self.is_running = False
        self.model = None
        self.model_name = "yolov8n.pt"

        self.current_ctk_img = None
        self.reload_model_requested = False

        self.setup_ui()

    def setup_ui(self):
        # Header bar
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
            text="🎯 YOLO Real-Time Object Detector",
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

        # Main Workspace (Split into Controls & Video Frame)
        main_content = ctk.CTkFrame(self, fg_color="transparent")
        main_content.pack(fill="both", expand=True, padx=5)

        # Left Control Panel (Options)
        control_panel = ctk.CTkFrame(
            main_content,
            fg_color=("#FFFFFF", "#1E1E2E"),
            corner_radius=12,
            width=280
        )
        control_panel.pack(side="left", fill="y", padx=(0, 10), pady=0)
        control_panel.pack_propagate(False)

        ctrl_title = ctk.CTkLabel(
            control_panel,
            text="⚙ Detector Settings",
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

        # Confidence Threshold Slider
        conf_lbl_frame = ctk.CTkFrame(control_panel, fg_color="transparent")
        conf_lbl_frame.pack(fill="x", padx=15, pady=(15, 2))

        ct_lbl = ctk.CTkLabel(
            conf_lbl_frame,
            text="Confidence Threshold:",
            font=ctk.CTkFont(size=12),
            text_color=("#4C4F69", "#CDD6F4")
        )
        ct_lbl.pack(side="left")

        self.conf_val_lbl = ctk.CTkLabel(
            conf_lbl_frame,
            text="50%",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=("#1E66F5", "#89B4FA")
        )
        self.conf_val_lbl.pack(side="right")

        self.conf_slider = ctk.CTkSlider(
            control_panel,
            from_=0.15,
            to=0.95,
            number_of_steps=16,
            command=self.on_conf_change
        )
        self.conf_slider.set(0.50)
        self.conf_slider.pack(fill="x", padx=15, pady=5)

        # Model Selector
        model_lbl = ctk.CTkLabel(
            control_panel,
            text="YOLO Model Variant:",
            font=ctk.CTkFont(size=12),
            text_color=("#4C4F69", "#CDD6F4")
        )
        model_lbl.pack(anchor="w", padx=15, pady=(15, 2))

        self.model_dropdown = ctk.CTkOptionMenu(
            control_panel,
            values=["yolov8n.pt (Nano - Fast)", "yolov8s.pt (Small - Accurate)"],
            fg_color=("#1E66F5", "#89B4FA"),
            text_color=("#FFFFFF", "#11111B"),
            command=self.on_model_change
        )
        self.model_dropdown.pack(fill="x", padx=15, pady=5)

        # Draw Options Checkboxes
        self.show_labels_var = ctk.BooleanVar(value=True)
        show_labels_chk = ctk.CTkCheckBox(
            control_panel,
            text="Show Class Labels",
            variable=self.show_labels_var,
            text_color=("#4C4F69", "#CDD6F4")
        )
        show_labels_chk.pack(anchor="w", padx=15, pady=10)

        self.show_boxes_var = ctk.BooleanVar(value=True)
        show_boxes_chk = ctk.CTkCheckBox(
            control_panel,
            text="Draw Bounding Boxes",
            variable=self.show_boxes_var,
            text_color=("#4C4F69", "#CDD6F4")
        )
        show_boxes_chk.pack(anchor="w", padx=15, pady=5)

        # Detection Summary Display Box
        summary_box = ctk.CTkFrame(
            control_panel,
            fg_color=("#F2F4F8", "#11111B"),
            corner_radius=10
        )
        summary_box.pack(fill="both", expand=True, padx=15, pady=15)

        sum_title = ctk.CTkLabel(
            summary_box,
            text="📊 Live Detections",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=("#1E66F5", "#89B4FA")
        )
        sum_title.pack(anchor="w", padx=12, pady=(10, 4))

        self.summary_lbl = ctk.CTkLabel(
            summary_box,
            text="Webcam is off.\nClick Start Webcam to detect objects.",
            font=ctk.CTkFont(size=11),
            text_color=("#5C5F77", "#A6ADC8"),
            justify="left",
            wraplength=230
        )
        self.summary_lbl.pack(anchor="w", padx=12, pady=(0, 10))

        # Right Live Video View Frame
        self.video_container = ctk.CTkFrame(
            main_content,
            fg_color=("#F2F4F8", "#181825"),
            corner_radius=12
        )
        self.video_container.pack(side="right", fill="both", expand=True)

        self.video_lbl = ctk.CTkLabel(
            self.video_container,
            text="🎥 Live Camera Feed\nClick '▶ Start Webcam' to start detection.",
            font=ctk.CTkFont(size=15),
            text_color=("#5C5F77", "#A6ADC8"),
            justify="center",
            wraplength=540
        )
        self.video_lbl.pack(expand=True, padx=20, pady=20)

    def on_conf_change(self, val):
        self.conf_val_lbl.configure(text=f"{int(val * 100)}%")

    def on_model_change(self, selected_text):
        new_name = "yolov8n.pt" if "yolov8n" in selected_text else "yolov8s.pt"
        if new_name != self.model_name:
            self.model_name = new_name
            self.reload_model_requested = True

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
            text="Detecting Live 🔴",
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
        self.summary_lbl.configure(text="Webcam is off.")

    def on_back_click(self):
        self.stop_webcam()
        self.on_back_callback()

    def _webcam_loop(self):
        try:
            import cv2
            from PIL import Image

            # Patch PyTorch 2.6 weights_only=True default behavior for YOLO models
            try:
                import torch
                _orig_load = torch.load
                def _patched_torch_load(*args, **kwargs):
                    if 'weights_only' not in kwargs:
                        kwargs['weights_only'] = False
                    return _orig_load(*args, **kwargs)
                torch.load = _patched_torch_load
            except Exception:
                pass

            from ultralytics import YOLO

            if self.model is None or self.reload_model_requested:
                self.after(0, lambda: self.summary_lbl.configure(text=f"⏳ Loading {self.model_name}..."))
                self.model = YOLO(self.model_name)
                self.reload_model_requested = False

            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.after(0, lambda: self._on_cam_error("Could not open webcam (Camera index 0). Please check camera permissions or close other apps using the camera."))
                return

            while self.is_running and self.cap.isOpened():
                if self.reload_model_requested:
                    self.after(0, lambda: self.summary_lbl.configure(text=f"⏳ Loading {self.model_name}..."))
                    self.model = YOLO(self.model_name)
                    self.reload_model_requested = False

                ret, frame = self.cap.read()
                if not ret or frame is None:
                    break

                # Get confidence threshold
                conf_val = self.conf_slider.get()

                # Perform YOLO inference
                results = self.model(frame, conf=conf_val, verbose=False)
                res = results[0]

                # Draw bounding boxes / labels according to options
                show_labels = self.show_labels_var.get()
                show_boxes = self.show_boxes_var.get()

                if show_boxes:
                    annotated_frame = res.plot(labels=show_labels, conf=show_labels)
                else:
                    annotated_frame = frame.copy()

                # Count detected objects by class name
                counts = {}
                for box in res.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = self.model.names[cls_id]
                    counts[cls_name] = counts.get(cls_name, 0) + 1

                if counts:
                    summary_str = "Detected Objects:\n" + "\n".join(f"• {k}: {v}" for k, v in counts.items())
                else:
                    summary_str = "No objects detected above confidence threshold."

                # Convert OpenCV BGR image to RGB for CustomTkinter
                rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_frame)

                # Dynamically calculate frame size based on current video container dimensions
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
            self.summary_lbl.configure(text=summary_text)
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
            self.video_lbl.configure(text=f"❌ Camera / Model Error:\n\n{clean_msg}", image=None)
        except Exception:
            pass
