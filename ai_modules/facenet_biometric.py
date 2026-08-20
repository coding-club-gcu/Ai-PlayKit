import os
import time
import threading
import customtkinter as ctk
from PIL import Image

# Metadata describing this module for AI PlayKit Hub
PROJECT_INFO = {
    "id": "facenet_biometric",
    "title": "FaceNet Biometric Matcher",
    "description": "Deep-learning facial recognition & biometric security system powered by FaceNet PyTorch & MTCNN. Register face profiles, extract 512-D neural embeddings, and match live webcam streams.",
    "icon": "🔐",
    "category": "Biometrics & AI Security",
    "required_packages": ["facenet_pytorch", "torch", "cv2", "PIL"],
    "install_command": "pip install facenet-pytorch torch torchvision opencv-python pillow",
    "guide": """# 🔐 FaceNet Biometric Matcher Guide

### Overview
This project implements a state-of-the-art **Deep Learning Facial Recognition & Biometric System** using **FaceNet (InceptionResnetV1)** and **MTCNN (Multi-task Cascaded Convolutional Networks)** in PyTorch.

---

### Step 1: Install Required Packages
Run the following command in your terminal:
```bash
pip install facenet-pytorch torch torchvision opencv-python pillow
```

---

### Step 2: How FaceNet Biometrics Work
1. **Face Detection & Alignment (MTCNN)**:
   Detects facial bounding boxes and 5 facial keypoints (eyes, nose, mouth corners) to align and crop faces to a standard $160 \\times 160$ input grid.
2. **512-Dimensional Deep Embeddings (FaceNet)**:
   The pre-trained **InceptionResnetV1** network maps face images into a 512-dimensional vector space where distances directly correspond to facial similarity (Triplet Loss architecture).
3. **Similarity Metrics**:
   - **Cosine Similarity ($S$)**: Measure of vector angle ($S = \\mathbf{a} \\cdot \\mathbf{b}$). Scores $\\ge 0.60$ (60%) indicate a biometric identity match.
   - **Euclidean L2 Distance ($D$)**: Distance between normalized embedding vectors ($D = \\|\\mathbf{a} - \\mathbf{b}\\|_2$). Values $< 0.90$ signal identical subjects.

---

### Step 3: Beginner Python Code Example
```python
import cv2
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

# 1. Initialize MTCNN Face Detector & FaceNet Model
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(image_size=160, margin=14, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# 2. Extract embedding from an image
def extract_embedding(image_path):
    img = Image.open(image_path).convert('RGB')
    face_tensor = mtcnn(img) # Crop and align face
    if face_tensor is not None:
        face_tensor = face_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = resnet(face_tensor).cpu().numpy()[0]
        return embedding / (embedding**2).sum()**0.5 # L2 Normalize
    return None

# 3. Compare two faces
emb1 = extract_embedding("alice.jpg")
emb2 = extract_embedding("alice_test.jpg")

if emb1 is not None and emb2 is not None:
    cosine_sim = (emb1 * emb2).sum()
    print(f"Biometric Match Score: {cosine_sim * 100:.1f}%")
```
"""
}

def check_dependencies():
    """Returns True if facenet_pytorch, torch, cv2, and PIL can be imported."""
    import importlib.util
    return (
        importlib.util.find_spec("facenet_pytorch") is not None and
        importlib.util.find_spec("torch") is not None and
        importlib.util.find_spec("cv2") is not None and
        importlib.util.find_spec("PIL") is not None
    )


class FacenetBiometricUI(ctk.CTkFrame):
    """CustomTkinter UI for FaceNet Biometric Registration & Verification."""

    def __init__(self, parent, on_back_callback):
        super().__init__(parent, fg_color="transparent")
        self.on_back_callback = on_back_callback

        self.cap = None
        self.is_running = False
        self.engine = None

        self.current_ctk_img = None
        self.latest_frame_bgr = None
        self.active_tab = "🎥 Live Scanner"

        self.setup_ui()

    def setup_ui(self):
        # 1. Top Header Bar
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
            text="🔐 FaceNet Biometric",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=("#1E1E2E", "#F5E0DC")
        )
        title_lbl.pack(side="left", padx=10)

        self.status_badge = ctk.CTkLabel(
            header_frame,
            text="Webcam Off ⏸",
            fg_color=("#FE640B", "#FAB387"),
            text_color=("#FFFFFF", "#11111B"),
            corner_radius=8,
            padx=10,
            pady=4,
            font=ctk.CTkFont(size=12, weight="bold")
        )
        self.status_badge.pack(side="right", padx=15)

        # 2. Main Work Area Container
        self.main_content = ctk.CTkFrame(self, fg_color="transparent")
        self.main_content.pack(fill="both", expand=True, padx=5)

        # 3. Top Mode Navigation Bar (Segmented Button)
        self.nav_bar = ctk.CTkSegmentedButton(
            self.main_content,
            values=["🎥 Live Scanner", "👤 Enroll Face", "🧪 1-to-1 Verification", "🗃️ Enrolled Database"],
            selected_color=("#1E66F5", "#89B4FA"),
            selected_hover_color=("#7287FD", "#B4BEFE"),
            unselected_color=("#E6E9EF", "#181825"),
            unselected_hover_color=("#CCD0DA", "#313244"),
            text_color=("black", "white"),
            font=ctk.CTkFont(size=13, weight="bold"),
            height=36,
            command=self.on_tab_change
        )
        self.nav_bar.set("🎥 Live Scanner")
        self.nav_bar.pack(fill="x", pady=(0, 10))

        # Tab Views Container
        self.view_container = ctk.CTkFrame(self.main_content, fg_color="transparent")
        self.view_container.pack(fill="both", expand=True)

        # Initialize View Frames
        self.setup_live_scanner_view()

    def on_tab_change(self, tab_name):
        self.active_tab = tab_name
        # Clear current views inside view_container
        for child in self.view_container.winfo_children():
            child.destroy()

        if tab_name == "🎥 Live Scanner":
            self.setup_live_scanner_view()
        elif tab_name == "👤 Enroll Face":
            self.setup_enroll_view()
        elif tab_name == "🧪 1-to-1 Verification":
            self.setup_verification_view()
        elif tab_name == "🗃️ Enrolled Database":
            self.setup_database_view()

    # ----------------------------------------------------
    # TAB 1: LIVE SCANNER & RECOGNITION
    # ----------------------------------------------------
    def setup_live_scanner_view(self):
        # Container frame
        scanner_frame = ctk.CTkFrame(self.view_container, fg_color="transparent")
        scanner_frame.pack(fill="both", expand=True)

        # Left Control Panel
        control_panel = ctk.CTkFrame(
            scanner_frame,
            fg_color=("#FFFFFF", "#1E1E2E"),
            corner_radius=12,
            width=290
        )
        control_panel.pack(side="left", fill="y", padx=(0, 10))
        control_panel.pack_propagate(False)

        ctrl_title = ctk.CTkLabel(
            control_panel,
            text="⚙ Scanner Controls",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=("#1E66F5", "#89B4FA")
        )
        ctrl_title.pack(anchor="w", padx=15, pady=(15, 10))

        # Start / Stop Webcam Button
        self.toggle_cam_btn = ctk.CTkButton(
            control_panel,
            text="▶ Start Scanner Camera",
            fg_color=("#40A02B", "#A6E3A1"),
            hover_color=("#207015", "#94E2D5"),
            text_color=("#FFFFFF", "#11111B"),
            font=ctk.CTkFont(size=14, weight="bold"),
            height=40,
            command=self.toggle_webcam
        )
        self.toggle_cam_btn.pack(fill="x", padx=15, pady=5)

        # Match Threshold Slider
        thresh_lbl_frame = ctk.CTkFrame(control_panel, fg_color="transparent")
        thresh_lbl_frame.pack(fill="x", padx=15, pady=(15, 2))

        t_lbl = ctk.CTkLabel(
            thresh_lbl_frame,
            text="Match Threshold:",
            font=ctk.CTkFont(size=12),
            text_color=("#4C4F69", "#CDD6F4")
        )
        t_lbl.pack(side="left")

        self.thresh_val_lbl = ctk.CTkLabel(
            thresh_lbl_frame,
            text="60%",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=("#1E66F5", "#89B4FA")
        )
        self.thresh_val_lbl.pack(side="right")

        self.thresh_slider = ctk.CTkSlider(
            control_panel,
            from_=0.40,
            to=0.85,
            number_of_steps=18,
            command=self.on_thresh_slider_change
        )
        self.thresh_slider.set(0.60)
        self.thresh_slider.pack(fill="x", padx=15, pady=5)

        # Show Facial Landmarks Checkbox
        self.show_landmarks_var = ctk.BooleanVar(value=True)
        landmarks_chk = ctk.CTkCheckBox(
            control_panel,
            text="Show MTCNN Landmarks (5-Pts)",
            variable=self.show_landmarks_var,
            text_color=("#4C4F69", "#CDD6F4")
        )
        landmarks_chk.pack(anchor="w", padx=15, pady=12)

        # Quick Register Button
        quick_reg_btn = ctk.CTkButton(
            control_panel,
            text="➕ Enroll Current Live Face",
            fg_color=("#EA76CB", "#F5C2E7"),
            hover_color=("#E64553", "#F2CDCD"),
            text_color=("#FFFFFF", "#11111B"),
            font=ctk.CTkFont(weight="bold"),
            command=lambda: self.nav_bar.set("👤 Enroll Face")
        )
        quick_reg_btn.pack(fill="x", padx=15, pady=10)

        # Live Status Card Box
        status_box = ctk.CTkFrame(
            control_panel,
            fg_color=("#F2F4F8", "#11111B"),
            corner_radius=10
        )
        status_box.pack(fill="both", expand=True, padx=15, pady=15)

        sum_title = ctk.CTkLabel(
            status_box,
            text="📊 Recognition Feed",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=("#1E66F5", "#89B4FA")
        )
        sum_title.pack(anchor="w", padx=12, pady=(10, 4))

        self.scanner_stats_lbl = ctk.CTkLabel(
            status_box,
            text="Webcam scanner is off.\nClick '▶ Start Scanner Camera' to begin biometrics.",
            font=ctk.CTkFont(size=11),
            text_color=("#5C5F77", "#A6ADC8"),
            justify="left",
            wraplength=230
        )
        self.scanner_stats_lbl.pack(anchor="w", padx=12, pady=(0, 10))

        # Right Live Video Display
        self.video_container = ctk.CTkFrame(
            scanner_frame,
            fg_color=("#F2F4F8", "#181825"),
            corner_radius=12
        )
        self.video_container.pack(side="right", fill="both", expand=True)

        self.video_lbl = ctk.CTkLabel(
            self.video_container,
            text="🎥 Live Biometric Scanner Feed\nClick '▶ Start Scanner Camera' to begin.",
            font=ctk.CTkFont(size=15),
            text_color=("#5C5F77", "#A6ADC8"),
            justify="center",
            wraplength=500
        )
        self.video_lbl.pack(expand=True, padx=20, pady=20)

    def on_thresh_slider_change(self, val):
        self.thresh_val_lbl.configure(text=f"{int(val * 100)}%")

    # ----------------------------------------------------
    # TAB 2: ENROLL / REGISTER NEW FACE
    # ----------------------------------------------------
    def setup_enroll_view(self):
        enroll_frame = ctk.CTkFrame(self.view_container, fg_color="transparent")
        enroll_frame.pack(fill="both", expand=True)

        # Left Registration Form
        form_panel = ctk.CTkFrame(
            enroll_frame,
            fg_color=("#FFFFFF", "#1E1E2E"),
            corner_radius=12,
            width=320
        )
        form_panel.pack(side="left", fill="y", padx=(0, 10))
        form_panel.pack_propagate(False)

        form_title = ctk.CTkLabel(
            form_panel,
            text="👤 Enrollee Details",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=("#1E66F5", "#89B4FA")
        )
        form_title.pack(anchor="w", padx=18, pady=(18, 10))

        # Full Name Input
        name_lbl = ctk.CTkLabel(form_panel, text="Full Name *", font=ctk.CTkFont(size=12, weight="bold"), text_color=("#4C4F69", "#CDD6F4"))
        name_lbl.pack(anchor="w", padx=18, pady=(8, 2))
        self.name_entry = ctk.CTkEntry(form_panel, placeholder_text="Name", fg_color=("#F2F4F8", "#11111B"))
        self.name_entry.pack(fill="x", padx=18, pady=(0, 15))

        # Quality Check Box
        quality_box = ctk.CTkFrame(form_panel, fg_color=("#F2F4F8", "#11111B"), corner_radius=10)
        quality_box.pack(fill="x", padx=18, pady=(0, 15))

        q_title = ctk.CTkLabel(quality_box, text="🔍 Alignment Check", font=ctk.CTkFont(size=12, weight="bold"), text_color=("#1E66F5", "#89B4FA"))
        q_title.pack(anchor="w", padx=12, pady=(8, 2))

        self.quality_lbl = ctk.CTkLabel(
            quality_box,
            text="• Face Detected: Waiting...\n• Position: Align face in frame",
            font=ctk.CTkFont(size=11),
            text_color=("#5C5F77", "#A6ADC8"),
            justify="left"
        )
        self.quality_lbl.pack(anchor="w", padx=12, pady=(0, 8))

        # Register Action Buttons
        self.capture_btn = ctk.CTkButton(
            form_panel,
            text="📸 Capture & Register Face",
            fg_color=("#40A02B", "#A6E3A1"),
            hover_color=("#207015", "#94E2D5"),
            text_color=("#FFFFFF", "#11111B"),
            font=ctk.CTkFont(size=14, weight="bold"),
            height=42,
            command=self.perform_registration
        )
        self.capture_btn.pack(fill="x", padx=18, pady=5)

        # Upload Image File Button
        file_upload_btn = ctk.CTkButton(
            form_panel,
            text="📁 Upload Image File",
            fg_color=("#DCE0E8", "#313244"),
            hover_color=("#BCC0CC", "#45475A"),
            text_color=("#4C4F69", "#CDD6F4"),
            font=ctk.CTkFont(weight="bold"),
            command=self.register_from_file
        )
        file_upload_btn.pack(fill="x", padx=18, pady=5)

        # Right Camera View Frame for Enrollment
        self.enroll_video_container = ctk.CTkFrame(
            enroll_frame,
            fg_color=("#F2F4F8", "#181825"),
            corner_radius=12
        )
        self.enroll_video_container.pack(side="right", fill="both", expand=True)

        self.enroll_video_lbl = ctk.CTkLabel(
            self.enroll_video_container,
            text="📷 Live Camera Preview\nMake sure your camera is turned on or click 'Start Camera'",
            font=ctk.CTkFont(size=14),
            text_color=("#5C5F77", "#A6ADC8"),
            justify="center"
        )
        self.enroll_video_lbl.pack(expand=True, padx=20, pady=20)

        if not self.is_running:
            self.start_webcam()

    def perform_registration(self):
        name = self.name_entry.get().strip()
        if not name:
            self.quality_lbl.configure(text="⚠️ Please enter full name first!")
            return

        if self.latest_frame_bgr is None:
            self.quality_lbl.configure(text="⚠️ No active webcam frame!")
            return

        if self.engine is None:
            from ai_modules.biometric_helpers.facenet_engine import BiometricFaceEngine
            self.engine = BiometricFaceEngine()

        frame = self.latest_frame_bgr.copy()
        boxes, probs, landmarks = self.engine.detect_faces(frame)

        if boxes is None or len(boxes) == 0:
            self.quality_lbl.configure(text="❌ No face detected in frame!\nPlease look directly into the camera.")
            return

        # Extract face embedding and face crop
        box = boxes[0]
        embedding, face_crop = self.engine.extract_face_embedding(frame, box)

        if embedding is None:
            self.quality_lbl.configure(text="❌ Face crop error. Position face clearly.")
            return

        user_rec = self.engine.register_user(
            name=name,
            embedding=embedding,
            face_crop_bgr=face_crop
        )

        self.quality_lbl.configure(text=f"✓ Registered: {user_rec['name']}\n512-D Embedding saved!")
        self.capture_btn.configure(text="Success! ✓", fg_color=("#40A02B", "#A6E3A1"))
        self.after(2000, lambda: self.capture_btn.configure(text="📸 Capture & Register Face", fg_color=("#40A02B", "#A6E3A1")))

    def register_from_file(self):
        from tkinter import filedialog
        file_path = filedialog.askopenfilename(
            title="Select Face Image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )
        if not file_path:
            return

        name = self.name_entry.get().strip()
        if not name:
            self.quality_lbl.configure(text="⚠️ Please enter full name first!")
            return

        try:
            import cv2
            img_bgr = cv2.imread(file_path)
            if img_bgr is None:
                self.quality_lbl.configure(text="❌ Could not open image file.")
                return

            if self.engine is None:
                from ai_modules.biometric_helpers.facenet_engine import BiometricFaceEngine
                self.engine = BiometricFaceEngine()

            boxes, _, _ = self.engine.detect_faces(img_bgr)
            if boxes is None or len(boxes) == 0:
                self.quality_lbl.configure(text="❌ No face detected in selected image.")
                return

            box = boxes[0]
            embedding, face_crop = self.engine.extract_face_embedding(img_bgr, box)

            if embedding is None:
                self.quality_lbl.configure(text="❌ Could not process facial crop.")
                return

            user_rec = self.engine.register_user(
                name=name,
                embedding=embedding,
                face_crop_bgr=face_crop
            )

            self.quality_lbl.configure(text=f"✓ Registered from File: {user_rec['name']}\n512-D Embedding saved!")
        except Exception as e:
            self.quality_lbl.configure(text=f"Error: {e}")

    # ----------------------------------------------------
    # TAB 3: 1-TO-1 VERIFICATION & COMPARISON
    # ----------------------------------------------------
    def setup_verification_view(self):
        verif_frame = ctk.CTkFrame(self.view_container, fg_color="transparent")
        verif_frame.pack(fill="both", expand=True)

        top_bar = ctk.CTkFrame(verif_frame, fg_color=("#FFFFFF", "#1E1E2E"), corner_radius=10)
        top_bar.pack(fill="x", pady=(0, 10))

        v_title = ctk.CTkLabel(
            top_bar,
            text="🧪 Face Verification (1-to-1 Match & 512-D Cosine Similarity)",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=("#1E66F5", "#89B4FA")
        )
        v_title.pack(side="left", padx=15, pady=12)

        compare_btn = ctk.CTkButton(
            top_bar,
            text="⚡ Run Comparison",
            fg_color=("#1E66F5", "#89B4FA"),
            hover_color=("#7287FD", "#B4BEFE"),
            text_color=("#FFFFFF", "#11111B"),
            font=ctk.CTkFont(weight="bold"),
            command=self.run_1to1_comparison
        )
        compare_btn.pack(side="right", padx=15, pady=10)

        # Image Containers (Left & Right)
        imgs_row = ctk.CTkFrame(verif_frame, fg_color="transparent")
        imgs_row.pack(fill="both", expand=True)

        # Image 1 Box
        box1 = ctk.CTkFrame(imgs_row, fg_color=("#FFFFFF", "#1E1E2E"), corner_radius=12)
        box1.pack(side="left", fill="both", expand=True, padx=(0, 5))

        b1_lbl = ctk.CTkLabel(box1, text="📷 Subject Image 1", font=ctk.CTkFont(size=14, weight="bold"), text_color=("#4C4F69", "#CDD6F4"))
        b1_lbl.pack(anchor="w", padx=15, pady=(12, 5))

        self.img1_display = ctk.CTkLabel(box1, text="Click 'Select Image 1' to load face", font=ctk.CTkFont(size=13), text_color=("#5C5F77", "#A6ADC8"))
        self.img1_display.pack(expand=True, padx=10, pady=10)

        self.img1_path = None
        self.img1_bgr = None

        load1_btn = ctk.CTkButton(box1, text="📁 Select Image 1", fg_color=("#DCE0E8", "#313244"), text_color=("#4C4F69", "#CDD6F4"), command=lambda: self.load_verify_img(1))
        load1_btn.pack(fill="x", padx=15, pady=12)

        # Image 2 Box
        box2 = ctk.CTkFrame(imgs_row, fg_color=("#FFFFFF", "#1E1E2E"), corner_radius=12)
        box2.pack(side="right", fill="both", expand=True, padx=(5, 0))

        b2_lbl = ctk.CTkLabel(box2, text="📷 Subject Image 2", font=ctk.CTkFont(size=14, weight="bold"), text_color=("#4C4F69", "#CDD6F4"))
        b2_lbl.pack(anchor="w", padx=15, pady=(12, 5))

        self.img2_display = ctk.CTkLabel(box2, text="Click 'Select Image 2' to load face", font=ctk.CTkFont(size=13), text_color=("#5C5F77", "#A6ADC8"))
        self.img2_display.pack(expand=True, padx=10, pady=10)

        self.img2_path = None
        self.img2_bgr = None

        load2_btn = ctk.CTkButton(box2, text="📁 Select Image 2", fg_color=("#DCE0E8", "#313244"), text_color=("#4C4F69", "#CDD6F4"), command=lambda: self.load_verify_img(2))
        load2_btn.pack(fill="x", padx=15, pady=12)

        # Bottom Results Meter Frame
        self.res_card = ctk.CTkFrame(verif_frame, fg_color=("#FFFFFF", "#1E1E2E"), corner_radius=12, height=100)
        self.res_card.pack(fill="x", pady=(10, 0))

        self.res_text_lbl = ctk.CTkLabel(
            self.res_card,
            text="Load 2 face images above and click 'Run Comparison' to verify biometric similarity.",
            font=ctk.CTkFont(size=13),
            text_color=("#5C5F77", "#A6ADC8")
        )
        self.res_text_lbl.pack(expand=True, padx=20, pady=15)

    def load_verify_img(self, num):
        from tkinter import filedialog
        file_path = filedialog.askopenfilename(
            title=f"Select Face Image {num}",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )
        if not file_path:
            return

        try:
            import cv2
            img_bgr = cv2.imread(file_path)
            if img_bgr is None:
                return

            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(220, 200))

            if num == 1:
                self.img1_path = file_path
                self.img1_bgr = img_bgr
                self.img1_display.configure(text="", image=ctk_img)
            else:
                self.img2_path = file_path
                self.img2_bgr = img_bgr
                self.img2_display.configure(text="", image=ctk_img)
        except Exception as e:
            print(f"Error loading image: {e}")

    def run_1to1_comparison(self):
        if self.img1_bgr is None or self.img2_bgr is None:
            self.res_text_lbl.configure(text="⚠️ Please select both Image 1 and Image 2 first!")
            return

        self.res_text_lbl.configure(text="⏳ Extracting FaceNet 512-D vectors & comparing...")
        
        def _worker():
            if self.engine is None:
                from ai_modules.biometric_helpers.facenet_engine import BiometricFaceEngine
                self.engine = BiometricFaceEngine()

            res = self.engine.compare_two_images(self.img1_bgr, self.img2_bgr)
            
            if res["status"] != "success":
                msg = f"❌ Error: {res['message']}"
                self.after(0, lambda: self.res_text_lbl.configure(text=msg))
                return

            sim_pct = res["similarity_pct"]
            cos_sim = res["cosine_sim"]
            l2_dist = res["l2_distance"]
            is_match = res["is_match"]

            match_str = "MATCH CONFIRMED ✓ (SAME INDIVIDUAL)" if is_match else "DIFFERENT INDIVIDUALS ❌"
            color_text = "#40A02B" if is_match else "#FE640B"

            result_msg = (
                f"VERDICT: {match_str}\n"
                f"• Cosine Similarity: {sim_pct:.1f}%  (Raw: {cos_sim:.4f})\n"
                f"• Euclidean L2 Distance: {l2_dist:.4f}  (Threshold: < 0.90)"
            )

            self.after(0, lambda: self.res_text_lbl.configure(
                text=result_msg,
                font=ctk.CTkFont(size=14, weight="bold"),
                text_color=color_text
            ))

        threading.Thread(target=_worker, daemon=True).start()

    # ----------------------------------------------------
    # TAB 4: ENROLLED BIOMETRICS DATABASE
    # ----------------------------------------------------
    def setup_database_view(self):
        db_frame = ctk.CTkFrame(self.view_container, fg_color="transparent")
        db_frame.pack(fill="both", expand=True)

        # Header Bar inside tab
        db_header = ctk.CTkFrame(db_frame, fg_color=("#FFFFFF", "#1E1E2E"), corner_radius=10)
        db_header.pack(fill="x", pady=(0, 10))

        title = ctk.CTkLabel(
            db_header,
            text="🗃️ Enrolled Face Biometrics Registry",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=("#1E66F5", "#89B4FA")
        )
        title.pack(side="left", padx=15, pady=12)

        refresh_btn = ctk.CTkButton(
            db_header,
            text="🔄 Refresh List",
            width=100,
            fg_color=("#DCE0E8", "#313244"),
            text_color=("#4C4F69", "#CDD6F4"),
            command=self.refresh_database_grid
        )
        refresh_btn.pack(side="right", padx=15, pady=10)

        # Scrollable Cards Grid
        self.db_scroll = ctk.CTkScrollableFrame(
            db_frame,
            fg_color="transparent",
            scrollbar_button_color=("#CBD5E1", "#2B2C3B")
        )
        self.db_scroll.pack(fill="both", expand=True)

        self.refresh_database_grid()

    def refresh_database_grid(self):
        for child in self.db_scroll.winfo_children():
            child.destroy()

        if self.engine is None:
            from ai_modules.biometric_helpers.facenet_engine import BiometricFaceEngine
            self.engine = BiometricFaceEngine()

        users = self.engine.database

        if not users:
            empty_lbl = ctk.CTkLabel(
                self.db_scroll,
                text="No face profiles enrolled yet.\nSwitch to '👤 Enroll Face' tab to add your first profile!",
                font=ctk.CTkFont(size=14),
                text_color=("#5C5F77", "#A6ADC8")
            )
            empty_lbl.pack(pady=50)
            return

        col_count = 2
        for idx, user in enumerate(users):
            row = idx // col_count
            col = idx % col_count

            card = ctk.CTkFrame(
                self.db_scroll,
                fg_color=("#FFFFFF", "#1E1E2E"),
                corner_radius=12,
                border_width=1,
                border_color=("#CCD0DA", "#313244")
            )
            card.grid(row=row, column=col, padx=8, pady=8, sticky="nsew")
            self.db_scroll.grid_columnconfigure(col, weight=1)

            # Avatar image
            avatar_path = os.path.join(self.engine.avatars_dir, user.get("avatar_filename", ""))
            avatar_img = None
            if os.path.exists(avatar_path):
                try:
                    pil_av = Image.open(avatar_path)
                    avatar_img = ctk.CTkImage(light_image=pil_av, dark_image=pil_av, size=(70, 70))
                except Exception:
                    pass

            left_av_lbl = ctk.CTkLabel(
                card,
                image=avatar_img,
                text="👤" if avatar_img is None else "",
                font=ctk.CTkFont(size=30)
            )
            left_av_lbl.pack(side="left", padx=12, pady=12)

            info_col = ctk.CTkFrame(card, fg_color="transparent")
            info_col.pack(side="left", fill="both", expand=True, padx=5, pady=10)

            u_name = ctk.CTkLabel(
                info_col,
                text=user["name"],
                font=ctk.CTkFont(size=15, weight="bold"),
                text_color=("#1E1E2E", "#F5E0DC"),
                anchor="w"
            )
            u_name.pack(fill="x", anchor="w")

            reg_date = user.get("registered_at", "").split()[0] if user.get("registered_at") else ""
            u_date = ctk.CTkLabel(
                info_col,
                text=f"Registered: {reg_date}",
                font=ctk.CTkFont(size=11),
                text_color=("#1E66F5", "#89B4FA"),
                anchor="w"
            )
            u_date.pack(fill="x", anchor="w", pady=(2, 0))

            vec_sample = ", ".join(f"{x:.2f}" for x in user["embedding"][:4]) + "..."
            u_vec = ctk.CTkLabel(
                info_col,
                text=f"512-D Vector: [{vec_sample}]",
                font=ctk.CTkFont(size=10, family="Consolas"),
                text_color=("#5C5F77", "#A6ADC8"),
                anchor="w"
            )
            u_vec.pack(fill="x", anchor="w", pady=(2, 0))

            # Delete Profile Button
            del_btn = ctk.CTkButton(
                card,
                text="🗑",
                width=34,
                height=34,
                fg_color=("#FE640B", "#F38BA8"),
                hover_color=("#D20F39", "#E64553"),
                text_color=("#FFFFFF", "#11111B"),
                command=lambda uid=user["id"]: self.delete_profile(uid)
            )
            del_btn.pack(side="right", padx=12, pady=12)

    def delete_profile(self, user_id):
        if self.engine:
            self.engine.delete_user(user_id)
            self.refresh_database_grid()

    # ----------------------------------------------------
    # WEBCAM & RECOGNITION LOOP THREADING
    # ----------------------------------------------------
    def toggle_webcam(self):
        if self.is_running:
            self.stop_webcam()
        else:
            self.start_webcam()

    def start_webcam(self):
        if self.is_running:
            return

        self.is_running = True
        if hasattr(self, "toggle_cam_btn") and self.toggle_cam_btn.winfo_exists():
            self.toggle_cam_btn.configure(
                text="⏹ Stop Scanner Camera",
                fg_color=("#FE640B", "#F38BA8"),
                hover_color=("#D20F39", "#E64553"),
                text_color=("#FFFFFF", "#11111B")
            )
        self.status_badge.configure(
            text="Scanning Live 🟢",
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

        if hasattr(self, "toggle_cam_btn") and self.toggle_cam_btn.winfo_exists():
            self.toggle_cam_btn.configure(
                text="▶ Start Scanner Camera",
                fg_color=("#40A02B", "#A6E3A1"),
                hover_color=("#207015", "#94E2D5"),
                text_color=("#FFFFFF", "#11111B")
            )
        self.status_badge.configure(
            text="Webcam Off ⏸",
            fg_color=("#FE640B", "#FAB387"),
            text_color=("#FFFFFF", "#11111B")
        )
        self.current_ctk_img = None
        self.latest_frame_bgr = None

    def on_back_click(self):
        self.stop_webcam()
        self.on_back_callback()

    def _webcam_loop(self):
        try:
            import cv2

            if self.engine is None:
                if hasattr(self, "scanner_stats_lbl") and self.scanner_stats_lbl.winfo_exists():
                    self.after(0, lambda: self.scanner_stats_lbl.configure(text="⏳ Initializing FaceNet PyTorch Engine..."))
                from ai_modules.biometric_helpers.facenet_engine import BiometricFaceEngine
                self.engine = BiometricFaceEngine()

            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.after(0, lambda: self._on_cam_error("Could not open webcam (Camera 0). Please check device permissions."))
                return

            last_time = time.time()
            fps = 30.0

            while self.is_running and self.cap.isOpened():
                start_proc_time = time.time()
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    break

                # Mirror view for natural interaction
                frame = cv2.flip(frame, 1)
                self.latest_frame_bgr = frame.copy()

                # Get sliders and check states
                thresh_val = self.thresh_slider.get() if hasattr(self, "thresh_slider") else 0.60
                show_landmarks = self.show_landmarks_var.get() if hasattr(self, "show_landmarks_var") else True

                # MTCNN Face Detection
                boxes, probs, landmarks = self.engine.detect_faces(frame)

                match_results = []
                num_matched = 0

                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        embedding, _ = self.engine.extract_face_embedding(frame, box)
                        if embedding is not None:
                            user, sim, dist = self.engine.match_face(embedding, similarity_threshold=thresh_val)
                            match_results.append((user, sim, dist))
                            if user is not None:
                                num_matched += 1
                        else:
                            match_results.append((None, 0.0, 999.0))

                # Draw annotations
                annotated_frame = self.engine.draw_annotations(
                    frame,
                    boxes,
                    probs,
                    landmarks,
                    match_results,
                    show_landmarks=show_landmarks
                )

                # FPS & Performance stats
                proc_time_ms = (time.time() - start_proc_time) * 1000.0
                curr_time = time.time()
                fps = 0.9 * fps + 0.1 * (1.0 / max(0.001, curr_time - last_time))
                last_time = curr_time

                detected_count = len(boxes) if boxes is not None else 0
                stats_text = (
                    f"• Faces Detected: {detected_count}\n"
                    f"• Biometric Matches: {num_matched}\n"
                    f"• Processing Time: {proc_time_ms:.1f} ms\n"
                    f"• Live Frame Rate: {int(fps)} FPS\n"
                    f"• Registered DB Users: {len(self.engine.database)}"
                )

                # Convert frame for CustomTkinter GUI
                rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_frame)

                # Update live view in active tab
                if self.active_tab == "🎥 Live Scanner" and hasattr(self, "video_container") and self.video_container.winfo_exists():
                    c_w = self.video_container.winfo_width()
                    c_h = self.video_container.winfo_height()
                    if c_w > 100 and c_h > 100:
                        img_w, img_h = pil_img.size
                        aspect = img_w / img_h
                        target_w = max(150, c_w - 30)
                        target_h = int(target_w / aspect)
                        if target_h > c_h - 30:
                            target_h = max(150, c_h - 30)
                            target_w = int(target_h * aspect)
                    else:
                        target_w, target_h = 580, 420

                    ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(target_w, target_h))

                    if self.is_running and self.winfo_exists():
                        self.after(0, lambda img=ctk_img, txt=stats_text: self._update_scanner_gui(img, txt))

                elif self.active_tab == "👤 Enroll Face" and hasattr(self, "enroll_video_container") and self.enroll_video_container.winfo_exists():
                    c_w = self.enroll_video_container.winfo_width()
                    c_h = self.enroll_video_container.winfo_height()
                    if c_w > 100 and c_h > 100:
                        img_w, img_h = pil_img.size
                        aspect = img_w / img_h
                        target_w = max(150, c_w - 30)
                        target_h = int(target_w / aspect)
                        if target_h > c_h - 30:
                            target_h = max(150, c_h - 30)
                            target_w = int(target_h * aspect)
                    else:
                        target_w, target_h = 540, 400

                    ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(target_w, target_h))

                    if self.is_running and self.winfo_exists():
                        self.after(0, lambda img=ctk_img: self._update_enroll_gui(img))

                time.sleep(0.03) # ~30 FPS

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

    def _update_scanner_gui(self, ctk_img, stats_text):
        if not self.is_running or not self.winfo_exists():
            return
        try:
            self.video_lbl.configure(text="", image=ctk_img)
            self.scanner_stats_lbl.configure(text=stats_text)
        except Exception:
            pass

    def _update_enroll_gui(self, ctk_img):
        if not self.is_running or not self.winfo_exists():
            return
        try:
            self.enroll_video_lbl.configure(text="", image=ctk_img)
        except Exception:
            pass

    def _on_cam_error(self, err_msg):
        if not self.winfo_exists():
            return
        self.stop_webcam()
        clean_msg = str(err_msg).replace("\t", " ").strip()
        if len(clean_msg) > 280:
            clean_msg = clean_msg[:280] + "..."
        try:
            if hasattr(self, "video_lbl"):
                self.video_lbl.configure(text=f"❌ Camera / FaceNet Error:\n\n{clean_msg}", image=None)
        except Exception:
            pass
