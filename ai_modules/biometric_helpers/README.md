# 🔐 FaceNet Biometric 

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.2+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch 2.2+" />
  <img src="https://img.shields.io/badge/FaceNet-InceptionResnetV1-1E66F5?style=for-the-badge" alt="FaceNet" />
  <img src="https://img.shields.io/badge/Vision-MTCNN%20%7C%20OpenCV-00599C?style=for-the-badge&logo=opencv&logoColor=white" alt="MTCNN & OpenCV" />
</p>

---

## 🌟 Overview

The **FaceNet Biometric Register & Matcher** is a high-performance computer vision and deep learning application built for real-time facial recognition, biometric identity enrollment, and 1-to-1 face verification. 

Unlike legacy face recognition algorithms that rely on raw pixel colors or geometric distance between landmarks, FaceNet uses a **Deep Convolutional Neural Network (InceptionResnetV1)** to project face images into a compact **512-dimensional vector space**. In this space, vector distances directly quantify facial similarity.

---

## ✨ Key Features

- 🎥 **Live Webcam Biometric Scanner**: Real-time camera feed with futuristic sci-fi HUD bounding boxes, MTCNN 5-point facial landmark visualizer, live FPS counter, and match confidence pill labels.
- 👤 **1-Click Enrollee Registration**: Simple face profile enrollment with Name input. Capture faces live from a webcam feed or upload image files. Automatically crops avatar thumbnails and saves 512-D embeddings.
- 🧪 **1-to-1 Verification Tool**: Compare any two face images side-by-side to compute exact **Cosine Similarity percentage** and **Euclidean $L_2$ distance** with a PASS/FAIL match verdict.
- 🗃️ **Enrolled Biometrics Registry**: Scrollable database view displaying enrolled user cards, avatar thumbnails, registration dates, 512-D vector previews, and 1-click profile deletion.
- 🎛️ **Customizable Match Threshold**: Interactive slider ($40\% - 85\%$) to adjust security matching strictness on the fly.

---

## 🧠 Deep Learning Pipeline & Mathematical Foundations

```
[ 🎥 Input Image / Video Frame ]
               │
               ▼
┌──────────────────────────────────────────────┐
│  1. MTCNN Face Detection & Alignment         │  --> Locates bounding boxes & 5 landmarks
└──────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│  2. InceptionResnetV1 Embedding Network      │  --> Maps 160x160 face crop to 512-D vector
└──────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│  3. L2 Vector Normalization                  │  --> Normalizes vector length to ||v|| = 1.0
└──────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│  4. Cosine Similarity & L2 Distance Search   │  --> S = q · d  |  D = ||q - d||
└──────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│  5. Real-Time HUD Match Output               │  --> "✓ Alex Morgan (94%)" or "UNKNOWN"
└──────────────────────────────────────────────┘
```

### 1. MTCNN Face Detection & Alignment
* **Multi-Task Cascaded Convolutional Network (MTCNN)** detects face bounding boxes and **5 facial keypoints**:
  * Left Eye Center
  * Right Eye Center
  * Nose Tip
  * Left Mouth Corner
  * Right Mouth Corner
* Faces are aligned and cropped to a standard $160 \times 160 \times 3$ RGB input tensor.

### 2. 512-Dimensional Deep Embeddings (FaceNet)
* The pre-trained **InceptionResnetV1** network (trained on VGGFace2) processes the aligned $160 \times 160$ crop and produces a 512-element floating-point vector:
  $$\mathbf{v} = [v_1, v_2, v_3, \dots, v_{512}] \in \mathbb{R}^{512}$$

### 3. $L_2$ Vector Normalization
* The raw vector is normalized to lie on a 512-D unit hypersphere:
  $$\hat{\mathbf{v}} = \frac{\mathbf{v}}{\|\mathbf{v}\|_2}$$

### 4. Similarity Metrics
* **Cosine Similarity ($S$)**: Measures the angle between query vector $\mathbf{q}$ and database vector $\mathbf{d}$:
  $$S = \mathbf{q} \cdot \mathbf{d} = \sum_{i=1}^{512} q_i d_i$$
  * Scores $\ge 0.60$ ($60\%$) represent a verified identity match.
* **Euclidean $L_2$ Distance ($D$)**: Measures geometric distance in 512-D space:
  $$D = \|\mathbf{q} - \mathbf{d}\|_2 = \sqrt{\sum_{i=1}^{512} (q_i - d_i)^2}$$
  * Values $< 0.90$ signal identical individuals.

---

## 📁 File Structure

```
AI PlayKit/
├── ai_modules/
│   ├── facenet_biometric.py           # CustomTkinter Tabbed UI View
│   └── biometric_helpers/
│       ├── README.md                  # Detailed Documentation (This File)
│       └── facenet_engine.py          # Core MTCNN & FaceNet Engine Class
└── biometric_db/                      # Local Biometric Registry (Auto-created)
    ├── database.json                  # JSON Database of Enrolled Profiles
    └── avatars/                       # Cropped Face Avatar PNG Images
```

---

## 🛠️ Required Dependencies

To run the FaceNet module independently, install:

```bash
pip install facenet-pytorch torch torchvision opencv-python pillow customtkinter
```

---

## 💻 Standalone Python Code Example

```python
import cv2
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

# 1. Initialize MTCNN & FaceNet Models
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(image_size=160, margin=14, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def get_face_embedding(image_path):
    img = Image.open(image_path).convert('RGB')
    face_tensor = mtcnn(img) # Crop & align face
    if face_tensor is not None:
        face_tensor = face_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = resnet(face_tensor).cpu().numpy()[0]
        # L2 Normalize
        return embedding / np.linalg.norm(embedding)
    return None

# 2. Extract 512-D Embeddings for two face images
emb1 = get_face_embedding("person_a.jpg")
emb2 = get_face_embedding("person_b.jpg")

if emb1 is not None and emb2 is not None:
    # 3. Calculate Cosine Similarity
    cosine_sim = np.dot(emb1, emb2)
    l2_dist = np.linalg.norm(emb1 - emb2)
    
    print(f"Similarity: {cosine_sim * 100:.1f}%")
    print(f"L2 Distance: {l2_dist:.4f}")
    if cosine_sim >= 0.60:
        print("VERDICT: MATCH CONFIRMED ✓")
    else:
        print("VERDICT: DIFFERENT INDIVIDUALS ❌")
```

---

## 📜 Credits

- **FaceNet Paper**: *FaceNet: A Unified Embedding for Face Recognition and Clustering*
- **FaceNet PyTorch Implementation**: `facenet-pytorch` by Tim Esler
