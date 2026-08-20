# 🎓 Codénix AI PlayKit Hub

<p align="center">
  <img width="100" height="100" src="https://github.com/user-attachments/assets/e9721516-22d3-4a19-9180-8bb66f146be8" />
</p>

<p align="center">
  <b>An interactive, student-friendly Python framework & modern GUI hub for exploring, building, and running cutting-edge AI & Computer Vision applications!</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.9+" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License" />
</p>

---

## 🌟 Overview

**AI PlayKit Hub** is a plug-and-play desktop application engineered for students and educators. It provides a central, beautifully styled interface powered by **CustomTkinter** that lets you run various fun Ai projects with **zero setup hassle**.

<img width="1917" height="1033" alt="image" src="https://github.com/user-attachments/assets/4b1a3473-7892-44c4-b3d6-1137e935fb06" />

---

## 🛠️ Technical AI Modules Included

AI PlayKit comes pre-packed with **10 feature-rich AI applications**:

### 1. 🤗 Local LLM Studio (`ai_modules/huggingface_llm.py`)
- **Description**: Download, load, and chat with open-source Large Language Models locally on GPU or CPU with 100% privacy and zero API fees.
- **Key Features**: Model downloader for Hugging Face repos (Qwen 2.5 0.5B, SmolLM2 360M, TinyLlama 1.1B, Llama 3.2), word-by-word streaming token output, parameter tuning (Max Tokens, Temperature, Top-P, System Prompt), hardware acceleration selector (CUDA/CPU).
- **Dependencies**: `pip install transformers torch huggingface_hub`

### 2. 🤖 Gemini AI Chatbot (`ai_modules/gemini_chat.py`)
- **Description**: Interactive conversational AI assistant powered by Google's Gemini API with real-time response generation.
- **Key Features**: API key manager, conversation history context, customizable system instructions, instant clearing.
- **Dependencies**: `pip install google-genai`

### 3. 🎯 YOLO Real-Time Object Detector (`ai_modules/object_detector.py`)
- **Description**: High-speed real-time webcam computer vision object detector powered by Ultralytics YOLOv8 and OpenCV.
- **Key Features**: Live FPS counter, class filter toggles, bounding box visualizer, threshold confidence adjustment slider.
- **Dependencies**: `pip install opencv-python ultralytics pillow`

### 4. 🖐️ AI Webcam Gesture Controller (`ai_modules/gesture_controller.py`)
- **Description**: 3D hand tracking & gesture recognition engine powering an **Interactive Flower Garden** canvas.
- **Key Features**: 
  - 🌸 **Pinch / Point Gesture**: Plant persistent blooming flower art on the virtual canvas.
  - ✊ **Fist Gesture**: Erase flowers within your fist's energy field.
  - 👆 **Index Pointer**: Draw continuous trails in 3D space.
- **Dependencies**: `pip install mediapipe opencv-python pillow`

### 5. ✂️ AI Live Background Remover (`ai_modules/background_remover.py`)
- **Description**: Real-time human portrait segmentation and backdrop replacer using MediaPipe Selfie Segmenter.
- **Key Features**: Virtual Backdrops (Cyberpunk Grid, Neon Sunset, Cosmic Galaxy, Tropical Bokeh, Modern Office), Background Blur slider, Green Screen mode, Cyber Halo outline glow.
- **Dependencies**: `pip install mediapipe opencv-python pillow`

### 6. 🎨 AI Neural Style Transfer Studio (`ai_modules/style_transfer.py`)
- **Description**: Transform photos and live webcam feeds into masterpiece paintings using Convolutional Neural Networks (CNNs).
- **Key Features**: 7 built-in artistic styles (Starry Night, Candy, Stained Glass, Udnie, Peacock, La Muse, The Scream), live webcam preview, image file import & export.
- **Dependencies**: `pip install opencv-python pillow`

### 7. 🔍 AI Text OCR Scanner (`ai_modules/text_ocr_scanner.py`)
- **Description**: Deep learning Optical Character Recognition (EasyOCR) to extract text live from images, documents, or webcam feeds.
- **Key Features**: Multi-language support (English, Spanish, French, German, Hindi), bounding box highlights, 1-click clipboard copy, and offline Text-to-Speech (TTS) read aloud.
- **Dependencies**: `pip install easyocr pyttsx3 opencv-python pillow`

### 8. 🎙️ AI Live Speech Transcriber (`ai_modules/speech_transcriber.py`)
- **Description**: Offline speech-to-text transcriber powered by OpenAI Whisper paired with an animated audio oscilloscope visualization.
- **Key Features**: Real-time live mic streaming, audio file upload (`.mp3`, `.wav`), animated 32-channel frequency spectrum, subtitle exporter (`.srt` / `.txt`), WPM telemetry.
- **Dependencies**: `pip install openai-whisper sounddevice numpy pillow`

### 9. 👁️ AI Eye Gaze Tracker & Laser (`ai_modules/eye_gaze_tracker.py`)
- **Description**: Iris tracking and head pose estimation using MediaPipe 478-point Face Mesh.
- **Key Features**: Interactive Cyber Laser Target game canvas, gaze velocity charts, blink counter telemetry, and drowsiness/attention monitors.
- **Dependencies**: `pip install mediapipe opencv-python pillow`

### 10. 📄 AI Document Summarizer (`ai_modules/document_summarizer.py`)
- **Description**: 100% offline NLP engine utilizing TextRank & TF-IDF scoring for document summarization and text analytics.
- **Key Features**: Multi-format parser (PDF, DOCX, TXT, MD), custom compression slider, sentiment mood gauge, top keyword extractor, offline TTS read aloud.
- **Dependencies**: `pip install pypdf python-docx nltk textblob pillow`

### 11. 🔐 FaceNet Biometric Matcher (`ai_modules/facenet_biometric.py`)
- **Description**: Deep-learning facial recognition & biometric authentication engine powered by FaceNet (InceptionResnetV1) PyTorch & MTCNN face alignment.
- **Key Features**: Live webcam face matching HUD with MTCNN 5-point landmark visualizer, 512-D neural embedding extractor, Enrollee Profile Registration with face crop avatars, 1-to-1 face verification tool with Cosine similarity % and L2 distance metrics, and searchable biometric database registry.
- **Dependencies**: `pip install facenet-pytorch torch torchvision opencv-python pillow`
---

## 🚀 Quick Start

### 1. Prerequisites
Ensure you have **Python 3.9** or higher installed. Check your version with:
```bash
python --version
```

### 2. Clone Repository & Install Core Requirements
```bash
git clone https://github.com/your-username/AI-PlayKit.git
cd AI-PlayKit

# Core UI package requirement:
pip install customtkinter pillow
```

### 3. Launch AI PlayKit Hub
```bash
python main.py
```

> **Note**: Heavy AI modules (PyTorch, MediaPipe, Ultralytics, Whisper, etc.) do NOT need to be installed upfront. You can install packages individually as you explore each project card inside the app!

---

## 📜 Credits & Acknowledgments

- **Developed by**: Codénix Coding Club
- **Organisation**: Girijananda Chowdhury University
- **Authors**: Akash Bora
- **Version**: 1.1.0

---


<p align="center">
  <b>Happy Coding & Exploring AI! 🚀</b>
</p>
