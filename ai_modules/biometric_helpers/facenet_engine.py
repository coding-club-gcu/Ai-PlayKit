import os
import json
import time
import datetime
import numpy as np
import cv2
from PIL import Image
import torch

class BiometricFaceEngine:
    """
    Biometric Face Recognition Engine using MTCNN for face alignment
    and FaceNet (InceptionResnetV1) for 512-D feature vector extraction.
    """

    def __init__(self, db_dir="biometric_db"):
        self.db_dir = db_dir
        self.db_file = os.path.join(self.db_dir, "database.json")
        self.avatars_dir = os.path.join(self.db_dir, "avatars")
        os.makedirs(self.avatars_dir, exist_ok=True)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.mtcnn = None
        self.resnet = None
        self._models_loaded = False

        self.database = self.load_db()

    def load_models(self):
        """Lazy loader for PyTorch MTCNN and InceptionResnetV1 models."""
        if self._models_loaded:
            return

        from facenet_pytorch import MTCNN, InceptionResnetV1

        # Patch PyTorch weights_only issue if present
        try:
            _orig_load = torch.load
            def _patched_load(*args, **kwargs):
                if 'weights_only' not in kwargs:
                    kwargs['weights_only'] = False
                return _orig_load(*args, **kwargs)
            torch.load = _patched_load
        except Exception:
            pass

        self.mtcnn = MTCNN(
            image_size=160,
            margin=14,
            keep_all=True,
            min_face_size=35,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,
            device=self.device
        )

        self.resnet = InceptionResnetV1(pretrained="vggface2", classify=False).eval().to(self.device)
        self._models_loaded = True

    def load_db(self):
        """Loads registered face database from JSON file."""
        if os.path.exists(self.db_file):
            try:
                with open(self.db_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data
            except Exception as e:
                print(f"Error reading biometric database: {e}")
        return []

    def save_db(self):
        """Saves registered face database to JSON file."""
        try:
            with open(self.db_file, "w", encoding="utf-8") as f:
                json.dump(self.database, f, indent=2)
        except Exception as e:
            print(f"Error saving biometric database: {e}")

    def detect_faces(self, frame_bgr):
        """
        Detects faces in BGR OpenCV frame using MTCNN.
        Returns:
            boxes: array of bounding boxes [[x1, y1, x2, y2], ...] or None
            probs: confidence scores for each box or None
            landmarks: array of 5 facial landmarks per face or None
        """
        self.load_models()
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)

        try:
            boxes, probs, landmarks = self.mtcnn.detect(pil_img, landmarks=True)
            return boxes, probs, landmarks
        except Exception as e:
            print(f"MTCNN detection error: {e}")
            return None, None, None

    def get_embedding_from_crop(self, face_pil):
        """
        Calculates normalized 512-D embedding tensor for a cropped PIL face image.
        """
        self.load_models()
        # Transform PIL face to PyTorch tensor [-1, 1] normalized
        face_pil = face_pil.resize((160, 160), Image.BILINEAR)
        img_np = np.array(face_pil, dtype=np.float32)
        
        # Standard FaceNet normalization: (x - 127.5) / 128.0
        img_np = (img_np - 127.5) / 128.0
        img_tensor = torch.tensor(img_np).permute(2, 0, 1).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            embedding = self.resnet(img_tensor).cpu().numpy()[0]

        # L2 normalize the embedding vector
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def extract_face_embedding(self, frame_bgr, box):
        """
        Crops face from frame according to bounding box and extracts 512-D embedding.
        """
        h, w, _ = frame_bgr.shape
        x1, y1, x2, y2 = [int(v) for v in box]

        # Add 10% margin around box
        box_w = x2 - x1
        box_h = y2 - y1
        pad_x = int(box_w * 0.1)
        pad_y = int(box_h * 0.1)

        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)

        if x2 - x1 < 10 or y2 - y1 < 10:
            return None, None

        crop_bgr = frame_bgr[y1:y2, x1:x2]
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        crop_pil = Image.fromarray(crop_rgb)

        embedding = self.get_embedding_from_crop(crop_pil)
        return embedding, crop_bgr

    def match_face(self, query_embedding, similarity_threshold=0.60):
        """
        Compares query 512-D embedding vector against all registered embeddings.
        Uses Cosine Similarity and Euclidean L2 distance.
        Returns:
            best_match: user record dict or None
            max_sim: similarity score float (0.0 to 1.0)
            min_dist: Euclidean L2 distance float
        """
        if not self.database or query_embedding is None:
            return None, 0.0, 999.0

        best_match = None
        max_sim = -1.0
        min_dist = 999.0

        q_vec = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm

        for user in self.database:
            db_vec = np.array(user["embedding"], dtype=np.float32)
            db_norm = np.linalg.norm(db_vec)
            if db_norm > 0:
                db_vec = db_vec / db_norm

            # Cosine similarity: dot product of normalized vectors
            sim = float(np.dot(q_vec, db_vec))
            # L2 Euclidean distance
            dist = float(np.linalg.norm(q_vec - db_vec))

            if sim > max_sim:
                max_sim = sim
                min_dist = dist
                best_match = user

        if max_sim >= similarity_threshold:
            return best_match, max_sim, min_dist
        else:
            return None, max_sim, min_dist

    def compare_two_images(self, img1_bgr, img2_bgr):
        """
        Compares 1-to-1 faces in two images.
        Returns dictionary with detection status, similarity, L2 distance, and verdict.
        """
        boxes1, _, _ = self.detect_faces(img1_bgr)
        boxes2, _, _ = self.detect_faces(img2_bgr)

        if boxes1 is None or len(boxes1) == 0:
            return {"status": "error", "message": "No face detected in Image 1"}
        if boxes2 is None or len(boxes2) == 0:
            return {"status": "error", "message": "No face detected in Image 2"}

        emb1, crop1 = self.extract_face_embedding(img1_bgr, boxes1[0])
        emb2, crop2 = self.extract_face_embedding(img2_bgr, boxes2[0])

        if emb1 is None or emb2 is None:
            return {"status": "error", "message": "Could not process facial embeddings."}

        # Cosine similarity and L2 distance
        v1 = emb1 / np.linalg.norm(emb1)
        v2 = emb2 / np.linalg.norm(emb2)

        sim = float(np.dot(v1, v2))
        l2_dist = float(np.linalg.norm(v1 - v2))
        is_match = sim >= 0.60

        return {
            "status": "success",
            "similarity_pct": max(0.0, min(100.0, sim * 100.0)),
            "cosine_sim": sim,
            "l2_distance": l2_dist,
            "is_match": is_match,
            "crop1": crop1,
            "crop2": crop2
        }

    def register_user(self, name, embedding, face_crop_bgr):
        """
        Registers a new user profile with face embedding and avatar crop image.
        """
        user_id = f"usr_{int(time.time() * 1000)}"
        avatar_filename = f"{user_id}.png"
        avatar_path = os.path.join(self.avatars_dir, avatar_filename)

        # Save avatar thumbnail crop
        try:
            cv2.imwrite(avatar_path, face_crop_bgr)
        except Exception as e:
            print(f"Error saving avatar image: {e}")

        user_record = {
            "id": user_id,
            "name": name.strip(),
            "embedding": [float(x) for x in embedding],
            "registered_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "avatar_filename": avatar_filename
        }

        self.database.append(user_record)
        self.save_db()
        return user_record

    def delete_user(self, user_id):
        """Deletes user profile and avatar from database."""
        user_to_del = None
        for u in self.database:
            if u["id"] == user_id:
                user_to_del = u
                break

        if user_to_del:
            self.database.remove(user_to_del)
            self.save_db()
            avatar_path = os.path.join(self.avatars_dir, user_to_del.get("avatar_filename", ""))
            if os.path.exists(avatar_path):
                try:
                    os.remove(avatar_path)
                except Exception:
                    pass
            return True
        return False

    def draw_annotations(self, frame_bgr, boxes, probs, landmarks, match_results, show_landmarks=True):
        """
        Draws professional futuristic HUD annotations for detected faces.
        """
        annotated = frame_bgr.copy()
        if boxes is None or len(boxes) == 0:
            return annotated

        for idx, box in enumerate(boxes):
            prob = probs[idx] if probs is not None else 1.0
            if prob < 0.60:
                continue

            x1, y1, x2, y2 = [int(v) for v in box]
            w = x2 - x1
            h = y2 - y1

            match_info = match_results[idx] if idx < len(match_results) else (None, 0.0, 999.0)
            user, sim, dist = match_info

            is_match = user is not None
            box_color = (0, 230, 115) if is_match else (50, 115, 255) # Green vs Neon Red/Orange (BGR)

            # Draw futuristic corner brackets instead of simple rectangle
            line_len = min(w, h) // 4
            thick = 2

            # Top-left corner
            cv2.line(annotated, (x1, y1), (x1 + line_len, y1), box_color, thick)
            cv2.line(annotated, (x1, y1), (x1, y1 + line_len), box_color, thick)
            # Top-right corner
            cv2.line(annotated, (x2, y1), (x2 - line_len, y1), box_color, thick)
            cv2.line(annotated, (x2, y1), (x2, y1 + line_len), box_color, thick)
            # Bottom-left corner
            cv2.line(annotated, (x1, y2), (x1 + line_len, y2), box_color, thick)
            cv2.line(annotated, (x1, y2), (x1, y2 - line_len), box_color, thick)
            # Bottom-right corner
            cv2.line(annotated, (x2, y2), (x2 - line_len, y2), box_color, thick)
            cv2.line(annotated, (x2, y2), (x2, y2 - line_len), box_color, thick)

            # Draw translucent box fill for subtle sci-fi effect
            overlay = annotated.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), box_color, -1)
            cv2.addWeighted(overlay, 0.08, annotated, 0.92, 0, annotated)

            # Draw facial landmarks (5 keypoints: eyes, nose, mouth corners)
            if show_landmarks and landmarks is not None and idx < len(landmarks):
                pts = landmarks[idx]
                for px, py in pts:
                    cv2.circle(annotated, (int(px), int(py)), 3, (255, 255, 0), -1) # Yellow dots
                    cv2.circle(annotated, (int(px), int(py)), 5, (255, 255, 0), 1)

            # Label Header Banner
            sim_pct = int(max(0.0, sim) * 100.0)
            if is_match:
                label_text = f"✓ {user['name']} ({sim_pct}%)"
                sub_text = f"L2 Dist: {dist:.2f}"
            else:
                label_text = f"UNKNOWN FACE ({sim_pct}%)"
                sub_text = f"No DB Match | L2: {dist:.2f}"

            # Calculate label background pill
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw1, th1), _ = cv2.getTextSize(label_text, font, 0.55, 2)
            (tw2, th2), _ = cv2.getTextSize(sub_text, font, 0.40, 1)

            banner_w = max(tw1, tw2) + 20
            banner_h = th1 + th2 + 16

            by1 = max(0, y1 - banner_h - 4)
            by2 = by1 + banner_h
            bx1 = x1
            bx2 = min(annotated.shape[1], x1 + banner_w)

            cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (20, 20, 30), -1)
            cv2.rectangle(annotated, (bx1, by1), (bx2, by2), box_color, 1)

            # Render text
            cv2.putText(annotated, label_text, (bx1 + 10, by1 + th1 + 4), font, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated, sub_text, (bx1 + 10, by1 + th1 + th2 + 10), font, 0.40, (200, 220, 255), 1, cv2.LINE_AA)

        return annotated
