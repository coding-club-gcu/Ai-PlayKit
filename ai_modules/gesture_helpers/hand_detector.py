import os
import math
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")

def ensure_model_downloaded():
    """Download hand_landmarker.task if not present locally."""
    if not os.path.exists(MODEL_PATH):
        try:
            print(f"[HandDetector] Downloading MediaPipe model to {MODEL_PATH}...")
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            print("[HandDetector] Downloaded successfully.")
        except Exception as e:
            print(f"[HandDetector] Download error: {e}")

class HandDetector:
    def __init__(self, num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        ensure_model_downloaded()
        
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_tracking_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

        self.HAND_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),        # Index
            (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
            (9, 13), (13, 14), (14, 15), (15, 16), # Ring
            (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
            (0, 17)                                # Palm base
        ]

    def detect(self, frame_bgr):
        """Process a BGR webcam frame and return list of detected hand dictionaries."""
        h, w, _ = frame_bgr.shape
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        results = self.detector.detect(mp_image)

        hands_data = []
        if not results.hand_landmarks:
            return hands_data

        for i, hand_landmarks in enumerate(results.hand_landmarks):
            handedness = "Right"
            score = 0.99
            if results.handedness and i < len(results.handedness):
                handedness_info = results.handedness[i][0]
                handedness = handedness_info.category_name
                score = handedness_info.score

            px_coords = []
            norm_coords = []
            xs, ys = [], []
            for lm in hand_landmarks:
                px_x = int(lm.x * w)
                px_y = int(lm.y * h)
                px_coords.append((px_x, px_y))
                norm_coords.append((lm.x, lm.y, lm.z))
                xs.append(px_x)
                ys.append(px_y)

            bbox = (max(0, min(xs) - 15), max(0, min(ys) - 15), 
                    min(w, max(xs) + 15), min(h, max(ys) + 15))

            # Palm Center (average of wrist 0, index MCP 5, pinky MCP 17)
            palm_x = int((px_coords[0][0] + px_coords[5][0] + px_coords[17][0]) / 3)
            palm_y = int((px_coords[0][1] + px_coords[5][1] + px_coords[17][1]) / 3)
            palm_center = (palm_x, palm_y)

            # Pinch Point (midpoint between Thumb tip 4 & Index tip 8)
            thumb_tip = px_coords[4]
            index_tip = px_coords[8]
            pinch_x = int((thumb_tip[0] + index_tip[0]) / 2)
            pinch_y = int((thumb_tip[1] + index_tip[1]) / 2)
            pinch_point = (pinch_x, pinch_y)

            # Classify Gesture
            gesture, confidence = self._classify_gesture(px_coords, norm_coords, w, h)

            hands_data.append({
                'handedness': handedness,
                'score': score,
                'landmarks_px': px_coords,
                'landmarks_norm': norm_coords,
                'gesture': gesture,
                'gesture_confidence': confidence,
                'pinch_point': pinch_point,
                'palm_center': palm_center,
                'bbox': bbox,
                'index_tip': px_coords[8]
            })

        return hands_data

    def _classify_gesture(self, px, norm, frame_w, frame_h):
        """Classifies hand gesture using landmark geometric distances."""
        wrist = np.array(px[0])
        middle_mcp = np.array(px[9])
        hand_scale = max(20.0, np.linalg.norm(wrist - middle_mcp))

        # 1. Check PINCH (Thumb tip 4 & Index tip 8 close)
        thumb_tip = np.array(px[4])
        index_tip = np.array(px[8])
        pinch_dist = np.linalg.norm(thumb_tip - index_tip)
        pinch_ratio = pinch_dist / hand_scale

        if pinch_ratio < 0.38:
            confidence = min(1.0, max(0.0, 1.0 - (pinch_ratio / 0.38)))
            return 'PINCH', confidence

        # Finger extension test comparing Tip to PIP joint relative to Wrist
        # Finger tips: [4, 8, 12, 16, 20], PIP joints: [2, 6, 10, 14, 18]
        tips = [4, 8, 12, 16, 20]
        pips = [2, 6, 10, 14, 18]

        finger_extended = []
        for i in range(5):
            tip_dist = np.linalg.norm(np.array(px[tips[i]]) - wrist)
            pip_dist = np.linalg.norm(np.array(px[pips[i]]) - wrist)
            finger_extended.append(tip_dist > pip_dist * 1.12)

        extended_count = sum(finger_extended)

        # OPEN PALM: ALL 5 fingers extended straight out & wide spread
        if extended_count == 5 and finger_extended[1] and finger_extended[2] and finger_extended[3] and finger_extended[4]:
            spread_dist = np.linalg.norm(np.array(px[4]) - np.array(px[20]))
            if spread_dist > hand_scale * 1.05:
                return 'OPEN_PALM', 0.95

        # FIST: 0 extended fingers or only thumb slightly open
        if extended_count == 0 or (extended_count == 1 and not finger_extended[1] and not finger_extended[2]):
            return 'FIST', 0.95

        # PEACE / VICTORY: Index & Middle extended, Ring & Pinky folded
        if finger_extended[1] and finger_extended[2] and not finger_extended[3] and not finger_extended[4]:
            return 'PEACE', 0.90

        # POINT: Index extended, Middle, Ring, Pinky folded
        if finger_extended[1] and not finger_extended[2] and not finger_extended[3] and not finger_extended[4]:
            return 'POINT', 0.90

        return 'NEUTRAL', 0.50

    def draw_skeleton(self, frame, hand_data, draw_labels=True):
        """Renders hand skeleton overlay onto frame."""
        px = hand_data['landmarks_px']
        gesture = hand_data['gesture']
        handedness = hand_data['handedness']

        if gesture == 'OPEN_PALM':
            main_color = (255, 200, 50)
            joint_color = (255, 255, 100)
        elif gesture == 'PINCH':
            main_color = (180, 105, 255)
            joint_color = (230, 150, 255)
        elif gesture == 'PEACE':
            main_color = (50, 255, 150)
            joint_color = (150, 255, 200)
        elif gesture == 'FIST':
            main_color = (80, 80, 255)
            joint_color = (150, 150, 255)
        else:
            main_color = (0, 220, 255)
            joint_color = (200, 240, 255)

        overlay = frame.copy()

        for p1_idx, p2_idx in self.HAND_CONNECTIONS:
            pt1 = px[p1_idx]
            pt2 = px[p2_idx]
            cv2.line(overlay, pt1, pt2, main_color, 5, cv2.LINE_AA)
            cv2.line(frame, pt1, pt2, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        for idx, (x, y) in enumerate(px):
            if idx in [4, 8, 12, 16, 20]:
                cv2.circle(frame, (x, y), 7, main_color, -1, cv2.LINE_AA)
                cv2.circle(frame, (x, y), 11, joint_color, 2, cv2.LINE_AA)
                cv2.circle(frame, (x, y), 3, (255, 255, 255), -1, cv2.LINE_AA)
            else:
                cv2.circle(frame, (x, y), 4, main_color, -1, cv2.LINE_AA)
                cv2.circle(frame, (x, y), 2, (255, 255, 255), -1, cv2.LINE_AA)

        if draw_labels:
            wrist = px[0]
            label_text = f"{handedness}: {gesture}"
            cv2.putText(frame, label_text, (wrist[0] - 30, wrist[1] + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, label_text, (wrist[0] - 30, wrist[1] + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, main_color, 1, cv2.LINE_AA)

    def close(self):
        if hasattr(self, 'detector'):
            try:
                self.detector.close()
            except Exception:
                pass
