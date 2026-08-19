import os
import math
import time
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")


def ensure_model_downloaded():
    """Ensure face_landmarker.task model is available locally."""
    if not os.path.exists(MODEL_PATH):
        try:
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            print(f"[EyeTracker] Downloading MediaPipe Face Landmarker model to {MODEL_PATH}...")
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            print("[EyeTracker] Face Landmarker model downloaded successfully.")
        except Exception as e:
            print(f"[EyeTracker] Model download failed: {e}")


class EyeGazeDetector:
    """MediaPipe FaceLandmarker wrapper to detect 3D mesh, irises, blinks, winks, EAR, and gaze vector."""

    def __init__(self, ear_threshold=0.20, smoothing_factor=0.25):
        ensure_model_downloaded()

        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

        self.ear_threshold = ear_threshold
        self.smoothing_factor = smoothing_factor

        # Eye Landmark Indices (MediaPipe Face Mesh 478 landmarks)
        self.LEFT_EYE_OUTER = 33
        self.LEFT_EYE_INNER = 133
        self.LEFT_EYE_TOP1 = 159
        self.LEFT_EYE_BOTTOM1 = 145
        self.LEFT_EYE_TOP2 = 158
        self.LEFT_EYE_BOTTOM2 = 153
        self.LEFT_IRIS_CENTER = 468
        self.LEFT_IRIS_CONTOUR = [469, 470, 471, 472]

        self.RIGHT_EYE_OUTER = 263
        self.RIGHT_EYE_INNER = 362
        self.RIGHT_EYE_TOP1 = 386
        self.RIGHT_EYE_BOTTOM1 = 374
        self.RIGHT_EYE_TOP2 = 385
        self.RIGHT_EYE_BOTTOM2 = 380
        self.RIGHT_IRIS_CENTER = 473
        self.RIGHT_IRIS_CONTOUR = [474, 475, 476, 477]

        # Face Oval / Mesh Key Points for aesthetic cyber radar
        self.FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
        self.NOSE_BRIDGE = [6, 197, 195, 5, 4]

        # Smoothing & Stats Tracking
        self.smooth_gaze_x = 0.5
        self.smooth_gaze_y = 0.5

        self.blink_count = 0
        self.left_wink_count = 0
        self.right_wink_count = 0
        
        self.prev_eye_state = "OPEN"
        self.closed_start_time = None
        self.drowsy_alert = False

        self.start_time = time.time()
        self.blink_history = []  # list of timestamps for BPM calculation

    def _dist(self, p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def detect(self, frame_bgr):
        """Processes BGR frame and returns eye gaze metrics, landmarks, and state analysis."""
        h, w, _ = frame_bgr.shape
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        results = self.landmarker.detect(mp_image)

        if not results.face_landmarks or len(results.face_landmarks) == 0:
            return None

        landmarks = results.face_landmarks[0]
        blendshapes = {}
        if results.face_blendshapes and len(results.face_blendshapes) > 0:
            for category in results.face_blendshapes[0]:
                blendshapes[category.category_name] = category.score

        # Convert normalized 3D landmarks to pixel coords
        px_landmarks = [(int(lm.x * w), int(lm.y * h), lm.z) for lm in landmarks]

        # 1. Compute Eye Aspect Ratio (EAR)
        left_ear = self._compute_ear(px_landmarks, self.LEFT_EYE_OUTER, self.LEFT_EYE_INNER,
                                      self.LEFT_EYE_TOP1, self.LEFT_EYE_BOTTOM1,
                                      self.LEFT_EYE_TOP2, self.LEFT_EYE_BOTTOM2)

        right_ear = self._compute_ear(px_landmarks, self.RIGHT_EYE_INNER, self.RIGHT_EYE_OUTER,
                                       self.RIGHT_EYE_TOP1, self.RIGHT_EYE_BOTTOM1,
                                       self.RIGHT_EYE_TOP2, self.RIGHT_EYE_BOTTOM2)

        avg_ear = (left_ear + right_ear) / 2.0

        # 2. Eye State Classification & Blink Detection
        eye_state, is_blink_event = self._update_eye_state(left_ear, right_ear)

        # 3. Compute Iris Position & Eye Gaze Vector Angles
        left_iris_px = px_landmarks[self.LEFT_IRIS_CENTER][:2] if len(px_landmarks) > self.LEFT_IRIS_CENTER else (0, 0)
        right_iris_px = px_landmarks[self.RIGHT_IRIS_CENTER][:2] if len(px_landmarks) > self.RIGHT_IRIS_CENTER else (0, 0)

        # Calculate Iris Horizontal & Vertical Ratios relative to Eye Corners
        left_gaze_h, left_gaze_v = self._iris_ratios(px_landmarks, left_iris_px,
                                                       self.LEFT_EYE_OUTER, self.LEFT_EYE_INNER,
                                                       self.LEFT_EYE_TOP1, self.LEFT_EYE_BOTTOM1)

        right_gaze_h, right_gaze_v = self._iris_ratios(px_landmarks, right_iris_px,
                                                         self.RIGHT_EYE_INNER, self.RIGHT_EYE_OUTER,
                                                         self.RIGHT_EYE_TOP1, self.RIGHT_EYE_BOTTOM1)

        gaze_h = (left_gaze_h + right_gaze_h) / 2.0
        gaze_v = (left_gaze_v + right_gaze_v) / 2.0

        # Map iris ratio (range roughly 0.25 to 0.75) to normalized screen coordinates [0.0, 1.0]
        # Invert horizontal so looking right moves cursor right in mirrored webcam frame
        norm_raw_x = np.clip((gaze_h - 0.28) / (0.72 - 0.28), 0.0, 1.0)
        norm_raw_y = np.clip((gaze_v - 0.25) / (0.70 - 0.25), 0.0, 1.0)

        # Exponential Smoothing filter for jitter-free laser pointer control
        self.smooth_gaze_x += self.smoothing_factor * (norm_raw_x - self.smooth_gaze_x)
        self.smooth_gaze_y += self.smoothing_factor * (norm_raw_y - self.smooth_gaze_y)

        laser_px_x = int(self.smooth_gaze_x * w)
        laser_px_y = int(self.smooth_gaze_y * h)

        # Yaw and Pitch Estimation (in degrees)
        yaw_deg = float((self.smooth_gaze_x - 0.5) * 60.0)
        pitch_deg = float((self.smooth_gaze_y - 0.5) * 45.0)

        # Gaze Direction Label
        direction_label = self._classify_direction(self.smooth_gaze_x, self.smooth_gaze_y)

        # Compute Blinks per Minute (BPM)
        now = time.time()
        self.blink_history = [t for t in self.blink_history if now - t <= 60.0]
        blinks_per_min = len(self.blink_history)

        return {
            'landmarks_px': px_landmarks,
            'landmarks_raw': landmarks,
            'left_ear': left_ear,
            'right_ear': right_ear,
            'avg_ear': avg_ear,
            'eye_state': eye_state,
            'is_blink_event': is_blink_event,
            'drowsy_alert': self.drowsy_alert,
            'blink_count': self.blink_count,
            'left_wink_count': self.left_wink_count,
            'right_wink_count': self.right_wink_count,
            'blinks_per_min': blinks_per_min,
            'left_iris_px': left_iris_px,
            'right_iris_px': right_iris_px,
            'laser_norm': (self.smooth_gaze_x, self.smooth_gaze_y),
            'laser_px': (laser_px_x, laser_px_y),
            'yaw_deg': yaw_deg,
            'pitch_deg': pitch_deg,
            'gaze_direction': direction_label,
            'blendshapes': blendshapes,
            'frame_size': (w, h)
        }

    def _compute_ear(self, px_landmarks, outer_idx, inner_idx, top1_idx, bottom1_idx, top2_idx, bottom2_idx):
        p_outer = px_landmarks[outer_idx][:2]
        p_inner = px_landmarks[inner_idx][:2]
        p_top1 = px_landmarks[top1_idx][:2]
        p_bot1 = px_landmarks[bottom1_idx][:2]
        p_top2 = px_landmarks[top2_idx][:2]
        p_bot2 = px_landmarks[bottom2_idx][:2]

        v1 = self._dist(p_top1, p_bot1)
        v2 = self._dist(p_top2, p_bot2)
        horiz = self._dist(p_outer, p_inner)

        if horiz < 1e-4:
            return 0.0
        return (v1 + v2) / (2.0 * horiz)

    def _iris_ratios(self, px_landmarks, iris_center, outer_idx, inner_idx, top_idx, bottom_idx):
        p_outer = px_landmarks[outer_idx][:2]
        p_inner = px_landmarks[inner_idx][:2]
        p_top = px_landmarks[top_idx][:2]
        p_bot = px_landmarks[bottom_idx][:2]

        width = self._dist(p_outer, p_inner)
        height = self._dist(p_top, p_bot)

        if width < 1e-4 or height < 1e-4:
            return 0.5, 0.5

        # Horizontal ratio (0 = at outer corner, 1 = at inner corner)
        h_ratio = self._dist(iris_center, p_outer) / width
        # Vertical ratio (0 = at top eyelid, 1 = at bottom eyelid)
        v_ratio = self._dist(iris_center, p_top) / height

        return h_ratio, v_ratio

    def _update_eye_state(self, left_ear, right_ear):
        left_closed = left_ear < self.ear_threshold
        right_closed = right_ear < self.ear_threshold
        is_blink_event = False

        if left_closed and right_closed:
            state = "BOTH_CLOSED"
            if self.closed_start_time is None:
                self.closed_start_time = time.time()
            elif time.time() - self.closed_start_time >= 1.5:
                self.drowsy_alert = True
        elif left_closed and not right_closed:
            state = "LEFT_WINK"
            self.closed_start_time = None
            self.drowsy_alert = False
        elif right_closed and not left_closed:
            state = "RIGHT_WINK"
            self.closed_start_time = None
            self.drowsy_alert = False
        else:
            state = "OPEN"
            self.closed_start_time = None
            self.drowsy_alert = False

        # Transition detection for blink counter
        if self.prev_eye_state in ["OPEN", "LEFT_WINK", "RIGHT_WINK"] and state == "BOTH_CLOSED":
            self.blink_count += 1
            self.blink_history.append(time.time())
            is_blink_event = True
        elif self.prev_eye_state == "OPEN" and state == "LEFT_WINK":
            self.left_wink_count += 1
        elif self.prev_eye_state == "OPEN" and state == "RIGHT_WINK":
            self.right_wink_count += 1

        self.prev_eye_state = state
        return state, is_blink_event

    def _classify_direction(self, gx, gy):
        if gx < 0.38:
            h_str = "LEFT"
        elif gx > 0.62:
            h_str = "RIGHT"
        else:
            h_str = "CENTER"

        if gy < 0.38:
            v_str = "UP"
        elif gy > 0.62:
            v_str = "DOWN"
        else:
            v_str = "CENTER"

        if h_str == "CENTER" and v_str == "CENTER":
            return "🎯 CENTER LOCK"
        elif h_str == "CENTER":
            return f"⬆️ {v_str}" if v_str == "UP" else f"⬇️ {v_str}"
        elif v_str == "CENTER":
            return f"⬅️ {h_str}" if h_str == "LEFT" else f"➡️ {h_str}"
        else:
            return f"{v_str}-{h_str}"

    def draw_eye_mesh(self, frame_bgr, eye_data):
        """Draws aesthetic 3D Face Mesh & Iris contours onto the frame."""
        if not eye_data or 'landmarks_px' not in eye_data:
            return

        px = eye_data['landmarks_px']
        w, h = eye_data['frame_size']

        overlay = frame_bgr.copy()

        # Face Oval Contour
        oval_pts = np.array([px[i][:2] for i in self.FACE_OVAL if i < len(px)], dtype=np.int32)
        cv2.polylines(overlay, [oval_pts], isClosed=True, color=(137, 180, 250), thickness=2, lineType=cv2.LINE_AA)

        # Nose Bridge Line
        nose_pts = np.array([px[i][:2] for i in self.NOSE_BRIDGE if i < len(px)], dtype=np.int32)
        cv2.polylines(overlay, [nose_pts], isClosed=False, color=(245, 194, 231), thickness=2, lineType=cv2.LINE_AA)

        # Left Eye Contour
        left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        left_eye_pts = np.array([px[i][:2] for i in left_eye_indices if i < len(px)], dtype=np.int32)
        cv2.polylines(overlay, [left_eye_pts], isClosed=True, color=(166, 227, 161), thickness=2, lineType=cv2.LINE_AA)

        # Right Eye Contour
        right_eye_indices = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
        right_eye_pts = np.array([px[i][:2] for i in right_eye_indices if i < len(px)], dtype=np.int32)
        cv2.polylines(overlay, [right_eye_pts], isClosed=True, color=(166, 227, 161), thickness=2, lineType=cv2.LINE_AA)

        # Iris Circles
        left_iris_center = px[self.LEFT_IRIS_CENTER][:2]
        right_iris_center = px[self.RIGHT_IRIS_CENTER][:2]

        cv2.circle(overlay, left_iris_center, 9, (243, 139, 168), 2, cv2.LINE_AA)
        cv2.circle(overlay, left_iris_center, 3, (245, 224, 220), -1, cv2.LINE_AA)

        cv2.circle(overlay, right_iris_center, 9, (243, 139, 168), 2, cv2.LINE_AA)
        cv2.circle(overlay, right_iris_center, 3, (245, 224, 220), -1, cv2.LINE_AA)

        cv2.addWeighted(overlay, 0.45, frame_bgr, 0.55, 0, frame_bgr)

    def draw_gaze_vectors(self, frame_bgr, eye_data):
        """Draws 3D Gaze direction ray vectors projecting out from each pupil."""
        if not eye_data or 'laser_norm' not in eye_data:
            return

        lx, ly = eye_data['left_iris_px']
        rx, ry = eye_data['right_iris_px']
        gx, gy = eye_data['laser_norm']
        w, h = eye_data['frame_size']

        target_x, target_y = int(gx * w), int(gy * h)

        # Ray lines from irises to gaze point
        cv2.arrowedLine(frame_bgr, (lx, ly), (target_x, target_y), (148, 226, 213), 2, cv2.LINE_AA, tipLength=0.08)
        cv2.arrowedLine(frame_bgr, (rx, ry), (target_x, target_y), (148, 226, 213), 2, cv2.LINE_AA, tipLength=0.08)

    def close(self):
        if hasattr(self, 'landmarker'):
            try:
                self.landmarker.close()
            except Exception:
                pass
