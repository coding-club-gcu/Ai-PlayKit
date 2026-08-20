import cv2
import numpy as np

class DrawCanvasEngine:
    """Manages persistent light blue pen drawing canvas and palm eraser interactions."""

    def __init__(self):
        self.canvas = None  # RGBA canvas uint8 (H, W, 4)
        self.prev_points = {}  # {handedness: (x, y)}
        # Light blue pen BGR color: (250, 180, 137) ~ Sky Light Blue (RGB 137, 180, 250)
        self.pen_color = (250, 180, 137, 255)
        self.pen_thickness = 7
        self.erase_radius = 70

    def _ensure_canvas(self, frame_shape):
        h, w, _ = frame_shape
        if self.canvas is None or self.canvas.shape[0] != h or self.canvas.shape[1] != w:
            self.canvas = np.zeros((h, w, 4), dtype=np.uint8)

    def process_hand_gestures(self, hands_data, frame_shape):
        self._ensure_canvas(frame_shape)
        active_hands = set()

        for hand in hands_data:
            handedness = hand.get('handedness', 'Right')
            active_hands.add(handedness)
            gesture = hand['gesture']

            # 1. DRAW: Index finger pointing gesture (POINT)
            if gesture == 'POINT':
                curr_pt = hand['index_tip']
                prev_pt = self.prev_points.get(handedness)

                if prev_pt is not None:
                    # Draw continuous line from previous point to current index tip
                    cv2.line(self.canvas, prev_pt, curr_pt, self.pen_color, self.pen_thickness, cv2.LINE_AA)
                else:
                    # Initial dot
                    cv2.circle(self.canvas, curr_pt, self.pen_thickness // 2, self.pen_color, -1, cv2.LINE_AA)

                self.prev_points[handedness] = curr_pt
            else:
                # Reset tracking point if not actively pointing
                self.prev_points[handedness] = None

            # 2. ERASE: Open Palm gesture (OPEN_PALM)
            if gesture == 'OPEN_PALM':
                eraser_pos = hand['palm_center']
                # Erase pixels on RGBA canvas within eraser radius
                cv2.circle(self.canvas, eraser_pos, self.erase_radius, (0, 0, 0, 0), -1, cv2.LINE_AA)

        # Clear previous points for hands no longer present
        for h_key in list(self.prev_points.keys()):
            if h_key not in active_hands:
                self.prev_points[h_key] = None

    def draw(self, frame, hands_data=None):
        if self.canvas is None:
            return

        # Alpha blend drawing canvas onto frame
        alpha = self.canvas[:, :, 3] / 255.0
        alpha_3d = alpha[:, :, np.newaxis]
        canvas_bgr = self.canvas[:, :, :3]

        # Blend: frame = canvas * alpha + frame * (1 - alpha)
        frame[:] = (canvas_bgr * alpha_3d + frame * (1.0 - alpha_3d)).astype(np.uint8)

        # Render active gesture feedback overlays (Eraser circle / Pointing tip)
        if hands_data:
            for hand in hands_data:
                gesture = hand['gesture']

                # Visual feedback for Erase
                if gesture == 'OPEN_PALM':
                    cx, cy = hand['palm_center']
                    overlay = frame.copy()
                    color = (255, 120, 50)
                    cv2.circle(overlay, (cx, cy), self.erase_radius, color, -1, cv2.LINE_AA)
                    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
                    cv2.circle(frame, (cx, cy), self.erase_radius, color, 2, cv2.LINE_AA)

                    text = "🖐️ ERASING CANVAS"
                    cv2.putText(frame, text, (cx - 70, cy - self.erase_radius - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(frame, text, (cx - 70, cy - self.erase_radius - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

                # Visual feedback for Point / Draw
                elif gesture == 'POINT':
                    ix, iy = hand['index_tip']
                    cv2.circle(frame, (ix, iy), self.pen_thickness + 3, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.circle(frame, (ix, iy), self.pen_thickness, (250, 180, 137), -1, cv2.LINE_AA)

        # Legend at bottom of video frame
        h, w, _ = frame.shape
        legend_text = "🎨 Draw Mode: 👉 Point Index to Draw (Light Blue)  |  🖐️ Open Palm to Erase"
        cv2.putText(frame, legend_text, (15, h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, legend_text, (15, h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 235, 180), 1, cv2.LINE_AA)

    def clear(self):
        """Clears drawing canvas and resets gesture tracking points."""
        if self.canvas is not None:
            self.canvas.fill(0)
        self.prev_points.clear()
