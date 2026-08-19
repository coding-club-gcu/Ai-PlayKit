import os
import math
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "selfie_segmenter.tflite")

def ensure_model_downloaded():
    """Download selfie_segmenter.tflite if not present locally."""
    if not os.path.exists(MODEL_PATH):
        try:
            print(f"[SelfieSegmenter] Downloading MediaPipe model to {MODEL_PATH}...")
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            print("[SelfieSegmenter] Model downloaded successfully.")
        except Exception as e:
            print(f"[SelfieSegmenter] Download error: {e}")

class SelfieSegmenterEngine:
    """MediaPipe Selfie Segmenter wrapper for real-time human detection and segmentation."""

    def __init__(self):
        ensure_model_downloaded()
        
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.ImageSegmenterOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            output_confidence_masks=True
        )
        self.segmenter = vision.ImageSegmenter.create_from_options(options)

    def process_frame(self, frame_bgr, threshold=0.5, softness=0.1, edge_blur=3):
        """
        Processes BGR webcam frame and produces a smoothed alpha mask (0.0 to 1.0)
        where 1.0 indicates human presence.
        """
        h, w, _ = frame_bgr.shape
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        result = self.segmenter.segment(mp_image)
        
        if not result.confidence_masks:
            alpha = np.ones((h, w), dtype=np.float32)
            return alpha, alpha

        conf_mask = result.confidence_masks[0].numpy_view()
        if len(conf_mask.shape) == 3:
            conf_mask = conf_mask[:, :, 0]

        # Resize mask if it differs from frame dimensions
        if conf_mask.shape[:2] != (h, w):
            conf_mask = cv2.resize(conf_mask, (w, h), interpolation=cv2.INTER_LINEAR)

        # Calculate threshold & softness curve for smooth edges
        s_val = max(0.01, softness)
        alpha = np.clip((conf_mask - threshold + s_val) / (2.0 * s_val), 0.0, 1.0).astype(np.float32)

        # Optional edge feathering via Gaussian blur
        if edge_blur > 0:
            k_size = edge_blur * 2 + 1
            alpha = cv2.GaussianBlur(alpha, (k_size, k_size), 0)
            alpha = np.clip(alpha, 0.0, 1.0)

        return alpha, conf_mask

    def close(self):
        if hasattr(self, 'segmenter') and self.segmenter:
            try:
                self.segmenter.close()
            except Exception:
                pass


class BackgroundEffectsEngine:
    """Procedural generator and alpha blender for virtual backdrops and video effects."""

    def __init__(self):
        self.frame_count = 0
        self._stars_cache = None

    def render_effect(
        self,
        frame_bgr,
        alpha_mask,
        mode="Virtual Backdrop",
        backdrop_type="Sci-Fi Cyber Grid",
        blur_amount=25,
        halo_color="Cyan",
        halo_width=5,
        custom_color_bgr=(255, 0, 128)
    ):
        """Main rendering entry point combining human cutout with chosen background mode."""
        self.frame_count += 1
        h, w, _ = frame_bgr.shape
        alpha_3d = np.dstack([alpha_mask, alpha_mask, alpha_mask])

        if mode == "🔲 Transparent Cutout":
            bg = self.create_checkerboard(w, h)
            output = (frame_bgr * alpha_3d + bg * (1.0 - alpha_3d)).astype(np.uint8)

        elif mode == "🌫️ Background Blur":
            # Apply Gaussian blur to original camera frame as background
            ksize = max(3, int(blur_amount))
            if ksize % 2 == 0:
                ksize += 1
            bg_blurred = cv2.GaussianBlur(frame_bgr, (ksize, ksize), 0)
            output = (frame_bgr * alpha_3d + bg_blurred * (1.0 - alpha_3d)).astype(np.uint8)

        elif mode == "🟩 Chroma Key Green":
            bg = np.zeros((h, w, 3), dtype=np.uint8)
            bg[:, :] = (0, 255, 0)  # Pure Green
            output = (frame_bgr * alpha_3d + bg * (1.0 - alpha_3d)).astype(np.uint8)

        elif mode == "🎨 Color Pop (B&W Back)":
            # Convert background to grayscale
            gray_bg = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            bg_bw = cv2.cvtColor(gray_bg, cv2.COLOR_GRAY2BGR)
            output = (frame_bgr * alpha_3d + bg_bw * (1.0 - alpha_3d)).astype(np.uint8)

        elif mode == "✨ Cyber Halo Glow":
            # Render virtual backdrop or blurred background, then add glowing halo around contour
            bg = self.generate_backdrop(backdrop_type, w, h, custom_color_bgr)
            blended = (frame_bgr * alpha_3d + bg * (1.0 - alpha_3d)).astype(np.uint8)
            output = self.apply_cyber_halo(blended, alpha_mask, color_name=halo_color, width=halo_width)

        else: # "🎭 Virtual Backdrop"
            bg = self.generate_backdrop(backdrop_type, w, h, custom_color_bgr)
            output = (frame_bgr * alpha_3d + bg * (1.0 - alpha_3d)).astype(np.uint8)

        return output

    def generate_backdrop(self, backdrop_type, w, h, custom_color_bgr):
        if "Sci-Fi" in backdrop_type:
            return self.create_scifi_grid(w, h)
        elif "Neon Sunset" in backdrop_type:
            return self.create_neon_sunset(w, h)
        elif "Cosmic" in backdrop_type:
            return self.create_cosmic_space(w, h)
        elif "Nature" in backdrop_type:
            return self.create_nature_bokeh(w, h)
        elif "Modern Office" in backdrop_type:
            return self.create_modern_office(w, h)
        elif "Solid Color" in backdrop_type:
            bg = np.zeros((h, w, 3), dtype=np.uint8)
            bg[:, :] = custom_color_bgr
            return bg
        else:
            return self.create_scifi_grid(w, h)

    def create_checkerboard(self, w, h, square_size=24):
        bg = np.zeros((h, w, 3), dtype=np.uint8)
        c1, c2 = (220, 220, 220), (160, 160, 160)
        for y in range(0, h, square_size):
            for x in range(0, w, square_size):
                color = c1 if ((x // square_size) + (y // square_size)) % 2 == 0 else c2
                bg[y:y+square_size, x:x+square_size] = color
        return bg

    def create_scifi_grid(self, w, h):
        bg = np.zeros((h, w, 3), dtype=np.uint8)
        # Deep navy/purple background gradient
        for y in range(h):
            ratio = y / h
            r = int(15 + ratio * 25)
            g = int(10 + ratio * 15)
            b = int(45 + ratio * 60)
            bg[y, :] = (b, g, r)

        # Dynamic perspective energy grid lines
        horizon = int(h * 0.45)
        pulse = (math.sin(self.frame_count * 0.08) + 1.0) * 0.5
        grid_color = (int(255 * (0.6 + 0.4 * pulse)), int(180 * (0.6 + 0.4 * pulse)), 40)

        # Horizontal perspective lines
        for i in range(1, 14):
            y = int(horizon + (h - horizon) * (i / 14.0) ** 2)
            cv2.line(bg, (0, y), (w, y), grid_color, 1, cv2.LINE_AA)

        # Radial perspective lines
        center_x = w // 2
        for x in range(-w, w * 2, int(w / 12)):
            cv2.line(bg, (center_x, horizon), (x, h), grid_color, 1, cv2.LINE_AA)

        # Glowing Horizon Line
        cv2.line(bg, (0, horizon), (w, horizon), (255, 200, 100), 2, cv2.LINE_AA)
        return bg

    def create_neon_sunset(self, w, h):
        bg = np.zeros((h, w, 3), dtype=np.uint8)
        # Synthwave gradient: Magenta (top) to Dark Purple/Orange (bottom)
        for y in range(h):
            ratio = y / h
            r = int(220 - ratio * 140)
            g = int(30 + ratio * 80)
            b = int(160 - ratio * 100)
            bg[y, :] = (b, g, r)

        # Sun in the middle background
        sun_center = (w // 2, int(h * 0.45))
        sun_radius = int(min(w, h) * 0.22)
        cv2.circle(bg, sun_center, sun_radius, (30, 210, 255), -1, cv2.LINE_AA)

        # Sun horizontal stripes
        for y in range(sun_center[1] - sun_radius, sun_center[1] + sun_radius, 10):
            if y > sun_center[1] and 0 <= y < h:
                stripe_h = int(3 + (y - sun_center[1]) * 0.08)
                bg[y:y+stripe_h, :] = (bg[y:y+stripe_h, :] * 0.4).astype(np.uint8)

        return bg

    def create_cosmic_space(self, w, h):
        bg = np.zeros((h, w, 3), dtype=np.uint8)
        # Galactic space background with nebula glow
        for y in range(h):
            ratio = y / h
            b = int(30 + math.sin(ratio * math.pi) * 45)
            g = int(10 + ratio * 20)
            r = int(20 + ratio * 35)
            bg[y, :] = (b, g, r)

        # Draw static star field
        if self._stars_cache is None or self._stars_cache.shape[:2] != (h, w):
            np.random.seed(42)
            self._stars_cache = np.zeros((h, w, 3), dtype=np.uint8)
            num_stars = int(w * h * 0.0006)
            for _ in range(num_stars):
                sx = np.random.randint(0, w)
                sy = np.random.randint(0, h)
                brightness = np.random.randint(150, 255)
                c_idx = np.random.choice([0, 1, 2])
                color = [brightness, brightness, brightness]
                if c_idx == 0: color = [255, 200, 180] # Warm star
                elif c_idx == 1: color = [180, 220, 255] # Cyan star
                self._stars_cache[sy, sx] = color

        bg = cv2.add(bg, self._stars_cache)
        return bg

    def create_nature_bokeh(self, w, h):
        bg = np.zeros((h, w, 3), dtype=np.uint8)
        # Forest green & warm sunlight gradient
        for y in range(h):
            ratio = y / h
            r = int(30 + ratio * 40)
            g = int(90 + ratio * 60)
            b = int(40 + ratio * 30)
            bg[y, :] = (b, g, r)

        # Soft bokeh circles
        t = self.frame_count * 0.03
        bokeh_centers = [
            (int(w * 0.2 + math.sin(t) * 20), int(h * 0.3)),
            (int(w * 0.7 + math.cos(t * 0.8) * 25), int(h * 0.25)),
            (int(w * 0.5 + math.sin(t * 1.2) * 15), int(h * 0.7)),
            (int(w * 0.85), int(h * 0.65)),
            (int(w * 0.15), int(h * 0.8))
        ]
        for idx, (cx, cy) in enumerate(bokeh_centers):
            radius = int(40 + (idx * 15))
            overlay = bg.copy()
            color = (120 + idx * 25, 220, 150 + idx * 20)
            cv2.circle(overlay, (cx, cy), radius, color, -1, cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.25, bg, 0.75, 0, bg)

        return bg

    def create_modern_office(self, w, h):
        bg = np.zeros((h, w, 3), dtype=np.uint8)
        # Warm wood and soft architectural lighting
        for y in range(h):
            ratio = y / h
            r = int(65 - ratio * 20)
            g = int(55 - ratio * 15)
            b = int(50 - ratio * 10)
            bg[y, :] = (b, g, r)

        # Soft ambient window panel highlights
        cv2.rectangle(bg, (int(w * 0.05), int(h * 0.1)), (int(w * 0.4), int(h * 0.7)), (90, 80, 75), -1)
        cv2.rectangle(bg, (int(w * 0.45), int(h * 0.1)), (int(w * 0.95), int(h * 0.7)), (80, 70, 65), -1)
        bg = cv2.GaussianBlur(bg, (51, 51), 0)
        return bg

    def apply_cyber_halo(self, frame_bgr, alpha_mask, color_name="Cyan", width=5):
        """Draws a neon glowing contour line around the segmented human silhouette."""
        h, w, _ = frame_bgr.shape
        binary_mask = (alpha_mask > 0.45).astype(np.uint8) * 255

        # Dilate mask to compute border outline
        kernel_size = max(3, width * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        dilated = cv2.dilate(binary_mask, kernel, iterations=1)
        contour_mask = cv2.subtract(dilated, binary_mask)

        # Select neon halo color in BGR
        colors = {
            "Cyan": (255, 255, 0),
            "Neon Pink": (255, 0, 255),
            "Electric Gold": (0, 215, 255),
            "Matrix Green": (0, 255, 0),
            "Vibrant Violet": (255, 50, 180)
        }
        bgr_color = colors.get(color_name, (255, 255, 0))

        # Create glow layer
        glow_layer = np.zeros((h, w, 3), dtype=np.uint8)
        glow_layer[contour_mask > 0] = bgr_color

        # Soften glow
        glow_layer = cv2.GaussianBlur(glow_layer, (15, 15), 0)

        # Blend glow over output frame
        output = cv2.add(frame_bgr, glow_layer)
        return output
