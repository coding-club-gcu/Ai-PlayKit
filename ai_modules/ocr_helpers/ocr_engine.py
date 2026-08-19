import os
import cv2
import numpy as np

class OCREngine:
    """Robust AI OCR Engine powered by EasyOCR with OpenCV pre-processing fallbacks."""

    def __init__(self):
        self.readers = {}
        self._tts_engine = None
        self._is_speaking = False
        import threading
        self._tts_lock = threading.Lock()

    def get_reader(self, lang_code="en"):
        """Lazy loads and caches EasyOCR reader instances for requested languages."""
        if lang_code in self.readers:
            return self.readers[lang_code]

        try:
            import easyocr
            print(f"[OCREngine] Initializing EasyOCR reader for '{lang_code}'...")
            reader = easyocr.Reader([lang_code], gpu=False)
            self.readers[lang_code] = reader
            return reader
        except Exception as e:
            print(f"[OCREngine] Error initializing EasyOCR reader: {e}")
            return None

    def process_image(
        self,
        image_bgr,
        lang_code="en",
        contrast_enhance=True,
        min_confidence=0.20,
        target_size=None,
        progress_callback=None
    ):
        """
        Processes BGR image for text recognition.
        Returns:
            processed_bgr (numpy array with drawn bounding boxes),
            extracted_text (str),
            detections (list of dicts containing bbox, text, confidence)
        """
        if image_bgr is None:
            return None, "", []

        h, w = image_bgr.shape[:2]

        # Resize for faster performance if specified
        if target_size and max(h, w) > target_size:
            aspect = w / h
            if w > h:
                new_w = target_size
                new_h = int(target_size / aspect)
            else:
                new_h = target_size
                new_w = int(target_size * aspect)
            proc_img = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
            scale_x = w / new_w
            scale_y = h / new_h
        else:
            proc_img = image_bgr
            scale_x = 1.0
            scale_y = 1.0

        # Pre-process image for higher OCR accuracy
        if contrast_enhance:
            gray = cv2.cvtColor(proc_img, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced_gray = clahe.apply(gray)
            prep_bgr = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
        else:
            prep_bgr = proc_img

        reader = self.get_reader(lang_code)
        detections = []
        full_text_lines = []

        if reader is not None:
            try:
                if progress_callback:
                    progress_callback("🔍 Scanning text with AI Neural Network...")
                results = reader.readtext(prep_bgr)

                for bbox, text, prob in results:
                    if prob >= min_confidence and text.strip():
                        # Scale bbox back to original image coordinates if resized
                        pts = []
                        for pt in bbox:
                            pts.append([int(pt[0] * scale_x), int(pt[1] * scale_y)])

                        pts_np = np.array(pts, dtype=np.int32)
                        x_min = int(min(pt[0] for pt in pts) * scale_x)
                        y_min = int(min(pt[1] for pt in pts) * scale_y)
                        x_max = int(max(pt[0] for pt in pts) * scale_x)
                        y_max = int(max(pt[1] for pt in pts) * scale_y)

                        detections.append({
                            "text": text.strip(),
                            "confidence": float(prob),
                            "polygon": pts_np,
                            "bbox": (x_min, y_min, x_max - x_min, y_max - y_min)
                        })
                        full_text_lines.append(text.strip())
            except Exception as e:
                print(f"[OCREngine] EasyOCR detection error: {e}")

        # Fallback OpenCV morphological text detector if no EasyOCR detections found
        if not detections:
            detections, full_text_lines = self._opencv_fallback_text_detect(image_bgr)

        # Draw bounding boxes and text overlays
        annotated_bgr = self.draw_bounding_boxes(image_bgr.copy(), detections)
        extracted_text = "\n".join(full_text_lines)

        return annotated_bgr, extracted_text, detections

    def draw_bounding_boxes(self, image_bgr, detections):
        """Draws glowing bounding boxes and confidence tags over text detections."""
        overlay = image_bgr.copy()

        for det in detections:
            text = det["text"]
            conf = det["confidence"]
            color = (255, 102, 30) if conf > 0.6 else (30, 200, 255)

            if "polygon" in det and len(det["polygon"]) == 4:
                pts = det["polygon"].reshape((-1, 1, 2))
                cv2.polylines(overlay, [pts], isClosed=True, color=color, thickness=2)
                x_min, y_min = pts[0][0][0], pts[0][0][1]
            else:
                x, y, w, h = det["bbox"]
                cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
                x_min, y_min = x, y

            label = f"{text} ({int(conf * 100)}%)"
            font_scale = 0.45
            thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

            # Draw background tag for text readability
            tag_y1 = max(0, y_min - text_h - 6)
            tag_y2 = max(text_h + 6, y_min)
            cv2.rectangle(overlay, (x_min, tag_y1), (x_min + text_w + 6, tag_y2), (20, 20, 30), -1)
            cv2.putText(overlay, label, (x_min + 3, tag_y2 - 3), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        return overlay

    def speak_text(self, text, on_finished_callback=None):
        """Synthesizes offline Text-to-Speech (TTS) using pyttsx3."""
        if not text.strip():
            return False

        self.stop_speaking()

        def _tts_thread():
            import pyttsx3
            with self._tts_lock:
                self._is_speaking = True
                try:
                    self._tts_engine = pyttsx3.init()
                    self._tts_engine.setProperty('rate', 160)
                    self._tts_engine.say(text[:800])
                    self._tts_engine.runAndWait()
                except Exception as ex:
                    print(f"TTS Thread Exception: {ex}")
                finally:
                    self._is_speaking = False
                    self._tts_engine = None
                    if on_finished_callback:
                        try:
                            on_finished_callback()
                        except Exception:
                            pass

        import threading
        threading.Thread(target=_tts_thread, daemon=True).start()
        return True

    def stop_speaking(self):
        """Stops active speech synthesis immediately."""
        with self._tts_lock:
            if self._is_speaking and self._tts_engine is not None:
                try:
                    self._tts_engine.stop()
                except Exception as e:
                    print(f"Stop TTS Error: {e}")
            self._is_speaking = False
            self._tts_engine = None

    def _opencv_fallback_text_detect(self, image_bgr):
        """Morphological text region detection fallback when deep learning OCR is unavailable."""
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, morph_kernel)
        _, thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        lines = []

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 30 and h > 10 and (w / h) > 1.2:
                detections.append({
                    "text": "[Text Region]",
                    "confidence": 0.50,
                    "bbox": (x, y, w, h)
                })
                lines.append("[Text Region Detected]")

        return detections, lines
