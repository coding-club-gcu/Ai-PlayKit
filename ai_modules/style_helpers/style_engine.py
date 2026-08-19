import os
import urllib.request
import cv2
import numpy as np

# Dictionary of available fast neural style transfer models
MODEL_METADATA = {
    "🌌 Starry Night (Van Gogh)": {
        "file": "starry_night.t7",
        "url": "https://cs.stanford.edu/people/jcjohns/fast-neural-style/models/instance_norm/starry_night.t7",
        "preview_file": "starry_night.jpg",
        "preview_url": "https://raw.githubusercontent.com/jcjohnson/fast-neural-style/master/images/styles/starry_night.jpg",
        "description": "Swirling vibrant blue and yellow impressionist brushstrokes of Van Gogh."
    },
    "🍬 Candy Art (Abstract)": {
        "file": "candy.t7",
        "url": "https://cs.stanford.edu/people/jcjohns/fast-neural-style/models/instance_norm/candy.t7",
        "preview_file": "candy.jpg",
        "preview_url": "https://raw.githubusercontent.com/jcjohnson/fast-neural-style/master/images/styles/candy.jpg",
        "description": "Bright, colorful abstract paint blobs and psychedelic candy shapes."
    },
    "🪞 Stained Glass Mosaic": {
        "file": "mosaic.t7",
        "url": "https://cs.stanford.edu/people/jcjohns/fast-neural-style/models/instance_norm/mosaic.t7",
        "preview_file": "mosaic.jpg",
        "preview_url": "https://raw.githubusercontent.com/jcjohnson/fast-neural-style/master/images/styles/mosaic.jpg",
        "description": "Intricate stained glass window tiles and rich tessellations."
    },
    "🎨 Udnie (Expressionism)": {
        "file": "udnie.t7",
        "url": "https://cs.stanford.edu/people/jcjohns/fast-neural-style/models/instance_norm/udnie.t7",
        "preview_file": "udnie.jpg",
        "preview_url": "https://raw.githubusercontent.com/jcjohnson/fast-neural-style/master/images/styles/udnie.jpg",
        "description": "Francis Picabia's abstract cubist dancer textures."
    },
    "🪶 Peacock Feathers": {
        "file": "feathers.t7",
        "url": "https://cs.stanford.edu/people/jcjohns/fast-neural-style/models/instance_norm/feathers.t7",
        "preview_file": "feathers.jpg",
        "preview_url": "https://raw.githubusercontent.com/jcjohnson/fast-neural-style/master/images/styles/feathers.jpg",
        "description": "Detailed organic patterns inspired by peacock feather plumage."
    },
    "🎭 La Muse (Picasso)": {
        "file": "la_muse.t7",
        "url": "https://cs.stanford.edu/people/jcjohns/fast-neural-style/models/instance_norm/la_muse.t7",
        "preview_file": "la_muse.jpg",
        "preview_url": "https://raw.githubusercontent.com/jcjohnson/fast-neural-style/master/images/styles/la_muse.jpg",
        "description": "Picasso's bold cubist lines and pastel tones."
    },
    "😱 The Scream (Munch)": {
        "file": "the_scream.t7",
        "url": "https://cs.stanford.edu/people/jcjohns/fast-neural-style/models/instance_norm/the_scream.t7",
        "preview_file": "the_scream.jpg",
        "preview_url": "https://raw.githubusercontent.com/jcjohnson/fast-neural-style/master/images/styles/the_scream.jpg",
        "description": "Wavy dramatic expressionist skies and intense emotional colors."
    }
}

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
THUMBNAILS_DIR = os.path.join(os.path.dirname(__file__), "thumbnails")

class NeuralStyleEngine:
    """Fast Feedforward Neural Style Transfer Engine powered by OpenCV DNN."""

    def __init__(self):
        os.makedirs(MODELS_DIR, exist_ok=True)
        os.makedirs(THUMBNAILS_DIR, exist_ok=True)
        self.current_model_name = None
        self.net = None

    def get_style_preview_path(self, style_name):
        """Returns local file path to the original painting thumbnail image for the specified style."""
        if style_name not in MODEL_METADATA:
            style_name = "🌌 Starry Night (Van Gogh)"
        meta = MODEL_METADATA[style_name]
        filename = meta.get("preview_file")
        if not filename:
            return None

        filepath = os.path.join(THUMBNAILS_DIR, filename)
        url = meta.get("preview_url")

        if not os.path.exists(filepath) and url:
            try:
                print(f"[NeuralStyleEngine] Downloading preview {filename}...")
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req) as resp, open(filepath, 'wb') as f:
                    f.write(resp.read())
            except Exception as e:
                print(f"[NeuralStyleEngine] Error downloading preview image: {e}")
                return None

        return filepath if os.path.exists(filepath) else None

    def load_model(self, style_name, progress_callback=None):
        """Loads or downloads the requested PyTorch neural style model."""
        if style_name not in MODEL_METADATA:
            style_name = "🌌 Starry Night (Van Gogh)"

        if self.current_model_name == style_name and self.net is not None:
            return True

        meta = MODEL_METADATA[style_name]
        filename = meta["file"]
        filepath = os.path.join(MODELS_DIR, filename)
        url = meta["url"]

        if not os.path.exists(filepath):
            if progress_callback:
                progress_callback(f"⏳ Downloading AI Model ({filename})...")
            try:
                print(f"[NeuralStyleEngine] Downloading {filename} from {url}...")
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req) as resp, open(filepath, 'wb') as f:
                    f.write(resp.read())
                print(f"[NeuralStyleEngine] Downloaded {filename} successfully.")
            except Exception as e:
                print(f"[NeuralStyleEngine] Error downloading model: {e}")
                if os.path.exists(filepath):
                    os.remove(filepath)
                return False

        try:
            if progress_callback:
                progress_callback(f"🧠 Initializing Neural Network...")
            self.net = cv2.dnn.readNetFromTorch(filepath)
            self.current_model_name = style_name
            return True
        except Exception as e:
            print(f"[NeuralStyleEngine] Error loading model into OpenCV dnn: {e}")
            self.net = None
            self.current_model_name = None
            return False

    def stylize_image(
        self,
        image_bgr,
        style_name,
        blend_ratio=1.0,
        preserve_color=False,
        target_size=None,
        progress_callback=None
    ):
        """
        Applies neural style transfer to a BGR image array.
        Returns stylized BGR output.
        """
        if not self.load_model(style_name, progress_callback):
            return image_bgr

        h, w = image_bgr.shape[:2]

        # Resize for inference if target_size specified (e.g. 640 for fast webcam)
        if target_size and max(h, w) > target_size:
            aspect = w / h
            if w > h:
                new_w = target_size
                new_h = int(target_size / aspect)
            else:
                new_h = target_size
                new_w = int(target_size * aspect)
            inf_img = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            inf_img = image_bgr

        ih, iw = inf_img.shape[:2]

        # Create 4D blob for VGG/Neural Style model
        # Mean subtraction values: (103.939, 116.779, 123.68) for BGR
        blob = cv2.dnn.blobFromImage(
            inf_img,
            1.0,
            (iw, ih),
            (103.939, 116.779, 123.68),
            swapRB=False,
            crop=False
        )

        self.net.setInput(blob)
        out = self.net.forward()

        # Reshape tensor back to 3D image format (3, H, W)
        out = out.reshape(3, out.shape[2], out.shape[3])

        # Add back channel mean values
        out[0] += 103.939
        out[1] += 116.779
        out[2] += 123.68

        # Normalize and transpose to (H, W, 3)
        out = out / 255.0
        out = out.transpose(1, 2, 0)
        out_bgr = np.clip(out * 255.0, 0, 255).astype(np.uint8)

        # Resize output back to original size if scaled
        if (out_bgr.shape[1], out_bgr.shape[0]) != (w, h):
            out_bgr = cv2.resize(out_bgr, (w, h), interpolation=cv2.INTER_CUBIC)

        # Preserve original colors (HSV color transfer)
        if preserve_color:
            out_bgr = self._transfer_color(image_bgr, out_bgr)

        # Alpha blend with original image according to blend_ratio
        if blend_ratio < 0.99:
            alpha = float(np.clip(blend_ratio, 0.0, 1.0))
            output = cv2.addWeighted(out_bgr, alpha, image_bgr, 1.0 - alpha, 0)
        else:
            output = out_bgr

        return output

    def _transfer_color(self, source_bgr, styled_bgr):
        """Transfers luminance texture from styled image to original source colors."""
        source_yuv = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2YUV)
        styled_yuv = cv2.cvtColor(styled_bgr, cv2.COLOR_BGR2YUV)

        # Combine Y channel from stylized with U & V channels from original
        merged_yuv = cv2.merge([styled_yuv[:, :, 0], source_yuv[:, :, 1], source_yuv[:, :, 2]])
        color_preserved_bgr = cv2.cvtColor(merged_yuv, cv2.COLOR_YUV2BGR)
        return color_preserved_bgr
