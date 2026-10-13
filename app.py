import os

# Render Free has no GPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import io
import traceback

import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, send_file, render_template

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024

# Largest side (px) of the crop that is sent to LaMa. Keeps CPU/RAM usage low.
MODEL_MAX_SIDE = 1024

lama = None
lama_error = None


# ------------------------------------------------------------
# LaMa loader
def get_lama():
    global lama, lama_error
    if lama is not None:
        return lama
    try:
        import torch
        from simple_lama_inpainting import SimpleLama
        import os
        import gc

        print("Loading LaMa on CPU (Memory Optimized)...")
        
        # تفريغ الذاكرة المتاحة قبل التحميل
        gc.collect()
        
        hub_dir = torch.hub.get_dir()
        model_path = os.path.join(hub_dir, "checkpoints", "big-lama.pt")
        
        if os.path.exists(model_path):
            # تحميل النموذج مع تعطيل التتبع (no_grad) لتوفير RAM
            with torch.no_grad():
                loaded_model = torch.jit.load(model_path, map_location=torch.device("cpu"))
                loaded_model.eval()
            
            lama = SimpleLama.__new__(SimpleLama)
            lama.model = loaded_model
            lama.device = torch.device("cpu")
        else:
            lama = SimpleLama(device=torch.device("cpu"))
            
        lama_error = None
        print("LaMa loaded successfully with low memory footprint.")
        return lama
    except Exception as e:
        lama = None
        lama_error = repr(e)
        import traceback
        traceback.print_exc()
        return None
# ------------------------------------------------------------
# Input reading
# ------------------------------------------------------------
def read_uploaded_image(file):
    data = file.read() if file else None
    if not data:
        return None
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def read_mask(file, width, height):
    """The browser mask is a transparent PNG with white strokes.
    The ALPHA channel is the real mask (erased parts keep white RGB but
    become transparent, so reading RGB/gray alone is wrong)."""
    data = file.read() if file else None
    if not data:
        return None
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None

    if img.ndim == 3 and img.shape[2] == 4:
        mask = img[:, :, 3]
    elif img.ndim == 3:
        mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        mask = img

    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return np.where(mask > 20, 255, 0).astype(np.uint8)


def improve_mask(mask):
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    # grow a bit so shadows / edge halos of the object are covered
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=3)
    return mask


# ------------------------------------------------------------
# Inpainting (crop around the mask -> LaMa -> feathered paste back)
# ------------------------------------------------------------
def run_lama(image, mask):
    model = get_lama()
    if model is None:
        raise RuntimeError("LaMa is not available. " + str(lama_error))

    h, w = image.shape[:2]

    ys, xs = np.where(mask > 0)
    x1, x2 = xs.min(), xs.max() + 1
    y1, y2 = ys.min(), ys.max() + 1

    # extra context around the object so LaMa sees the surrounding background
    margin = max(64, int(0.5 * max(x2 - x1, y2 - y1)))
    cx1, cy1 = max(0, x1 - margin), max(0, y1 - margin)
    cx2, cy2 = min(w, x2 + margin), min(h, y2 + margin)

    crop_img = image[cy1:cy2, cx1:cx2]
    crop_mask = mask[cy1:cy2, cx1:cx2]
    ch, cw = crop_img.shape[:2]

    # downscale only if the crop is big
    scale = min(1.0, MODEL_MAX_SIDE / max(ch, cw))
    if scale < 1.0:
        mw, mh = max(8, int(cw * scale)), max(8, int(ch * scale))
        in_img = cv2.resize(crop_img, (mw, mh), interpolation=cv2.INTER_AREA)
        in_mask = cv2.resize(crop_mask, (mw, mh), interpolation=cv2.INTER_NEAREST)
    else:
        in_img, in_mask = crop_img, crop_mask

    pil_img = Image.fromarray(cv2.cvtColor(in_img, cv2.COLOR_BGR2RGB))
    pil_mask = Image.fromarray(in_mask).convert("L")

    print("Running LaMa on crop:", pil_img.size)
    out = model(pil_img, pil_mask)
    out = cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR)

    # back to crop size
    if out.shape[:2] != (ch, cw):
        out = cv2.resize(out, (cw, ch), interpolation=cv2.INTER_CUBIC)

    # feathered blend: inside the mask -> AI result, soft edge -> no visible seam
    m = crop_mask.astype(np.float32) / 255.0
    soft = cv2.GaussianBlur(m, (0, 0), sigmaX=4)
    alpha = np.maximum(m, soft)[:, :, None]
    blended = out.astype(np.float32) * alpha + crop_img.astype(np.float32) * (1.0 - alpha)

    result = image.copy()
    result[cy1:cy2, cx1:cx2] = np.clip(blended, 0, 255).astype(np.uint8)
    return result


def run_classic(image, mask):
    mask = cv2.dilate(mask, np.ones((7, 7), np.uint8), iterations=2)
    return cv2.inpaint(image, mask, inpaintRadius=5, flags=cv2.INPAINT_NS)


# ------------------------------------------------------------
# Routes
# ------------------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "device": "cpu",
        "lama_loaded": lama is not None,
        "lama_error": lama_error,
    })


@app.route("/process", methods=["POST"])
def process():
    try:
        image_file = request.files.get("image")
        if image_file is None:
            return jsonify({"success": False, "error": "Image is required."}), 400

        image = read_uploaded_image(image_file)
        if image is None:
            return jsonify({"success": False, "error": "Could not read image."}), 400

        height, width = image.shape[:2]

        mask_file = request.files.get("mask")
        if mask_file is None:
            return jsonify({"success": False, "error": "Mask is required."}), 400

        mask = read_mask(mask_file, width, height)
        if mask is None:
            return jsonify({"success": False, "error": "Could not read mask."}), 400

        if cv2.countNonZero(mask) == 0:
            return jsonify({"success": False, "error": "Mask is empty. Please select an object."}), 400

        mask = improve_mask(mask)

        method = request.form.get("method", "ai").lower()

        if method in ("ai", "lama", "smart"):
            try:
                result = run_lama(image, mask)
            except Exception as e:
                traceback.print_exc()
                import gc
                gc.collect()
                return jsonify({"success": False, "error": "LaMa processing failed.", "details": repr(e)}), 500
        elif method in ("classic", "opencv"):
            result = run_classic(image, mask)
        else:
            return jsonify({"success": False, "error": "Unknown method."}), 400

        ok, encoded = cv2.imencode(".png", result)
        if not ok:
            raise RuntimeError("Could not encode result.")

        # --- تفريغ الذاكرة المؤقتة فور الانتهاء لمنع تجاوز حد 512MB على Render ---
        import gc
        gc.collect()

        return send_file(io.BytesIO(encoded.tobytes()), mimetype="image/png",
                         as_attachment=False, download_name="result.png")

    except Exception as e:
        import gc
        gc.collect()
        traceback.print_exc()
        return jsonify({"success": False, "error": repr(e)}), 500

@app.errorhandler(413)
def too_large(error):
    return jsonify({"success": False, "error": "Image is too large. Maximum size is 25 MB."}), 413


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
