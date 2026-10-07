import io
import os
import urllib.request
import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, send_file

app = Flask(__name__)

# Maximum uploaded image size = 12 MB
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024

# =========================================================
# LaMa AI MODEL CONFIGURATION
# =========================================================

MODEL_URL = (
    "https://huggingface.co/opencv/inpainting_lama/"
    "resolve/main/inpainting_lama_2025jan.onnx"
)

MODEL_PATH = "/tmp/inpainting_lama_2025jan.onnx"
lama_net = None


def get_lama_model():
    global lama_net

    if lama_net is not None:
        return lama_net

    if not os.path.exists(MODEL_PATH):
        print("Downloading LaMa AI model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("LaMa model downloaded.")

    print("Loading LaMa AI model...")
    lama_net = cv2.dnn.readNetFromONNX(MODEL_PATH)
    lama_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    lama_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    print("LaMa AI model ready.")

    return lama_net


# =========================================================
# IMAGE DECODING HELPERS
# =========================================================


def decode_uploaded_image(file):
    data = file.read()
    if not data:
        return None
    array = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(array, cv2.IMREAD_COLOR)


def decode_uploaded_mask(file):
    data = file.read()
    if not data:
        return None
    array = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(array, cv2.IMREAD_GRAYSCALE)


# =========================================================
# AUTO-FILL ENCLOSED CIRCLES / CONTOURS & MASK CLEANING
# =========================================================


def prepare_mask(mask):
    """Binarizes the mask, automatically fills any drawn loop/circle,

    and dilates it slightly for smooth object removal.
    """
    # 1. Convert mask to binary (0 or 255)
    _, binary_mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)

    # 2. Find contours (outer shapes drawn by user)
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # 3. Fill inside all drawn loops/circles
    filled_mask = np.zeros_like(binary_mask)
    for cnt in contours:
        # Draw and fill the entire area bounded by the circle/stroke
        cv2.drawContours(filled_mask, [cnt], -1, 255, thickness=cv2.FILLED)

    # Combine original strokes with filled interior
    final_mask = cv2.bitwise_or(binary_mask, filled_mask)

    # 4. Expand mask slightly to cover object edges cleanly
    kernel = np.ones((7, 7), np.uint8)
    final_mask = cv2.dilate(final_mask, kernel, iterations=2)

    return final_mask


# =========================================================
# FAST CLASSIC REMOVAL
# =========================================================


def classic_remove(image, mask):
    clean_mask = prepare_mask(mask)
    # Fast Inpainting using Telea algorithm
    result = cv2.inpaint(image, clean_mask, 5, cv2.INPAINT_TELEA)
    return result


# =========================================================
# AI LAMA REMOVAL (OPTIMIZED FOR RENDER / LOW RAM)
# =========================================================


def ai_remove(image, mask):
    clean_mask = prepare_mask(mask)

    # Locate selection coordinates
    points = cv2.findNonZero(clean_mask)
    if points is None:
        raise ValueError("Please select an object first.")

    x, y, w, h = cv2.boundingRect(points)
    image_h, image_w = image.shape[:2]

    # Crop target region with dynamic padding
    padding = 60
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(image_w, x + w + padding)
    y2 = min(image_h, y + h + padding)

    crop = image[y1:y2, x1:x2].copy()
    crop_mask = clean_mask[y1:y2, x1:x2].copy()
    crop_h, crop_w = crop.shape[:2]

    # Resize crop to 512x512 for AI execution
    ai_image = cv2.resize(crop, (512, 512), interpolation=cv2.INTER_AREA)
    ai_mask = cv2.resize(
        crop_mask, (512, 512), interpolation=cv2.INTER_NEAREST
    )

    # Prepare DNN blobs
    image_blob = cv2.dnn.blobFromImage(
        ai_image,
        scalefactor=1.0 / 255.0,
        size=(512, 512),
        mean=(0, 0, 0),
        swapRB=False,
        crop=False,
    )

    mask_blob = cv2.dnn.blobFromImage(
        ai_mask,
        scalefactor=1.0,
        size=(512, 512),
        mean=(0,),
        swapRB=False,
        crop=False,
    )
    mask_blob = (mask_blob > 0).astype(np.float32)

    # Load & execute model
    net = get_lama_model()
    net.setInput(image_blob, "image")
    net.setInput(mask_blob, "mask")
    output = net.forward()

    # Post-process output
    result = output[0]
    result = np.transpose(result, (1, 2, 0))
    result = np.clip(result, 0, 1)
    result = (result * 255).astype(np.uint8)

    # Resize back to original crop dimension
    result = cv2.resize(
        result, (crop_w, crop_h), interpolation=cv2.INTER_CUBIC
    )

    # Blend result back into full image
    final_image = image.copy()
    soft_mask = cv2.GaussianBlur(crop_mask, (15, 15), 0)
    soft_mask = (soft_mask.astype(np.float32) / 255.0)[..., np.newaxis]

    original_crop = final_image[y1:y2, x1:x2].astype(np.float32)
    ai_crop = result.astype(np.float32)

    blended = ai_crop * soft_mask + original_crop * (1.0 - soft_mask)
    final_image[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)

    return final_image


# =========================================================
# ROUTES
# =========================================================


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process_image():
    try:
        if "image" not in request.files or "mask" not in request.files:
            return jsonify({"error": "Missing image or mask data."}), 400

        image_file = request.files["image"]
        mask_file = request.files["mask"]
        method = request.form.get("method", "ai")

        image = decode_uploaded_image(image_file)
        mask = decode_uploaded_mask(mask_file)

        if image is None or mask is None:
            return jsonify({"error": "Invalid image or mask payload."}), 400

        # Resize mask if dimensions mismatch
        if (
            mask.shape[0] != image.shape[0]
            or mask.shape[1] != image.shape[1]
        ):
            mask = cv2.resize(
                mask,
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        if cv2.countNonZero(mask) == 0:
            return jsonify(
                {"error": "Please paint over or circle an object first."}
            ), 400

        # Choose processing method
        if method == "ai":
            result = ai_remove(image, mask)
        else:
            result = classic_remove(image, mask)

        # Encode JPEG
        success, encoded = cv2.imencode(
            ".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, 90]
        )
        if not success:
            return jsonify({"error": "Failed to encode output image."}), 500

        return send_file(
            io.BytesIO(encoded.tobytes()),
            mimetype="image/jpeg",
            as_attachment=False,
            download_name="erased_result.jpg",
        )

    except Exception as e:
        print("PROCESS ERROR:", repr(e))
        return jsonify(
            {
                "error": "Processing failed due to memory or execution timeout.",
                "details": str(e),
            }
        ), 500


@app.errorhandler(413)
def file_too_large(error):
    return jsonify({"error": "File size exceeds 12MB limit."}), 413


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
