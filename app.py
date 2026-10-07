import io
import os
import urllib.request
import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, send_file

app = Flask(__name__)

# Maximum uploaded image size = 16 MB
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

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
    """Lazy loader for the ONNX LaMa Model."""
    global lama_net

    if lama_net is not None:
        return lama_net

    if not os.path.exists(MODEL_PATH):
        print("Downloading LaMa AI model weights...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("LaMa AI model weights downloaded successfully.")

    print("Loading LaMa AI ONNX network into OpenCV DNN...")
    lama_net = cv2.dnn.readNetFromONNX(MODEL_PATH)
    lama_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    lama_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    print("LaMa AI model ready for inference.")

    return lama_net


# =========================================================
# IMAGE DECODING UTILITIES
# =========================================================


def decode_image_file(file):
    """Converts uploaded file payload to BGR numpy array."""
    data = file.read()
    if not data:
        return None
    array = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(array, cv2.IMREAD_COLOR)


def decode_mask_file(file):
    """Converts uploaded mask payload to Grayscale numpy array."""
    data = file.read()
    if not data:
        return None
    array = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(array, cv2.IMREAD_GRAYSCALE)


# =========================================================
# AUTO HOLE-FILLING & CONTOUR DILATION
# =========================================================


def process_mask(mask):
    """Binarizes the user mask, detects closed loops/circles,

    fills interior areas automatically, and dilates boundaries.
    """
    # Thresholding
    _, binary_mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)

    # Find contours drawn by user
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Fill internal region if user drew a loop/circle around object
    filled_mask = np.zeros_like(binary_mask)
    for cnt in contours:
        cv2.drawContours(filled_mask, [cnt], -1, 255, thickness=cv2.FILLED)

    final_mask = cv2.bitwise_or(binary_mask, filled_mask)

    # Dilate mask slightly to prevent seam artifacts around object edges
    kernel = np.ones((7, 7), np.uint8)
    final_mask = cv2.dilate(final_mask, kernel, iterations=2)

    return final_mask


# =========================================================
# INPAINTING ENGINES
# =========================================================


def run_classic_inpainting(image, mask):
    """Fast Telea Inpainting algorithm for low resource consumption."""
    clean_mask = process_mask(mask)
    return cv2.inpaint(image, clean_mask, 5, cv2.INPAINT_TELEA)


def run_ai_inpainting(image, mask):
    """High quality AI LaMa Inpainting with localized bounding-box crop."""
    clean_mask = process_mask(mask)

    # Locate masked bounding box
    points = cv2.findNonZero(clean_mask)
    if points is None:
        raise ValueError("No object selected in the mask layer.")

    x, y, w, h = cv2.boundingRect(points)
    img_h, img_w = image.shape[:2]

    # Crop target region with dynamic padding
    padding = 60
    x1, y1 = max(0, x - padding), max(0, y - padding)
    x2, y2 = min(img_w, x + w + padding), min(img_h, y + h + padding)

    crop_img = image[y1:y2, x1:x2].copy()
    crop_mask = clean_mask[y1:y2, x1:x2].copy()
    crop_h, crop_w = crop_img.shape[:2]

    # Resize crop to 512x512 expected tensor input
    ai_img = cv2.resize(crop_img, (512, 512), interpolation=cv2.INTER_AREA)
    ai_mask = cv2.resize(
        crop_mask, (512, 512), interpolation=cv2.INTER_NEAREST
    )

    # Construct DNN input blobs
    img_blob = cv2.dnn.blobFromImage(
        ai_img, 1.0 / 255.0, (512, 512), (0, 0, 0), False, False
    )
    mask_blob = cv2.dnn.blobFromImage(
        ai_mask, 1.0, (512, 512), (0,), False, False
    )
    mask_blob = (mask_blob > 0).astype(np.float32)

    # Perform LaMa AI Inference
    net = get_lama_model()
    net.setInput(img_blob, "image")
    net.setInput(mask_blob, "mask")
    output = net.forward()

    # Process model tensor output
    result = output[0]
    result = np.transpose(result, (1, 2, 0))
    result = np.clip(result, 0, 1)
    result = (result * 255).astype(np.uint8)

    # Resize back to original crop resolution
    result = cv2.resize(
        result, (crop_w, crop_h), interpolation=cv2.INTER_CUBIC
    )

    # Blend result back into full original image using soft Gaussian mask
    final_image = image.copy()
    soft_mask = cv2.GaussianBlur(crop_mask, (15, 15), 0)
    soft_mask = (soft_mask.astype(np.float32) / 255.0)[..., np.newaxis]

    orig_crop_float = final_image[y1:y2, x1:x2].astype(np.float32)
    ai_crop_float = result.astype(np.float32)

    blended = ai_crop_float * soft_mask + orig_crop_float * (1.0 - soft_mask)
    final_image[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)

    return final_image


# =========================================================
# ROUTES
# =========================================================


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process_image():
    try:
        if "image" not in request.files or "mask" not in request.files:
            return jsonify({"error": "Missing image or mask payload."}), 400

        image_file = request.files["image"]
        mask_file = request.files["mask"]
        method = request.form.get("method", "ai")

        image = decode_image_file(image_file)
        mask = decode_mask_file(mask_file)

        if image is None or mask is None:
            return jsonify({"error": "Failed to decode input images."}), 400

        # Resize mask if dimensions mismatch image
        if (
            mask.shape[0] != image.shape[0]
            or mask.shape[1] != image.shape[1]
        ):
            mask = cv2.resize(
                mask,
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        # Check if selection exists
        if cv2.countNonZero(mask) == 0:
            return jsonify(
                {"error": "Please paint over an object to select it first."}
            ), 400

        # Execute removal according to method
        if method == "ai":
            result = run_ai_inpainting(image, mask)
        else:
            result = run_classic_inpainting(image, mask)

        # Encode JPEG output
        success, encoded_img = cv2.imencode(
            ".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, 92]
        )
        if not success:
            return jsonify({"error": "Encoding output failed."}), 500

        return send_file(
            io.BytesIO(encoded_img.tobytes()),
            mimetype="image/jpeg",
            as_attachment=False,
            download_name="erased_result.jpg",
        )

    except Exception as e:
        print("EXECUTION ERROR:", str(e))
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500


@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({"error": "Uploaded image size exceeds 16MB limit."}), 413


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
