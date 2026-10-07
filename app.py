import io
import os
import urllib.request
import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, send_file

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

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

    lama_net = cv2.dnn.readNetFromONNX(MODEL_PATH)
    lama_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    lama_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return lama_net


def decode_image_file(file):
    data = file.read()
    if not data:
        return None
    return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)


def decode_mask_file(file):
    data = file.read()
    if not data:
        return None
    return cv2.imdecode(
        np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_GRAYSCALE
    )


def process_mask(mask):
    """Clean mask thresholding and boundary expansion without central artifacts."""
    _, binary_mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)

    # Fill closed user contours cleanly
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    filled_mask = binary_mask.copy()
    for cnt in contours:
        cv2.drawContours(filled_mask, [cnt], -1, 255, thickness=cv2.FILLED)

    # Mild dilation without creating heavy circular centers
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    final_mask = cv2.dilate(filled_mask, kernel, iterations=2)
    return final_mask


def run_ai_inpainting_fixed(image, mask):
    """Square Aspect-Ratio Preserving LaMa Inpainting."""
    clean_mask = process_mask(mask)

    points = cv2.findNonZero(clean_mask)
    if points is None:
        raise ValueError("No selection found.")

    x, y, w, h = cv2.boundingRect(points)
    img_h, img_w = image.shape[:2]

    # Force crop to be a SQUARE with sufficient background context
    max_dim = max(w, h)
    padding = max(max_dim // 2, 80)
    square_size = max_dim + (padding * 2)

    center_x, center_y = x + w // 2, y + h // 2

    x1 = max(0, center_x - square_size // 2)
    y1 = max(0, center_y - square_size // 2)
    x2 = min(img_w, x1 + square_size)
    y2 = min(img_h, y1 + square_size)

    # Adjust x1, y1 if near image boundaries
    x1 = max(0, x2 - square_size)
    y1 = max(0, y2 - square_size)

    crop_img = image[y1:y2, x1:x2].copy()
    crop_mask = clean_mask[y1:y2, x1:x2].copy()
    crop_h, crop_w = crop_img.shape[:2]

    # Resize cleanly to 512x512
    ai_img = cv2.resize(crop_img, (512, 512), interpolation=cv2.INTER_AREA)
    ai_mask = cv2.resize(
        crop_mask, (512, 512), interpolation=cv2.INTER_NEAREST
    )

    img_blob = cv2.dnn.blobFromImage(
        ai_img, 1.0 / 255.0, (512, 512), (0, 0, 0), False, False
    )
    mask_blob = (
        cv2.dnn.blobFromImage(ai_mask, 1.0, (512, 512), (0,), False, False) > 0
    ).astype(np.float32)

    net = get_lama_model()
    net.setInput(img_blob, "image")
    net.setInput(mask_blob, "mask")
    output = net.forward()[0]

    result = np.clip(np.transpose(output, (1, 2, 0)), 0, 1) * 255
    result = result.astype(np.uint8)

    # Scale back to original cropped square dimensions
    res_crop = cv2.resize(
        result, (crop_w, crop_h), interpolation=cv2.INTER_LANCZOS4
    )

    # Smooth blending mask
    soft_mask = cv2.GaussianBlur(crop_mask, (15, 15), 0)
    soft_mask_3d = (soft_mask.astype(np.float32) / 255.0)[..., np.newaxis]

    orig_crop_float = crop_img.astype(np.float32)
    ai_crop_float = res_crop.astype(np.float32)

    blended = ai_crop_float * soft_mask_3d + orig_crop_float * (
        1.0 - soft_mask_3d
    )

    final_image = image.copy()
    final_image[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)

    return final_image


def run_classic_inpainting(image, mask):
    clean_mask = process_mask(mask)
    return cv2.inpaint(image, clean_mask, 5, cv2.INPAINT_TELEA)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process_image():
    try:
        if "image" not in request.files or "mask" not in request.files:
            return jsonify({"error": "Missing image or mask data."}), 400

        image = decode_image_file(request.files["image"])
        mask = decode_mask_file(request.files["mask"])

        if image is None or mask is None:
            return jsonify({"error": "Failed to decode files."}), 400

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
            return jsonify({"error": "Please paint over an object first."}), 400

        method = request.form.get("method", "ai")
        if method == "ai":
            result = run_ai_inpainting_fixed(image, mask)
        else:
            result = run_classic_inpainting(image, mask)

        _, encoded_img = cv2.imencode(
            ".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, 98]
        )
        return send_file(
            io.BytesIO(encoded_img.tobytes()),
            mimetype="image/jpeg",
            as_attachment=False,
            download_name="erased_result.jpg",
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
