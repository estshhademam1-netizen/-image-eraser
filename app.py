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
        print("Downloading High-Res LaMa AI model...")
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
    """Refines mask, fills shapes automatically, and expands boundaries smoothly."""
    _, binary_mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    filled_mask = np.zeros_like(binary_mask)
    for cnt in contours:
        cv2.drawContours(filled_mask, [cnt], -1, 255, thickness=cv2.FILLED)

    final_mask = cv2.bitwise_or(binary_mask, filled_mask)

    # Expanding mask boundary slightly to cover object edges cleanly
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    final_mask = cv2.dilate(final_mask, kernel, iterations=3)
    return final_mask


def run_ai_inpainting_high_res(image, mask):
    """High-Quality Anti-Pixelation AI Object Eraser."""
    clean_mask = process_mask(mask)

    points = cv2.findNonZero(clean_mask)
    if points is None:
        raise ValueError("No object selected in mask.")

    x, y, w, h = cv2.boundingRect(points)
    img_h, img_w = image.shape[:2]

    # Dynamic padding to preserve surrounding background context
    padding = max(w, h) // 2 + 50
    x1, y1 = max(0, x - padding), max(0, y - padding)
    x2, y2 = min(img_w, x + w + padding), min(img_h, y + h + padding)

    crop_img = image[y1:y2, x1:x2].copy()
    crop_mask = clean_mask[y1:y2, x1:x2].copy()
    crop_h, crop_w = crop_img.shape[:2]

    # AI Model Inference at 512x512
    ai_img = cv2.resize(crop_img, (512, 512), interpolation=cv2.INTER_LANCZOS4)
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

    # High-Quality Resize back using INTER_LANCZOS4 to avoid pixelation
    res_crop = cv2.resize(
        result, (crop_w, crop_h), interpolation=cv2.INTER_LANCZOS4
    )

    # High quality multi-stage feathering mask
    soft_mask = cv2.GaussianBlur(crop_mask, (21, 21), 0)
    soft_mask_3d = (soft_mask.astype(np.float32) / 255.0)[..., np.newaxis]

    orig_crop_float = crop_img.astype(np.float32)
    ai_crop_float = res_crop.astype(np.float32)

    # Seamless blending
    blended = ai_crop_float * soft_mask_3d + orig_crop_float * (
        1.0 - soft_mask_3d
    )

    final_image = image.copy()
    final_image[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)

    return final_image


def run_classic_inpainting(image, mask):
    clean_mask = process_mask(mask)
    return cv2.inpaint(image, clean_mask, 7, cv2.INPAINT_TELEA)


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
            result = run_ai_inpainting_high_res(image, mask)
        else:
            result = run_classic_inpainting(image, mask)

        # High Quality JPEG output without artifacts
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
