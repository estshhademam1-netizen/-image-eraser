import io
import os
import urllib.request
import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, send_file

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024

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


def decode_image(file):
    data = file.read()
    if not data:
        return None
    return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)


def prepare_mask(mask):
    _, binary_mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    filled_mask = np.zeros_like(binary_mask)
    for cnt in contours:
        cv2.drawContours(filled_mask, [cnt], -1, 255, thickness=cv2.FILLED)

    final_mask = cv2.bitwise_or(binary_mask, filled_mask)
    kernel = np.ones((7, 7), np.uint8)
    return cv2.dilate(final_mask, kernel, iterations=2)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/magic-select", methods=["POST"])
def magic_select():
    """Smart Selection via Click (FloodFill algorithm)"""
    try:
        image_file = request.files.get("image")
        x = int(request.form.get("x", 0))
        y = int(request.form.get("y", 0))
        tolerance = int(request.form.get("tolerance", 20))

        img = decode_image(image_file)
        if img is None:
            return jsonify({"error": "Invalid Image"}), 400

        h, w = img.shape[:2]
        if x < 0 or x >= w or y < 0 or y >= h:
            return jsonify({"error": "Out of bounds click"}), 400

        # Flood Fill Mask Generation
        flood_mask = np.zeros((h + 2, w + 2), np.uint8)
        flags = 4 | (255 << 8) | cv2.FLOODFILL_MASK_ONLY
        cv2.floodFill(
            img,
            flood_mask,
            (x, y),
            (255, 255, 255),
            (tolerance,) * 3,
            (tolerance,) * 3,
            flags,
        )

        final_mask = flood_mask[1 : h + 1, 1 : w + 1]
        kernel = np.ones((5, 5), np.uint8)
        final_mask = cv2.dilate(final_mask, kernel, iterations=2)

        success, encoded = cv2.imencode(".png", final_mask)
        return send_file(io.BytesIO(encoded.tobytes()), mimetype="image/png")

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/process", methods=["POST"])
def process_image():
    try:
        image_file = request.files["image"]
        mask_file = request.files["mask"]
        method = request.form.get("method", "ai")

        image = decode_image(image_file)
        mask_data = mask_file.read()
        mask = cv2.imdecode(
            np.frombuffer(mask_data, dtype=np.uint8), cv2.IMREAD_GRAYSCALE
        )

        if (
            mask.shape[0] != image.shape[0]
            or mask.shape[1] != image.shape[1]
        ):
            mask = cv2.resize(
                mask,
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        clean_mask = prepare_mask(mask)

        if method == "ai":
            # Crop & Process via AI
            points = cv2.findNonZero(clean_mask)
            if points is None:
                return jsonify({"error": "Please select an object first."}), 400

            x, y, w, h = cv2.boundingRect(points)
            padding = 60
            x1, y1 = max(0, x - padding), max(0, y - padding)
            x2, y2 = min(image.shape[1], x + w + padding), min(
                image.shape[0], y + h + padding
            )

            crop = image[y1:y2, x1:x2]
            crop_mask = clean_mask[y1:y2, x1:x2]

            ai_img = cv2.resize(crop, (512, 512), interpolation=cv2.INTER_AREA)
            ai_m = cv2.resize(
                crop_mask, (512, 512), interpolation=cv2.INTER_NEAREST
            )

            img_blob = cv2.dnn.blobFromImage(
                ai_img, 1.0 / 255.0, (512, 512), (0, 0, 0), False
            )
            mask_blob = (
                cv2.dnn.blobFromImage(ai_m, 1.0, (512, 512), (0,), False) > 0
            ).astype(np.float32)

            net = get_lama_model()
            net.setInput(img_blob, "image")
            net.setInput(mask_blob, "mask")
            output = net.forward()[0]

            res = (
                np.clip(np.transpose(output, (1, 2, 0)), 0, 1) * 255
            ).astype(np.uint8)
            res = cv2.resize(
                res,
                (crop.shape[1], crop.shape[0]),
                interpolation=cv2.INTER_CUBIC,
            )

            soft_mask = cv2.GaussianBlur(crop_mask, (15, 15), 0)[
                ..., np.newaxis
            ] / 255.0
            blended = res.astype(np.float32) * soft_mask + crop.astype(
                np.float32
            ) * (1.0 - soft_mask)

            final_image = image.copy()
            final_image[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(
                np.uint8
            )
        else:
            final_image = cv2.inpaint(image, clean_mask, 5, cv2.INPAINT_TELEA)

        success, encoded = cv2.imencode(
            ".jpg", final_image, [cv2.IMWRITE_JPEG_QUALITY, 92]
        )
        return send_file(io.BytesIO(encoded.tobytes()), mimetype="image/jpeg")

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
