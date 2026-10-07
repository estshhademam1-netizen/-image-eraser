from flask import Flask, render_template, request, jsonify, send_file
from PIL import Image
import cv2
import numpy as np
import io
import os

app = Flask(__name__)

MAX_FILE_SIZE = 12 * 1024 * 1024  # 12 MB
MAX_IMAGE_DIMENSION = 2400

app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process_image():
    try:
        if "image" not in request.files:
            return jsonify({"error": "Image is missing"}), 400

        if "mask" not in request.files:
            return jsonify({"error": "Selection mask is missing"}), 400

        image_file = request.files["image"]
        mask_file = request.files["mask"]

        # Read original image
        image_bytes = image_file.read()

        if not image_bytes:
            return jsonify({"error": "Empty image"}), 400

        image_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({"error": "Invalid image"}), 400

        original_height, original_width = image.shape[:2]

        # Prevent extremely large images from exhausting free server memory
        scale = min(
            1.0,
            MAX_IMAGE_DIMENSION / max(original_width, original_height)
        )

        if scale < 1.0:
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)

            image = cv2.resize(
                image,
                (new_width, new_height),
                interpolation=cv2.INTER_AREA
            )
        else:
            new_width = original_width
            new_height = original_height

        # Read mask
        mask_bytes = mask_file.read()
        mask_array = np.frombuffer(mask_bytes, np.uint8)
        mask = cv2.imdecode(mask_array, cv2.IMREAD_GRAYSCALE)

        if mask is None:
            return jsonify({"error": "Invalid mask"}), 400

        # Make mask match image dimensions
        mask = cv2.resize(
            mask,
            (new_width, new_height),
            interpolation=cv2.INTER_NEAREST
        )

        # Clean and strengthen the mask
        _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

        # Inpainting
        result = cv2.inpaint(
            image,
            mask,
            5,
            cv2.INPAINT_TELEA
        )

        # Encode result
        success, encoded = cv2.imencode(
            ".jpg",
            result,
            [cv2.IMWRITE_JPEG_QUALITY, 95]
        )

        if not success:
            return jsonify({"error": "Could not create result"}), 500

        return send_file(
            io.BytesIO(encoded.tobytes()),
            mimetype="image/jpeg",
            as_attachment=False,
            download_name="image-eraser-result.jpg"
        )

    except Exception as e:
        return jsonify({
            "error": "Processing failed",
            "details": str(e)
        }), 500


@app.errorhandler(413)
def too_large(error):
    return jsonify({
        "error": "Image is too large. Maximum size is 12 MB."
    }), 413


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
