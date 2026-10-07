from flask import Flask, render_template, request, jsonify, send_file
import cv2
import numpy as np
import io
import os
import urllib.request

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

    # Download model only when needed
    if not os.path.exists(MODEL_PATH):
        print("Downloading LaMa model...")

        urllib.request.urlretrieve(
            MODEL_URL,
            MODEL_PATH
        )

        print("LaMa model downloaded.")

    print("Loading LaMa model...")

    lama_net = cv2.dnn.readNetFromONNX(
        MODEL_PATH
    )

    # CPU
    lama_net.setPreferableBackend(
        cv2.dnn.DNN_BACKEND_OPENCV
    )

    lama_net.setPreferableTarget(
        cv2.dnn.DNN_TARGET_CPU
    )

    print("LaMa model ready.")

    return lama_net


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process_image():

    try:

        if "image" not in request.files:
            return jsonify({
                "error": "Image is missing."
            }), 400

        if "mask" not in request.files:
            return jsonify({
                "error": "Mask is missing."
            }), 400

        image_file = request.files["image"]
        mask_file = request.files["mask"]

        image_bytes = image_file.read()
        mask_bytes = mask_file.read()

        if not image_bytes:
            return jsonify({
                "error": "Empty image."
            }), 400

        if not mask_bytes:
            return jsonify({
                "error": "Empty mask."
            }), 400

        # Decode image
        image_array = np.frombuffer(
            image_bytes,
            np.uint8
        )

        image = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )

        if image is None:
            return jsonify({
                "error": "Invalid image."
            }), 400

        # Decode mask
        mask_array = np.frombuffer(
            mask_bytes,
            np.uint8
        )

        mask = cv2.imdecode(
            mask_array,
            cv2.IMREAD_GRAYSCALE
        )

        if mask is None:
            return jsonify({
                "error": "Invalid mask."
            }), 400

        original_height, original_width = image.shape[:2]

        # -------------------------------------------------
        # Resize for LaMa
        # -------------------------------------------------

        MAX_SIDE = 1600

        scale = min(
            1.0,
            MAX_SIDE / max(
                original_width,
                original_height
            )
        )

        if scale < 1:

            new_width = int(
                original_width * scale
            )

            new_height = int(
                original_height * scale
            )

            image = cv2.resize(
                image,
                (new_width, new_height),
                interpolation=cv2.INTER_AREA
            )

            mask = cv2.resize(
                mask,
                (new_width, new_height),
                interpolation=cv2.INTER_NEAREST
            )

        else:

            new_width = original_width
            new_height = original_height

        # -------------------------------------------------
        # Clean mask
        # -------------------------------------------------

        _, mask = cv2.threshold(
            mask,
            10,
            255,
            cv2.THRESH_BINARY
        )

        # Slightly expand selection
        kernel = np.ones(
            (7, 7),
            np.uint8
        )

        mask = cv2.dilate(
            mask,
            kernel,
            iterations=1
        )

        # -------------------------------------------------
        # Load LaMa
        # -------------------------------------------------

        net = get_lama_model()

        # -------------------------------------------------
        # Prepare 512x512 input
        # -------------------------------------------------

        image_blob = cv2.dnn.blobFromImage(
            image,
            scalefactor=1.0 / 255.0,
            size=(512, 512),
            mean=(0, 0, 0),
            swapRB=False,
            crop=False
        )

        mask_blob = cv2.dnn.blobFromImage(
            mask,
            scalefactor=1.0,
            size=(512, 512),
            mean=(0,),
            swapRB=False,
            crop=False
        )

        mask_blob = (
            mask_blob > 0
        ).astype(np.float32)

        # -------------------------------------------------
        # AI Inpainting
        # -------------------------------------------------

        net.setInput(
            image_blob,
            "image"
        )

        net.setInput(
            mask_blob,
            "mask"
        )

        output = net.forward()

        # -------------------------------------------------
        # Convert output
        # -------------------------------------------------

        result = output[0]

        result = np.transpose(
            result,
            (1, 2, 0)
        )

        result = np.clip(
            result,
            0,
            255
        ).astype(np.uint8)

        # Resize result back to working size
        result = cv2.resize(
            result,
            (new_width, new_height),
            interpolation=cv2.INTER_CUBIC
        )

        # -------------------------------------------------
        # Restore original size
        # -------------------------------------------------

        if (
            new_width != original_width
            or
            new_height != original_height
        ):

            result = cv2.resize(
                result,
                (
                    original_width,
                    original_height
                ),
                interpolation=cv2.INTER_CUBIC
            )

        # -------------------------------------------------
        # Encode
        # -------------------------------------------------

        success, encoded = cv2.imencode(
            ".jpg",
            result,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                95
            ]
        )

        if not success:

            return jsonify({
                "error": "Could not create result."
            }), 500

        return send_file(
            io.BytesIO(
                encoded.tobytes()
            ),
            mimetype="image/jpeg",
            as_attachment=False,
            download_name="image-eraser-result.jpg"
        )

    except Exception as e:

        print("ERROR:", str(e))

        return jsonify({
            "error": "Image processing failed.",
            "details": str(e)
        }), 500


@app.errorhandler(413)
def too_large(error):

    return jsonify({
        "error":
        "Image is too large. Maximum size is 12 MB."
    }), 413


if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
