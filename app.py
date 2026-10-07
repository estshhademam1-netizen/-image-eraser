import io
import os
import urllib.request

import cv2
import numpy as np

from flask import Flask, jsonify, render_template, request, send_file


app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024


# =========================================================
# LAMA MODEL
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

        print("Downloading LaMa model...")

        urllib.request.urlretrieve(
            MODEL_URL,
            MODEL_PATH
        )

        print("LaMa download complete.")

    print("Loading LaMa...")

    lama_net = cv2.dnn.readNetFromONNX(
        MODEL_PATH
    )

    lama_net.setPreferableBackend(
        cv2.dnn.DNN_BACKEND_OPENCV
    )

    lama_net.setPreferableTarget(
        cv2.dnn.DNN_TARGET_CPU
    )

    print("LaMa loaded.")

    return lama_net


# =========================================================
# READ IMAGE
# =========================================================

def decode_image(file):

    data = file.read()

    if not data:
        return None

    array = np.frombuffer(
        data,
        dtype=np.uint8
    )

    return cv2.imdecode(
        array,
        cv2.IMREAD_COLOR
    )


# =========================================================
# READ MASK
# =========================================================

def decode_mask(file):

    data = file.read()

    if not data:
        return None

    array = np.frombuffer(
        data,
        dtype=np.uint8
    )

    return cv2.imdecode(
        array,
        cv2.IMREAD_GRAYSCALE
    )


# =========================================================
# LIMIT HUGE PHONE PHOTOS
# =========================================================

def resize_if_needed(image, max_side=1800):

    height, width = image.shape[:2]

    largest = max(
        height,
        width
    )

    if largest <= max_side:
        return image

    scale = max_side / float(largest)

    new_width = int(
        width * scale
    )

    new_height = int(
        height * scale
    )

    return cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )


# =========================================================
# CLEAN MASK
# =========================================================

def prepare_mask(mask):

    # Binary mask:
    # 0   = keep
    # 255 = remove

    mask = np.where(
        mask > 10,
        255,
        0
    ).astype(np.uint8)

    if cv2.countNonZero(mask) == 0:
        raise ValueError(
            "No selected object."
        )

    # Fill tiny gaps inside the user's selection
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1
    )

    return mask


# =========================================================
# LAMA
# =========================================================

def lama_inpaint(image, mask):

    clean_mask = prepare_mask(
        mask
    )

    original = image.copy()

    # =====================================================
    # IMPORTANT
    #
    # We now follow OpenCV's official LaMa inference
    # pipeline directly.
    #
    # No square crop.
    # No radial crop.
    # No giant generated patch.
    # =====================================================

    image_blob = cv2.dnn.blobFromImage(
        image,
        0.00392,
        (512, 512),
        (0, 0, 0),
        False,
        False
    )

    mask_blob = cv2.dnn.blobFromImage(
        clean_mask,
        scalefactor=1.0,
        size=(512, 512),
        mean=(0,),
        swapRB=False,
        crop=False
    )

    mask_blob = (
        mask_blob > 0
    ).astype(
        np.float32
    )

    net = get_lama_model()

    net.setInput(
        image_blob,
        "image"
    )

    net.setInput(
        mask_blob,
        "mask"
    )

    print("Running LaMa inference...")

    output = net.forward()

    # =====================================================
    # OFFICIAL OPENCV POST PROCESSING
    # =====================================================

    result = output[0]

    result = np.transpose(
        result,
        (1, 2, 0)
    )

    result = result.astype(
        np.uint8
    )

    # =====================================================
    # Resize result back to original dimensions
    # =====================================================

    result = cv2.resize(
        result,
        (
            original.shape[1],
            original.shape[0]
        ),
        interpolation=cv2.INTER_LINEAR
    )

    # =====================================================
    # IMPORTANT COMPOSITING
    #
    # We NEVER replace the entire AI result.
    #
    # Only the selected object area is taken from AI.
    # Everything else comes from ORIGINAL.
    # =====================================================

    hard_mask = (
        clean_mask > 0
    ).astype(
        np.uint8
    )

    # Slight edge expansion to make sure object border
    # disappears completely.
    edge_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )

    hard_mask = cv2.dilate(
        hard_mask,
        edge_kernel,
        iterations=1
    )

    # Very small feather
    soft_mask = cv2.GaussianBlur(
        hard_mask.astype(np.float32),
        (7, 7),
        0
    )

    soft_mask = np.clip(
        soft_mask,
        0.0,
        1.0
    )

    soft_mask = soft_mask[..., None]

    original_float = (
        original.astype(np.float32)
    )

    result_float = (
        result.astype(np.float32)
    )

    final = (
        result_float * soft_mask
        +
        original_float * (1.0 - soft_mask)
    )

    final = np.clip(
        final,
        0,
        255
    ).astype(
        np.uint8
    )

    return final


# =========================================================
# CLASSIC
# =========================================================

def classic_inpaint(image, mask):

    clean_mask = prepare_mask(
        mask
    )

    return cv2.inpaint(
        image,
        clean_mask,
        7,
        cv2.INPAINT_TELEA
    )


# =========================================================
# PROCESS
# =========================================================

@app.route(
    "/process",
    methods=["POST"]
)
def process():

    try:

        if "image" not in request.files:

            return jsonify({
                "error": "Image is missing."
            }), 400


        if "mask" not in request.files:

            return jsonify({
                "error": "Mask is missing."
            }), 400


        # -------------------------------------------------
        # IMAGE
        # -------------------------------------------------

        image = decode_image(
            request.files["image"]
        )

        if image is None:

            return jsonify({
                "error": "Could not read image."
            }), 400


        # -------------------------------------------------
        # MASK
        # -------------------------------------------------

        mask = decode_mask(
            request.files["mask"]
        )

        if mask is None:

            return jsonify({
                "error": "Could not read mask."
            }), 400


        # -------------------------------------------------
        # Keep Render memory reasonable
        # -------------------------------------------------

        image = resize_if_needed(
            image,
            1800
        )


        # -------------------------------------------------
        # Mask must have exact same dimensions
        # -------------------------------------------------

        mask = cv2.resize(
            mask,
            (
                image.shape[1],
                image.shape[0]
            ),
            interpolation=cv2.INTER_NEAREST
        )


        if cv2.countNonZero(mask) == 0:

            return jsonify({
                "error": "Please select an object first."
            }), 400


        method = request.form.get(
            "method",
            "ai"
        )


        print(
            "================================"
        )

        print(
            "METHOD:",
            method
        )

        print(
            "IMAGE:",
            image.shape
        )

        print(
            "MASK PIXELS:",
            cv2.countNonZero(mask)
        )

        print(
            "================================"
        )


        # =================================================
        # AI
        # =================================================

        if method == "ai":

            result = lama_inpaint(
                image,
                mask
            )


        # =================================================
        # CLASSIC
        # =================================================

        else:

            result = classic_inpaint(
                image,
                mask
            )


        # =================================================
        # OUTPUT JPEG
        # =================================================

        success, encoded = cv2.imencode(
            ".jpg",
            result,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                98
            ]
        )


        if not success:

            raise ValueError(
                "Could not encode result."
            )


        return send_file(
            io.BytesIO(
                encoded.tobytes()
            ),
            mimetype="image/jpeg",
            as_attachment=False,
            download_name="erased_result.jpg"
        )


    except Exception as e:

        print(
            "PROCESS ERROR:",
            repr(e)
        )

        return jsonify({
            "error": str(e)
        }), 500


# =========================================================
# HOME
# =========================================================

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


# =========================================================
# RUN
# =========================================================

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
