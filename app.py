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

        print("LaMa model downloaded.")

    print("Loading LaMa model...")

    lama_net = cv2.dnn.readNetFromONNX(
        MODEL_PATH
    )

    lama_net.setPreferableBackend(
        cv2.dnn.DNN_BACKEND_OPENCV
    )

    lama_net.setPreferableTarget(
        cv2.dnn.DNN_TARGET_CPU
    )

    print("LaMa model ready.")

    return lama_net


# =========================================================
# FILE DECODING
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
# MASK CLEANING
# =========================================================

def clean_mask(mask, mode):

    # Binary
    _, mask = cv2.threshold(
        mask,
        10,
        255,
        cv2.THRESH_BINARY
    )

    # -----------------------------------------------------
    # OUTLINE MODE
    #
    # The frontend already closes/fills the polygon.
    # Here we simply clean it.
    # -----------------------------------------------------

    if mode == "outline":

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (5, 5)
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=2
        )

        mask = cv2.dilate(
            mask,
            kernel,
            iterations=1
        )

        return mask

    # -----------------------------------------------------
    # BRUSH MODE
    # -----------------------------------------------------

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

    mask = cv2.dilate(
        mask,
        kernel,
        iterations=1
    )

    return mask


# =========================================================
# GET CROP AROUND OBJECT
# =========================================================

def get_object_crop(image, mask):

    points = cv2.findNonZero(mask)

    if points is None:

        raise ValueError(
            "No selected object."
        )

    x, y, w, h = cv2.boundingRect(
        points
    )

    image_h, image_w = image.shape[:2]

    # More context around the selected object
    padding = max(
        80,
        int(max(w, h) * 0.55)
    )

    x1 = max(
        0,
        x - padding
    )

    y1 = max(
        0,
        y - padding
    )

    x2 = min(
        image_w,
        x + w + padding
    )

    y2 = min(
        image_h,
        y + h + padding
    )

    crop = image[
        y1:y2,
        x1:x2
    ].copy()

    crop_mask = mask[
        y1:y2,
        x1:x2
    ].copy()

    return (
        crop,
        crop_mask,
        x1,
        y1,
        x2,
        y2
    )


# =========================================================
# AI LAMA
# =========================================================

def ai_inpaint(image, mask, mode):

    mask = clean_mask(
        mask,
        mode
    )

    (
        crop,
        crop_mask,
        x1,
        y1,
        x2,
        y2
    ) = get_object_crop(
        image,
        mask
    )

    crop_h, crop_w = crop.shape[:2]

    print(
        "AI crop:",
        crop_w,
        "x",
        crop_h
    )

    # -----------------------------------------------------
    # Resize to LaMa input
    # -----------------------------------------------------

    ai_image = cv2.resize(
        crop,
        (512, 512),
        interpolation=cv2.INTER_AREA
    )

    ai_mask = cv2.resize(
        crop_mask,
        (512, 512),
        interpolation=cv2.INTER_NEAREST
    )

    # -----------------------------------------------------
    # Official LaMa preprocessing
    # -----------------------------------------------------

    image_blob = cv2.dnn.blobFromImage(
        ai_image,
        0.00392,
        (512, 512),
        (0, 0, 0),
        False,
        False
    )

    mask_blob = cv2.dnn.blobFromImage(
        ai_mask,
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

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    net = get_lama_model()

    net.setInput(
        image_blob,
        "image"
    )

    net.setInput(
        mask_blob,
        "mask"
    )

    output = net.forward()

    # -----------------------------------------------------
    # Official LaMa postprocessing
    # -----------------------------------------------------

    result = output[0]

    result = np.transpose(
        result,
        (1, 2, 0)
    )

    result = result.astype(
        np.uint8
    )

    # Resize back
    result = cv2.resize(
        result,
        (crop_w, crop_h),
        interpolation=cv2.INTER_CUBIC
    )

    # -----------------------------------------------------
    # Smooth edge
    # -----------------------------------------------------

    soft_mask = cv2.GaussianBlur(
        crop_mask,
        (9, 9),
        0
    )

    soft_mask = (
        soft_mask.astype(
            np.float32
        ) / 255.0
    )

    soft_mask = soft_mask[
        ...,
        np.newaxis
    ]

    original = crop.astype(
        np.float32
    )

    generated = result.astype(
        np.float32
    )

    blended = (
        generated * soft_mask
        +
        original * (1.0 - soft_mask)
    )

    blended = np.clip(
        blended,
        0,
        255
    ).astype(
        np.uint8
    )

    # -----------------------------------------------------
    # Put crop back
    # -----------------------------------------------------

    final_image = image.copy()

    final_image[
        y1:y2,
        x1:x2
    ] = blended

    return final_image


# =========================================================
# CLASSIC
# =========================================================

def classic_inpaint(image, mask, mode):

    mask = clean_mask(
        mask,
        mode
    )

    return cv2.inpaint(
        image,
        mask,
        7,
        cv2.INPAINT_TELEA
    )


# =========================================================
# ROUTES
# =========================================================

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


@app.route(
    "/process",
    methods=["POST"]
)
def process_image():

    try:

        if (
            "image" not in request.files
            or
            "mask" not in request.files
        ):

            return jsonify({
                "error":
                "Image or mask is missing."
            }), 400

        image = decode_image(
            request.files["image"]
        )

        mask = decode_mask(
            request.files["mask"]
        )

        if image is None:

            return jsonify({
                "error":
                "Could not read image."
            }), 400

        if mask is None:

            return jsonify({
                "error":
                "Could not read mask."
            }), 400

        # Match mask to image
        if (
            mask.shape[:2]
            != image.shape[:2]
        ):

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
                "error":
                "Please select an object first."
            }), 400

        method = request.form.get(
            "method",
            "ai"
        )

        mode = request.form.get(
            "selection_mode",
            "brush"
        )

        print(
            "Method:",
            method,
            "| Selection:",
            mode
        )

        # -------------------------------------------------
        # PROCESS
        # -------------------------------------------------

        if method == "ai":

            result = ai_inpaint(
                image,
                mask,
                mode
            )

        else:

            result = classic_inpaint(
                image,
                mask,
                mode
            )

        # -------------------------------------------------
        # JPEG
        # -------------------------------------------------

        success, encoded = cv2.imencode(
            ".jpg",
            result,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                97
            ]
        )

        if not success:

            return jsonify({
                "error":
                "Could not create result."
            }), 500

        return send_file(
            io.BytesIO(
                encoded.tobytes()
            ),
            mimetype="image/jpeg",
            as_attachment=False,
            download_name="cleaned_image.jpg"
        )

    except Exception as e:

        print(
            "PROCESS ERROR:",
            repr(e)
        )

        return jsonify({
            "error":
            "Processing failed.",
            "details":
            str(e)
        }), 500


@app.errorhandler(413)
def too_large(error):

    return jsonify({
        "error":
        "Image is too large. Maximum 16 MB."
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
