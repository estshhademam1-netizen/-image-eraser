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
# DECODE FILES
# =========================================================

def decode_image_file(file):

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


def decode_mask_file(file):

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
# MASK PROCESSING
# =========================================================

def prepare_mask(mask):

    # Binary mask
    _, mask = cv2.threshold(
        mask,
        10,
        255,
        cv2.THRESH_BINARY
    )

    # -----------------------------------------------------
    # Fill closed contours
    # This allows the user to circle an object.
    # -----------------------------------------------------

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    filled = np.zeros_like(mask)

    for contour in contours:

        area = cv2.contourArea(contour)

        if area > 20:

            cv2.drawContours(
                filled,
                [contour],
                -1,
                255,
                thickness=cv2.FILLED
            )

    # If there were no usable contours,
    # keep original painted mask.
    if cv2.countNonZero(filled) == 0:

        filled = mask.copy()

    # -----------------------------------------------------
    # Small expansion
    # -----------------------------------------------------

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    filled = cv2.dilate(
        filled,
        kernel,
        iterations=1
    )

    # -----------------------------------------------------
    # Close tiny holes
    # -----------------------------------------------------

    filled = cv2.morphologyEx(
        filled,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1
    )

    return filled


# =========================================================
# FIND GOOD AI CROP
# =========================================================

def get_ai_crop(image, mask):

    points = cv2.findNonZero(mask)

    if points is None:

        raise ValueError(
            "No selected object was found."
        )

    x, y, w, h = cv2.boundingRect(
        points
    )

    img_h, img_w = image.shape[:2]

    # -----------------------------------------------------
    # Context around object
    # -----------------------------------------------------

    largest_side = max(
        w,
        h
    )

    # More context for natural reconstruction
    padding = max(
        int(largest_side * 0.65),
        80
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
        img_w,
        x + w + padding
    )

    y2 = min(
        img_h,
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
# AI LAMA INPAINTING
# =========================================================

def run_ai_inpainting(image, mask):

    clean_mask = prepare_mask(
        mask
    )

    (
        crop,
        crop_mask,
        x1,
        y1,
        x2,
        y2
    ) = get_ai_crop(
        image,
        clean_mask
    )

    crop_h, crop_w = crop.shape[:2]

    print(
        "AI crop:",
        crop_w,
        "x",
        crop_h
    )

    # -----------------------------------------------------
    # Resize image and mask
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
    # IMPORTANT:
    # This follows the official OpenCV LaMa preprocessing.
    # -----------------------------------------------------

    image_blob = cv2.dnn.blobFromImage(
        ai_image,
        scalefactor=1.0 / 255.0,
        size=(512, 512),
        mean=(0, 0, 0),
        swapRB=False,
        crop=False
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
    # Run model
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
    # IMPORTANT:
    # Official OpenCV LaMa post-processing
    # does NOT multiply the output by 255.
    # -----------------------------------------------------

    result = output[0]

    result = np.transpose(
        result,
        (1, 2, 0)
    )

    result = result.astype(
        np.uint8
    )

    # -----------------------------------------------------
    # Resize back
    # -----------------------------------------------------

    result = cv2.resize(
        result,
        (crop_w, crop_h),
        interpolation=cv2.INTER_CUBIC
    )

    # -----------------------------------------------------
    # Create soft transition
    # -----------------------------------------------------

    blend_mask = cv2.GaussianBlur(
        crop_mask,
        (9, 9),
        0
    )

    blend_mask = (
        blend_mask.astype(
            np.float32
        ) / 255.0
    )

    blend_mask = blend_mask[
        ...,
        np.newaxis
    ]

    original_crop = crop.astype(
        np.float32
    )

    generated_crop = result.astype(
        np.float32
    )

    # -----------------------------------------------------
    # Blend only the selected area
    # -----------------------------------------------------

    blended = (
        generated_crop * blend_mask
        +
        original_crop * (
            1.0 - blend_mask
        )
    )

    blended = np.clip(
        blended,
        0,
        255
    ).astype(
        np.uint8
    )

    # -----------------------------------------------------
    # Put result back into original image
    # -----------------------------------------------------

    final_image = image.copy()

    final_image[
        y1:y2,
        x1:x2
    ] = blended

    return final_image


# =========================================================
# CLASSIC OPENCV
# =========================================================

def run_classic_inpainting(image, mask):

    clean_mask = prepare_mask(
        mask
    )

    result = cv2.inpaint(
        image,
        clean_mask,
        7,
        cv2.INPAINT_TELEA
    )

    return result


# =========================================================
# MAIN PROCESS
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
                "Missing image or mask data."
            }), 400

        # -------------------------------------------------
        # Decode
        # -------------------------------------------------

        image = decode_image_file(
            request.files["image"]
        )

        mask = decode_mask_file(
            request.files["mask"]
        )

        if image is None:

            return jsonify({
                "error":
                "Could not read the image."
            }), 400

        if mask is None:

            return jsonify({
                "error":
                "Could not read the mask."
            }), 400

        # -------------------------------------------------
        # Match dimensions
        # -------------------------------------------------

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

        # -------------------------------------------------
        # Check selection
        # -------------------------------------------------

        if cv2.countNonZero(mask) == 0:

            return jsonify({
                "error":
                "Please select an object first."
            }), 400

        # -------------------------------------------------
        # Selected method
        # -------------------------------------------------

        method = request.form.get(
            "method",
            "ai"
        )

        print(
            "Selected method:",
            method
        )

        # -------------------------------------------------
        # AI
        # -------------------------------------------------

        if method == "ai":

            print(
                "Starting LaMa..."
            )

            result = run_ai_inpainting(
                image,
                mask
            )

            print(
                "LaMa finished."
            )

        # -------------------------------------------------
        # Classic
        # -------------------------------------------------

        else:

            print(
                "Starting OpenCV inpainting..."
            )

            result = run_classic_inpainting(
                image,
                mask
            )

            print(
                "Classic finished."
            )

        # -------------------------------------------------
        # Encode
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
                "Failed to encode result."
            }), 500

        return send_file(
            io.BytesIO(
                encoded.tobytes()
            ),
            mimetype="image/jpeg",
            as_attachment=False,
            download_name=
            "cleaned_image.jpg"
        )

    except Exception as e:

        print(
            "PROCESS ERROR:",
            repr(e)
        )

        return jsonify({
            "error":
            "Image processing failed.",
            "details":
            str(e)
        }), 500


# =========================================================
# FILE SIZE ERROR
# =========================================================

@app.errorhandler(413)
def file_too_large(error):

    return jsonify({
        "error":
        "Image is too large. Maximum size is 16 MB."
    }), 413


# =========================================================
# LOCAL RUN
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
