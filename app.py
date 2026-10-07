from flask import Flask, render_template, request, jsonify, send_file
import cv2
import numpy as np
import io
import os
import urllib.request

app = Flask(__name__)

# Maximum uploaded image size = 12 MB
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024


# =========================================================
# LaMa AI MODEL
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

    # Download model when AI is used for the first time
    if not os.path.exists(MODEL_PATH):

        print("Downloading LaMa AI model...")

        urllib.request.urlretrieve(
            MODEL_URL,
            MODEL_PATH
        )

        print("LaMa model downloaded.")

    print("Loading LaMa AI model...")

    lama_net = cv2.dnn.readNetFromONNX(
        MODEL_PATH
    )

    lama_net.setPreferableBackend(
        cv2.dnn.DNN_BACKEND_OPENCV
    )

    lama_net.setPreferableTarget(
        cv2.dnn.DNN_TARGET_CPU
    )

    print("LaMa AI model ready.")

    return lama_net


# =========================================================
# BASIC ROUTES
# =========================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# =========================================================
# IMAGE HELPERS
# =========================================================

def decode_uploaded_image(file):

    data = file.read()

    if not data:
        return None

    array = np.frombuffer(
        data,
        dtype=np.uint8
    )

    image = cv2.imdecode(
        array,
        cv2.IMREAD_COLOR
    )

    return image


def decode_uploaded_mask(file):

    data = file.read()

    if not data:
        return None

    array = np.frombuffer(
        data,
        dtype=np.uint8
    )

    mask = cv2.imdecode(
        array,
        cv2.IMREAD_GRAYSCALE
    )

    return mask


# =========================================================
# MASK CLEANING
# =========================================================

def prepare_mask(mask):

    # Make mask binary
    _, mask = cv2.threshold(
        mask,
        10,
        255,
        cv2.THRESH_BINARY
    )

    # Close tiny holes
    kernel_close = np.ones(
        (5, 5),
        np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel_close
    )

    # Slight expansion
    kernel_dilate = np.ones(
        (5, 5),
        np.uint8
    )

    mask = cv2.dilate(
        mask,
        kernel_dilate,
        iterations=1
    )

    return mask


# =========================================================
# CLASSIC OPENCV REMOVAL
# =========================================================

def classic_remove(image, mask):

    # Stronger mask for better object removal
    clean_mask = prepare_mask(mask)

    # Telea is fast and works well for small/simple objects
    result = cv2.inpaint(
        image,
        clean_mask,
        7,
        cv2.INPAINT_TELEA
    )

    return result


# =========================================================
# AI LAMA REMOVAL
# =========================================================

def ai_remove(image, mask):

    clean_mask = prepare_mask(mask)

    # Find selected area
    points = cv2.findNonZero(
        clean_mask
    )

    if points is None:

        raise ValueError(
            "Please select an object first."
        )

    x, y, w, h = cv2.boundingRect(
        points
    )

    image_h, image_w = image.shape[:2]

    # -----------------------------------------------------
    # Add padding around selected object
    # -----------------------------------------------------

    padding = 80

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

    crop_mask = clean_mask[
        y1:y2,
        x1:x2
    ].copy()

    crop_h, crop_w = crop.shape[:2]

    # -----------------------------------------------------
    # Resize only the selected region to 512x512
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

    # Normalize image
    image_blob = cv2.dnn.blobFromImage(
        ai_image,
        scalefactor=1.0 / 255.0,
        size=(512, 512),
        mean=(0, 0, 0),
        swapRB=False,
        crop=False
    )

    # Mask
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
    # Load model
    # -----------------------------------------------------

    net = get_lama_model()

    # -----------------------------------------------------
    # AI inference
    # -----------------------------------------------------

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
    # Convert output
    # -----------------------------------------------------

    result = output[0]

    result = np.transpose(
        result,
        (1, 2, 0)
    )

    result = np.clip(
        result,
        0,
        1
    )

    result = (
        result * 255
    ).astype(
        np.uint8
    )

    # Resize AI result back to crop size
    result = cv2.resize(
        result,
        (crop_w, crop_h),
        interpolation=cv2.INTER_CUBIC
    )

    # -----------------------------------------------------
    # Blend result into original image
    # -----------------------------------------------------

    final_image = image.copy()

    # Soft edge for natural transition
    soft_mask = cv2.GaussianBlur(
        crop_mask,
        (11, 11),
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

    original_crop = final_image[
        y1:y2,
        x1:x2
    ].astype(
        np.float32
    )

    ai_crop = result.astype(
        np.float32
    )

    blended = (
        ai_crop * soft_mask
        +
        original_crop * (1.0 - soft_mask)
    )

    blended = np.clip(
        blended,
        0,
        255
    ).astype(
        np.uint8
    )

    final_image[
        y1:y2,
        x1:x2
    ] = blended

    return final_image


# =========================================================
# PROCESS IMAGE
# =========================================================

@app.route(
    "/process",
    methods=["POST"]
)
def process_image():

    try:

        # -------------------------------------------------
        # Validate request
        # -------------------------------------------------

        if "image" not in request.files:

            return jsonify({
                "error":
                "Image is missing."
            }), 400

        if "mask" not in request.files:

            return jsonify({
                "error":
                "Mask is missing."
            }), 400

        image_file = request.files[
            "image"
        ]

        mask_file = request.files[
            "mask"
        ]

        method = request.form.get(
            "method",
            "classic"
        )

        # -------------------------------------------------
        # Decode
        # -------------------------------------------------

        image = decode_uploaded_image(
            image_file
        )

        mask = decode_uploaded_mask(
            mask_file
        )

        if image is None:

            return jsonify({
                "error":
                "Invalid image."
            }), 400

        if mask is None:

            return jsonify({
                "error":
                "Invalid mask."
            }), 400

        # -------------------------------------------------
        # Make sure mask matches image
        # -------------------------------------------------

        if (
            mask.shape[0] != image.shape[0]
            or
            mask.shape[1] != image.shape[1]
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
                "Please paint over the object first."
            }), 400

        # -------------------------------------------------
        # Choose processing method
        # -------------------------------------------------

        if method == "ai":

            print(
                "Starting AI LaMa removal..."
            )

            result = ai_remove(
                image,
                mask
            )

            print(
                "AI removal finished."
            )

        else:

            print(
                "Starting Classic OpenCV removal..."
            )

            result = classic_remove(
                image,
                mask
            )

            print(
                "Classic removal finished."
            )

        # -------------------------------------------------
        # Encode result
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
                "error":
                "Could not create result."
            }), 500

        return send_file(
            io.BytesIO(
                encoded.tobytes()
            ),
            mimetype="image/jpeg",
            as_attachment=False,
            download_name=
            "image-eraser-result.jpg"
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
# FILE TOO LARGE
# =========================================================

@app.errorhandler(413)
def too_large(error):

    return jsonify({
        "error":
        "Image is too large. Maximum size is 12 MB."
    }), 413


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
