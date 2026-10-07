import io
import os
import urllib.request

import cv2
import numpy as np

from flask import Flask, jsonify, render_template, request, send_file


app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024


# =========================================================
# LAMA
# =========================================================

MODEL_URL = (
    "https://huggingface.co/opencv/inpainting_lama/"
    "resolve/main/inpainting_lama_2025jan.onnx"
)

MODEL_PATH = "/tmp/inpainting_lama_2025jan.onnx"

lama_net = None


def get_lama():

    global lama_net

    if lama_net is not None:
        return lama_net

    if not os.path.exists(MODEL_PATH):

        print("Downloading LaMa model...")

        urllib.request.urlretrieve(
            MODEL_URL,
            MODEL_PATH
        )

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

    print("LaMa ready.")

    return lama_net


# =========================================================
# READ IMAGE
# =========================================================

def read_image(file):

    data = file.read()

    if not data:
        return None

    arr = np.frombuffer(
        data,
        np.uint8
    )

    return cv2.imdecode(
        arr,
        cv2.IMREAD_COLOR
    )


def read_mask(file):

    data = file.read()

    if not data:
        return None

    arr = np.frombuffer(
        data,
        np.uint8
    )

    return cv2.imdecode(
        arr,
        cv2.IMREAD_GRAYSCALE
    )


# =========================================================
# RESIZE LARGE PHONE IMAGES
# =========================================================

def resize_image(image, max_side=1800):

    h, w = image.shape[:2]

    longest = max(h, w)

    if longest <= max_side:
        return image

    scale = max_side / float(longest)

    nw = int(w * scale)
    nh = int(h * scale)

    return cv2.resize(
        image,
        (nw, nh),
        interpolation=cv2.INTER_AREA
    )


# =========================================================
# CLEAN MASK
# =========================================================

def clean_mask(mask):

    # Binary
    mask = np.where(
        mask > 10,
        255,
        0
    ).astype(np.uint8)

    if cv2.countNonZero(mask) == 0:
        raise ValueError(
            "No selected area."
        )

    # Fill tiny holes
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
# GET OBJECT BOX
# =========================================================

def get_object_box(mask, image):

    points = cv2.findNonZero(mask)

    if points is None:
        raise ValueError(
            "No object selected."
        )

    x, y, w, h = cv2.boundingRect(
        points
    )

    ih, iw = image.shape[:2]

    # Context around the object.
    #
    # Enough background for LaMa to understand
    # grass / fence / wall / texture.
    pad_x = max(
        100,
        int(w * 0.9)
    )

    pad_y = max(
        100,
        int(h * 0.9)
    )

    x1 = max(
        0,
        x - pad_x
    )

    y1 = max(
        0,
        y - pad_y
    )

    x2 = min(
        iw,
        x + w + pad_x
    )

    y2 = min(
        ih,
        y + h + pad_y
    )

    return x1, y1, x2, y2


# =========================================================
# MAKE SQUARE CONTEXT
# =========================================================

def make_square_context(
    image,
    mask
):

    h, w = image.shape[:2]

    points = cv2.findNonZero(mask)

    x, y, bw, bh = cv2.boundingRect(
        points
    )

    # Center around the actual selected object
    cx = x + bw // 2
    cy = y + bh // 2

    # Context must be significantly larger
    # than the object.
    side = int(
        max(bw, bh) * 2.4
    )

    # Minimum context
    side = max(
        side,
        320
    )

    # Don't exceed image dimensions
    side = min(
        side,
        max(h, w)
    )

    half = side // 2

    sx1 = cx - half
    sy1 = cy - half
    sx2 = sx1 + side
    sy2 = sy1 + side

    # Shift inside image
    if sx1 < 0:
        sx2 -= sx1
        sx1 = 0

    if sy1 < 0:
        sy2 -= sy1
        sy1 = 0

    if sx2 > w:
        shift = sx2 - w
        sx1 -= shift
        sx2 = w

    if sy2 > h:
        shift = sy2 - h
        sy1 -= shift
        sy2 = h

    sx1 = max(
        0,
        sx1
    )

    sy1 = max(
        0,
        sy1
    )

    sx2 = min(
        w,
        sx2
    )

    sy2 = min(
        h,
        sy2
    )

    crop = image[
        sy1:sy2,
        sx1:sx2
    ].copy()

    crop_mask = mask[
        sy1:sy2,
        sx1:sx2
    ].copy()

    return (
        crop,
        crop_mask,
        sx1,
        sy1
    )


# =========================================================
# AI LAMA
# =========================================================

def lama_remove(
    image,
    mask
):

    mask = clean_mask(
        mask
    )

    # -----------------------------------------------------
    # Get context
    # -----------------------------------------------------

    crop, crop_mask, offset_x, offset_y = \
        make_square_context(
            image,
            mask
        )

    ch, cw = crop.shape[:2]

    if ch < 10 or cw < 10:
        raise ValueError(
            "Invalid crop."
        )

    # -----------------------------------------------------
    # Resize to 512x512
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

    ai_mask = np.where(
        ai_mask > 0,
        255,
        0
    ).astype(np.uint8)

    # -----------------------------------------------------
    # IMPORTANT:
    # Slightly expand mask.
    #
    # This helps remove the edge of the object.
    # -----------------------------------------------------

    expand_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    ai_mask = cv2.dilate(
        ai_mask,
        expand_kernel,
        iterations=1
    )

    # -----------------------------------------------------
    # LAMA INPUT
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
    # RUN MODEL
    # -----------------------------------------------------

    net = get_lama()

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
    # OFFICIAL OPENCV LAMA OUTPUT FORMAT
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
    # Resize result to original crop
    # -----------------------------------------------------

    generated = cv2.resize(
        result,
        (cw, ch),
        interpolation=cv2.INTER_LANCZOS4
    )

    # -----------------------------------------------------
    # IMPORTANT:
    #
    # Do NOT replace the whole crop.
    #
    # Only pixels inside the original user mask
    # are allowed to change.
    # -----------------------------------------------------

    original = crop.astype(
        np.float32
    )

    generated = generated.astype(
        np.float32
    )

    original_mask = (
        crop_mask > 0
    ).astype(
        np.float32
    )

    # -----------------------------------------------------
    # Soft edge
    # -----------------------------------------------------

    soft = cv2.GaussianBlur(
        original_mask,
        (11, 11),
        0
    )

    # Make sure only selected region changes.
    soft = np.clip(
        soft,
        0,
        1
    )

    soft = soft[..., None]

    # -----------------------------------------------------
    # Composite
    # -----------------------------------------------------

    final_crop = (
        generated * soft
        +
        original * (1.0 - soft)
    )

    final_crop = np.clip(
        final_crop,
        0,
        255
    ).astype(
        np.uint8
    )

    # -----------------------------------------------------
    # PUT PATCH BACK
    # -----------------------------------------------------

    result_image = image.copy()

    result_image[
        offset_y:offset_y + ch,
        offset_x:offset_x + cw
    ] = final_crop

    return result_image


# =========================================================
# CLASSIC
# =========================================================

def classic_remove(
    image,
    mask
):

    mask = clean_mask(
        mask
    )

    # Larger radius gives more background reconstruction
    return cv2.inpaint(
        image,
        mask,
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

        image = read_image(
            request.files["image"]
        )

        mask = read_mask(
            request.files["mask"]
        )

        if image is None:
            return jsonify({
                "error": "Could not read image."
            }), 400

        if mask is None:
            return jsonify({
                "error": "Could not read mask."
            }), 400

        # -------------------------------------------------
        # Resize phone image if extremely large
        # -------------------------------------------------

        image = resize_image(
            image,
            1800
        )

        # -------------------------------------------------
        # Mask must match image
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
                "error": "Please select an object."
            }), 400

        method = request.form.get(
            "method",
            "ai"
        )

        print(
            "Processing method:",
            method
        )

        # -------------------------------------------------
        # AI
        # -------------------------------------------------

        if method == "ai":

            result = lama_remove(
                image,
                mask
            )

        # -------------------------------------------------
        # CLASSIC
        # -------------------------------------------------

        else:

            result = classic_remove(
                image,
                mask
            )

        # -------------------------------------------------
        # OUTPUT
        # -------------------------------------------------

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
            "ERROR:",
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
