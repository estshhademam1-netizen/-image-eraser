import io
import os
import urllib.request

import cv2
import numpy as np

from flask import Flask, jsonify, render_template, request, send_file


app = Flask(__name__)

# Maximum uploaded file size: 16 MB
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
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("LaMa model downloaded.")

    print("Loading LaMa model...")

    lama_net = cv2.dnn.readNetFromONNX(MODEL_PATH)

    lama_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    lama_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    print("LaMa model ready.")

    return lama_net


# =========================================================
# IMAGE DECODING
# =========================================================

def decode_image(file):
    data = file.read()

    if not data:
        return None

    arr = np.frombuffer(data, dtype=np.uint8)

    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def decode_mask(file):
    data = file.read()

    if not data:
        return None

    arr = np.frombuffer(data, dtype=np.uint8)

    return cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)


# =========================================================
# IMAGE SIZE CONTROL
# =========================================================

def limit_image_size(image, max_side=2200):
    """
    Keeps very large phone photos from consuming too much RAM.
    """

    h, w = image.shape[:2]

    largest = max(h, w)

    if largest <= max_side:
        return image

    scale = max_side / float(largest)

    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    return cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )


# =========================================================
# MASK CLEANING
# =========================================================

def prepare_mask(mask, mode="lasso"):
    """
    Converts the user's selection into a clean binary mask.

    White = remove this area
    Black = keep this area
    """

    if mask is None:
        raise ValueError("Mask is missing.")

    # Binary mask
    _, mask = cv2.threshold(
        mask,
        20,
        255,
        cv2.THRESH_BINARY
    )

    if cv2.countNonZero(mask) == 0:
        raise ValueError("No selection found.")

    # Small closing to remove tiny holes/gaps
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        close_kernel,
        iterations=1
    )

    # For brush selections we expand slightly.
    # For lasso selections, don't destroy the user's boundary.
    if mode == "brush":
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (5, 5)
        )

        mask = cv2.dilate(
            mask,
            kernel,
            iterations=1
        )

    return mask


# =========================================================
# FIND OBJECT REGION
# =========================================================

def get_selection_box(mask, image_shape):
    """
    Finds the exact region containing the selected object,
    then adds background context around it.
    """

    points = cv2.findNonZero(mask)

    if points is None:
        raise ValueError("No selected area found.")

    x, y, w, h = cv2.boundingRect(points)

    image_h, image_w = image_shape[:2]

    # Context around object.
    # More context = better background reconstruction.
    largest_object_dimension = max(w, h)

    padding = max(
        80,
        int(largest_object_dimension * 0.75)
    )

    x1 = max(0, x - padding)
    y1 = max(0, y - padding)

    x2 = min(image_w, x + w + padding)
    y2 = min(image_h, y + h + padding)

    return x1, y1, x2, y2


# =========================================================
# AI INPAINTING
# =========================================================

def ai_inpaint(image, mask):
    """
    AI removal using OpenCV LaMa.

    Important:
    LaMa processes a contextual crop,
    but ONLY the selected mask is pasted back.
    """

    clean_mask = prepare_mask(mask, "lasso")

    x1, y1, x2, y2 = get_selection_box(
        clean_mask,
        image.shape
    )

    crop = image[y1:y2, x1:x2].copy()
    crop_mask = clean_mask[y1:y2, x1:x2].copy()

    crop_h, crop_w = crop.shape[:2]

    if crop_h < 2 or crop_w < 2:
        raise ValueError("Selection is too small.")

    # -----------------------------------------------------
    # Make a square context area.
    # This gives LaMa background information around object.
    # -----------------------------------------------------

    square_size = max(crop_h, crop_w)

    center_x = (crop_w // 2)
    center_y = (crop_h // 2)

    half = square_size // 2

    sx1 = max(0, center_x - half)
    sy1 = max(0, center_y - half)

    sx2 = min(crop_w, sx1 + square_size)
    sy2 = min(crop_h, sy1 + square_size)

    # Correct boundary
    sx1 = max(0, sx2 - square_size)
    sy1 = max(0, sy2 - square_size)

    square_crop = crop[sy1:sy2, sx1:sx2]
    square_mask = crop_mask[sy1:sy2, sx1:sx2]

    if square_crop.size == 0:
        raise ValueError("Invalid AI crop.")

    # -----------------------------------------------------
    # Resize to LaMa's expected 512x512 input.
    # -----------------------------------------------------

    ai_image = cv2.resize(
        square_crop,
        (512, 512),
        interpolation=cv2.INTER_AREA
    )

    ai_mask = cv2.resize(
        square_mask,
        (512, 512),
        interpolation=cv2.INTER_NEAREST
    )

    # Ensure binary mask
    ai_mask = np.where(
        ai_mask > 0,
        255,
        0
    ).astype(np.uint8)

    # -----------------------------------------------------
    # LaMa input
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
    ).astype(np.float32)

    # -----------------------------------------------------
    # Run AI
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

    result = output[0]

    result = np.transpose(
        result,
        (1, 2, 0)
    )

    # OpenCV LaMa model outputs 0-255
    result = np.clip(
        result,
        0,
        255
    ).astype(np.uint8)

    # -----------------------------------------------------
    # Resize AI result back to contextual crop.
    # -----------------------------------------------------

    generated_square = cv2.resize(
        result,
        (square_crop.shape[1], square_crop.shape[0]),
        interpolation=cv2.INTER_LANCZOS4
    )

    # -----------------------------------------------------
    # IMPORTANT:
    # Only replace the selected mask.
    # Everything outside it stays ORIGINAL.
    # -----------------------------------------------------

    original_float = square_crop.astype(np.float32)
    generated_float = generated_square.astype(np.float32)

    binary = (
        square_mask > 0
    ).astype(np.float32)

    # Small soft edge to avoid a hard seam.
    soft = cv2.GaussianBlur(
        binary,
        (9, 9),
        0
    )

    # Keep the inside strongly selected.
    soft = np.clip(
        soft,
        0.0,
        1.0
    )[..., None]

    blended = (
        generated_float * soft
        +
        original_float * (1.0 - soft)
    )

    blended = np.clip(
        blended,
        0,
        255
    ).astype(np.uint8)

    # -----------------------------------------------------
    # Put generated patch into crop.
    # -----------------------------------------------------

    crop_result = crop.copy()

    crop_result[
        sy1:sy2,
        sx1:sx2
    ] = blended

    # -----------------------------------------------------
    # Put ONLY this contextual patch back into original.
    # -----------------------------------------------------

    final_image = image.copy()

    final_image[
        y1:y2,
        x1:x2
    ] = crop_result

    return final_image


# =========================================================
# CLASSIC INPAINTING
# =========================================================

def classic_inpaint(image, mask):
    """
    Fast OpenCV inpainting.
    """

    clean_mask = prepare_mask(
        mask,
        "brush"
    )

    return cv2.inpaint(
        image,
        clean_mask,
        5,
        cv2.INPAINT_TELEA
    )


# =========================================================
# PROCESS ROUTE
# =========================================================

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

        image = decode_image(
            request.files["image"]
        )

        mask = decode_mask(
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

        # Keep memory reasonable on Render
        original_h, original_w = image.shape[:2]

        image = limit_image_size(
            image,
            2200
        )

        # Resize mask to current image size
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

        selection_mode = request.form.get(
            "selection_mode",
            "lasso"
        )

        print(
            f"Processing: method={method}, "
            f"selection={selection_mode}, "
            f"image={image.shape}"
        )

        # -------------------------------------------------
        # AI
        # -------------------------------------------------

        if method == "ai":

            result = ai_inpaint(
                image,
                mask
            )

        # -------------------------------------------------
        # CLASSIC
        # -------------------------------------------------

        else:

            result = classic_inpaint(
                image,
                mask
            )

        # -------------------------------------------------
        # JPEG response
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
