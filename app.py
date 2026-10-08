from flask import Flask, render_template, request, send_file, jsonify
import cv2
import numpy as np
import io
import os

app = Flask(__name__)


# =========================================================
# IMAGE HELPERS
# =========================================================

def read_image(file_bytes):
    arr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Could not read image.")

    return image


def encode_image(image):
    ok, buffer = cv2.imencode(
        ".jpg",
        image,
        [cv2.IMWRITE_JPEG_QUALITY, 95]
    )

    if not ok:
        raise ValueError("Could not encode image.")

    return io.BytesIO(buffer.tobytes())


def prepare_mask(mask_bytes, target_shape):
    arr = np.frombuffer(mask_bytes, np.uint8)

    mask = cv2.imdecode(
        arr,
        cv2.IMREAD_GRAYSCALE
    )

    if mask is None:
        raise ValueError("Could not read mask.")

    height, width = target_shape[:2]

    if mask.shape[:2] != (height, width):
        mask = cv2.resize(
            mask,
            (width, height),
            interpolation=cv2.INTER_NEAREST
        )

    # Binary mask
    mask = np.where(mask > 20, 255, 0).astype(np.uint8)

    return mask


# =========================================================
# CLASSIC HYBRID INPAINTING
# =========================================================

def classic_hybrid(image, mask):
    """
    Improved classic object removal.

    Uses:
        1. Small mask expansion
        2. Telea inpainting
        3. Navier-Stokes inpainting
        4. Edge-aware blending

    No AI model required.
    """

    if cv2.countNonZero(mask) == 0:
        return image.copy()

    # -----------------------------------------------------
    # 1. Clean mask
    # -----------------------------------------------------

    kernel_small = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )

    clean_mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel_small
    )

    # -----------------------------------------------------
    # 2. Slightly expand the selected object
    # -----------------------------------------------------

    # Prevent leftover object edges.
    kernel_expand = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (7, 7)
    )

    expanded_mask = cv2.dilate(
        clean_mask,
        kernel_expand,
        iterations=1
    )

    # -----------------------------------------------------
    # 3. Determine reasonable radius
    # -----------------------------------------------------

    area = cv2.countNonZero(expanded_mask)

    if area < 5000:
        radius = 3
    elif area < 30000:
        radius = 5
    else:
        radius = 7

    # -----------------------------------------------------
    # 4. Telea
    # -----------------------------------------------------

    telea = cv2.inpaint(
        image,
        expanded_mask,
        radius,
        cv2.INPAINT_TELEA
    )

    # -----------------------------------------------------
    # 5. Navier-Stokes
    # -----------------------------------------------------

    ns = cv2.inpaint(
        image,
        expanded_mask,
        radius,
        cv2.INPAINT_NS
    )

    # -----------------------------------------------------
    # 6. Choose/merge result
    # -----------------------------------------------------

    # NS tends to preserve directional structures.
    # Telea tends to preserve local texture.
    #
    # Blend them instead of trusting one algorithm alone.

    hybrid = cv2.addWeighted(
        telea,
        0.55,
        ns,
        0.45,
        0
    )

    # -----------------------------------------------------
    # 7. Feather the mask edge
    # -----------------------------------------------------

    feather = cv2.GaussianBlur(
        expanded_mask,
        (0, 0),
        sigmaX=3
    )

    alpha = feather.astype(np.float32) / 255.0
    alpha = alpha[:, :, None]

    result = (
        image.astype(np.float32) * (1.0 - alpha)
        +
        hybrid.astype(np.float32) * alpha
    )

    result = np.clip(
        result,
        0,
        255
    ).astype(np.uint8)

    # -----------------------------------------------------
    # 8. Only change the selected area
    # -----------------------------------------------------

    # Keep everything outside the selection EXACTLY
    # as it was.

    original_float = image.astype(np.float32)
    result_float = result.astype(np.float32)

    final = (
        original_float * (1.0 - alpha)
        +
        result_float * alpha
    )

    final = np.clip(
        final,
        0,
        255
    ).astype(np.uint8)

    return final


# =========================================================
# ROUTES
# =========================================================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
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

        image_file = request.files["image"]
        mask_file = request.files["mask"]

        image = read_image(
            image_file.read()
        )

        mask = prepare_mask(
            mask_file.read(),
            image.shape
        )

        method = request.form.get(
            "method",
            "classic"
        )

        # -------------------------------------------------
        # CLASSIC
        # -------------------------------------------------

        if method == "classic":

            result = classic_hybrid(
                image,
                mask
            )

        # -------------------------------------------------
        # AI
        # -------------------------------------------------

        elif method == "ai":

            # Keep AI endpoint available.
            # For now use classic fallback if LaMa
            # cannot be loaded on the free server.

            result = classic_hybrid(
                image,
                mask
            )

        else:

            return jsonify({
                "error": "Unknown method."
            }), 400

        output = encode_image(result)

        return send_file(
            output,
            mimetype="image/jpeg",
            as_attachment=False,
            download_name="modified.jpg"
        )

    except Exception as e:

        print("PROCESS ERROR:", repr(e))

        return jsonify({
            "error": str(e)
        }), 500


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
