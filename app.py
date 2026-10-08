from flask import Flask, render_template, request, send_file, jsonify
import cv2
import numpy as np
import io
import os

# =========================================================
# LAMA
# =========================================================

try:
    from simple_lama_inpainting import SimpleLama
    from PIL import Image

    print("Loading LaMa model...")
    lama = SimpleLama()
    print("LaMa model loaded successfully!")

except Exception as e:
    lama = None
    print("LaMa could not be loaded:")
    print(repr(e))


app = Flask(__name__)


# =========================================================
# IMAGE HELPERS
# =========================================================

def read_image(file_bytes):
    arr = np.frombuffer(file_bytes, np.uint8)

    image = cv2.imdecode(
        arr,
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise ValueError("Could not read image.")

    return image


def encode_image(image):

    ok, buffer = cv2.imencode(
        ".jpg",
        image,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            95
        ]
    )

    if not ok:
        raise ValueError("Could not encode image.")

    return io.BytesIO(
        buffer.tobytes()
    )


# =========================================================
# MASK
# =========================================================

def prepare_mask(mask_bytes, target_shape):

    arr = np.frombuffer(
        mask_bytes,
        np.uint8
    )

    mask = cv2.imdecode(
        arr,
        cv2.IMREAD_GRAYSCALE
    )

    if mask is None:
        raise ValueError(
            "Could not read mask."
        )

    height, width = target_shape[:2]

    # Make mask exactly same size as image
    if mask.shape != (height, width):

        mask = cv2.resize(
            mask,
            (width, height),
            interpolation=cv2.INTER_NEAREST
        )

    # -----------------------------------------------------
    # Binary mask
    # White = remove
    # Black = keep
    # -----------------------------------------------------

    mask = np.where(
        mask > 20,
        255,
        0
    ).astype(np.uint8)

    return mask


# =========================================================
# MASK CLEANING
# =========================================================

def improve_mask(mask):

    if cv2.countNonZero(mask) == 0:
        return mask

    # Fill tiny holes inside selected object
    kernel_close = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel_close
    )

    # Remove tiny noise
    kernel_open = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel_open
    )

    return mask


# =========================================================
# LAMA INPAINTING
# =========================================================

def lama_inpaint(image, mask):

    if lama is None:
        raise RuntimeError(
            "LaMa model is not available. "
            "Install simple-lama-inpainting."
        )

    if cv2.countNonZero(mask) == 0:
        return image.copy()

    # -----------------------------------------------------
    # IMPORTANT
    #
    # LaMa expects:
    #
    # image = RGB
    # mask  = grayscale
    #
    # 255 = area to remove
    # -----------------------------------------------------

    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    pil_image = Image.fromarray(
        image_rgb
    )

    pil_mask = Image.fromarray(
        mask
    ).convert("L")

    # -----------------------------------------------------
    # Run AI
    # -----------------------------------------------------

    result = lama(
        pil_image,
        pil_mask
    )

    # PIL -> numpy
    result = np.array(
        result
    )

    # RGB -> BGR
    result = cv2.cvtColor(
        result,
        cv2.COLOR_RGB2BGR
    )

    return result


# =========================================================
# CLASSIC FALLBACK
# =========================================================

def classic_inpaint(image, mask):

    if cv2.countNonZero(mask) == 0:
        return image.copy()

    # Slight expansion to remove object edges
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    expanded = cv2.dilate(
        mask,
        kernel,
        iterations=1
    )

    # Determine radius based on mask size
    area = cv2.countNonZero(
        expanded
    )

    if area < 5000:
        radius = 3

    elif area < 30000:
        radius = 5

    else:
        radius = 7

    telea = cv2.inpaint(
        image,
        expanded,
        radius,
        cv2.INPAINT_TELEA
    )

    ns = cv2.inpaint(
        image,
        expanded,
        radius,
        cv2.INPAINT_NS
    )

    result = cv2.addWeighted(
        telea,
        0.55,
        ns,
        0.45,
        0
    )

    return result


# =========================================================
# MAIN PROCESS
# =========================================================

@app.route(
    "/process",
    methods=["POST"]
)
def process():

    try:

        # -------------------------------------------------
        # Check files
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

        # -------------------------------------------------
        # Read image
        # -------------------------------------------------

        image_bytes = image_file.read()

        image = read_image(
            image_bytes
        )

        original_height, original_width = \
            image.shape[:2]

        print(
            f"Original image: "
            f"{original_width} x "
            f"{original_height}"
        )

        # -------------------------------------------------
        # Read mask
        # -------------------------------------------------

        mask = prepare_mask(
            mask_file.read(),
            image.shape
        )

        # -------------------------------------------------
        # Clean mask
        # -------------------------------------------------

        mask = improve_mask(
            mask
        )

        selected_pixels = cv2.countNonZero(
            mask
        )

        print(
            f"Selected pixels: "
            f"{selected_pixels}"
        )

        if selected_pixels == 0:

            return jsonify({
                "error":
                "The mask is empty. "
                "Select an object first."
            }), 400

        # -------------------------------------------------
        # Method
        # -------------------------------------------------

        method = request.form.get(
            "method",
            "ai"
        )

        print(
            f"Processing method: {method}"
        )

        # =================================================
        # AI
        # =================================================

        if method == "ai":

            if lama is None:

                return jsonify({
                    "error":
                    "LaMa is not available. "
                    "Install the required package."
                }), 500

            result = lama_inpaint(
                image,
                mask
            )

        # =================================================
        # CLASSIC
        # =================================================

        elif method == "classic":

            result = classic_inpaint(
                image,
                mask
            )

        else:

            return jsonify({
                "error":
                "Unknown method."
            }), 400

        # -------------------------------------------------
        # Safety check
        # -------------------------------------------------

        if result is None:

            raise ValueError(
                "Inpainting returned no result."
            )

        # Make sure result has same dimensions
        if result.shape[:2] != (
            original_height,
            original_width
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

        output = encode_image(
            result
        )

        print(
            "Processing completed successfully."
        )

        return send_file(
            output,
            mimetype="image/jpeg",
            as_attachment=False,
            download_name="removed_object.jpg"
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
# HEALTH CHECK
# =========================================================

@app.route("/health")
def health():

    return jsonify({
        "flask": True,
        "lama": lama is not None
    })


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

    print("")
    print("==============================")
    print(" Object Removal Flask App")
    print("==============================")
    print(
        "LaMa available:",
        lama is not None
    )
    print(
        f"Server running on port {port}"
    )
    print("==============================")
    print("")

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
