import os

# ============================================================
# FORCE CPU - Render Free does not have NVIDIA GPU
# ============================================================

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import io
import traceback

import cv2
import numpy as np

from PIL import Image

from flask import (
    Flask,
    request,
    jsonify,
    send_file,
    render_template
)


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024


# ============================================================
# LAMA STATE
# ============================================================

lama = None
lama_error = None


# ============================================================
# LOAD LAMA
# ============================================================

def get_lama():

    global lama
    global lama_error

    # Already loaded
    if lama is not None:
        return lama

    try:

        import torch
        from simple_lama_inpainting import SimpleLama

        print("=" * 60)
        print("Loading LaMa...")
        print("PyTorch:", torch.__version__)
        print("CUDA available:", torch.cuda.is_available())
        print("Device: CPU")
        print("=" * 60)

        # IMPORTANT:
        # Force LaMa to CPU
        device = torch.device("cpu")

        lama = SimpleLama(
            device=device
        )

        lama_error = None

        print("=" * 60)
        print("LaMa loaded successfully!")
        print("Running on CPU")
        print("=" * 60)

        return lama

    except Exception as e:

        lama = None
        lama_error = repr(e)

        print("=" * 60)
        print("LaMa loading failed:")
        print(repr(e))
        print("=" * 60)

        traceback.print_exc()

        return None


# ============================================================
# READ IMAGE
# ============================================================

def read_uploaded_image(file):

    if file is None:
        return None

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


# ============================================================
# READ MASK
# ============================================================

def read_mask(file, width, height):

    if file is None:
        return None

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

    if mask is None:
        return None

    # Make mask exactly same size as image
    mask = cv2.resize(
        mask,
        (width, height),
        interpolation=cv2.INTER_NEAREST
    )

    # White = remove
    # Black = keep
    mask = np.where(
        mask > 20,
        255,
        0
    ).astype(np.uint8)

    return mask


# ============================================================
# IMPROVE MASK
# ============================================================

def improve_mask(mask):

    # Close small holes
    kernel_close = np.ones(
        (5, 5),
        np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel_close,
        iterations=1
    )

    # Remove tiny noise
    kernel_open = np.ones(
        (3, 3),
        np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel_open,
        iterations=1
    )

    # Slight expansion around object edges
    kernel_dilate = np.ones(
        (3, 3),
        np.uint8
    )

    mask = cv2.dilate(
        mask,
        kernel_dilate,
        iterations=1
    )

    return mask


# ============================================================
# LAMA INPAINTING
# ============================================================

def run_lama(image, mask):

    model = get_lama()

    if model is None:

        raise RuntimeError(
            "LaMa is not available. "
            + str(lama_error)
        )

    # OpenCV BGR -> RGB
    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    pil_image = Image.fromarray(
        rgb
    )

    # Mask
    pil_mask = Image.fromarray(
        mask
    ).convert("L")

    print(
        "Running LaMa...",
        pil_image.size
    )

    # AI INPAINTING
    result = model(
        pil_image,
        pil_mask
    )

    # PIL -> numpy
    result_np = np.array(
        result
    )

    # RGB -> BGR
    result_bgr = cv2.cvtColor(
        result_np,
        cv2.COLOR_RGB2BGR
    )

    return result_bgr


# ============================================================
# CLASSIC FALLBACK
# ============================================================

def run_classic(image, mask):

    kernel = np.ones(
        (3, 3),
        np.uint8
    )

    mask = cv2.dilate(
        mask,
        kernel,
        iterations=1
    )

    result = cv2.inpaint(
        image,
        mask,
        5,
        cv2.INPAINT_TELEA
    )

    return result


# ============================================================
# HOME
# ============================================================

@app.route("/", methods=["GET"])
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# HEALTH
# ============================================================

@app.route("/health", methods=["GET"])
def health():

    try:

        import torch

        return jsonify({
            "status": "ok",
            "pytorch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": "cpu",
            "lama_loaded": lama is not None,
            "lama_error": lama_error
        })

    except Exception as e:

        return jsonify({
            "status": "ok",
            "lama_loaded": lama is not None,
            "lama_error": lama_error,
            "error": repr(e)
        })


# ============================================================
# PROCESS
# ============================================================

@app.route("/process", methods=["POST"])
def process():

    try:

        print("=" * 60)
        print("NEW IMAGE PROCESS REQUEST")
        print("=" * 60)

        # ----------------------------------------------------
        # IMAGE
        # ----------------------------------------------------

        image_file = request.files.get(
            "image"
        )

        if image_file is None:

            return jsonify({
                "success": False,
                "error": "Image is required."
            }), 400

        image = read_uploaded_image(
            image_file
        )

        if image is None:

            return jsonify({
                "success": False,
                "error": "Could not read image."
            }), 400

        height, width = image.shape[:2]

        print(
            f"Image size: {width} x {height}"
        )


        # ----------------------------------------------------
        # MASK
        # ----------------------------------------------------

        mask_file = request.files.get(
            "mask"
        )

        if mask_file is None:

            return jsonify({
                "success": False,
                "error": "Mask is required."
            }), 400

        mask = read_mask(
            mask_file,
            width,
            height
        )

        if mask is None:

            return jsonify({
                "success": False,
                "error": "Could not read mask."
            }), 400


        # ----------------------------------------------------
        # CHECK MASK
        # ----------------------------------------------------

        mask_pixels = cv2.countNonZero(
            mask
        )

        print(
            "Mask pixels:",
            mask_pixels
        )

        if mask_pixels == 0:

            return jsonify({
                "success": False,
                "error": "Mask is empty. Please select an object."
            }), 400


        # ----------------------------------------------------
        # CLEAN MASK
        # ----------------------------------------------------

        mask = improve_mask(
            mask
        )


        # ----------------------------------------------------
        # METHOD
        # ----------------------------------------------------

        method = request.form.get(
            "method",
            "ai"
        ).lower()

        print(
            "Method:",
            method
        )


        # ----------------------------------------------------
        # LAMA
        # ----------------------------------------------------

        if method in [
            "ai",
            "lama",
            "smart"
        ]:

            try:

                result = run_lama(
                    image,
                    mask
                )

                used_method = "lama"

            except Exception as e:

                print("=" * 60)
                print("LAMA PROCESSING ERROR")
                print(repr(e))
                print("=" * 60)

                traceback.print_exc()

                return jsonify({
                    "success": False,
                    "error": "LaMa processing failed.",
                    "details": repr(e)
                }), 500


        # ----------------------------------------------------
        # CLASSIC
        # ----------------------------------------------------

        elif method in [
            "classic",
            "opencv"
        ]:

            result = run_classic(
                image,
                mask
            )

            used_method = "classic"


        else:

            return jsonify({
                "success": False,
                "error": "Unknown method."
            }), 400


        # ----------------------------------------------------
        # KEEP ORIGINAL SIZE
        # ----------------------------------------------------

        if result.shape[:2] != image.shape[:2]:

            result = cv2.resize(
                result,
                (width, height),
                interpolation=cv2.INTER_LANCZOS4
            )


        # ----------------------------------------------------
        # ENCODE PNG
        # ----------------------------------------------------

        success, encoded = cv2.imencode(
            ".png",
            result
        )

        if not success:

            raise RuntimeError(
                "Could not encode result."
            )

        output = encoded.tobytes()


        print(
            "=" * 60
        )

        print(
            "SUCCESS"
        )

        print(
            "Method:",
            used_method
        )

        print(
            "Output:",
            width,
            "x",
            height
        )

        print(
            "=" * 60
        )


        # ----------------------------------------------------
        # SEND IMAGE
        # ----------------------------------------------------

        return send_file(
            io.BytesIO(output),
            mimetype="image/png",
            as_attachment=False,
            download_name="result.png"
        )


    except Exception as e:

        print("=" * 60)
        print("SERVER ERROR")
        print(repr(e))
        print("=" * 60)

        traceback.print_exc()

        return jsonify({
            "success": False,
            "error": repr(e)
        }), 500


# ============================================================
# FILE TOO LARGE
# ============================================================

@app.errorhandler(413)
def too_large(error):

    return jsonify({
        "success": False,
        "error": "Image is too large. Maximum size is 25 MB."
    }), 413


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    print("=" * 60)
    print("AI IMAGE ERASER")
    print("Flask server starting...")
    print("CPU MODE")
    print("Port:", port)
    print("=" * 60)

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
