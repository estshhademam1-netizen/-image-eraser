import os

# =========================================================
# IMPORTANT:
# Render Free/CPU has no NVIDIA GPU.
# Force PyTorch to stay away from CUDA.
# =========================================================
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import io
import traceback

import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, send_file


# =========================================================
# Flask
# =========================================================

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25 MB


# =========================================================
# LaMa
# =========================================================

lama = None
lama_error = None


def load_lama():
    """
    Load LaMa explicitly on CPU.

    Render Free does not have an NVIDIA GPU.
    """

    global lama
    global lama_error

    if lama is not None:
        return lama

    try:
        import torch
        from simple_lama_inpainting import SimpleLama

        print("=" * 60)
        print("Loading LaMa...")
        print("PyTorch version:", torch.__version__)
        print("CUDA available:", torch.cuda.is_available())
        print("Using device: CPU")
        print("=" * 60)

        # FORCE CPU
        device = torch.device("cpu")

        lama = SimpleLama(device=device)

        print("=" * 60)
        print("LaMa loaded successfully on CPU!")
        print("=" * 60)

        lama_error = None

        return lama

    except Exception as e:
        lama = None
        lama_error = repr(e)

        print("=" * 60)
        print("LaMa could not be loaded:")
        print(repr(e))
        print("=" * 60)

        traceback.print_exc()

        return None


# =========================================================
# Image helpers
# =========================================================

def read_image(file_storage):
    """
    Read uploaded image into OpenCV BGR format.
    """

    if file_storage is None:
        return None

    data = file_storage.read()

    if not data:
        return None

    array = np.frombuffer(data, dtype=np.uint8)

    image = cv2.imdecode(array, cv2.IMREAD_COLOR)

    return image


def encode_image(image, extension=".png"):
    """
    Encode OpenCV image to bytes.
    """

    success, encoded = cv2.imencode(
        extension,
        image
    )

    if not success:
        raise ValueError("Could not encode output image.")

    return encoded.tobytes()


# =========================================================
# Mask preparation
# =========================================================

def prepare_mask(mask_file, target_size):
    """
    Prepare uploaded mask.

    White   = remove
    Black   = keep

    target_size = (width, height)
    """

    if mask_file is None:
        raise ValueError("Mask is required.")

    data = mask_file.read()

    if not data:
        raise ValueError("Mask file is empty.")

    array = np.frombuffer(data, dtype=np.uint8)

    mask = cv2.imdecode(
        array,
        cv2.IMREAD_GRAYSCALE
    )

    if mask is None:
        raise ValueError("Invalid mask image.")

    width, height = target_size

    # Resize mask to original image dimensions
    if mask.shape[1] != width or mask.shape[0] != height:

        mask = cv2.resize(
            mask,
            (width, height),
            interpolation=cv2.INTER_NEAREST
        )

    # Convert everything into a clean binary mask
    mask = np.where(
        mask > 20,
        255,
        0
    ).astype(np.uint8)

    return mask


# =========================================================
# Mask improvement
# =========================================================

def improve_mask(mask):
    """
    Clean small holes/noise in the mask.

    White = object to remove.
    """

    kernel_small = np.ones(
        (3, 3),
        np.uint8
    )

    kernel_medium = np.ones(
        (5, 5),
        np.uint8
    )

    # Remove tiny holes/noise
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel_medium,
        iterations=1
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel_small,
        iterations=1
    )

    # Small dilation helps avoid leaving object edges
    mask = cv2.dilate(
        mask,
        kernel_small,
        iterations=1
    )

    return mask


# =========================================================
# LaMa Inpainting
# =========================================================

def lama_inpaint(image, mask):
    """
    Run LaMa on CPU.
    """

    model = load_lama()

    if model is None:
        raise RuntimeError(
            "LaMa is not available. "
            + str(lama_error)
        )

    # OpenCV BGR -> PIL RGB
    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    pil_image = Image.fromarray(
        rgb
    )

    # Mask must be grayscale
    pil_mask = Image.fromarray(
        mask
    ).convert("L")

    print(
        "Running LaMa...",
        "Image:",
        pil_image.size
    )

    # Run model
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


# =========================================================
# Classic fallback
# =========================================================

def classic_inpaint(image, mask):
    """
    OpenCV fallback.
    """

    # Slightly enlarge mask to remove edge remnants
    kernel = np.ones(
        (3, 3),
        np.uint8
    )

    clean_mask = cv2.dilate(
        mask,
        kernel,
        iterations=1
    )

    # Telea
    telea = cv2.inpaint(
        image,
        clean_mask,
        5,
        cv2.INPAINT_TELEA
    )

    # Navier-Stokes
    ns = cv2.inpaint(
        image,
        clean_mask,
        5,
        cv2.INPAINT_NS
    )

    # Blend both
    result = cv2.addWeighted(
        telea,
        0.65,
        ns,
        0.35,
        0
    )

    return result


# =========================================================
# Health
# =========================================================

@app.route("/health", methods=["GET"])
def health():

    try:
        import torch

        return jsonify({
            "status": "ok",
            "service": "image-eraser",
            "pytorch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": "cpu",
            "lama_loaded": lama is not None,
            "lama_error": lama_error
        })

    except Exception as e:

        return jsonify({
            "status": "ok",
            "service": "image-eraser",
            "lama_loaded": lama is not None,
            "lama_error": lama_error,
            "error": repr(e)
        })


# =========================================================
# Home
# =========================================================

@app.route("/", methods=["GET"])
def home():

    return """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport"
              content="width=device-width, initial-scale=1.0">

        <title>AI Image Eraser</title>

        <style>
            body {
                font-family: Arial, sans-serif;
                padding: 30px;
                text-align: center;
            }

            h1 {
                margin-bottom: 10px;
            }

            p {
                color: #666;
            }

            .status {
                margin-top: 20px;
                padding: 15px;
                border-radius: 10px;
                background: #f3f3f3;
            }
        </style>
    </head>

    <body>

        <h1>AI Image Eraser</h1>

        <p>
            LaMa AI Inpainting Server
        </p>

        <div class="status">
            Backend is running.
        </div>

    </body>
    </html>
    """


# =========================================================
# Process
# =========================================================

@app.route("/process", methods=["POST"])
def process():

    try:

        # -------------------------------------------------
        # Get image
        # -------------------------------------------------

        image_file = request.files.get("image")

        if image_file is None:
            return jsonify({
                "success": False,
                "error": "Image is required."
            }), 400

        image = read_image(
            image_file
        )

        if image is None:
            return jsonify({
                "success": False,
                "error": "Could not read image."
            }), 400

        original_height, original_width = image.shape[:2]

        print(
            f"Input image: "
            f"{original_width}x{original_height}"
        )

        # -------------------------------------------------
        # Get mask
        # -------------------------------------------------

        mask_file = request.files.get("mask")

        if mask_file is None:
            return jsonify({
                "success": False,
                "error": "Mask is required."
            }), 400

        mask = prepare_mask(
            mask_file,
            (
                original_width,
                original_height
            )
        )

        # -------------------------------------------------
        # Improve mask
        # -------------------------------------------------

        mask = improve_mask(
            mask
        )

        # -------------------------------------------------
        # Method
        # -------------------------------------------------

        method = request.form.get(
            "method",
            "ai"
        ).lower()

        print(
            "Requested method:",
            method
        )

        # -------------------------------------------------
        # AI / LaMa
        # -------------------------------------------------

        if method in [
            "ai",
            "lama",
            "smart"
        ]:

            try:

                result = lama_inpaint(
                    image,
                    mask
                )

                used_method = "lama"

            except Exception as lama_exception:

                print("=" * 60)
                print("LaMa processing failed:")
                print(repr(lama_exception))
                print("=" * 60)

                traceback.print_exc()

                # Do NOT silently pretend classic is AI.
                return jsonify({
                    "success": False,
                    "error": "LaMa processing failed.",
                    "details": repr(lama_exception)
                }), 500

        # -------------------------------------------------
        # Classic
        # -------------------------------------------------

        elif method in [
            "classic",
            "opencv"
        ]:

            result = classic_inpaint(
                image,
                mask
            )

            used_method = "classic"

        else:

            return jsonify({
                "success": False,
                "error": (
                    "Unknown method. "
                    "Use 'ai' or 'classic'."
                )
            }), 400

        # -------------------------------------------------
        # Make sure output has original dimensions
        # -------------------------------------------------

        if (
            result.shape[1] != original_width
            or result.shape[0] != original_height
        ):

            result = cv2.resize(
                result,
                (
                    original_width,
                    original_height
                ),
                interpolation=cv2.INTER_LANCZOS4
            )

        # -------------------------------------------------
        # Encode
        # -------------------------------------------------

        output_bytes = encode_image(
            result,
            ".png"
        )

        print(
            "Processing completed:",
            used_method
        )

        # -------------------------------------------------
        # Return image
        # -------------------------------------------------

        return send_file(
            io.BytesIO(output_bytes),
            mimetype="image/png",
            as_attachment=False,
            download_name="result.png"
        )

    except Exception as e:

        print("=" * 60)
        print("PROCESS ERROR:")
        print(repr(e))
        print("=" * 60)

        traceback.print_exc()

        return jsonify({
            "success": False,
            "error": repr(e)
        }), 500


# =========================================================
# Error handlers
# =========================================================

@app.errorhandler(413)
def file_too_large(error):

    return jsonify({
        "success": False,
        "error": "Image is too large. Maximum size is 25 MB."
    }), 413


@app.errorhandler(500)
def internal_error(error):

    return jsonify({
        "success": False,
        "error": "Internal server error."
    }), 500


# =========================================================
# Start
# =========================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    print("=" * 60)
    print("AI IMAGE ERASER")
    print("Starting Flask server...")
    print("Device: CPU")
    print("Port:", port)
    print("=" * 60)

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
