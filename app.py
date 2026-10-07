from flask import Flask, render_template, request, send_file
from PIL import Image
import cv2
import numpy as np
import os
import uuid

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        return "No image uploaded", 400

    file = request.files["image"]

    if file.filename == "":
        return "No image selected", 400

    # Generate unique filename
    file_id = str(uuid.uuid4())
    input_path = os.path.join(UPLOAD_FOLDER, file_id + ".png")
    output_path = os.path.join(RESULT_FOLDER, file_id + "_result.png")

    # Save uploaded image
    image = Image.open(file).convert("RGB")
    image.save(input_path)

    # OpenCV processing
    img = cv2.imread(input_path)

    # Temporary test:
    # The uploaded image is currently returned unchanged.
    # Later we will add the actual object-removal algorithm here.
    cv2.imwrite(output_path, img)

    return send_file(
        output_path,
        mimetype="image/png",
        as_attachment=False
    )


if __name__ == "__main__":
    app.run(debug=True)
