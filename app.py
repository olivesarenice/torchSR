import base64
import os
from io import BytesIO

import backend_model
from flask import Flask, jsonify, request
from PIL import Image

app = Flask(__name__)

UPLOAD_DIR = "./tmp/uploaded"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.route("/")
def home():
    return """
    <!DOCTYPE html>
    <html>
    <body>
        <h1>Image Upload and Inference</h1>
        <form id="upload-form" enctype="multipart/form-data">
            <label for="file">Choose a PNG file:</label>
            <input type="file" id="file" name="file" accept="image/*">
            <button type="button" id="upload-button">Upload and Process</button>
        </form>
        <div id="images-container" style="display: flex; gap: 20px; margin-top: 20px;">
            <div>
                <h3>Original Image</h3>
                <img id="original-image" style="max-width: 300px; border: 1px solid #000;" />
            </div>
            <div>
                <h3>Bilinear Resize</h3>
                <img id="bilinear-image" style="max-width: 300px; border: 1px solid #000;" />
            </div>


            <div>
                <h3>Lanczos Resize</h3>
                <img id="lanczos-image" style="max-width: 300px; border: 1px solid #000;" />
            </div>
            <div>
                <h3>Processed Image</h3>
                <img id="processed-image" style="max-width: 300px; border: 1px solid #000;" />
            </div>
        </div>

        <script>
            document.getElementById('upload-button').addEventListener('click', async () => {
                const formData = new FormData();
                const fileInput = document.getElementById('file');
                const file = fileInput.files[0];
                if (!file) {
                    alert('Please select a file before uploading.');
                    return;
                }
                formData.append('file', file);

                try {
                    const response = await fetch('/upload_and_process', {
                        method: 'POST',
                        body: formData
                    });
                    const data = await response.json();
                    if (data.original && data.processed && data.bilinear && data.lanczos) {
                        document.getElementById('original-image').src = `data:image/png;base64,${data.original}`;
                        document.getElementById('processed-image').src = `data:image/png;base64,${data.processed}`;
                        document.getElementById('bilinear-image').src = `data:image/png;base64,${data.bilinear}`;
                        document.getElementById('lanczos-image').src = `data:image/png;base64,${data.lanczos}`;
                    } else {
                        alert('Processing failed.');
                    }
                } catch (error) {
                    alert('Error uploading and processing image.');
                }
            });
        </script>
    </body>
    </html>
    """


@app.route("/upload_and_process", methods=["POST"])
def upload_and_process():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    filename = "file_name.png"
    filepath = os.path.join(UPLOAD_DIR, filename)

    # Convert to PNG if necessary
    if not file.filename.lower().endswith(".png"):
        img = Image.open(file)
        img.save(filepath, "PNG")
    else:
        file.save(filepath)

    # Load the uploaded image as base64
    with open(filepath, "rb") as img_file:
        original_base64 = base64.b64encode(img_file.read()).decode("utf-8")
        # Load the original image
    original_img = Image.open(filepath)

    # Nearest neighbor resizing
    nearest_img = original_img.resize(
        (original_img.width * 8, original_img.height * 8), Image.NEAREST
    )
    nearest_buffered = BytesIO()
    nearest_img.save(nearest_buffered, format="PNG")
    nearest_base64 = base64.b64encode(nearest_buffered.getvalue()).decode("utf-8")

    # Bilinear resizing
    bilinear_img = original_img.resize(
        (original_img.width * 8, original_img.height * 8), Image.BILINEAR
    )
    bilinear_buffered = BytesIO()
    bilinear_img.save(bilinear_buffered, format="PNG")
    bilinear_base64 = base64.b64encode(bilinear_buffered.getvalue()).decode("utf-8")

    # Lanczos resizing
    lanczos_img = original_img.resize(
        (original_img.width * 8, original_img.height * 8), Image.LANCZOS
    )
    lanczos_buffered = BytesIO()
    lanczos_img.save(lanczos_buffered, format="PNG")
    lanczos_base64 = base64.b64encode(lanczos_buffered.getvalue()).decode("utf-8")

    # Process the image using the backend model
    processed_base64 = backend_model.run_inference(original_base64, backend_model.MODEL)

    return jsonify(
        {
            "original": nearest_base64,
            "processed": processed_base64,
            "bilinear": bilinear_base64,
            "lanczos": lanczos_base64,
        }
    )


@app.route("/inference", methods=["POST"])
def inference():
    data = request.get_json()
    img_base64 = data.get("img_base64")
    if not img_base64:
        return "Invalid input", 400

    result_base64 = backend_model.run_inference(img_base64, backend_model.MODEL)
    return jsonify({"result": result_base64})


# def run_inference(img_base64):
#     # Decode the base64 string back to an image
#     img_data = base64.b64decode(img_base64)
#     img = Image.open(BytesIO(img_data))

#     # Placeholder for actual model inference (here we just convert to grayscale)
#     processed_img = img.convert("L")

#     # Convert the processed image back to base64
#     buffered = BytesIO()
#     processed_img.save(buffered, format="PNG")
#     result_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

#     return result_base64


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
