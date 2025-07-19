import numpy as np
import cv2
import os
from flask import Flask, request, redirect, url_for, render_template_string, send_from_directory
from werkzeug.utils import secure_filename
import uuid

# --- Flask App Setup ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['RESULT_FOLDER'] = 'static/results/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB upload limit

# --- HTML Templates ---
# We define the HTML directly in the Python script for simplicity.

HTML_FORM = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <title>Image Colorizer</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; background-color: #f0f2f5; color: #333; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; }
        .container { background-color: #fff; padding: 2rem 3rem; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.1); text-align: center; }
        h1 { color: #d9534f; margin-bottom: 1.5rem; }
        .upload-btn-wrapper { position: relative; overflow: hidden; display: inline-block; }
        .btn { border: 2px solid #5cb85c; color: #5cb85c; background-color: white; padding: 10px 25px; border-radius: 8px; font-size: 16px; font-weight: bold; cursor: pointer; transition: all 0.3s ease; }
        .upload-btn-wrapper input[type=file] { font-size: 100px; position: absolute; left: 0; top: 0; opacity: 0; cursor: pointer; }
        .btn:hover { background-color: #5cb85c; color: white; }
        input[type=submit] { margin-top: 1rem; background-color: #5bc0de; color: white; border: none; padding: 12px 30px; border-radius: 8px; font-size: 16px; cursor: pointer; transition: background-color 0.3s ease; }
        input[type=submit]:hover { background-color: #31b0d5; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Colorize Your Grayscale Image</h1>
        <form method="post" action="/upload" enctype="multipart/form-data">
            <div class="upload-btn-wrapper">
                <button class="btn">Choose a file</button>
                <input type="file" name="file" onchange="form.submit()">
            </div>
        </form>
    </div>
</body>
</html>
"""

HTML_RESULT = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <title>Colorization Result</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; background-color: #333; color: #f0f2f5; margin: 0; padding: 2rem; }
        .container { max-width: 1200px; margin: 0 auto; }
        h1, h2 { text-align: center; color: #d9534f; }
        .images { display: flex; justify-content: space-around; align-items: flex-start; flex-wrap: wrap; gap: 2rem; margin-top: 2rem; }
        .image-card { background-color: #444; padding: 1rem; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.2); text-align: center; }
        .image-card img { max-width: 100%; height: auto; border-radius: 8px; max-height: 70vh; }
        .actions { text-align: center; margin-top: 2rem; }
        a { color: #5bc0de; text-decoration: none; font-size: 1.2rem; transition: color 0.3s ease; }
        a:hover { color: #31b0d5; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Colorization Result</h1>
        <div class="images">
            <div class="image-card">
                <h2>Original</h2>
                <img src="{{ url_for('static', filename=original_path) }}">
            </div>
            <div class="image-card">
                <h2>Colorized</h2>
                <img src="{{ url_for('static', filename=colorized_path) }}">
            </div>
        </div>
        <div class="actions">
            <a href="/">Try another image</a>
        </div>
    </div>
</body>
</html>
"""

# --- Colorization Logic ---
def colorize_image(image_path, output_path):
    """
    This function takes a path to a grayscale image, colorizes it using a
    pre-trained Caffe model, and saves the result.
    """
    try:
        # Get the absolute path to the directory where this script is located.
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Define model paths relative to the script's directory.
        prototxt_path = os.path.join(script_dir, 'models', 'colorization_deploy_v2.prototxt')
        caffemodel_path = os.path.join(script_dir, 'models', 'colorization_release_v2.caffemodel')
        points_path = os.path.join(script_dir, 'models', 'pts_in_hull.npy')

        # Load the Caffe model from disk.
        net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
        pts = np.load(points_path)

        # Add the cluster centers as 1x1 convolutions to the model.
        class8 = net.getLayerId("class8_ab")
        conv8 = net.getLayerId("conv8_313_rh")
        pts = pts.transpose().reshape(2, 313, 1, 1)
        net.getLayer(class8).blobs.append(pts)
        net.getLayer(conv8).blobs.append(np.full([1, 313], 2.606, dtype="float32"))

        # Load the input image, scale it, and convert to LAB color space.
        image = cv2.imread(image_path)
        if image is None:
            print(f"[ERROR] Could not read image from path: {image_path}")
            return False

        scaled = image.astype("float32") / 255.0
        lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
        resized = cv2.resize(lab, (224, 224))
        L = cv2.split(resized)[0]
        L -= 50

        # Predict the 'a' and 'b' channels.
        net.setInput(cv2.dnn.blobFromImage(L))
        ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
        ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

        # Reconstruct the image and save it.
        L = cv2.split(lab)[0]
        colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
        colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
        colorized = np.clip(colorized, 0, 1)
        colorized = (255 * colorized).astype("uint8")
        cv2.imwrite(output_path, colorized)
        return True

    except Exception as e:
        print(f"An error occurred during colorization: {e}")
        return False

# --- Web Routes ---
@app.route('/', methods=['GET'])
def index():
    """Renders the main upload page."""
    return render_template_string(HTML_FORM)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handles file upload, processing, and displays the result."""
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        # Generate a unique filename to avoid conflicts
        ext = os.path.splitext(file.filename)[1]
        unique_id = uuid.uuid4().hex
        
        original_filename = secure_filename(f"{unique_id}_original{ext}")
        colorized_filename = secure_filename(f"{unique_id}_colorized{ext}")

        input_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
        output_path = os.path.join(app.config['RESULT_FOLDER'], colorized_filename)
        
        file.save(input_path)

        # Run the colorization
        success = colorize_image(input_path, output_path)

        if success:
            # Prepare paths for the template (relative to the 'static' folder)
            original_url_path = os.path.join('uploads', original_filename)
            colorized_url_path = os.path.join('results', colorized_filename)
            
            return render_template_string(HTML_RESULT, 
                                          original_path=original_url_path, 
                                          colorized_path=colorized_url_path)
        else:
            return "Error during colorization.", 500

    return redirect('/')

if __name__ == '__main__':
    # Create necessary directories before running
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)
    # Run the app
    # For development, use debug=True. For production, use a proper WSGI server.
    app.run(debug=True)
