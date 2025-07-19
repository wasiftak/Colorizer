from flask import Flask, render_template, request, send_from_directory
import os
from main import colorize_image  # We use your refactored function

app = Flask(__name__)

# Configure upload and results folders
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
RESULT_FOLDER = os.path.join(os.getcwd(), 'results')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/colorize', methods=['POST'])
def colorize():
    if 'image' not in request.files:
        return "No file part", 400
    file = request.files['image']
    if file.filename == '':
        return "No selected file", 400

    # Save uploaded image
    input_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(input_path)

    # Create output filename
    base_name, ext = os.path.splitext(file.filename)
    output_filename = f"{base_name}_colorized{ext}"
    output_path = os.path.join(RESULT_FOLDER, output_filename)

    # Colorize
    colorize_image(input_path, output_path)

    return render_template('result.html', original_file=file.filename, colorized_file=output_filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/results/<filename>')
def result_file(filename):
    return send_from_directory(RESULT_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
