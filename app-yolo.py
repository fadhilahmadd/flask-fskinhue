from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import numpy as np
from ultralytics import YOLO
import stone
from PIL import Image
import cv2
from routes.categories import categories_blueprint
from routes.detail import detail_blueprint
import time  # Add this import for timestamp generation

app = Flask(__name__, static_folder='uploads')
CORS(app)

# YOLO v8
model_path = '/Users/fadhilahmad/Documents/filry/model yolo/runs/classify/train4/weights/best.pt'
model = YOLO(model_path)

def classify_image(image_path):
    results = model(image_path)
    names_dict = results[0].names
    probs = results[0].probs.data.tolist()
    predicted_class = names_dict[np.argmax(probs)]
    return predicted_class

def process_skin_tone(image_path):
    result = stone.process(image_path, image_type="color", return_report_image=True)
    report_images = result.pop("report_images")  # Obtain and remove the report image from the `result`
    return report_images, result

# Ensure the uploads folder exists
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/uploads/<path:filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/yolo', methods=['POST'])
def upload_files():
    if 'files' not in request.files:
        return jsonify({'error': 'No files uploaded'}), 400

    files = request.files.getlist('files')
    if not files:
        return jsonify({'error': 'No files uploaded'}), 400

    results = []

    for file in files:
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Save the uploaded file with a secure filename
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # Classify the image and process the skin tone
        predicted_class = classify_image(filepath)
        report_images, skin_tone_result = process_skin_tone(filepath)

        # Save the report image with a unique filename
        report_image_path = save_report_image(report_images, filename)

        result = {
            'class': predicted_class,
            'skin_tone_result': skin_tone_result,
            'report_image_path': report_image_path
        }

        results.append(result)

    return jsonify(results)

def save_report_image(report_images, original_filename):
    face_id = 1  # Assuming you want the first face's report image

    # Generate a unique filename using the current timestamp
    unique_suffix = str(int(time.time()))  # Or you can use uuid.uuid4() for UUID
    unique_filename = f'report_{unique_suffix}_{original_filename}'
    report_image_path = os.path.join(UPLOAD_FOLDER, unique_filename)
    
    # Convert the numpy array (BGR) to RGB before saving
    report_image_rgb = cv2.cvtColor(report_images[face_id], cv2.COLOR_BGR2RGB)
    report_image_pil = Image.fromarray(report_image_rgb)
    report_image_pil.save(report_image_path)
    
    return unique_filename

app.register_blueprint(categories_blueprint)
app.register_blueprint(detail_blueprint)

if __name__ == "__main__":
    app.run(host="192.168.26.16", debug=True)
