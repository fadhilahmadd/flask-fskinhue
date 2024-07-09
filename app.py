from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import joblib
from skimage.io import imread
from skimage.transform import resize

from routes.categories import categories_blueprint
from routes.detail import detail_blueprint

app = Flask(__name__, static_folder='img')
CORS(app)

# CNN
MODEL_PATH = '/Users/fadhilahmad/Documents/filry/flask/model/Warna Kulit-Warna Kulit -64.75.h5'

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# K-means
kmeans_model = joblib.load('/Users/fadhilahmad/Documents/filry/flask/model/kmeans_model.joblib')
label_encoder = joblib.load('/Users/fadhilahmad/Documents/filry/flask/model/label_encoder.joblib')

# Image preprocessing function k-means
def preprocess_image(filepath):
    image = imread(filepath)
    image = resize(image, (224, 224), anti_aliasing=True)
    return image.flatten()

# Ensure the uploads folder exists
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/kmeans')
def kmeans():
    return render_template('kmeans.html')

@app.route('/img/<path:filename>')
def send_file(filename):
    return send_from_directory(app.static_folder, filename)

@app.route('/predict', methods=['POST'])
def predict():
    files = request.files.getlist('files')
    if not files:
        return "No files part in the request", 400
    
    results = []
    class_indices = {0: 'Tidak Diketahui', 1: 'Warm', 2: 'Neutral', 3: 'Cool'}
    
    for file in files:
        if file.filename == '':
            return "No selected file", 400
        
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        img = image.load_img(file_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        
        predictions = model.predict(img_array)
        predicted_class = class_indices[np.argmax(predictions)]
        confidence = np.max(predictions)
        
        result = {
            'class': predicted_class,
            'confidence': float(confidence)
        }
        results.append(result)
    
    return jsonify(results)

@app.route('/predict-kmeans', methods=['POST'])
def predict_kmeans():
    if 'files' not in request.files:
        return jsonify({'error': 'No files uploaded'}), 400

    files = request.files.getlist('files')
    if not files:
        return jsonify({'error': 'No files uploaded'}), 400

    results = []

    for file in files:
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Save the uploaded file
        filepath = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
        file.save(filepath)

        # Preprocess the image
        image = preprocess_image(filepath)
        image = np.expand_dims(image, axis=0)

        # Predict the class using K-means model
        pred = kmeans_model.predict(image)
        class_name = label_encoder.inverse_transform(pred)[0]

        results.append({
            'class': class_name
        })

    return jsonify(results)

app.register_blueprint(categories_blueprint)
app.register_blueprint(detail_blueprint)

if __name__ == "__main__":
    app.run(host="192.168.26.16",debug=True)
