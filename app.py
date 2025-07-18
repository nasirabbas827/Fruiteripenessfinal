from flask import Flask, render_template, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io
import os

app = Flask(__name__)

# Load the trained model
model = load_model('best_model.h5')  # Replace with your model path

# Class labels
CLASS_NAMES = ["Unripe", "Ripe"]

def preprocess_image(image):
    """Resize and normalize the uploaded image."""
    image = image.resize((128, 128))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400
    
    try:
        # Read and preprocess the image
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        processed_image = preprocess_image(image)
        
        # Predict
        prediction = model.predict(processed_image)[0][0]
        class_idx = int(prediction > 0.5)
        confidence = float(prediction if class_idx == 1 else 1 - prediction)
        
        return jsonify({
            "class": CLASS_NAMES[class_idx],
            "confidence": f"{confidence:.2%}"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)