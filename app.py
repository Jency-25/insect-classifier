import os
import json
import base64
from io import BytesIO
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from PIL import Image

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow all origins (Vercel frontend → Render backend)

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- Load ML Model and Labels ---
print("Loading model and labels...")
MODEL_PATH = 'model/insect_model.keras'
LABELS_PATH = 'model/labels.npy'

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    labels = np.load(LABELS_PATH)
    
    with open('insect_details.json', 'r', encoding='utf-8') as f:
        import json
        insect_db = json.load(f)

    
    # Try to determine target size from model input shape
    # E.g. (None, 224, 224, 3)
    input_shape = model.input_shape
    if input_shape and len(input_shape) >= 3 and input_shape[1] is not None:
        TARGET_SIZE = (input_shape[1], input_shape[2])
    else:
        TARGET_SIZE = (224, 224)  # Default fallback shape
        
    print(f"Model and labels loaded successfully! Expected image shape: {TARGET_SIZE}")
except Exception as e:
    print(f"--- ERROR LOADING MODEL ---")
    print(e)
    model = None
    labels = []
    TARGET_SIZE = (224, 224)


@app.route('/')
def home():
    """Serve the Home page."""
    import json
    return render_template('home.html', species_json=json.dumps(insect_db))

@app.route('/scanner')
def scanner():
    """Serve the Scanner interface page."""
    return render_template('scanner.html')

@app.route('/api/species')
def api_species():
    """Return the full species database as JSON."""
    from flask import jsonify
    return jsonify(insect_db)


@app.route('/predict', methods=['POST'])
def predict():
    """Handle image analysis."""
    if not model:
        return jsonify({'success': False, 'error': 'Model failed to load on the server.'}), 500

    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({'success': False, 'error': 'No valid image provided'}), 400

        # Decode base64 image coming from the frontend canvas
        image_data = data['image']
        if "base64," in image_data:
            image_data = image_data.split("base64,")[1]

        image_bytes = base64.b64decode(image_data)
        
        # Open with PIL
        img = Image.open(BytesIO(image_bytes))
        
        # Ensure it's 3-channel RGB (ignore alpha/transparency from WebRTC captures)
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        # Resize to network's expected spatial dimensions
        img = img.resize(TARGET_SIZE)
        
        # The model has a built-in Rescaling layer, so keep values in [0, 255]
        img_array = np.array(img).astype('float32')
        
        # Expand dims to match (batch_size, height, width, channels)
        img_array = np.expand_dims(img_array, axis=0)

        # Run inference
        predictions = model.predict(img_array)
        
        # Get highest probability index
        class_index = np.argmax(predictions[0])
        confidence_val = float(predictions[0][class_index])
        
        # Lookup label mapped from numpy list
        pred_class_name = str(labels[class_index])
        
        # Get details from DB
        details = insect_db.get(pred_class_name, "Details not found for this species.")
        
        return jsonify({
            'success': True,
            'prediction': pred_class_name,
            'confidence': confidence_val * 100,
            'details': details
        })

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    print("Starting Flask Server...")
    print("Access the app at: http://127.0.0.1:5000")
    # Using host 0.0.0.0 so it can be accessed on the local network (requires HTTPS for camera, though)
    app.run(host='0.0.0.0', port=5000, debug=False)
